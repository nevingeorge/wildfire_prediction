import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from tfrecord.reader import tfrecord_loader
import numpy as np
from glob import glob
import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import argparse
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
import csv
import os
from sklearn.metrics import precision_score, recall_score, average_precision_score, precision_recall_curve, f1_score
import seaborn as sns
import pandas as pd
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from tqdm import tqdm

IMG_SHAPE = (64,64)
NUM_CLASSES = 2  # fire vs no-fire (uncertain is ignored)
input_keys = ['erc', 'tmmx', 'elevation', 'PrevFireMask', 'th', 'population',
              'pdsi', 'vs', 'NDVI', 'sph', 'pr', 'tmmn']
target_key = 'FireMask'

# If the batch size changes from 16, update these values accordingly
AVAILABLE_TRAINING_BATCHES = 937
AVAILABLE_VAL_BATCHES = 118
BATCH_SIZE = 16
BATCHES_BEFORE_VAL = 200

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# weighted cross entropy loss
class_counts = np.array([6643368, 84331])
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()  # normalize
weights1 = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
weights2 = torch.tensor(np.array([0.1, 0.9]), dtype=torch.float32, device=DEVICE)
weights3 = torch.tensor(np.array([0.25, 0.75]), dtype=torch.float32, device=DEVICE)
weights4 = torch.tensor(np.array([0.5, 0.5]), dtype=torch.float32, device=DEVICE)
WEIGHT_OPTIONS = [weights1, weights2, weights3, weights4]

# display n sample
DISPLAY_ORDER = ['erc', 'sph', 'pr', 'tmmn', 'tmmx', 'NDVI', 'pdsi', 'vs', 'th', 'population', 'elevation', 'PrevFireMask', 'FireMask']
COL_TO_NAME_DICT = {
    'erc' : 'Energy Release Component', 
    'sph' : 'Specific Humidity', 
    'pr' : 'Precipitation Amount', 
    'tmmn' : 'Temperature Minimum', 
    'tmmx' : 'Temperature Maximum', 
    'NDVI' : 'Vegetation Index', 
    'pdsi' : 'Drought Severity Index', 
    'vs' : 'Wind Speed', 
    'th' : 'Wind Direction', 
    'population' : 'Population Density', 
    'elevation' : 'Elevation', 
    'PrevFireMask' : 'Fire Mask', 
    'FireMask' : 'Next Day Fire Mask'
}
def display_tfrecord(tfrecord_path, n=1):
    loader = tfrecord_loader(tfrecord_path, index_path=None)
    for i, record in enumerate(itertools.islice(loader, n), 1):
        print("Keys in TFRecord:", list(record.keys()))
        
        num_keys = len(record)
        rows = 1
        cols = int(np.ceil(num_keys / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3))

        for ax, key in zip(axes.flat, DISPLAY_ORDER):
            try:
                value = record[key]
                arr = np.array(value, dtype=np.float32).reshape(IMG_SHAPE)
                if 'FireMask' in key :
                    firemask_cmap = ListedColormap(['dimgray', 'lightgray', 'red'])
                    firemask_norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], firemask_cmap.N)
                    ax.imshow(arr, cmap=firemask_cmap, norm=firemask_norm)
                else :
                    ax.imshow(arr, cmap='viridis')

                ax.set_title(COL_TO_NAME_DICT[key].replace(' ', '\n'), fontsize=9)
                ax.axis('off')
            except Exception as e:
                ax.set_title(f"{key} (Error)")
                ax.text(0.5, 0.5, str(e), ha='center', va='center')
                ax.axis('off')

        # Hide extra axes
        for ax in axes.flat[len(record):]:
            ax.axis('off')

        plt.tight_layout()
        plt.show()

def display_tensor_record(x) :
    """
    Display each of the 12 input channels in a tensor like display_tfrecord.
    
    Args:
        x: Tensor of shape (1, 12, H, W)
    """
    x = x.squeeze(0).cpu().numpy()  # Shape: (12, H, W)

    input_keys = DISPLAY_ORDER[:-2]  # Remove PrevFireMask and FireMask
    rows = 1
    cols = len(input_keys)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3))

    for ax, key in zip(axes.flat, input_keys):
        idx = DISPLAY_ORDER.index(key)
        try:
            ax.imshow(x[idx], cmap='viridis')
            ax.set_title(COL_TO_NAME_DICT[key].replace(' ', '\n'), fontsize=9)
            ax.axis('off')
        except Exception as e:
            ax.set_title(f"{key} (Error)")
            ax.text(0.5, 0.5, str(e), ha='center', va='center')
            ax.axis('off')

    plt.tight_layout()
    plt.show()

def display_saliency_map(saliency, collapsed=False):
    """
    Display saliency maps in the same layout as display_tfrecord.
    
    Args:
        saliency: Tensor or ndarray of shape (12, H, W)
    """
    if collapsed:
        plt.figure(figsize=(6, 5))
        plt.imshow(saliency, cmap='hot')
        plt.colorbar()
        plt.title('Saliency')
        plt.axis('off')
        plt.show()
    else :
        saliency = saliency.numpy() if isinstance(saliency, torch.Tensor) else saliency
    
        rows = 1
        cols = len(DISPLAY_ORDER) - 2  # exclude PrevFireMask and FireMask
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3))
    
        for ax, key in zip(axes.flat, DISPLAY_ORDER[:-2]):  # skip FireMask/PrevFireMask
            idx = DISPLAY_ORDER.index(key)
            try:
                ax.imshow(saliency[idx], cmap='hot')
                ax.set_title(COL_TO_NAME_DICT[key].replace(' ', '\n'), fontsize=9)
                ax.axis('off')
            except Exception as e:
                ax.set_title(f"{key} (Error)")
                ax.text(0.5, 0.5, str(e), ha='center', va='center')
                ax.axis('off')
    
        plt.tight_layout()
        plt.show()

def smoothgrad(model, x, target_class=1, n_samples=50, noise_std=0.1, collapse=False):
    """
    Compute SmoothGrad saliency map.

    Args:
        model: PyTorch model
        x: Input tensor of shape (1, 12, H, W)
        target_class: class index to compute gradient with respect to
        n_samples: number of noisy samples
        noise_std: std of Gaussian noise added to input

    Returns:
        SmoothGrad saliency map of shape (H, W)
    """
    model.eval()
    x = x.clone().detach().to(DEVICE).requires_grad_(True)

    _, _, H, W = x.shape
    grads = torch.zeros_like(x).to(DEVICE)

    for i in range(n_samples):
        noise = torch.randn_like(x) * noise_std
        x_noisy = (x + noise).clamp(0, 1)
        x_noisy.requires_grad_()
        x_noisy.retain_grad()

        output = model(x_noisy)  # (1, C, H, W)
        score = output[0, target_class].mean()
        model.zero_grad()
        score.backward()

        grads += x_noisy.grad.data

    grads /= n_samples
    if collapse:
        saliency = grads.abs().sum(dim=1).squeeze().cpu().numpy()  # shape (H, W)
    else :
        saliency = grads.abs().squeeze().cpu().numpy()  # shape (C, H, W)

    return saliency

# === Dataset ===
class WildfireTFRecordDataset(IterableDataset):
    def __init__(self, tfrecord_paths):
        self.paths = tfrecord_paths
        self.input_keys = DISPLAY_ORDER[:-1]
        self.target_key = target_key

    def __iter__(self):
        for path in self.paths:
            for record in tfrecord_loader(path, index_path=None):
                try:
                    input_images = []
                    for key in self.input_keys:
                        if key not in record:
                            raise ValueError(f"Missing key: {key}")
                        arr = np.array(record[key], dtype=np.float32).reshape(IMG_SHAPE)
                        input_images.append(arr)

                    x = np.stack(input_images, axis=0)  # (12, 64, 64)

                    y = np.array(record[target_key], dtype=np.uint8).reshape(IMG_SHAPE)

                    yield torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

                except Exception as e:
                    print(f"Skipping record due to error: {e}")
                    continue

class UNet(nn.Module):
    def __init__(self, in_channels=12, num_classes=2, dropout=0.3):
        super(UNet, self).__init__()

        # --- Encoder ---
        self.enc1 = self.conv_block(in_channels, 32, dropout)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = self.conv_block(32, 64, dropout)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = self.conv_block(64, 128, dropout)
        self.pool3 = nn.MaxPool2d(2)

        # --- Bottleneck ---
        self.bottleneck = self.conv_block(128, 256, dropout)

        # --- Decoder ---
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256, 128, dropout)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 64, dropout)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(64, 32, dropout)

        # --- Output ---
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels, dropout):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # 64x64
        e2 = self.enc2(self.pool1(e1))  # 32x32
        e3 = self.enc3(self.pool2(e2))  # 16x16

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))  # 8x8

        # Decoder
        d3 = self.up3(b)  # 8x8 → 16x16
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)  # 16x16 → 32x32
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)  # 32x32 → 64x64
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.classifier(d1)

class SegFormerB0(nn.Module):
    def __init__(self, in_channels=12, num_classes=2, image_size=64):
        super().__init__()

        # MiT-B0 backbone
        config = SegformerConfig(
            num_labels=num_classes,
            image_size=image_size,
            hidden_sizes=[32, 64, 160, 256],
            depths=[2, 2, 2, 2],
            attention_heads=[1, 2, 5, 8],
            decoder_hidden_size=256,
            classifier_dropout_prob=0.1,
            hidden_act="gelu",
            initializer_range=0.02
        )

        # Create model from scratch (no pretraining)
        self.model = SegformerForSemanticSegmentation(config)

        # Replace first Conv2D to accept in_channels (e.g., 12 instead of 3)
        self.model.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(
            in_channels,
            config.hidden_sizes[0],
            kernel_size=7,
            stride=4,
            padding=3
        )

        # Init input layer
        nn.init.kaiming_normal_(self.model.segformer.encoder.patch_embeddings[0].proj.weight, nonlinearity='relu')

        self.output_size = (image_size, image_size)

    def forward(self, x):
        logits = self.model(pixel_values=x).logits  # shape: (B, C, H_out, W_out)

        # Upsample to match label resolution (64x64)
        if logits.shape[-2:] != self.output_size:
            logits = nn.functional.interpolate(
                logits,
                size=self.output_size,
                mode="bilinear",
                align_corners=False
            )

        return logits  # (B, num_classes, 64, 64)
    
class SegFormerB1(nn.Module):
    def __init__(self, in_channels=12, num_classes=2, image_size=64):
        super().__init__()

        # MiT-B1 backbone
        config = SegformerConfig(
            num_labels=num_classes,
            image_size=image_size,
            hidden_sizes=[64, 128, 320, 512],
            depths=[2, 2, 2, 2],
            attention_heads=[1, 2, 5, 8],
            decoder_hidden_size=256,
            classifier_dropout_prob=0.1,
            hidden_act="gelu",
            initializer_range=0.02
        )

        # Create model from scratch (no pretraining)
        self.model = SegformerForSemanticSegmentation(config)

        # Replace first Conv2D to accept in_channels (e.g., 12 instead of 3)
        self.model.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(
            in_channels,
            config.hidden_sizes[0],
            kernel_size=7,
            stride=4,
            padding=3
        )

        # Init input layer
        nn.init.kaiming_normal_(self.model.segformer.encoder.patch_embeddings[0].proj.weight, nonlinearity='relu')

        self.output_size = (image_size, image_size)

    def forward(self, x):
        logits = self.model(pixel_values=x).logits  # shape: (B, C, H_out, W_out)

        # Upsample to match label resolution (64x64)
        if logits.shape[-2:] != self.output_size:
            logits = nn.functional.interpolate(
                logits,
                size=self.output_size,
                mode="bilinear",
                align_corners=False
            )

        return logits  # (B, num_classes, 64, 64)
    
def get_data(train_path, val_path, max_train_batches, max_val_batches):
    train_paths = sorted(glob(train_path))
    val_paths = sorted(glob(val_path))

    train_loader = DataLoader(WildfireTFRecordDataset(train_paths), batch_size=BATCH_SIZE)
    val_loader = DataLoader(WildfireTFRecordDataset(val_paths), batch_size=BATCH_SIZE)

    num_training_batches = min(max_train_batches, AVAILABLE_TRAINING_BATCHES)
    num_validation_batches = min(max_val_batches, AVAILABLE_VAL_BATCHES)

    return train_loader, val_loader, num_training_batches, num_validation_batches

def get_model(model_type):
    if model_type == "UNet":
        return UNet().to(DEVICE)
    elif model_type == "SegFormerB0":
        return SegFormerB0().to(DEVICE)
    elif model_type == "SegFormerB1":
        return SegFormerB1().to(DEVICE)
    else:
        raise ValueError("Invalid model type. Choose 'UNet', 'SegFormerB0', or 'SegFormerB1'.")

def get_val_loss(model, criterion, val_loader, num_validation_batches):
    model.eval()
    val_loss, count = 0.0, 0.0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(itertools.islice(val_loader, num_validation_batches)):
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            val_loss += loss.item()
            count += 1.0

    val_loss /= count
    return val_loss

def log_results(model, criterion, val_loader, num_validation_batches, avg_train_loss, batch_idx, epoch, num_training_batches, optimizer, log_rows, save_results):
    val_loss = get_val_loss(model, criterion, val_loader, num_validation_batches)
    epoch_exact = round((epoch - 1) + (batch_idx + 1) / num_training_batches, 2)
    print(f"[Epoch {epoch_exact}] Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

    if save_results:
        current_lr = optimizer.param_groups[0]["lr"]
        log_rows.append((epoch_exact, avg_train_loss, val_loss, current_lr))
    return val_loss

def train(model_type, train_loader, val_loader, num_training_batches, num_validation_batches, lr, weight_decay, num_epochs, weights, save_results=True):
    log_rows = None

    if save_results:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        directory = f"models/model_{timestamp}"
        os.makedirs(directory, exist_ok=True)
        model_save_path = f"{directory}/{model_type}.pth"
        training_log_path = f"{directory}/training_log.csv"
        training_loss_plot_path = f"{directory}/training_loss_plot.png"
        print(f"Model will be saved to: {model_save_path}")

        log_rows = [("epoch", "train_loss", "val_loss", "lr")]

    model = get_model(model_type)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=255)

    early_stop_patience = 5
    epochs_without_improvement = 0

    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        # === Train ===
        model.train()
        train_loss, count = 0.0, 0.0

        for batch_idx, (x, y) in enumerate(itertools.islice(train_loader, num_training_batches)):
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            count += 1.0

            if batch_idx % BATCHES_BEFORE_VAL == BATCHES_BEFORE_VAL - 1:
                _ = log_results(model, criterion, val_loader, num_validation_batches, train_loss / count, 
                            batch_idx, epoch, num_training_batches, optimizer, log_rows, save_results)
                model.train()

        val_loss = log_results(model, criterion, val_loader, num_validation_batches, train_loss / count, 
                            batch_idx, epoch, num_training_batches, optimizer, log_rows, save_results)
        scheduler.step(val_loss)

        # Save best model and track early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            if save_results:
                torch.save(model.state_dict(), model_save_path)
                print(f"Saved new best model at epoch {epoch} with val loss {val_loss:.4f}")

            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stop_patience:
            print("Early stopping: no improvement in validation loss.")
            break
    
    if save_results:
        with open(training_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(log_rows)
        print(f"Saved training log to {training_log_path}")

        # Extract values
        epochs = [row[0] for row in log_rows[1:]]
        train_losses = [row[1] for row in log_rows[1:]]
        val_losses = [row[2] for row in log_rows[1:]]

        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(training_loss_plot_path)
        print(f"Saved loss plot to {training_loss_plot_path}")
    else:
        return model, best_val_loss
    
    return None, None

def grid_search(model_type, search_space, train_loader, val_loader, num_training_batches, num_validation_batches, num_epochs, weights):
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    best_val_loss = float("inf")
    best_hparams = None
    results = []

    for lr in search_space["lr"]:
        for wd in search_space["weight_decay"]:
            print(f"\n--- Training with lr={lr}, weight_decay={wd} ---")
            _, best_trial_val_loss = train(model_type, train_loader, val_loader, num_training_batches, num_validation_batches, lr, wd, num_epochs, weights, save_results=False)
            print(f"Final Val Loss for lr={lr}, wd={wd}: {best_trial_val_loss:.4f}")
            results.append({"lr": lr, "weight_decay": wd, "val_loss": best_trial_val_loss})

            if best_trial_val_loss < best_val_loss:
                best_val_loss = best_trial_val_loss
                best_hparams = {"lr": lr, "weight_decay": wd}
    
    print(f"Best hyperparameters: {best_hparams}")

    save_results_path = f"grid_search/results_{model_type}_{timestamp}"
    os.makedirs(save_results_path, exist_ok=True)

    # === Save results to txt file ===
    results_txt_path = os.path.join(save_results_path, "gridsearch_results.txt")
    with open(results_txt_path, "w") as f:
        f.write("Grid Search Results:\n")
        for result in results:
            f.write(f"lr: {result['lr']}, weight_decay: {result['weight_decay']}, val_loss: {result['val_loss']:.4f}\n")
        f.write(f"\nBest Hyperparameters:\n{best_hparams}\n")
        f.write(f"Best Validation Loss: {best_val_loss:.4f}\n")
    print(f"Saved grid search results to {results_txt_path}")

    # === Plot heatmap ===
    results_df = pd.DataFrame(results)
    heatmap_data = results_df.pivot(index="lr", columns="weight_decay", values="val_loss")

    vmin = max(heatmap_data.min().min(), 0)
    vmax = min(heatmap_data.max().max(), 1)

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="viridis_r", vmin=vmin, vmax=vmax)
    plt.title("Grid Search Validation Loss")
    plt.xlabel("Weight Decay")
    plt.ylabel("Learning Rate")
    plt.tight_layout()
    heatmap_path = os.path.join(save_results_path, "gridsearch_heatmap.png")
    plt.savefig(heatmap_path)
    print(f"Saved heatmap to {heatmap_path}")

def get_predictions(model, loader, weights, maxsamples=None):
    model.eval()
    total_loss = 0.0
    batch_count = 0
    all_preds = []
    all_labels = []
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=255)

    with torch.no_grad():
        if maxsamples is not None:
            iterator = enumerate(tqdm(loader, desc="Evaluating", unit="batch", total=maxsamples))
        else:
            iterator = enumerate(tqdm(loader, desc="Evaluating", unit="batch"))
        for batch_idx, (x, y) in iterator:
            if maxsamples is not None and batch_idx >= maxsamples:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)  # y: (B, H, W)
            logits = model(x)  # (B, C, H, W)
            loss = criterion(logits, y)
            total_loss += loss.item()
            batch_count += 1

            # Get probabilities for class 1 (fire)
            probs = torch.softmax(logits, dim=1)[:, 1, :, :]  # (B, H, W)

            # Flatten and mask out uncertain labels
            y_flat = y.reshape(-1)
            p_flat = probs.reshape(-1)

            mask = y_flat != 255
            y_valid = y_flat[mask].cpu().numpy()
            p_valid = p_flat[mask].cpu().numpy()

            all_labels.extend(y_valid)
            all_preds.extend(p_valid)

    # Convert to NumPy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    loss_avg = total_loss / batch_count
    return all_labels, all_preds, loss_avg

def get_statistics(model, loader, weights, threshold=0.5, maxsamples=None):
    all_labels, all_preds, test_loss_avg = get_predictions(model, loader, weights, maxsamples)

    pred_binary = (all_preds >= threshold).astype(int)
    auc_pr = average_precision_score(all_labels, all_preds)
    precision = precision_score(all_labels, pred_binary)
    recall = recall_score(all_labels, pred_binary)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    # print("Max fire prob:", np.max(all_preds))
    # print("Mean fire prob:", np.mean(all_preds))
    # unique, counts = np.unique(all_labels, return_counts=True)
    # print(dict(zip(unique, counts)))

    print(f"Loss: {test_loss_avg:.4f}, AUC (PR): {auc_pr:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")

    return (test_loss_avg, auc_pr, precision, recall, f1_score)

def loss_function_search(model_type, train_loader, val_loader, num_training_batches, num_validation_batches, num_epochs, lr, weight_decay):
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    best_auc_pr = float("-inf")
    best_weight_option = None
    results = []

    for i in range(len(WEIGHT_OPTIONS)):
        weights = WEIGHT_OPTIONS[i]
        print(f"\n--- Training with weights={weights} ---")
        model, _ = train(model_type, train_loader, val_loader, num_training_batches, num_validation_batches, lr, weight_decay, num_epochs, weights, save_results=False)
        val_loss, auc_pr, precision, recall, f1_score = get_statistics(model, val_loader, weights)
        results.append({"weight_option": i, "val_loss": val_loss, "auc_pr": auc_pr, "precision": precision, "recall": recall, "f1_score": f1_score})

        if auc_pr > best_auc_pr:
            best_auc_pr = auc_pr
            best_weight_option = i
    
    print(f"Best weight option: {best_weight_option}")

    save_results_path = f"loss_function_search/results_{model_type}_{timestamp}"
    os.makedirs(save_results_path, exist_ok=True)

    # === Save results to txt file ===
    results_txt_path = os.path.join(save_results_path, "loss_function_search_results.txt")
    with open(results_txt_path, "w") as f:
        f.write("Loss Function Search Results:\n")
        for result in results:
            f.write(f"weight_option: {result['weight_option']}, val_loss: {result['val_loss']:.4f}, auc_pr: {result['auc_pr']:.4f}, precision: {result['precision']:.4f}, recall: {result['recall']:.4f}, f1_score: {result['f1_score']:.4f}\n")
        f.write(f"\nBest weight option: {best_weight_option}\n")
        f.write(f"Best AUC (PR): {best_auc_pr:.4f}\n")
    print(f"Saved loss function search results to {results_txt_path}")

def get_loader_model(model_type, path, model_load_path, weights):
    paths = sorted(glob(path))
    loader = DataLoader(WildfireTFRecordDataset(paths), batch_size=BATCH_SIZE)

    num_batches = 0
    for batch in loader:
         num_batches += 1

    model = get_model(model_type)
    model.load_state_dict(torch.load(f"{model_load_path}/{model_type}.pth", map_location=DEVICE))
    print(f"Loaded model from {model_load_path}. Beginning testing on {num_batches} batches with loss function weights {weights}...")
    return loader, model

def threshold_search(model_type, val_path, model_load_path, weights):
    val_loader, model = get_loader_model(model_type, val_path, model_load_path, weights)
    all_labels, all_preds, _ = get_predictions(model, val_loader, weights)
    
    save_results_path = f"{model_load_path}/threshold_search_results.txt"
    with open(save_results_path, "w") as f:
        precision, recall, thresholds = precision_recall_curve(all_labels, all_preds)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)  # Add epsilon to avoid divide-by-zero

        best_index = np.argmax(f1_scores)
        best_threshold = thresholds[best_index]
        best_f1 = f1_scores[best_index]

        print(f"Best threshold: {best_threshold:.4f}")
        print(f"Best F1 score: {best_f1:.4f}")
        f.write(f"\nBest Threshold: {best_threshold:.4f}, Best F1 score: {best_f1:.4f}\n")
    print(f"Results saved to {save_results_path}")

def test(model_type, test_path, model_load_path, weights, threshold=0.5):
    test_loader, model = get_loader_model(model_type, test_path, model_load_path, weights)

    test_loss_avg, auc_pr, precision, recall, f1_score = get_statistics(model, test_loader, weights, threshold)

    save_results_path = f"{model_load_path}/test_results.txt"
    with open(save_results_path, "w") as f:
        f.write(f"Test Loss: {test_loss_avg:.4f}\n")
        f.write(f"AUC (PR):   {auc_pr:.4f}\n")
        f.write(f"F1 score: {f1_score:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")

    print(f"Test results saved to {save_results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test wildfire segmentation model.")
    parser.add_argument("mode", choices=["train", "test", "search", "loss", "threshold"], help="Mode: train, test, search, loss, or threshold")
    parser.add_argument("--model_type", type=str, required=False, help="UNet, SegFormerB0, or SegFormerB1", default="UNet")
    parser.add_argument("--model_path", type=str, required=False, help="Path to load model weights")
    parser.add_argument("--lr", type=float, required=False, help="Learning rate", default=1e-3)
    parser.add_argument("--wd", type=float, required=False, help="Weight decay", default=1e-2)
    parser.add_argument("--epochs", type=int, required=False, help="Number of epochs", default=30)
    parser.add_argument("--weights", type=int, required=False, help="Weight option: 0 (inverse class counts), 1 (0.1, 0.9), 2 (0.25, 0.75), or 3 (0.5, 0.5)", default=3)
    parser.add_argument("--threshold", type=float, required=False, help="Threshold", default=0.5)
    parser.add_argument("--max_train_batches", type=int, required=False, help="Maximum number of training batches", default=100000)
    parser.add_argument("--max_val_batches", type=int, required=False, help="Maximum number of validation batches", default=100000)
    args = parser.parse_args()

    weights = WEIGHT_OPTIONS[args.weights]

    train_path = "archive/next_day_wildfire_spread_train_*.tfrecord"
    val_path = "archive/next_day_wildfire_spread_eval_*.tfrecord"
    test_path = "archive/next_day_wildfire_spread_test_*.tfrecord"
    print(f"Model selection: {args.model_type}\n")

    if args.mode == "train" or args.mode == "search" or args.mode == "loss":
        train_loader, val_loader, num_training_batches, num_validation_batches = get_data(train_path, val_path, args.max_train_batches, args.max_val_batches)
        print(f"Training on {num_training_batches} batches, validating on {num_validation_batches} batches, batch size {BATCH_SIZE}")

        if args.mode == "train":
            print(f"Model will be trained for {args.epochs} epochs with learning rate {args.lr}, weight decay {args.wd}, and loss function weights {weights}")
            print("\n--------------------------------------------------------------------\n")
            train(args.model_type, train_loader, val_loader, num_training_batches, num_validation_batches, args.lr, args.wd, args.epochs, weights)
        elif args.mode == "search":
            search_space = {
                "lr": [1e-4, 1e-3, 1e-2],
                "weight_decay": [1e-5, 1e-3, 1e-2]
            }

            print(f"Starting grid search with {args.epochs} epochs, loss function weights {weights}, and hyperparameter space: {search_space}")
            grid_search(args.model_type, search_space, train_loader, val_loader, num_training_batches, num_validation_batches, args.epochs, weights)
        elif args.mode == "loss":
            print(f"Starting loss function search with {args.epochs} epochs, learning rate {args.lr}, and weight decay {args.wd}")
            loss_function_search(args.model_type, train_loader, val_loader, num_training_batches, num_validation_batches, args.epochs, args.lr, args.wd)
    elif args.mode == "threshold":
        print(f"Starting threshold search for model at {args.model_path} with loss function weights {weights}")
        threshold_search(args.model_type, val_path, args.model_path, weights)
    elif args.mode == "test":
        print(f"Starting test for model at {args.model_path} with loss function weights {weights} and threshold {args.threshold}")
        test(args.model_type, test_path, args.model_path, weights, args.threshold)