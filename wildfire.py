import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from tfrecord.reader import tfrecord_loader
import numpy as np
from glob import glob
import itertools
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
import csv
import os
from sklearn.metrics import precision_score, recall_score, average_precision_score, precision_recall_curve, f1_score
import seaborn as sns
import pandas as pd

IMG_SHAPE = (64,64)
NUM_CLASSES = 2  # fire vs no-fire (uncertain is ignored)
input_keys = ['erc', 'tmmx', 'elevation', 'PrevFireMask', 'th', 'population',
              'pdsi', 'vs', 'NDVI', 'sph', 'pr', 'tmmn']
target_key = 'FireMask'

# If the batch size changes from 16, update these values accordingly
AVAILABLE_TRAINING_BATCHES = 937
AVAILABLE_VAL_BATCHES = 118
BATCH_SIZE = 16
BATCHES_BEFORE_PRINT = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# display n sample
def display_tfrecord(tfrecord_path, n=1):
    loader = tfrecord_loader(tfrecord_path, index_path=None)
    for i, record in enumerate(itertools.islice(loader, n), 1):
        print("Keys in TFRecord:", list(record.keys()))
        
        num_keys = len(record)
        rows = 1
        cols = int(np.ceil(num_keys / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3))

        for ax, (key, value) in zip(axes.flat, sorted(record.items())):
            try:
                arr = np.array(value, dtype=np.float32).reshape(IMG_SHAPE)
                ax.imshow(arr, cmap='viridis')
                ax.set_title(key)
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

# === Dataset ===
class WildfireTFRecordDataset(IterableDataset):
    def __init__(self, tfrecord_paths):
        self.paths = tfrecord_paths

    def __iter__(self):
        for path in self.paths:
            for record in tfrecord_loader(path, index_path=None):
                try:
                    input_images = [
                        np.array(record[k], dtype=np.float32).reshape(IMG_SHAPE)
                        for k in input_keys
                    ]
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

def get_data(train_path, val_path, max_train_batches, max_val_batches):
    train_paths = sorted(glob(train_path))
    val_paths = sorted(glob(val_path))

    train_loader = DataLoader(WildfireTFRecordDataset(train_paths), batch_size=BATCH_SIZE)
    val_loader = DataLoader(WildfireTFRecordDataset(val_paths), batch_size=BATCH_SIZE)

    num_training_batches = min(max_train_batches, AVAILABLE_TRAINING_BATCHES)
    num_validation_batches = min(max_val_batches, AVAILABLE_VAL_BATCHES)

    return train_loader, val_loader, num_training_batches, num_validation_batches

# === Train Function ===
def train(train_loader, val_loader, num_training_batches, num_validation_batches, lr, weight_decay, num_epochs, save_results=True):
    if save_results:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        directory = f"models/model_{timestamp}"
        os.makedirs(directory, exist_ok=True)
        model_save_path = directory + "/UNet.pth"
        training_log_path = directory + "/training_log.csv"
        training_loss_plot_path = directory + "/training_loss_plot.png"
        print(f"Model will be saved to: {model_save_path}")

        log_rows = [("epoch", "train_loss", "val_loss", "lr")]

    model = UNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # weighted cross entropy loss
    class_counts = np.array([6643368, 84331])
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()  # normalize
    weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
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

            if batch_idx % BATCHES_BEFORE_PRINT == 0:
                avg_loss = train_loss / count
                print(f"[Batch {batch_idx}/{num_training_batches}] Train Loss: {avg_loss:.4f}")
        train_loss /= count
        
        # === Validation ===
        model.eval()
        val_loss, count = 0.0, 0.0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(itertools.islice(val_loader, num_validation_batches)):
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                count += 1.0

                if batch_idx % BATCHES_BEFORE_PRINT == 0:
                    avg_loss = val_loss / count
                    print(f"[{batch_idx}/{num_validation_batches}] Val Loss: {avg_loss:.4f}")
        val_loss /= count
        
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)

        if save_results:
            current_lr = optimizer.param_groups[0]["lr"]
            log_rows.append((epoch, train_loss, val_loss, current_lr))

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
        plt.show()
        print(f"Saved loss plot to {training_loss_plot_path}")
    else:
        return best_val_loss
    
    return None

def grid_search(search_space, train_loader, val_loader, num_training_batches, num_validation_batches, num_epochs):
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    best_val_loss = float("inf")
    best_hparams = None
    results = []

    for lr in search_space["lr"]:
        for wd in search_space["weight_decay"]:
            print(f"\n--- Training with lr={lr}, weight_decay={wd} ---")
            best_trial_val_loss = train(train_loader, val_loader, num_training_batches, num_validation_batches, lr, wd, num_epochs, save_results=False)
            print(f"Final Val Loss for lr={lr}, wd={wd}: {best_trial_val_loss:.4f}")
            results.append({"lr": lr, "weight_decay": wd, "val_loss": best_trial_val_loss})

            if best_trial_val_loss < best_val_loss:
                best_val_loss = best_trial_val_loss
                best_hparams = {"lr": lr, "weight_decay": wd}
    
    print(f"Best hyperparameters: {best_hparams}")

    save_results_path = f"grid_search/results_{timestamp}"
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

def test(test_path, model_load_path):
    test_paths = sorted(glob(test_path))
    test_loader = DataLoader(WildfireTFRecordDataset(test_paths), batch_size=BATCH_SIZE)

    num_testing_batches = 0
    for batch in test_loader:
         num_testing_batches += 1

    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(model_load_path + "/UNet.pth", map_location=DEVICE))
    model.eval()

    print(f"Loaded model from {model_load_path}. Beginning testing on {num_testing_batches} batches...")

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    total_loss = 0.0
    batch_count = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
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
    threshold = 0.2
    pred_binary = (all_preds >= threshold).astype(int)

    test_loss_avg = total_loss / batch_count
    auc_pr = average_precision_score(all_labels, all_preds)
    precision = precision_score(all_labels, pred_binary)
    recall = recall_score(all_labels, pred_binary)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    # print("Max fire prob:", np.max(all_preds))
    # print("Mean fire prob:", np.mean(all_preds))

    # unique, counts = np.unique(all_labels, return_counts=True)
    # print(dict(zip(unique, counts)))

    # precision, recall, thresholds = precision_recall_curve(all_labels, all_preds)
    # f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    # best_idx = np.argmax(f1_scores)
    # best_threshold = thresholds[best_idx]
    # best_f1 = f1_scores[best_idx]

    print(f"\nTest Loss: {test_loss_avg:.4f}")
    print(f"AUC (PR):   {auc_pr:.4f}")
    # print(f"Best threshold: {best_threshold:.4f}")
    # print(f"F1 score: {best_f1:.4f}")
    # print(f"Precision: {precision[best_idx]:.4f}")
    # print(f"Recall: {recall[best_idx]:.4f}")
    print(f"F1 score: {f1_score:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

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
    parser.add_argument("mode", choices=["train", "test", "search"], help="Mode: train, test, or search")
    parser.add_argument("--model_path", type=str, required=False, help="Path to load model weights")
    parser.add_argument("--lr", type=float, required=False, help="Learning rate")
    parser.add_argument("--wd", type=float, required=False, help="Weight decay")
    parser.add_argument("--epochs", type=int, required=False, help="Number of epochs", default=30)
    parser.add_argument("--max_train_batches", type=int, required=False, help="Maximum number of training batches", default=100000)
    parser.add_argument("--max_val_batches", type=int, required=False, help="Maximum number of validation batches", default=100000)
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")

    if args.mode == "train" or args.mode == "search":
        train_path = "archive/next_day_wildfire_spread_train_*.tfrecord"
        val_path = "archive/next_day_wildfire_spread_eval_*.tfrecord"
        train_loader, val_loader, num_training_batches, num_validation_batches = get_data(train_path, val_path, args.max_train_batches, args.max_val_batches)
        print(f"Training on {num_training_batches} batches, validating on {num_validation_batches} batches, batch size {BATCH_SIZE}")

        if args.mode == "train":
            if args.lr is None or args.wd is None:
                parser.error("--lr and --wd are required when mode is 'train'")

            train(train_loader, val_loader, num_training_batches, num_validation_batches, args.lr, args.wd, args.epochs)
        elif args.mode == "search":
            search_space = {
                "lr": [1e-4, 1e-3, 1e-2],
                "weight_decay": [1e-5, 1e-3, 1e-2]
            }

            grid_search(search_space, train_loader, val_loader, num_training_batches, num_validation_batches, args.epochs)
    elif args.mode == "test":
        test_path = "archive/next_day_wildfire_spread_test_*.tfrecord"
        test(test_path, model_load_path=args.model_path)