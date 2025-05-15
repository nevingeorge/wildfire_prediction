import torch
import torch.nn as nn
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

IMG_SHAPE = (64,64)
NUM_CLASSES = 2  # fire vs no-fire (uncertain is ignored)
input_keys = ['erc', 'tmmx', 'elevation', 'PrevFireMask', 'th', 'population',
              'pdsi', 'vs', 'NDVI', 'sph', 'pr', 'tmmn']
target_key = 'FireMask'

# Training parameters
NUM_EPOCHS = 15
NUM_TRAINING_EXAMPLES = 2000
NUM_VALIDATION_EXAMPLES = 500
EXAMPLES_BEFORE_PRINT = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# display one sample
def display_tfrecord(tfrecord_path):
    for record in tfrecord_loader(tfrecord_path, index_path=None):
        print("Keys in TFRecord:", list(record.keys()))
        
        num_keys = len(record)
        cols = 4
        rows = int(np.ceil(num_keys / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))

        for ax, (key, value) in zip(axes.flat, record.items()):
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
        break  # Only process the first record

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

# === Train Function ===
def train():
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    directory = f"models/model_{timestamp}"
    os.makedirs(directory, exist_ok=True)
    model_save_path = directory + "/UNet.pth"
    training_log_path = directory + "/training_log.csv"
    training_loss_plot_path = directory + "/training_loss_plot.png"
    print(f"Model will be saved to: {model_save_path}")

    log_rows = [("epoch", "train_loss", "val_loss", "lr")]

    train_paths = sorted(glob("archive/next_day_wildfire_spread_train_*.tfrecord"))
    val_paths = sorted(glob("archive/next_day_wildfire_spread_eval_*.tfrecord"))

    # train_paths = sorted(glob("archive/next_day_wildfire_spread_train_00.tfrecord"))
    # val_paths = sorted(glob("archive/next_day_wildfire_spread_eval_00.tfrecord"))

    train_loader = DataLoader(WildfireTFRecordDataset(train_paths), batch_size=8)
    val_loader = DataLoader(WildfireTFRecordDataset(val_paths), batch_size=8)

    model = UNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    early_stop_patience = 5
    epochs_without_improvement = 0

    best_val_loss = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        # === Train ===
        model.train()
        train_loss = 0.0

        for batch_idx, (x, y) in enumerate(itertools.islice(train_loader, NUM_TRAINING_EXAMPLES)):
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % EXAMPLES_BEFORE_PRINT == 0:
                print(f"[{batch_idx}/{NUM_TRAINING_EXAMPLES}] Train Loss: {loss.item():.4f}")
        
        # === Validation ===
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(itertools.islice(val_loader, NUM_VALIDATION_EXAMPLES)):
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()

                if batch_idx % EXAMPLES_BEFORE_PRINT == 0:
                    print(f"[{batch_idx}/{NUM_VALIDATION_EXAMPLES}] Val Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        log_rows.append((epoch, train_loss, val_loss, current_lr))

        # Save best model and track early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model at epoch {epoch} with val loss {val_loss:.4f}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stop_patience:
            print("Early stopping: no improvement in validation loss.")
            break
    
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

def test(model_load_path):
    test_paths = sorted(glob("archive/next_day_wildfire_spread_test_*.tfrecord"))
    test_loader = DataLoader(WildfireTFRecordDataset(test_paths), batch_size=8)

    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(model_load_path, map_location=DEVICE))
    model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    total_loss = 0.0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()

    print(f"Test Loss: {total_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test wildfire segmentation model.")
    parser.add_argument("mode", choices=["train", "test"], help="Mode: train or test")
    parser.add_argument("--model_path", type=str, required=False, help="Path to load model weights")
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")

    if args.mode == "train":
        train()
    elif args.mode == "test":
        test(model_load_path=args.model_path)