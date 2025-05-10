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

IMG_SHAPE = (64,64)
NUM_EPOCHS = 10
NUM_CLASSES = 2  # fire vs no-fire (uncertain is ignored)
NUM_TRAINING_EXAMPLES = 100
NUM_VALIDATION_EXAMPLES = 50
EXAMPLES_BEFORE_PRINT = 20
input_keys = ['erc', 'tmmx', 'elevation', 'PrevFireMask', 'th', 'population',
              'pdsi', 'vs', 'NDVI', 'sph', 'pr', 'tmmn']
target_key = 'FireMask'
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

# === Model ===
class FireSegNet(nn.Module):
    def __init__(self, in_channels=12, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)

# === Train Function ===
def train():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = f"models/fire_segnet_best_{timestamp}.pth"
    print(f"Model will be saved to: {model_save_path}")

    train_paths = sorted(glob("archive/next_day_wildfire_spread_train_*.tfrecord"))
    val_paths = sorted(glob("archive/next_day_wildfire_spread_eval_*.tfrecord"))

    # train_paths = sorted(glob("archive/next_day_wildfire_spread_train_00.tfrecord"))
    # val_paths = sorted(glob("archive/next_day_wildfire_spread_eval_00.tfrecord"))

    train_loader = DataLoader(WildfireTFRecordDataset(train_paths), batch_size=8)
    val_loader = DataLoader(WildfireTFRecordDataset(val_paths), batch_size=8)

    model = FireSegNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

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
                print(f"[Batch {batch_idx}/{NUM_TRAINING_EXAMPLES}] Train Loss: {loss.item():.4f}")
        
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
                    print(f"[Batch {batch_idx}/{NUM_VALIDATION_EXAMPLES}] Val Loss: {loss.item():.4f}")

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model at epoch {epoch} with val loss {val_loss:.4f}")

def test(model_load_path):
    test_paths = sorted(glob("archive/next_day_wildfire_spread_test_*.tfrecord"))
    test_loader = DataLoader(WildfireTFRecordDataset(test_paths), batch_size=8)

    model = FireSegNet().to(DEVICE)
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