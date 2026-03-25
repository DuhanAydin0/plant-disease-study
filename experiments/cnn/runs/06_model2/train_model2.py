


# ============================================================
# IMPORTS
# ============================================================
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torch.optim.lr_scheduler import ReduceLROnPlateau


# ============================================================
# CONFIG
# ============================================================
BATCH_SIZE = 32
MAX_EPOCHS = 30
#normalde bitkilerin LR değerlerini 1e-3 olarak train ettim.
#corn ve tomato bitkisi doyuma ulaşmadan kesildiği için onların train configini LR = 1e-5 olarak güncelledim
# ve ReduceLROnPlateau kullandım (corn ve tomato için)
PATIENCE = 5
LR = 1e-5
IMAGE_SIZE = 224
RANDOM_SEED = 42
NUM_WORKERS = 4

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(RANDOM_SEED)


# ============================================================
# MODEL
# ============================================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMAGE_SIZE // 8) * (IMAGE_SIZE // 8), 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# TRAIN / VALIDATE
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()

    return total_loss / total_samples, correct / total_samples


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    return total_loss / total_samples, correct / total_samples


# ============================================================
# MAIN
# ============================================================
def main(plant: str):
    DATA_DIR = Path(f"/Users/duhanaydin/cursor/plant disease study/data/processed/model2_{plant}")
    SAVE_DIR = Path(f"/Users/duhanaydin/cursor/plant disease study/experiments/cnn/results/06_model2/{plant}")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=transform)
    val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    num_classes = len(train_dataset.classes)

    print(f"\n==============================")
    print(f"MODEL-2 TRAINING STARTED")
    print(f"Plant: {plant}")
    print(f"Classes ({num_classes}): {train_dataset.classes}")
    print(f"Device: {DEVICE}")
    print(f"==============================\n")

    model = SimpleCNN(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # REDUCE LR ON PLATEAU
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,

    )

    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)

        print(
            f"Epoch [{epoch:02d}/{MAX_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} || "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} "
            f"| LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # LR STEP (EARLY STOPPING'DEN ÖNCE)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": train_dataset.class_to_idx
                },
                SAVE_DIR / "best_model.pth"
            )
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n Early stopping triggered. Best epoch: {best_epoch}\n")
                break

    metrics = {
        "plant": plant,
        "num_classes": num_classes,
        "class_names": train_dataset.classes,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss
    }

    with open(SAVE_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Model saved to: {SAVE_DIR / 'best_model.pth'}")
    print(f"Metrics saved to: {SAVE_DIR / 'metrics.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant", required=True)
    args = parser.parse_args()
    main(args.plant)
