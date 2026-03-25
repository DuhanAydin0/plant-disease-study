import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

from model import CNN_02_Optimized
import config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(dataset):
    """
    Compute class weights based on training set distribution.
    """
    targets = [label for _, label in dataset.samples]
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    return torch.tensor(class_weights, dtype=torch.float)


def main():
    # -----------------------------
    # Reproducibility
    # -----------------------------
    set_seed(config.RANDOM_SEED)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Using device: {device}")

    # -----------------------------
    # Paths
    # -----------------------------
    results_dir = Path("experiments/cnn/results/02_optimized")
    results_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = results_dir / "cnn_02_optimized_20epochs_model.pth"

    # -----------------------------
    # Transforms (same as baseline)
    # -----------------------------
    train_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # -----------------------------
    # Datasets & Loaders
    # -----------------------------
    train_dataset = datasets.ImageFolder(
        root=config.DATA_DIR / "train",
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=config.DATA_DIR / "val",
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    # -----------------------------
    # Model, Loss, Optimizer
    # -----------------------------
    model = CNN_02_Optimized(num_classes=config.NUM_CLASSES).to(device)

    class_weights = compute_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.3,
        patience=2,
        
    )

    # -----------------------------
    # Training Loop
    # -----------------------------
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # -----------------------------
        # Validation
        # -----------------------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        scheduler.step(val_loss)

        print(
            f"Epoch [{epoch + 1}/{config.NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current LR: {current_lr:.6f}")

    # -----------------------------
    # Save Model
    # -----------------------------
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    main()
