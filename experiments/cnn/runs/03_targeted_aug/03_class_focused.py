import random
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# -------------------------------------------------
# Path setup
# -------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
RUNS_DIR = CURRENT_FILE.parents[1]          # experiments/cnn/runs
ALL_DATASET_DIR = RUNS_DIR / "03_all_dataset"

sys.path.append(str(ALL_DATASET_DIR))

from cnn_03_all_model import CNN_03_All_Dataset
import config

# -------------------------------------------------
# Config
# -------------------------------------------------
FOCUS_EPOCHS = 6
FOCUS_LR = 1e-4

FOCUS_CLASSES = {
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Tomato___Early_blight",
    "Potato___healthy",
}

# -------------------------------------------------
# Reproducibility
# -------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -------------------------------------------------
# Targeted Dataset Wrapper
# -------------------------------------------------
class TargetedAugDataset(Dataset):
    def __init__(self, base_dataset, focus_classes, base_transform, focus_transform):
        self.dataset = base_dataset
        self.focus_classes = focus_classes
        self.base_transform = base_transform
        self.focus_transform = focus_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[idx]
        image = self.dataset.loader(path)
        class_name = self.dataset.classes[label]

        if class_name in self.focus_classes:
            image = self.focus_transform(image)
        else:
            image = self.base_transform(image)

        return image, label

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    set_seed(config.RANDOM_SEED)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # -------------------------------------------------
    # Transforms
    # -------------------------------------------------
    base_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    focus_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # -------------------------------------------------
    # Datasets
    # -------------------------------------------------
    train_base = datasets.ImageFolder(
        root=config.DATA_DIR / "train",
        transform=None
    )

    val_dataset = datasets.ImageFolder(
        root=config.DATA_DIR / "val",
        transform=base_transform
    )

    train_dataset = TargetedAugDataset(
        base_dataset=train_base,
        focus_classes=FOCUS_CLASSES,
        base_transform=base_transform,
        focus_transform=focus_transform
    )

    # -------------------------------------------------
    # Loaders
    # -------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    # -------------------------------------------------
    # Model
    # -------------------------------------------------
    model = CNN_03_All_Dataset(
        num_classes=config.NUM_CLASSES
    ).to(device)

    model.load_state_dict(
        torch.load(config.MODEL_SAVE_PATH, map_location=device)
    )

    # 🔒 Freeze backbone
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    # -------------------------------------------------
    # Loss & Optimizer
    # -------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FOCUS_LR
    )

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------
    for epoch in range(FOCUS_EPOCHS):
        model.train()
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        print(
            f"[Targeted Aug] Epoch [{epoch+1}/{FOCUS_EPOCHS}] "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
        )

    # -------------------------------------------------
    # Save
    # -------------------------------------------------
    save_path = Path("experiments/cnn/results/03_targeted_aug")
    save_path.mkdir(parents=True, exist_ok=True)

    torch.save(
        model.state_dict(),
        save_path / "cnn_03_targeted_aug_model.pth"
    )

    print("Targeted augmentation model saved.")

# -------------------------------------------------
if __name__ == "__main__":
    main()
