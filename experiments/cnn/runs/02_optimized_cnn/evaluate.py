"""
Evaluation script for baseline CNN.

This script:
- Loads the trained model
- Evaluates ONLY on the test set
- Computes and prints detailed metrics:
  - Accuracy
  - Precision (macro)
  - Recall (macro)
  - F1-score (macro)
  - Confusion matrix

All metrics are printed explicitly and clearly labeled.
"""
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from model import CNN_02_Optimized
import config


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------
    # Paths
    # -----------------------------
    model_path = Path("experiments/cnn/results/02_optimized/cnn_02_optimized_20epochs_model.pth")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    # -----------------------------
    # Transforms
    # -----------------------------
    test_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # -----------------------------
    # Dataset & Loader
    # -----------------------------
    test_dataset = datasets.ImageFolder(
        root=config.DATA_DIR / "test",
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = CNN_02_Optimized(num_classes=config.NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # -----------------------------
    # Evaluation
    # -----------------------------
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # -----------------------------
    # Metrics
    # -----------------------------
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n===== Test Set Evaluation =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()
