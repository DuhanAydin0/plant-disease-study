# experiments/cnn/runs/06_model2/evaluate_model2.py

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from train_model2 import SimpleCNN  # aynı modeli reuse ediyoruz


def main(plant: str):
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    base_data = Path("data/processed") / f"model2_{plant}"
    results_dir = Path("experiments/cnn/results/06_model2") / plant
    model_path = results_dir / "best_model.pth"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    test_dataset = datasets.ImageFolder(base_data / "test", transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    num_classes = len(test_dataset.classes)
    model = SimpleCNN(num_classes=num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()


    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    report = classification_report(
        y_true, y_pred,
        target_names=test_dataset.classes,
        output_dict=True
    )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_recall": report["macro avg"]["recall"],
        "weighted_recall": report["weighted avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "num_classes": num_classes,
        "classes": test_dataset.classes
    }

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("=== Evaluation Complete ===")
    print(json.dumps(metrics, indent=2))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant", required=True)
    args = parser.parse_args()

    main(args.plant)
