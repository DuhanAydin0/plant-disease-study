# experiments/cnn/runs/06_model2/recall_analysis.py

import argparse
from pathlib import Path

import torch
from torchvision import datasets, transforms
from sklearn.metrics import recall_score

from train_model2 import SimpleCNN


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

    model = SimpleCNN(num_classes=len(test_dataset.classes)).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            y_true.extend(y.numpy())
            y_pred.extend(preds.cpu().numpy())

    recalls = recall_score(y_true, y_pred, average=None)

    print("=== Class-wise Recall ===")
    for cls, r in zip(test_dataset.classes, recalls):
        print(f"{cls:30s}: {r:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant", required=True)
    args = parser.parse_args()

    main(args.plant)
