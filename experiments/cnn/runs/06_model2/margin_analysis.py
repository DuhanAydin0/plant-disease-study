# experiments/cnn/runs/06_model2/margin_analysis.py

import argparse
from pathlib import Path
import numpy as np

import torch
from torchvision import datasets, transforms

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

    dataset = datasets.ImageFolder(base_data / "test", transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    model = SimpleCNN(num_classes=len(dataset.classes)).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    margins = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)

            for i in range(len(y)):
                true_logit = logits[i, y[i]].item()
                other_logits = torch.cat([logits[i, :y[i]], logits[i, y[i]+1:]])
                max_other = torch.max(other_logits).item()
                margins.append(true_logit - max_other)

    margins = np.array(margins)

    print("=== Logit Margin Analysis ===")
    print(f"Mean margin: {margins.mean():.4f}")
    print(f"Std margin : {margins.std():.4f}")
    print("Percentiles:")
    for p in [10, 25, 50, 75, 90]:
        print(f"  {p}%: {np.percentile(margins, p):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant", required=True)
    args = parser.parse_args()

    main(args.plant)
