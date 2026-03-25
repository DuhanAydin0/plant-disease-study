from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT / "experiments" / "cnn" / "runs" / "03_all_dataset"))

#  MODEL CLASS (AYNI)
from cnn_03_all_model import CNN_03_All_Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[4]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "full_split"
MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "cnn"
    / "results"
    / "03_focus_classes"
    / "cnn_03_focus_classes_model.pth"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 38


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Model path:", MODEL_PATH)

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_dataset = datasets.ImageFolder(
        root=DATA_DIR / "test",
        transform=transform,
    )

    loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = CNN_03_All_Dataset(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    margins = []
    correct_margins = []
    wrong_margins = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            sorted_logits, _ = torch.sort(logits, dim=1, descending=True)

            margin = (sorted_logits[:, 0] - sorted_logits[:, 1]).cpu().numpy()
            preds = logits.argmax(dim=1)

            margins.extend(margin)

            for m, p, y in zip(margin, preds, labels):
                if p == y:
                    correct_margins.append(m)
                else:
                    wrong_margins.append(m)

    margins = np.array(margins)
    correct_margins = np.array(correct_margins)
    wrong_margins = np.array(wrong_margins)

    print("\n===== Focused Model – Logit Margin Analysis =====")
    print(f"Mean margin (all): {margins.mean():.4f}")
    print(f"Mean margin (correct): {correct_margins.mean():.4f}")
    print(f"Mean margin (wrong): {wrong_margins.mean():.4f}")

    for p in [10, 25, 50, 75, 90]:
        print(f"{p}% percentile (correct): {np.percentile(correct_margins, p):.4f}")


if __name__ == "__main__":
    main()
