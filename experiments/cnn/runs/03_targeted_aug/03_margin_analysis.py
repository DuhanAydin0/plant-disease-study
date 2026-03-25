import sys
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# =================================================
#  ABSOLUTE & EXPLICIT PATH SETUP
# =================================================
PROJECT_ROOT = Path(__file__).resolve().parents[4]
# -> plant disease study/

ALL_DATASET_DIR = PROJECT_ROOT / "experiments/cnn/runs/03_all_dataset"
RESULTS_DIR = PROJECT_ROOT / "experiments/cnn/results/03_targeted_aug"

sys.path.insert(0, str(ALL_DATASET_DIR))

from cnn_03_all_model import CNN_03_All_Dataset
import config


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # =================================================
    #  MODEL PATH
    # =================================================
    MODEL_PATH = RESULTS_DIR / "cnn_03_targeted_aug_model.pth"

    print("Loading model from:", MODEL_PATH)
    print("Exists:", MODEL_PATH.exists())

    # =================================================
    # Model
    # =================================================
    model = CNN_03_All_Dataset(
        num_classes=config.NUM_CLASSES
    ).to(device)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )
    model.eval()

    # =================================================
    # Test dataset
    # =================================================
    test_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_dataset = datasets.ImageFolder(
        root=config.DATA_DIR / "test",
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    # =================================================
    # Margin computation
    # =================================================
    margins = []
    correct_flags = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            top2 = torch.topk(logits, k=2, dim=1).values
            margin = top2[:, 0] - top2[:, 1]

            preds = torch.argmax(logits, dim=1)
            correct = preds.eq(labels)

            margins.extend(margin.cpu().numpy())
            correct_flags.extend(correct.cpu().numpy())

    margins = np.array(margins)
    correct_flags = np.array(correct_flags)

    print("\n===== Logit Margin Analysis (Targeted Aug Model) =====")
    print(f"Mean margin (all): {margins.mean():.4f}")
    print(f"Mean margin (correct): {margins[correct_flags].mean():.4f}")
    print(f"Mean margin (wrong): {margins[~correct_flags].mean():.4f}")

    print("\nMargin percentiles (correct predictions):")
    for p in [10, 25, 50, 75, 90]:
        print(f"{p}%: {np.percentile(margins[correct_flags], p):.4f}")


if __name__ == "__main__":
    main()
