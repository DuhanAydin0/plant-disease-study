import sys
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# =================================================
# 🔧 ABSOLUTE & EXPLICIT PATH SETUP (NO MAGIC)
# =================================================
PROJECT_ROOT = Path(__file__).resolve().parents[4]  
# -> plant disease study/

ALL_DATASET_DIR = PROJECT_ROOT / "experiments/cnn/runs/03_all_dataset"
RESULTS_DIR = PROJECT_ROOT / "experiments/cnn/results/03_targeted_aug"

sys.path.insert(0, str(ALL_DATASET_DIR))

from cnn_03_all_model import CNN_03_All_Dataset
import config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed(config.RANDOM_SEED)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # =================================================
    #  MODEL PATH (TARGETED AUG MODEL)
    # =================================================
    MODEL_PATH = RESULTS_DIR / "cnn_03_targeted_aug_model.pth"

    print("Loading model from:", MODEL_PATH)
    print("Exists:", MODEL_PATH.exists())

    # =================================================
    # Dataset
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
    # Recall computation
    # =================================================
    num_classes = config.NUM_CLASSES
    correct_per_class = np.zeros(num_classes)
    total_per_class = np.zeros(num_classes)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            for y, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                total_per_class[y] += 1
                if y == p:
                    correct_per_class[y] += 1

    print("\n===== Class-wise Recall (Targeted Aug Model) =====")
    for idx, class_name in enumerate(test_dataset.classes):
        if total_per_class[idx] > 0:
            recall = correct_per_class[idx] / total_per_class[idx]
            print(f"{class_name:45s} Recall: {recall:.4f}")


if __name__ == "__main__":
    main()
