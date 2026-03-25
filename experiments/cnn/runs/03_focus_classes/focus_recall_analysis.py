import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT / "experiments" / "cnn" / "runs" / "03_all_dataset"))


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

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = CNN_03_All_Dataset(num_classes=len(test_dataset.classes)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\n===== CLASS-WISE RECALL (FOCUSED MODEL) =====\n")
    print(classification_report(
        y_true,
        y_pred,
        target_names=test_dataset.classes,
        digits=4
    ))

if __name__ == "__main__":
    main()
