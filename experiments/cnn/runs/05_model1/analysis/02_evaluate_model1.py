import sys
from pathlib import Path

# ============================================================
# PATH FIX (SimpleCNN import için)
# ============================================================
RUN_DIR = Path(
    "/Users/duhanaydin/cursor/plant disease study/experiments/cnn/runs/05_model1"
)
sys.path.append(str(RUN_DIR))

from baseline_cnn_model1 import SimpleCNN  # noqa

# ============================================================
# IMPORTS
# ============================================================
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# ============================================================
# PATH CONFIG (SENİN VERDİĞİN)
# ============================================================
MODEL_PATH = Path(
    "/Users/duhanaydin/cursor/plant disease study/experiments/cnn/results/05_model1/model1_baseline_cnn.pth"
)

DATA_DIR = Path(
    "/Users/duhanaydin/cursor/plant disease study/data/processed/model1_plant"
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
IMAGE_SIZE = 224
BATCH_SIZE = 32

# ============================================================
# LOAD MODEL
# ============================================================
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

class_to_idx = ckpt["class_to_idx"]
idx_to_class = {v: k for k, v in class_to_idx.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

model = SimpleCNN(num_classes=len(class_names)).to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# ============================================================
# LOAD TEST DATA
# ============================================================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=transform)

# kontrol
assert test_dataset.class_to_idx == class_to_idx, \
    "class_to_idx mismatch between training and test dataset!"

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# EVALUATION
# ============================================================
y_true, y_pred = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1).cpu().numpy()

        y_pred.extend(preds)
        y_true.extend(labels.numpy())

acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("\n==============================")
print(f" Test Accuracy: {acc:.4f}")
print("==============================\n")
print("Confusion Matrix:\n")
print(cm)
