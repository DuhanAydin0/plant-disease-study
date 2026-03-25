import sys
from pathlib import Path
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ============================================================
# PATH FIX (SimpleCNN import)
# ============================================================
RUN_DIR = Path(
    "/Users/duhanaydin/cursor/plant disease study/experiments/cnn/runs/05_model1"
)
sys.path.append(str(RUN_DIR))

from baseline_cnn_model1 import SimpleCNN  # noqa

# ============================================================
# PATH CONFIG
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

assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"

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

# Güvenlik kontrolü
assert test_dataset.class_to_idx == class_to_idx, \
    "class_to_idx mismatch between training and test dataset!"

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# MARGIN ANALYSIS
# ============================================================
margins_correct = []
margins_wrong = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)

        top2 = torch.topk(probs, 2, dim=1)
        margin = top2.values[:, 0] - top2.values[:, 1]

        preds = probs.argmax(dim=1)
        correct_mask = preds == labels

        margins_correct.extend(margin[correct_mask].cpu().numpy())
        margins_wrong.extend(margin[~correct_mask].cpu().numpy())

# ============================================================
# REPORT
# ============================================================
print("\n==============================")
print(" Logit Margin Analysis")
print("==============================")
print(f"Mean margin (correct): {np.mean(margins_correct):.4f}")
print(f"Mean margin (wrong)  : {np.mean(margins_wrong):.4f}")

for p in [10, 25, 50]:
    print(
        f"Correct margin {p}% percentile: "
        f"{np.percentile(margins_correct, p):.4f}"
    )
