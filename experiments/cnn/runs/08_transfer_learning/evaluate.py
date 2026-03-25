# experiments/cnn/runs/08_transfer_learning/evaluate.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import classification_report, confusion_matrix

from model import build_resnet18_classifier

# ==========================
# Project-fixed paths (experiments-only)
# ==========================
DATA_DIR = Path(r"/Users/duhanaydin/cursor/plant disease study/data/processed/full_split")
BASE_RESULTS_DIR = Path(r"/Users/duhanaydin/cursor/plant disease study/experiments/cnn/results/08_transfer_learning")
STAGE_A_DIR = BASE_RESULTS_DIR / "stageA"
STAGE_B_DIR = BASE_RESULTS_DIR / "stageB"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def get_stage_dir(stage: str) -> Path:
    stage = stage.upper().strip()
    if stage == "A":
        return STAGE_A_DIR
    if stage == "B":
        return STAGE_B_DIR
    raise ValueError("stage must be 'A' or 'B'")

def load_class_maps(stage_dir: Path) -> List[str]:
    classes_path = stage_dir / "classes.json"
    if not classes_path.exists():
        raise FileNotFoundError(
            f"Missing classes.json in {stage_dir} (did you train first?)."
        )
    return json.loads(classes_path.read_text())

def find_ckpt(stage_dir: Path) -> Path:
    best = stage_dir / "model_best.pth"
    last = stage_dir / "model_last.pth"
    if best.exists():
        return best
    if last.exists():
        return last
    raise FileNotFoundError(f"No checkpoint found in {stage_dir} (model_best.pth / model_last.pth).")

@torch.no_grad()
def forward_collect(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_true = []
    y_pred = []
    margins = []  # per-sample margin: p1 - p2
    probs_max = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        top2 = torch.topk(probs, k=2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1]).cpu().numpy()

        pred = probs.argmax(dim=1).cpu().numpy()

        y_true.extend(y.numpy().tolist())
        y_pred.extend(pred.tolist())
        margins.extend(margin.tolist())
        probs_max.extend(top2[:, 0].cpu().numpy().tolist())

    return np.array(y_true), np.array(y_pred), np.array(margins), np.array(probs_max)

def margin_stats(margins: np.ndarray) -> Dict[str, float]:
    return {
        "n": float(margins.size),
        "mean": float(np.mean(margins)) if margins.size else 0.0,
        "p50": float(np.percentile(margins, 50)) if margins.size else 0.0,
        "p10": float(np.percentile(margins, 10)) if margins.size else 0.0,
        "p01": float(np.percentile(margins, 1)) if margins.size else 0.0,
        "min": float(np.min(margins)) if margins.size else 0.0,
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["A", "B", "a", "b"], required=True)
    args = parser.parse_args()

    stage_dir = get_stage_dir(args.stage)
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Always evaluate on TEST split (project convention)
    test_dir = DATA_DIR / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Expected test split folder at: {test_dir}")

    classes = load_class_maps(stage_dir)
    num_classes = len(classes)

    test_ds = datasets.ImageFolder(str(test_dir), transform=build_eval_transform())
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    ckpt_path = find_ckpt(stage_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18_classifier(num_classes=num_classes, pretrained=False)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)

    y_true, y_pred, margins, probs_max = forward_collect(model, test_loader, device)

    acc = float((y_true == y_pred).mean()) if y_true.size else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    report = classification_report(
        y_true, y_pred, labels=list(range(num_classes)), target_names=classes, digits=4, zero_division=0
    )

    # Per-class recall
    recalls = []
    for i, cls in enumerate(classes):
        tp = float(cm[i, i])
        fn = float(cm[i, :].sum() - cm[i, i])
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append((cls, rec))

    # Save artifacts
    np.save(stage_dir / "confusion_matrix.npy", cm)

    import csv
    with (stage_dir / "per_class_recall.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "recall"])
        for cls, rec in recalls:
            w.writerow([cls, f"{rec:.6f}"])

    with (stage_dir / "per_sample_margin.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_idx", "y_true", "y_pred", "p_max", "margin_p1_p2"])
        for i in range(len(y_true)):
            w.writerow([i, int(y_true[i]), int(y_pred[i]), float(probs_max[i]), float(margins[i])])

    mstats = margin_stats(margins)
    (stage_dir / "margin_stats.json").write_text(json.dumps(mstats, indent=2, ensure_ascii=False))
    (stage_dir / "classification_report.txt").write_text(report)

    eval_summary = {
        "run_name": "08_transfer_learning",
        "stage": args.stage.upper(),
        "split": "test",
        "ckpt_path": str(ckpt_path),
        "data_dir": str(DATA_DIR),
        "n_test": int(len(test_ds)),
        "test_accuracy": acc,
        "macro_recall": float(np.mean([r for _, r in recalls])) if recalls else 0.0,
    }
    (stage_dir / "eval_summary.json").write_text(json.dumps(eval_summary, indent=2, ensure_ascii=False))

    print(f"[{args.stage.upper()}] ckpt: {ckpt_path}")
    print(f"[{args.stage.upper()}] test_accuracy={acc:.4f} macro_recall={eval_summary['macro_recall']:.4f}")
    print(f"[{args.stage.upper()}] Saved eval artifacts under: {stage_dir}")

if __name__ == "__main__":
    main()
