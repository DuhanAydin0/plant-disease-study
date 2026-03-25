

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# -------------------------
# Paths I define paths like that in order to establish project in github
# -------------------------
REPO_ROOT = Path(__file__).resolve().parents[5]

RESULTS_DIR = REPO_ROOT / "experiments" / "cnn" / "results" / "03_all_dataset"
MODEL_DEF_FILE = REPO_ROOT / "experiments" / "cnn" / "runs" / "03_all_dataset" / "cnn_03_all_model.py"
CKPT_PATH = RESULTS_DIR / "cnn_03_all_dataset_30epochs_model.pth"


def load_cnn_class(model_file: Path):
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found:\n{model_file}")

    spec = importlib.util.spec_from_file_location("cnn_03_all_model", str(model_file))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import from:\n{model_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore

    if not hasattr(module, "CNN_03_All_Dataset"):
        raise AttributeError(f"'CNN_03_All_Dataset' not found in:\n{model_file}")

    return module.CNN_03_All_Dataset


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_split_rel",
        type=str,
        default="data/processed/full_split",
        help="Relative to repo_root. Must contain test/ folder.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data_split_root = REPO_ROOT / Path(args.data_split_rel)
    test_dir = data_split_root / "test"

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found:\n{CKPT_PATH}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test folder not found:\n{test_dir}")

    print("\n" + "=" * 70)
    print("REPO_ROOT   :", REPO_ROOT)
    print("RESULTS_DIR :", RESULTS_DIR)
    print("MODEL_DEF   :", MODEL_DEF_FILE)
    print("CKPT        :", CKPT_PATH)
    print("TEST_DIR    :", test_dir)
    print("=" * 70 + "\n")

    device = get_device()
    print("[INFO] device:", device)

    # Dataset (ImageNet normalization)
    tfm = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_ds = datasets.ImageFolder(test_dir, transform=tfm)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    class_names = test_ds.classes
    num_classes = len(class_names)

    # Model load (dynamic import)
    CNN_03_All_Dataset = load_cnn_class(MODEL_DEF_FILE)
    model = CNN_03_All_Dataset(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
    model.eval()

    # Predict
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Summary metrics
    summary = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n_test": int(len(y_true)),
    }
    write_json(RESULTS_DIR / "eval_summary.json", summary)

    # classification_report.txt
    report_txt = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    (RESULTS_DIR / "classification_report.txt").write_text(report_txt, encoding="utf-8")

    # per_class_recall.csv (sorted by recall asc)
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    rows = []
    for name in class_names:
        rows.append((name, float(report_dict[name]["recall"]), int(report_dict[name]["support"])))
    rows.sort(key=lambda x: x[1])  # ascending recall

    lines = ["class,recall,support"]
    for name, rec, sup in rows:
        lines.append(f"{name},{rec:.6f},{sup}")
    (RESULTS_DIR / "per_class_recall.csv").write_text("\n".join(lines), encoding="utf-8")

    print("\n[SAVED]")
    print(" -", RESULTS_DIR / "eval_summary.json")
    print(" -", RESULTS_DIR / "classification_report.txt")
    print(" -", RESULTS_DIR / "per_class_recall.csv")
    print("\n[OK] summary:", summary)


if __name__ == "__main__":
    main()
