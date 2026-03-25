"""
GLOBAL CNN + SVM (TRAIN)

Run from repo root:
python experiments/cnn/runs/07_cnn_svm/train_cnn_svm.py

What it does:
1) Dynamically imports your CNN class from:
   repo_root/experiments/cnn/runs/03_all_dataset/cnn_03_all_model.py
2) Loads pretrained CNN checkpoint:
   repo_root/experiments/cnn/results/03_all_dataset/cnn_03_all_dataset_30epochs_model.pth
3) Extracts 128-d embeddings (fc1 + ReLU, pre-dropout)
4) Trains SVM on embeddings
5) Saves ALL outputs to:
   repo_root/experiments/cnn/results/07_cnn_svm
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# =========================
# 1) PATHS
# =========================
REPO_ROOT = Path(__file__).resolve().parents[4]

RESULTS_DIR = REPO_ROOT / "experiments" / "cnn" / "results" / "07_cnn_svm"
CNN_CKPT = (
    REPO_ROOT
    / "experiments"
    / "cnn"
    / "results"
    / "03_all_dataset"
    / "cnn_03_all_dataset_30epochs_model.pth"
)
MODEL_DEF_FILE = (
    REPO_ROOT
    / "experiments"
    / "cnn"
    / "runs"
    / "03_all_dataset"
    / "cnn_03_all_model.py"
)

# default split
DEFAULT_DATA_SPLIT = REPO_ROOT / "data" / "processed" / "full_split"


# =========================
# 2) DYNAMIC IMPORT
# =========================
def load_cnn_class(model_file: Path):
    if not model_file.exists():
        raise FileNotFoundError(f"Model definition file not found:\n{model_file}")

    spec = importlib.util.spec_from_file_location("cnn_03_all_model", str(model_file))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for:\n{model_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore

    if not hasattr(module, "CNN_03_All_Dataset"):
        raise AttributeError(
            f"'CNN_03_All_Dataset' not found in:\n{model_file}\n"
            "Make sure the class name matches exactly."
        )

    return getattr(module, "CNN_03_All_Dataset")


# =========================
# 3) FEATURE EXTRACTOR
# =========================
class CNNFeatureExtractor(nn.Module):
    """
    Exposes embedding = classifier[1] + ReLU output (pre-dropout).
    Assumes your model structure:
      conv1, conv2, conv3
      classifier = [Flatten, Linear(...->128), ReLU, Dropout, Linear(128->num_classes)]
    """

    def __init__(self, base: nn.Module):
        super().__init__()
        self.conv1 = base.conv1
        self.conv2 = base.conv2
        self.conv3 = base.conv3
        self.flatten = base.classifier[0]
        self.fc1 = base.classifier[1]
        self.relu = base.classifier[2]

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        return x


# =========================
# 4) UTILS
# =========================
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transform(image_size: Tuple[int, int]):
    # Must match your original evaluate.py normalization
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


@torch.no_grad()
def extract_embeddings(
    feat_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    l2_normalize: bool = True,
):
    feat_model.eval()
    Xs, ys = [], []

    for images, labels in loader:
        images = images.to(device)
        emb = feat_model(images).detach().cpu().numpy()

        if l2_normalize:
            emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

        Xs.append(emb)
        ys.append(labels.numpy())

    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# =========================
# 5) MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_split_rel",
        type=str,
        default=str(DEFAULT_DATA_SPLIT.relative_to(REPO_ROOT)),
        help="Dataset split root relative to repo_root (contains train/val/test)",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--no_cache", action="store_true", help="Do not save embeddings_cache_*.npz")
    parser.add_argument("--no_l2", action="store_true", help="Disable L2 normalization on embeddings")

    # SVM params
    parser.add_argument("--svm_c", type=float, default=10.0)
    parser.add_argument("--svm_gamma", type=str, default="scale")
    parser.add_argument("--svm_kernel", type=str, default="rbf")
    parser.add_argument("--svm_no_proba", action="store_true")

    args = parser.parse_args()

    # Resolve paths
    results_dir = RESULTS_DIR
    data_split_root = REPO_ROOT / Path(args.data_split_rel)

    # Ensure dirs
    results_dir.mkdir(parents=True, exist_ok=True)

    # Hard checks
    if not CNN_CKPT.exists():
        raise FileNotFoundError(f"CNN checkpoint not found:\n{CNN_CKPT}")
    if not MODEL_DEF_FILE.exists():
        raise FileNotFoundError(f"CNN model definition not found:\n{MODEL_DEF_FILE}")
    for split in ["train", "val", "test"]:
        p = data_split_root / split
        if not p.exists():
            raise FileNotFoundError(f"Missing split folder:\n{p}")

    # Print paths clearly
    print("\n" + "=" * 70)
    print("REPO_ROOT   :", REPO_ROOT)
    print("RESULTS_DIR :", results_dir)
    print("CNN_CKPT    :", CNN_CKPT)
    print("MODEL_DEF   :", MODEL_DEF_FILE)
    print("DATA_SPLIT  :", data_split_root)
    print("=" * 70 + "\n")

    device = get_device()
    print("[INFO] device:", device)

    # Load CNN class dynamically
    CNN_03_All_Dataset = load_cnn_class(MODEL_DEF_FILE)

    # Data
    image_size = (args.image_size, args.image_size)
    tfm = build_transform(image_size)

    train_ds = datasets.ImageFolder(data_split_root / "train", transform=tfm)
    val_ds = datasets.ImageFolder(data_split_root / "val", transform=tfm)
    test_ds = datasets.ImageFolder(data_split_root / "test", transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Save class maps
    class_to_idx = train_ds.class_to_idx
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    write_json(results_dir / "class_to_idx.json", class_to_idx)
    write_json(results_dir / "idx_to_class.json", idx_to_class)

    # Load pretrained CNN weights
    base = CNN_03_All_Dataset(num_classes=len(train_ds.classes))
    base.load_state_dict(torch.load(CNN_CKPT, map_location="cpu"))
    base.eval()

    feat_model = CNNFeatureExtractor(base).to(device)

    # Extract embeddings
    l2 = not args.no_l2
    print("[INFO] extracting embeddings (l2_normalize =", l2, ") ...")
    X_train, y_train = extract_embeddings(feat_model, train_loader, device, l2_normalize=l2)
    X_val, y_val = extract_embeddings(feat_model, val_loader, device, l2_normalize=l2)
    X_test, y_test = extract_embeddings(feat_model, test_loader, device, l2_normalize=l2)

    # Train SVM
    print("[INFO] training SVM ...")
    svm = SVC(
        C=args.svm_c,
        gamma=args.svm_gamma,
        kernel=args.svm_kernel,
        class_weight="balanced",
        probability=(not args.svm_no_proba),
    )
    svm.fit(X_train, y_train)

    # Quick metrics
    val_pred = svm.predict(X_val)
    test_pred = svm.predict(X_test)

    quick_metrics = {
        "val_accuracy": float(accuracy_score(y_val, val_pred)),
        "val_recall_macro": float(recall_score(y_val, val_pred, average="macro", zero_division=0)),
        "val_f1_macro": float(f1_score(y_val, val_pred, average="macro", zero_division=0)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_recall_macro": float(recall_score(y_test, test_pred, average="macro", zero_division=0)),
        "test_f1_macro": float(f1_score(y_test, test_pred, average="macro", zero_division=0)),
    }

    # Save artifacts (CLEAR paths)
    cnn_out = results_dir / "cnn_feature_extractor.pth"
    svm_out = results_dir / "svm_model.joblib"
    metrics_out = results_dir / "quick_metrics.json"
    cfg_out = results_dir / "embedding_config.json"

    torch.save(base.state_dict(), cnn_out)
    joblib.dump(svm, svm_out)
    write_json(metrics_out, quick_metrics)

    write_json(
        cfg_out,
        {
            "image_size": list(image_size),
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "embedding_dim": 128,
            "embedding_layer": "classifier[1]+ReLU (pre-dropout)",
            "l2_normalize": l2,
            "cnn_ckpt": str(CNN_CKPT.relative_to(REPO_ROOT)),
            "model_def_file": str(MODEL_DEF_FILE.relative_to(REPO_ROOT)),
            "data_split_rel": str(Path(args.data_split_rel)),
            "svm": {
                "C": args.svm_c,
                "gamma": args.svm_gamma,
                "kernel": args.svm_kernel,
                "class_weight": "balanced",
                "probability": (not args.svm_no_proba),
            },
            "dataloader": {
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "shuffle": False,
            },
        },
    )

    if not args.no_cache:
        c_train = results_dir / "embeddings_cache_train.npz"
        c_val = results_dir / "embeddings_cache_val.npz"
        c_test = results_dir / "embeddings_cache_test.npz"
        np.savez_compressed(c_train, X=X_train, y=y_train)
        np.savez_compressed(c_val, X=X_val, y=y_val)
        np.savez_compressed(c_test, X=X_test, y=y_test)

    # Print saved files clearly
    print("\n" + "-" * 70)
    print("SAVED FILES:")
    print("CNN  ->", cnn_out)
    print("SVM  ->", svm_out)
    print("METR ->", metrics_out)
    print("CONF ->", cfg_out)
    if not args.no_cache:
        print("CACH ->", results_dir / "embeddings_cache_train.npz")
        print("CACH ->", results_dir / "embeddings_cache_val.npz")
        print("CACH ->", results_dir / "embeddings_cache_test.npz")
    print("-" * 70)
    print("\n[OK] quick_metrics:", quick_metrics)


if __name__ == "__main__":
    main()
