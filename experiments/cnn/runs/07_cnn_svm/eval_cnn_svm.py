"""
CNN + SVM (EVAL) — Comparable to Global CNN 03 analysis

Produces:
- eval_summary.json  (accuracy, macro/weighted recall, f1)
- classification_report.txt  (same style as your MD)
- per_class_recall.csv  (sorted, easiest for "patlayan sınıflar" list)
- margin_stats.json  (mean margin correct vs wrong, like CNN logit margin section)
- per_sample_margin.csv  (optional deep dive)

Uses:
- repo_root/experiments/cnn/results/07_cnn_svm/svm_model.joblib
- repo_root/experiments/cnn/results/07_cnn_svm/idx_to_class.json
- repo_root/experiments/cnn/results/07_cnn_svm/embedding_config.json
- repo_root/experiments/cnn/results/07_cnn_svm/embeddings_cache_test.npz (if exists)

If cache is missing:
- dynamically imports CNN model class from:
  repo_root/experiments/cnn/runs/03_all_dataset/cnn_03_all_model.py
- loads CNN weights from:
  repo_root/experiments/cnn/results/07_cnn_svm/cnn_feature_extractor.pth
- recomputes embeddings from test split
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# -------------------------
# Paths (GitHub-friendly)
# -------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]  # per your requirement
RESULTS_DIR = REPO_ROOT / "experiments" / "cnn" / "results" / "07_cnn_svm"

MODEL_DEF_FILE = (
    REPO_ROOT
    / "experiments"
    / "cnn"
    / "runs"
    / "03_all_dataset"
    / "cnn_03_all_model.py"
)

CNN_WEIGHTS_FOR_THIS_RUN = RESULTS_DIR / "cnn_feature_extractor.pth"
SVM_MODEL = RESULTS_DIR / "svm_model.joblib"
IDX_TO_CLASS_PATH = RESULTS_DIR / "idx_to_class.json"
EMB_CFG_PATH = RESULTS_DIR / "embedding_config.json"
CACHE_TEST = RESULTS_DIR / "embeddings_cache_test.npz"


# -------------------------
# Dynamic import
# -------------------------
def load_cnn_class(model_file: Path):
    if not model_file.exists():
        raise FileNotFoundError(f"Model definition file not found:\n{model_file}")

    spec = importlib.util.spec_from_file_location("cnn_03_all_model", str(model_file))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for:\n{model_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore

    if not hasattr(module, "CNN_03_All_Dataset"):
        raise AttributeError(f"'CNN_03_All_Dataset' not found in:\n{model_file}")

    return module.CNN_03_All_Dataset


# -------------------------
# Feature extractor (must match train)
# -------------------------
class CNNFeatureExtractor(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.conv1 = base.conv1
        self.conv2 = base.conv2
        self.conv3 = base.conv3
        self.flatten = base.classifier[0]
        self.fc1 = base.classifier[1]
        self.relu = base.classifier[2]

    @torch.no_grad()
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        return x


# -------------------------
# Utils
# -------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def build_transform(image_size, mean, std):
    return transforms.Compose(
        [
            transforms.Resize(tuple(image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


@torch.no_grad()
def extract_embeddings(feat_model, loader, device, l2_normalize: bool):
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


def svm_margin_from_scores(scores: np.ndarray) -> np.ndarray:
    """
    For multi-class, use top1 - top2 score margin.
    For binary, use abs(score) (distance from hyperplane).
    """
    if scores.ndim == 1:
        return np.abs(scores)
    # scores: (N, K)
    top2 = np.sort(scores, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_split_rel",
        type=str,
        default="data/processed/full_split",
        help="Only used if embeddings cache is missing. Must contain test/ folder.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    # Hard checks
    for p in [RESULTS_DIR, SVM_MODEL, IDX_TO_CLASS_PATH, EMB_CFG_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file:\n{p}")

    print("\n" + "=" * 70)
    print("REPO_ROOT   :", REPO_ROOT)
    print("RESULTS_DIR :", RESULTS_DIR)
    print("SVM_MODEL   :", SVM_MODEL)
    print("CACHE_TEST  :", CACHE_TEST, "(exists)" if CACHE_TEST.exists() else "(missing)")
    print("=" * 70 + "\n")

    # Load artifacts
    svm = joblib.load(SVM_MODEL)
    idx_to_class = read_json(IDX_TO_CLASS_PATH)
    # keys might be strings; normalize
    idx_to_class = {int(k): v for k, v in idx_to_class.items()}

    emb_cfg = read_json(EMB_CFG_PATH)
    image_size = emb_cfg.get("image_size", [224, 224])
    mean = emb_cfg.get("normalize_mean", [0.485, 0.456, 0.406])
    std = emb_cfg.get("normalize_std", [0.229, 0.224, 0.225])
    l2_normalize = bool(emb_cfg.get("l2_normalize", True))

    device = get_device()
    print("[INFO] device:", device)

    # -------------------------
    # Get X_test, y_test
    # -------------------------
    if CACHE_TEST.exists():
        z = np.load(CACHE_TEST)
        X_test, y_test = z["X"], z["y"]
        print("[INFO] Using cached test embeddings:", CACHE_TEST)
    else:
        # recompute embeddings (must be consistent with train)
        if not CNN_WEIGHTS_FOR_THIS_RUN.exists():
            raise FileNotFoundError(
                "Test embeddings cache is missing AND CNN weights are missing:\n"
                f"{CNN_WEIGHTS_FOR_THIS_RUN}\n"
                "Fix: re-run train script with cache enabled, or ensure cnn_feature_extractor.pth exists."
            )

        CNN_03_All_Dataset = load_cnn_class(MODEL_DEF_FILE)
        base = CNN_03_All_Dataset(num_classes=len(idx_to_class))
        base.load_state_dict(torch.load(CNN_WEIGHTS_FOR_THIS_RUN, map_location="cpu"))
        base.eval()

        feat = CNNFeatureExtractor(base).to(device)

        data_split_root = REPO_ROOT / Path(args.data_split_rel)
        test_dir = data_split_root / "test"
        if not test_dir.exists():
            raise FileNotFoundError(f"Missing test dir:\n{test_dir}")

        tfm = build_transform(image_size, mean, std)
        test_ds = datasets.ImageFolder(test_dir, transform=tfm)
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )

        print("[INFO] Cache missing. Recomputing embeddings from:", test_dir)
        X_test, y_test = extract_embeddings(feat, test_loader, device, l2_normalize=l2_normalize)

    # -------------------------
    # Predictions + metrics
    # -------------------------
    y_pred = svm.predict(X_test)

    # Recall like your MD: macro + weighted
    summary = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "n_test": int(len(y_test)),
    }
    write_json(RESULTS_DIR / "eval_summary.json", summary)

    # classification report (same style)
    class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    report_txt = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    (RESULTS_DIR / "classification_report.txt").write_text(report_txt, encoding="utf-8")

    # per-class recall CSV (sorted asc => "patlayan sınıflar" direkt çıkar)
    report_dict = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    rows = []
    for name in class_names:
        rows.append((name, float(report_dict[name]["recall"]), int(report_dict[name]["support"])))

    rows_sorted = sorted(rows, key=lambda x: x[1])
    lines = ["class,recall,support"]
    for name, rec, sup in rows_sorted:
        lines.append(f"{name},{rec:.6f},{sup}")
    (RESULTS_DIR / "per_class_recall.csv").write_text("\n".join(lines), encoding="utf-8")

    # -------------------------
    # Margin analysis (SVM equivalent of "logit margin")
    # -------------------------
    scores = svm.decision_function(X_test)
    margin = svm_margin_from_scores(np.asarray(scores))

    correct_mask = (y_pred == y_test)
    wrong_mask = ~correct_mask

    margin_stats = {
        "mean_margin_correct": float(np.mean(margin[correct_mask])) if correct_mask.any() else None,
        "mean_margin_wrong": float(np.mean(margin[wrong_mask])) if wrong_mask.any() else None,
        "median_margin_correct": float(np.median(margin[correct_mask])) if correct_mask.any() else None,
        "median_margin_wrong": float(np.median(margin[wrong_mask])) if wrong_mask.any() else None,
        "n_correct": int(correct_mask.sum()),
        "n_wrong": int(wrong_mask.sum()),
        "margin_definition": "SVM decision_function top1-top2 (multiclass) or abs(score) (binary)",
    }
    write_json(RESULTS_DIR / "margin_stats.json", margin_stats)

    # per-sample margin CSV (debug / deep dive)
    # (This helps you inspect low-margin wrong predictions for specific classes)
    per_sample_lines = ["true_label,pred_label,correct,margin,true_name,pred_name"]
    for yt, yp, ok, m in zip(y_test, y_pred, correct_mask.astype(int), margin):
        yt_i = int(yt)
        yp_i = int(yp)
        per_sample_lines.append(
            f"{yt_i},{yp_i},{int(ok)},{float(m):.8f},{idx_to_class[yt_i]},{idx_to_class[yp_i]}"
        )
    (RESULTS_DIR / "per_sample_margin.csv").write_text("\n".join(per_sample_lines), encoding="utf-8")

    # Optional: confusion matrix as npy for later plotting/compare scripts
    cm = confusion_matrix(y_test, y_pred)
    np.save(RESULTS_DIR / "confusion_matrix.npy", cm)

    # Print key outputs (like your MD summary)
    print("\n===== CNN+SVM EVAL SUMMARY =====")
    print("Test Accuracy   :", round(summary["test_accuracy"], 4))
    print("Macro Recall    :", round(summary["macro_recall"], 4))
    print("Weighted Recall :", round(summary["weighted_recall"], 4))

    print("\n===== Margin Analysis (SVM) =====")
    print("Mean margin (correct):", margin_stats["mean_margin_correct"])
    print("Mean margin (wrong)  :", margin_stats["mean_margin_wrong"])

    worst5 = rows_sorted[:5]
    print("\n===== Worst-5 Classes by Recall (CNN+SVM) =====")
    for name, rec, sup in worst5:
        print(f"- {name}: recall={rec:.3f} (support={sup})")

    print("\n[SAVED] eval_summary.json")
    print("[SAVED] classification_report.txt")
    print("[SAVED] per_class_recall.csv")
    print("[SAVED] margin_stats.json")
    print("[SAVED] per_sample_margin.csv")
    print("[SAVED] confusion_matrix.npy")
    print("\n[OK] Results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
