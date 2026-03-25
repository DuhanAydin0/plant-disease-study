"""
Evaluate the trained SVM baseline on the test split.
Uses the same evaluation logic as train.py to avoid redundancy.
"""

from __future__ import annotations

import joblib
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn import metrics


# Paths
DATA_ROOT = Path("data/processed/tomato_split")
MODEL_PATH = Path("experiments/svm/svm_model.joblib")


def load_split(split: str, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Load images and labels from a given split.
    """
    split_dir = DATA_ROOT / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    features: list[np.ndarray] = []
    labels: list[str] = []

    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.iterdir()):
            if not img_path.is_file():
                continue
            image = Image.open(img_path).convert("RGB").resize(image_size)
            array = np.asarray(image, dtype=np.float32) / 255.0
            features.append(array.flatten())
            labels.append(class_dir.name)

    if not features:
        raise RuntimeError(f"No images found under split: {split_dir}")

    X = np.stack(features, axis=0)
    y = np.array(labels)
    return X, y


def print_metrics(y_true, y_pred, label_encoder, split_name):
    """
    Print accuracy, confusion matrix, precision, recall and F1-score.
    Shared evaluation logic with train.py.
    """
    print(f"\n=== {split_name} Metrics ===")

    acc = metrics.accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")

    cm = metrics.confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("Labels:", list(label_encoder.classes_))

    precision = metrics.precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    recall = metrics.recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    f1 = metrics.f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted):    {recall:.4f}")
    print(f"F1-score (weighted):  {f1:.4f}")

    print("\nClassification Report:")
    print(
        metrics.classification_report(
            y_true,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run train.py first."
        )

    # Load trained artifacts
    bundle = joblib.load(MODEL_PATH)
    svm_clf = bundle["model"]
    label_encoder = bundle["label_encoder"]
    image_size = bundle["image_size"]

    # Load test data
    X_test, y_test = load_split("test", image_size)
    y_test_enc = label_encoder.transform(y_test)

    # Predict
    y_pred_enc = svm_clf.predict(X_test)

    # Evaluate (single responsibility: TEST only)
    print_metrics(y_test_enc, y_pred_enc, label_encoder, "Test")


if __name__ == "__main__":
    main()

