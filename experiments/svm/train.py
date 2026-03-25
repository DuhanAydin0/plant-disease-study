"""
Train a simple SVM baseline for tomato leaf disease classification.

Workflow:
- Load train/val images from the split dataset.
- Resize to a fixed size, flatten to 1D feature vectors.
- Encode class labels.
- Train an SVM classifier (with feature scaling).
- Report train and validation accuracy.
- Report confusion matrix, Precision, Recall, F1-score.
- Save the fitted pipeline and label encoder to disk.
"""

from __future__ import annotations

import joblib
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


# Paths
DATA_ROOT = Path("data/processed/tomato_split")
MODEL_PATH = Path("experiments/svm/svm_model.joblib")

# Image processing
IMAGE_SIZE = (64, 64)  # (width, height)

# Model config (lightweight baseline)
SVM_KERNEL = "rbf"
SVM_C = 2.0
SVM_GAMMA = "scale"


def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load images and labels from a given split (train/val/test).
    Returns:
        X: np.ndarray of shape (n_samples, n_features)
        y: np.ndarray of shape (n_samples,)
    """
    split_dir = DATA_ROOT / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    features: list[np.ndarray] = []
    labels: list[str] = []

    # Deterministic ordering for reproducibility/traceability
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.iterdir()):
            if not img_path.is_file():
                continue
            image = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)
            array = np.asarray(image, dtype=np.float32) / 255.0  # scale to [0,1]
            features.append(array.flatten())
            labels.append(class_dir.name)

    if not features:
        raise RuntimeError(f"No images found under split: {split_dir}")

    X = np.stack(features, axis=0)
    y = np.array(labels)
    return X, y


def print_metrics(y_true, y_pred, label_encoder, split_name):
    print(f"\n=== {split_name} Metrics ===")
    acc = metrics.accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")

    cm = metrics.confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("Labels:", list(label_encoder.classes_))

    precision = metrics.precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted):    {recall:.4f}")
    print(f"F1-score (weighted):  {f1:.4f}")

    # Print detailed classification report as well
    print("\nClassification Report:")
    print(metrics.classification_report(
        y_true, y_pred, target_names=label_encoder.classes_, zero_division=0
    ))


def main() -> None:
    # Load train/val data
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)

    # Build pipeline: scale features then SVM classifier
    svm_clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA),
    )

    # Train
    svm_clf.fit(X_train, y_train_enc)

    # Evaluate on train and val
    train_preds = svm_clf.predict(X_train)
    val_preds = svm_clf.predict(X_val)
    

    print_metrics(y_train_enc, train_preds, label_encoder, "Train")
    print_metrics(y_val_enc, val_preds, label_encoder, "Validation")

    # Persist model and label encoder together
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": svm_clf,
            "label_encoder": label_encoder,
            "image_size": IMAGE_SIZE,
        },
        MODEL_PATH,
    )
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()


