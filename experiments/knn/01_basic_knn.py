"""
Basic KNN Baseline for Tomato Leaf Disease Classification

This script implements a simple KNN classifier as a baseline.
- Uses fixed parameters: n_neighbors=5, weights="uniform", metric="euclidean"
- No feature scaling (to demonstrate KNN's sensitivity to scale)
- Trains on train set, evaluates separately on validation and test sets
- Converts images to flat feature vectors (no CNN)
"""

from __future__ import annotations

import joblib
import numpy as np
from pathlib import Path
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


# Directory structure
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Paths
DATA_ROOT = Path("data/processed/tomato_split")

# Image processing
IMAGE_SIZE = (64, 64)  # (width, height)

# KNN parameters (basic baseline)
N_NEIGHBORS = 5
WEIGHTS = "uniform"
METRIC = "euclidean"

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_split_as_vectors(split: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load images from a split directory and convert to flat feature vectors.
    
    Uses torchvision.datasets.ImageFolder to load images, then converts
    them to flattened numpy arrays.
    
    Args:
        split: One of "train", "val", or "test"
        
    Returns:
        X: np.ndarray of shape (n_samples, n_features) - flattened image features
        y: np.ndarray of shape (n_samples,) - class labels as strings
    """
    split_dir = DATA_ROOT / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    
    # Define transform: resize and convert to tensor, then to numpy
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),  # Converts PIL Image to tensor [C, H, W] in [0, 1]
    ])
    
    # Load dataset using ImageFolder
    dataset = ImageFolder(root=str(split_dir), transform=transform)
    
    # Extract features and labels
    features: list[np.ndarray] = []
    labels: list[str] = []
    
    for idx in range(len(dataset)):
        image_tensor, label_idx = dataset[idx]
        # Convert tensor to numpy and flatten: [C, H, W] -> [C*H*W]
        image_array = image_tensor.numpy().flatten()
        features.append(image_array)
        # Get class name from dataset
        labels.append(dataset.classes[label_idx])
    
    X = np.stack(features, axis=0)
    y = np.array(labels)
    
    return X, y


def print_metrics(y_true, y_pred, label_encoder, split_name: str):
    """
    Print accuracy, classification report, and confusion matrix.
    
    Args:
        y_true: True labels (encoded)
        y_pred: Predicted labels (encoded)
        label_encoder: LabelEncoder used to encode labels
        split_name: Name of the split (for display)
    """
    print(f"\n{'='*60}")
    print(f"{split_name} Set Metrics")
    print(f"{'='*60}")
    
    # Accuracy
    accuracy = metrics.accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Classification Report
    print("\nClassification Report:")
    print(metrics.classification_report(
        y_true, 
        y_pred, 
        target_names=label_encoder.classes_,
        zero_division=0
    ))
    
    # Confusion Matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    print(f"\nClass labels (in order): {list(label_encoder.classes_)}")


def main() -> None:
    """Main training and evaluation pipeline."""
    print("="*60)
    print("Basic KNN Baseline - Tomato Leaf Disease Classification")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  n_neighbors: {N_NEIGHBORS}")
    print(f"  weights: {WEIGHTS}")
    print(f"  metric: {METRIC}")
    print(f"  image_size: {IMAGE_SIZE}")
    print(f"  random_seed: {RANDOM_SEED}")
    print(f"\nNote: No feature scaling applied (to demonstrate KNN sensitivity)")
    
    # Load data
    print("\nLoading data...")
    X_train, y_train = load_split_as_vectors("train")
    X_val, y_val = load_split_as_vectors("val")
    X_test, y_test = load_split_as_vectors("test")
    
    print(f"Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)
    
    # Create and train KNN classifier
    print("\nTraining KNN classifier...")
    knn = KNeighborsClassifier(
        n_neighbors=N_NEIGHBORS,
        weights=WEIGHTS,
        metric=METRIC,
        n_jobs=-1  # Use all available cores
    )
    
    knn.fit(X_train, y_train_enc)
    print("Training completed!")
    
    # Save the trained model
    model_path = MODELS_DIR / "knn_01_basic.joblib"
    joblib.dump(
        {
            "model": knn,
            "label_encoder": label_encoder,
            "image_size": IMAGE_SIZE,
        },
        model_path,
    )
    print(f"Saved model to {model_path}")
    
    # Evaluate on train set
    train_preds = knn.predict(X_train)
    print_metrics(y_train_enc, train_preds, label_encoder, "Train")
    
    # Evaluate on validation set
    val_preds = knn.predict(X_val)
    print_metrics(y_val_enc, val_preds, label_encoder, "Validation")
    
    # Evaluate on test set
    test_preds = knn.predict(X_test)
    print_metrics(y_test_enc, test_preds, label_encoder, "Test")
    
    # Print precision, recall, and F1-score for test set
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        y_test_enc, test_preds, average="weighted", zero_division=0
    )
    print(f"\nTest Set - Precision (weighted): {precision:.4f}")
    print(f"Test Set - Recall (weighted): {recall:.4f}")
    print(f"Test Set - F1-score (weighted): {f1:.4f}")
    
    # Calculate test accuracy
    test_accuracy = metrics.accuracy_score(y_test_enc, test_preds)
    
    # Save results summary to file
    results_path = RESULTS_DIR / "knn_01_basic_results.txt"
    with open(results_path, "w") as f:
        f.write("KNN 01 Basic - Final Test Results Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: 01_basic\n\n")
        f.write(f"Dataset Summary:\n")
        f.write(f"  Number of test samples: {X_test.shape[0]}\n")
        f.write(f"  Number of features: {X_test.shape[1]}\n\n")
        f.write(f"Final TEST Metrics:\n")
        f.write(f"  Accuracy: {test_accuracy:.4f}\n")
        f.write(f"  Precision (weighted): {precision:.4f}\n")
        f.write(f"  Recall (weighted): {recall:.4f}\n")
        f.write(f"  F1-score (weighted): {f1:.4f}\n\n")
        f.write("Note: This file was generated automatically after training.\n")
    
    print("\n" + "="*60)
    print("Basic KNN Baseline Evaluation Complete")
    print("="*60)


if __name__ == "__main__":
    main()

