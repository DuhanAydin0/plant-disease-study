"""
Optimized KNN for Tomato Leaf Disease Classification

This script demonstrates improvements over basic KNN:
- Uses StandardScaler to normalize features (critical for distance-based methods)
- Uses distance-weighted voting (weights="distance")
- Tries multiple n_neighbors values to find optimal k
- Explains overfitting risks, curse of dimensionality, and scaling sensitivity

Expected behavior:
- Training accuracy may decrease (less overfitting)
- Validation/Test performance should be more meaningful and stable
"""

from __future__ import annotations

import joblib
import numpy as np
from pathlib import Path
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

# KNN optimization parameters
N_NEIGHBORS_RANGE = range(3, 12)  # Try k from 3 to 11
WEIGHTS = "distance"  # Weight neighbors by inverse distance (better than uniform)

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


def evaluate_knn(k: int, X_train, y_train_enc, X_val, y_val_enc, X_test, y_test_enc, label_encoder):
    """
    Train and evaluate a KNN classifier with a specific k value.
    
    Returns:
        Dictionary with accuracies for train, val, and test sets
    """
    # Create pipeline: StandardScaler -> KNN
    # StandardScaler is CRITICAL for KNN because:
    # - KNN uses distance metrics (Euclidean, Manhattan, etc.)
    # - Features with larger scales dominate distance calculations
    # - Without scaling, pixel intensity differences can overwhelm spatial patterns
    knn_pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        KNeighborsClassifier(
            n_neighbors=k,
            weights=WEIGHTS,  # Distance weighting: closer neighbors have more influence
            metric="euclidean",
            n_jobs=-1
        )
    )
    
    # Train
    knn_pipeline.fit(X_train, y_train_enc)
    
    # Evaluate
    train_preds = knn_pipeline.predict(X_train)
    val_preds = knn_pipeline.predict(X_val)
    test_preds = knn_pipeline.predict(X_test)
    
    train_acc = metrics.accuracy_score(y_train_enc, train_preds)
    val_acc = metrics.accuracy_score(y_val_enc, val_preds)
    test_acc = metrics.accuracy_score(y_test_enc, test_preds)
    
    return {
        "k": k,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "pipeline": knn_pipeline
    }


def main() -> None:
    """Main training and evaluation pipeline with hyperparameter search."""
    print("="*60)
    print("Optimized KNN - Tomato Leaf Disease Classification")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  n_neighbors range: {list(N_NEIGHBORS_RANGE)}")
    print(f"  weights: {WEIGHTS}")
    print(f"  image_size: {IMAGE_SIZE}")
    print(f"  random_seed: {RANDOM_SEED}")
    print(f"\nImprovements over basic KNN:")
    print(f"StandardScaler applied (critical for distance-based methods)")
    print(f"Distance-weighted voting (closer neighbors have more influence)")
    print(f"Hyperparameter search over n_neighbors")
    
    # Load data
    print("\nLoading data...")
    X_train, y_train = load_split_as_vectors("train")
    X_val, y_val = load_split_as_vectors("val")
    X_test, y_test = load_split_as_vectors("test")
    
    print(f"Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Number of classes: {len(np.unique(y_train))}")
    print(f"\nFeature space dimensionality: {X_train.shape[1]} features")
    print("  Note: High dimensionality can cause 'curse of dimensionality'")
    print("        where distances become less meaningful")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)
    
    # Try different k values
    print("\n" + "="*60)
    print("Hyperparameter Search: Testing different k values")
    print("="*60)
    
    results = []
    for k in N_NEIGHBORS_RANGE:
        print(f"\nTesting k={k}...")
        result = evaluate_knn(
            k, X_train, y_train_enc, X_val, y_val_enc, 
            X_test, y_test_enc, label_encoder
        )
        results.append(result)
        print(f"  Train Acc: {result['train_acc']:.4f}")
        print(f"  Val Acc:   {result['val_acc']:.4f}")
        print(f"  Test Acc:  {result['test_acc']:.4f}")
    
    # Find best k based on validation accuracy
    best_result = max(results, key=lambda x: x['val_acc'])
    best_k = best_result['k']
    
    print("\n" + "="*60)
    print("Hyperparameter Search Results Summary")
    print("="*60)
    print(f"\nBest k={best_k} (based on validation accuracy)")
    print(f"  Validation Accuracy: {best_result['val_acc']:.4f}")
    print(f"  Test Accuracy:       {best_result['test_acc']:.4f}")
    
    print("\nFull results table:")
    print(f"{'k':<5} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12}")
    print("-" * 45)
    for r in results:
        print(f"{r['k']:<5} {r['train_acc']:<12.4f} {r['val_acc']:<12.4f} {r['test_acc']:<12.4f}")
    
    # Detailed evaluation with best model
    print("\n" + "="*60)
    print(f"Detailed Evaluation: Best Model (k={best_k})")
    print("="*60)
    
    best_pipeline = best_result['pipeline']
    
    # Detailed metrics for train set
    train_preds = best_pipeline.predict(X_train)
    print_metrics(y_train_enc, train_preds, label_encoder, "Train")
    
    # Detailed metrics for validation set
    val_preds = best_pipeline.predict(X_val)
    print_metrics(y_val_enc, val_preds, label_encoder, "Validation")
    
    # Detailed metrics for test set
    test_preds = best_pipeline.predict(X_test)
    print_metrics(y_test_enc, test_preds, label_encoder, "Test")
    
    # Print precision, recall, and F1-score for test set
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        y_test_enc, test_preds, average="weighted", zero_division=0
    )
    print(f"\nTest Set - Precision (weighted): {precision:.4f}")
    print(f"Test Set - Recall (weighted): {recall:.4f}")
    print(f"Test Set - F1-score (weighted): {f1:.4f}")
    
    # Save the best optimized pipeline
    model_path = MODELS_DIR / "knn_02_optimized.joblib"
    joblib.dump(
        {
            "model": best_pipeline,
            "label_encoder": label_encoder,
            "image_size": IMAGE_SIZE,
            "best_k": best_k,
        },
        model_path,
    )
    print(f"\nSaved best model (k={best_k}) to {model_path}")
    
    # Analysis
    print("\n" + "="*60)
    print("Performance Analysis")
    print("="*60)
    print(f"\nTraining accuracy: {best_result['train_acc']:.4f}")
    print(f"Validation accuracy: {best_result['val_acc']:.4f}")
    print(f"Test accuracy: {best_result['test_acc']:.4f}")
    
    train_val_gap = best_result['train_acc'] - best_result['val_acc']
    val_test_gap = best_result['val_acc'] - best_result['test_acc']
    
    print(f"\nTrain-Val gap: {train_val_gap:.4f}")
    if train_val_gap > 0.05:
        print("Large gap suggests some overfitting to training data")
    else:
        print("Gap is reasonable, model generalizes well")
    
    print(f"Val-Test gap: {val_test_gap:.4f}")
    if abs(val_test_gap) > 0.05:
        print("Significant difference between val and test sets")
    else:
        print("Val and test performance are consistent")
    
    # Calculate test accuracy
    test_accuracy = metrics.accuracy_score(y_test_enc, test_preds)
    
    # Save results summary to file
    results_path = RESULTS_DIR / "knn_02_optimized_results.txt"
    with open(results_path, "w") as f:
        f.write("KNN 02 Optimized - Final Test Results Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: 02_optimized\n\n")
        f.write(f"Dataset Summary:\n")
        f.write(f"  Number of test samples: {X_test.shape[0]}\n")
        f.write(f"  Number of features: {X_test.shape[1]}\n\n")
        f.write(f"  Final TEST Metrics:\n")
        f.write(f"  Accuracy: {test_accuracy:.4f}\n")
        f.write(f"  Precision (weighted): {precision:.4f}\n")
        f.write(f"  Recall (weighted): {recall:.4f}\n")
        f.write(f"  F1-score (weighted): {f1:.4f}\n")
        f.write(f"  Best k value: {best_k}\n\n")
        f.write("Note: This file was generated automatically after training.\n")
    
    print("\n" + "="*60)
    print("Optimized KNN Evaluation Complete")
    print("="*60)


if __name__ == "__main__":
    main()

"""
================================================================================
KNN ANALYSIS: STRENGTHS AND WEAKNESSES
================================================================================

STRENGTHS OF KNN:
-----------------
1. Simple and intuitive: No training phase required, just stores training data
2. Non-parametric: Makes no assumptions about data distribution
3. Naturally handles multi-class problems
4. Can be effective for small datasets with clear class boundaries
5. Lazy learning: Can adapt to new training data without retraining
6. Distance weighting (used here) gives more influence to closer neighbors,
   which can improve performance over uniform voting

WEAKNESSES OF KNN:
------------------
1. Computationally expensive: Requires computing distances to all training samples
   for each prediction (O(n) per prediction). With 12,707 training samples and
   12,288 features, this is very slow.
2. Curse of dimensionality: Performance degrades significantly in high-dimensional
   spaces. With 12,288 features (64*64*3), distances become less meaningful:
   - All points become approximately equidistant
   - The concept of "nearest neighbors" loses meaning
   - This is why KNN works better with lower-dimensional, meaningful features
3. Sensitive to feature scaling: Features with larger scales dominate distance
   calculations. This is why StandardScaler is CRITICAL (and used in this script):
   - Without scaling: pixel intensity differences (0-255) dominate
   - With scaling: all features contribute equally to distance
4. Sensitive to irrelevant features: All features contribute equally to distance,
   even if they don't help distinguish classes
5. Memory intensive: Must store entire training set (12,707 * 12,288 floats)
6. Slow prediction: No model compression, must search through all neighbors
7. Overfitting risk with small k: k=1 would memorize training data perfectly
   but generalize poorly. Larger k reduces overfitting but may underfit.

WHY CLASS IMBALANCE AND DIFFICULT CLASSES REDUCE PERFORMANCE:
---------------------------------------------------------------
1. Class Imbalance:
   - Even with distance weighting, majority classes dominate neighborhoods
   - Minority classes (e.g., Tomato_mosaic_virus with only 373 samples) are
     underrepresented in the training set
   - When querying a test sample, its k-nearest neighbors are more likely to
     come from larger classes simply due to probability
   - Distance weighting helps but doesn't fully solve the imbalance problem

2. Difficult Classes (e.g., Early Blight):
   - Visual similarity between disease classes creates overlapping feature spaces
   - High intra-class variation (different disease stages, lighting, angles,
     leaf positions) means samples from the same class can be far apart
   - Low inter-class distance: Early Blight and Late Blight may have similar
     visual patterns in raw pixel space
   - KNN relies on local neighborhoods, which may contain mixed classes when
     boundaries are not well-separated in the high-dimensional feature space

3. High-Dimensional Feature Space (Curse of Dimensionality):
   - With 12,288 features, the feature space is extremely sparse
   - In high dimensions, most training samples are approximately equidistant
     from any query point
   - This makes the concept of "nearest neighbors" less meaningful
   - Raw pixel values don't capture semantic features that distinguish diseases
   - Better features (e.g., from CNNs) would reduce dimensionality while
     preserving discriminative information

4. Overfitting vs Underfitting Trade-off:
   - Small k (k=1, k=3): Very sensitive to noise, overfits to training data
   - Large k (k=11+): May underfit, smooths out local patterns
   - Optimal k balances these concerns, but in high-dimensional spaces,
     even optimal k may not perform well

IMPROVEMENTS IN THIS SCRIPT:
----------------------------
1. StandardScaler: Normalizes features so all contribute equally to distance
2. Distance weighting: Closer neighbors have more influence than distant ones
3. Hyperparameter search: Finds optimal k value for the dataset
4. Validation-based selection: Chooses k that generalizes best (not just
   highest training accuracy)

Despite these improvements, KNN remains fundamentally limited for high-dimensional
image classification tasks. CNN-based approaches extract meaningful low-dimensional
features that are much better suited for this problem.
"""

