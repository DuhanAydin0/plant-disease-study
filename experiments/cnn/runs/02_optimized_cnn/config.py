from pathlib import Path

# config.py
# plant disease study/experiments/cnn/runs/02_optimized/config.py

# -------------------------------------------------
# Project root
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[4]

# -------------------------------------------------
# Dataset
# -------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "tomato_split"

# -------------------------------------------------
# Training parameters
# -------------------------------------------------
IMAGE_SIZE = (224, 224)        # SAME as baseline (critical for fair comparison)
BATCH_SIZE = 32                # SAME as baseline
NUM_EPOCHS = 20                # SAME as baseline (no extra epochs yet)
LEARNING_RATE = 1e-4           # Lower LR for more stable training
NUM_CLASSES = 10
RANDOM_SEED = 42               # SAME seed for reproducibility

# -------------------------------------------------
# Output paths
# -------------------------------------------------
RESULTS_DIR = PROJECT_ROOT / "experiments" / "cnn" / "results" / "02_optimized"
MODEL_SAVE_PATH = RESULTS_DIR / "cnn_02_optimized_20epochs_model.pth"
