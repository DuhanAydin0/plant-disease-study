from pathlib import Path

# config.py dosyası:
# plant disease study/experiments/cnn/runs/01_baseline_cnn/config.py

# Proje root'u
PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Dataset yolu
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "tomato_split"

# Eğitim parametreleri
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_CLASSES = 10
RANDOM_SEED = 42

# Çıktı yolları
RESULTS_DIR = PROJECT_ROOT / "experiments" / "cnn" / "results" / "01_baseline"
MODEL_SAVE_PATH = RESULTS_DIR / "baseline_cnn_model.pth"
