"""
Split FULL plant disease dataset into stratified train/val/test sets.

- Copies files (does NOT move or delete originals)
- Preserves class directory structure
- Excludes 'Background_without_leaves' class
- Reproducible via fixed random seed
"""

from pathlib import Path
import random
import shutil

# =========================
# CONFIG
# =========================
RAW_DIR = Path("data/raw/Plant_leave_diseases_dataset_without_augmentation")
OUTPUT_DIR = Path("data/processed/full_split")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42
EXCLUDED_CLASSES = {"Background_without_leaves"}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# =========================
# SETUP
# =========================
random.seed(RANDOM_SEED)

for split in ["train", "val", "test"]:
    (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)

# =========================
# SPLIT LOGIC
# =========================
for class_dir in sorted(RAW_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    class_name = class_dir.name

    if class_name in EXCLUDED_CLASSES:
        print(f"[SKIP] Excluding class: {class_name}")
        continue

    images = [
        p for p in class_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if len(images) == 0:
        print(f"[WARNING] No images found in {class_name}")
        continue

    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    splits = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs
    }

    for split_name, split_images in splits.items():
        target_dir = OUTPUT_DIR / split_name / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

        for img_path in split_images:
            shutil.copy2(img_path, target_dir / img_path.name)

    print(
        f"[OK] {class_name}: "
        f"train={len(train_imgs)}, "
        f"val={len(val_imgs)}, "
        f"test={len(test_imgs)}"
    )

print("\n Full dataset split completed successfully.")
