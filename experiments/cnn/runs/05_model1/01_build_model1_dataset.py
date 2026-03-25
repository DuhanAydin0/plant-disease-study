from pathlib import Path
import shutil
import random

# -----------------------------
# CONFIG
# -----------------------------
SOURCE_DIR = Path(
    "data/raw/Plant_leave_diseases_dataset_without_augmentation"
)
TARGET_DIR = Path(
    "data/processed/model1_plant"
)

SPLIT_RATIOS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

RANDOM_SEED = 42
EXCLUDE_CLASS = "Background_without_leaves"

# -----------------------------
# SETUP
# -----------------------------
random.seed(RANDOM_SEED)

for split in SPLIT_RATIOS:
    (TARGET_DIR / split).mkdir(parents=True, exist_ok=True)

# -----------------------------
# COLLECT IMAGES BY PLANT
# -----------------------------
plant_to_images = {}

for class_dir in SOURCE_DIR.iterdir():
    if not class_dir.is_dir():
        continue

    if class_dir.name == EXCLUDE_CLASS:
        continue

    # Example: Apple___Black_rot → Apple
    plant_name = class_dir.name.split("___")[0]

    image_files = list(class_dir.glob("*"))

    if plant_name not in plant_to_images:
        plant_to_images[plant_name] = []

    plant_to_images[plant_name].extend(image_files)

# -----------------------------
# SPLIT & COPY
# -----------------------------
for plant, images in plant_to_images.items():
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * SPLIT_RATIOS["train"])
    n_val = int(n_total * SPLIT_RATIOS["val"])

    split_map = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split, split_images in split_map.items():
        target_class_dir = TARGET_DIR / split / plant
        target_class_dir.mkdir(parents=True, exist_ok=True)

        for img_path in split_images:
            shutil.copy2(
                img_path,
                target_class_dir / img_path.name
            )

    print(
        f"{plant}: "
        f"train={len(split_map['train'])}, "
        f"val={len(split_map['val'])}, "
        f"test={len(split_map['test'])}"
    )

print("\n Model-1 dataset oluşturuldu.")
