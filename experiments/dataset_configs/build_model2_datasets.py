from pathlib import Path
import shutil
import random
from collections import defaultdict

# =========================
# CONFIG
# =========================
RAW_DIR = Path("data/raw/Plant_leave_diseases_dataset_without_augmentation")
OUT_DIR = Path("data/processed")

SPLIT_RATIOS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15,
}

SEED = 42
random.seed(SEED)

EXCLUDE_DIRS = {"Background_without_leaves"}

# =========================
# HELPERS
# =========================
def normalize_class_name(folder_name: str, plant: str) -> str:
    """
    Tomato___Early_blight -> Early_blight
    """
    return folder_name.replace(f"{plant}___", "").strip().replace(" ", "_")


def split_files(files):
    random.shuffle(files)
    n = len(files)

    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])

    return {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }


# =========================
# MAIN LOGIC
# =========================
def build_all_model2_datasets():
    # plant -> list of class folders
    plant_classes = defaultdict(list)

    for class_dir in RAW_DIR.iterdir():
        if not class_dir.is_dir():
            continue

        name = class_dir.name
        if name in EXCLUDE_DIRS:
            continue

        if "___" not in name:
            continue

        plant = name.split("___")[0]
        plant_classes[plant].append(class_dir)

    print(f"Found {len(plant_classes)} plants")

    for plant, class_dirs in plant_classes.items():
        print(f"\n=== Building MODEL-2 dataset for: {plant} ===")

        plant_out = OUT_DIR / f"model2_{plant.lower()}"
        for split in SPLIT_RATIOS:
            (plant_out / split).mkdir(parents=True, exist_ok=True)

        for class_dir in class_dirs:
            class_name = normalize_class_name(class_dir.name, plant)

            images =[
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]

            if len(images) == 0:
                continue

            splits = split_files(images)

            for split, files in splits.items():
                target_dir = plant_out / split / class_name
                target_dir.mkdir(parents=True, exist_ok=True)

                for img in files:
                    shutil.copy2(img, target_dir / img.name)

            print(f"  - {class_name}: {len(images)} images")

    print("\nAll MODEL-2 datasets built successfully.")


if __name__ == "__main__":
    build_all_model2_datasets()
