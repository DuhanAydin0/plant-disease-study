# tools/export_global_cnn_artifacts.py
#
# Global CNN için deploy artifacts üretir (retrain yok):
# - global_cnn_classes.json        (index sırasıyla class listesi)
# - global_cnn_class_to_idx.json
# - global_cnn_idx_to_class.json
# - global_cnn_config.json
#
# Ayrıca bazı kritik class isimlerinin dataset'te birebir varlığını assert eder
# (mapping/split/path kaymasını erken yakalamak için).

import json
from pathlib import Path

from torchvision import datasets, transforms




# -----------------------------
CKPT_PATH = Path(
    "/Users/duhanaydin/cursor/plant disease study/experiments/cnn/results/03_all_dataset/cnn_03_all_dataset_30epochs_model.pth"
)

FULL_SPLIT_DIR = Path(
    "/Users/duhanaydin/cursor/plant disease study/data/processed/full_split"
)

OUT_DIR = CKPT_PATH.parent

# -----------------------------
# Assumed training config
# -----------------------------
IMAGE_SIZE = 224
MODEL_NAME = "SimpleCNN"
NORMALIZE = None  # Global CNN train'inde normalize yoksa None
NOTES = "ImageFolder(full_split/train). Resize(224)+ToTensor. No normalization (assumed)."

# -----------------------------
# Sanity checks (RAW class names)
# -----------------------------
SANITY_CHECK_CLASSES = [
    "Pepper,_bell___Bacterial_spot",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
]

FOCUS_CLASSES = [
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Tomato___Early_blight",
    "Potato___healthy",
]


def _save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found:\n{CKPT_PATH}")

    train_dir = FULL_SPLIT_DIR / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"full_split train dir not found:\n{train_dir}")

    # Transform sadece ImageFolder'ı kurmak için; class mapping'i etkilemez.
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    ds = datasets.ImageFolder(train_dir, transform=transform)

    classes = ds.classes                 # index sırası önemli
    class_to_idx = ds.class_to_idx       # {class_name: idx}
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}

    # -----------------------------
    # Checks
    # -----------------------------
    missing_sanity = [c for c in SANITY_CHECK_CLASSES if c not in class_to_idx]
    if missing_sanity:
        raise ValueError(
            "SANITY_CHECK_CLASSES içinden dataset'te bulunamayanlar var.\n"
            "Path/split/klasör adlarını kontrol et:\n"
            + "\n".join(f"- {c}" for c in missing_sanity)
        )

    missing_focus = [c for c in FOCUS_CLASSES if c not in class_to_idx]
    if missing_focus:
        raise ValueError(
            "FOCUS_CLASSES içinden dataset'te bulunamayanlar var.\n"
            "Klasör adlarını kontrol et:\n"
            + "\n".join(f"- {c}" for c in missing_focus)
        )

    # -----------------------------
    # Output paths
    # -----------------------------
    classes_path = OUT_DIR / "global_cnn_classes.json"
    c2i_path = OUT_DIR / "global_cnn_class_to_idx.json"
    i2c_path = OUT_DIR / "global_cnn_idx_to_class.json"
    cfg_path = OUT_DIR / "global_cnn_config.json"

    # -----------------------------
    # Save artifacts
    # -----------------------------
    _save_json(classes, classes_path)
    _save_json(class_to_idx, c2i_path)
    _save_json(idx_to_class, i2c_path)

    config = {
        "model_name": MODEL_NAME,
        "image_size": IMAGE_SIZE,
        "normalize": NORMALIZE,
        "num_classes": len(classes),
        "dataset_root": str(FULL_SPLIT_DIR),
        "train_dir_used_for_mapping": str(train_dir),
        "class_naming": "Plant___Disease (raw folder name)",
        "checkpoint_path": str(CKPT_PATH),
        "out_dir": str(OUT_DIR),
        "notes": NOTES,
    }
    _save_json(config, cfg_path)

    # -----------------------------
    # Print summary
    # -----------------------------
    print("\n✅ Global CNN artifacts exported\n")
    print(f"Checkpoint:\n  {CKPT_PATH}")
    print(f"Dataset(train):\n  {train_dir}")
    print(f"Out dir:\n  {OUT_DIR}\n")

    print("Saved:")
    print(f"  - {classes_path.name}")
    print(f"  - {c2i_path.name}")
    print(f"  - {i2c_path.name}")
    print(f"  - {cfg_path.name}\n")

    print(f"Num classes: {len(classes)}")
    print("\nFocus class indices:")
    for c in FOCUS_CLASSES:
        print(f"  {class_to_idx[c]:03d} → {c}")

    print("\nDone.")


if __name__ == "__main__":
    main()
