"""
Split the prepared tomato dataset into stratified train/val/test folders.

Constraints:
- No augmentation or resizing.
- Preserve class directory structure.
- Copy files (do not move/delete originals).
- Use a fixed random seed for reproducibility.
"""

from __future__ import annotations

import math
import random
import shutil
from pathlib import Path


# Fixed seed for reproducible shuffling
RANDOM_SEED = 42

# Source and target dataset roots
SOURCE_ROOT = Path("data/processed/tomato")
TARGET_ROOT = Path("data/processed/tomato_split")

# Desired split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def collect_class_images(source_root: Path) -> dict[str, list[Path]]:
    """
    Collect image file paths for each class (subdirectory) under the source root.
    """
    class_to_images: dict[str, list[Path]] = {}
    # Walk only immediate subdirectories (each treated as a class)
    for class_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        images = [p for p in class_dir.iterdir() if p.is_file()]
        class_to_images[class_dir.name] = images
    return class_to_images


def stratified_split(
    items: list[Path], train_ratio: float, val_ratio: float
) -> tuple[list[Path], list[Path], list[Path]]:
    """
    Deterministically split items into train, val, and test lists using the given ratios.
    The remainder after flooring goes to the test split.
    """
    rng = random.Random(RANDOM_SEED)
    items_shuffled = items.copy()
    rng.shuffle(items_shuffled)

    n = len(items_shuffled)
    train_count = math.floor(n * train_ratio)
    val_count = math.floor(n * val_ratio)
    test_count = n - train_count - val_count  # remainder to test

    train_items = items_shuffled[:train_count]
    val_items = items_shuffled[train_count : train_count + val_count]
    test_items = items_shuffled[train_count + val_count :]

    # Sanity check to ensure no items lost/duplicated
    assert len(train_items) + len(val_items) + len(test_items) == n
    return train_items, val_items, test_items


def prepare_target_dirs(target_root: Path, class_names: list[str]) -> None:
    """
    Create the target directory structure (train/val/test with class subfolders).
    Does not delete existing contents; simply ensures directories exist.
    """
    for split in ("train", "val", "test"):
        for class_name in class_names:
            (target_root / split / class_name).mkdir(parents=True, exist_ok=True)


def copy_split(
    split_name: str, files: list[Path], target_root: Path, class_name: str
) -> None:
    """
    Copy the provided files into the appropriate split/class folder.
    """
    destination_dir = target_root / split_name / class_name
    for src in files:
        dst = destination_dir / src.name
        shutil.copy2(src, dst)


def main() -> None:
    # Validate source existence
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Source dataset not found: {SOURCE_ROOT}")

    # Gather class -> image list mapping
    class_to_images = collect_class_images(SOURCE_ROOT)
    class_names = sorted(class_to_images.keys())

    # Prepare destination folders
    prepare_target_dirs(TARGET_ROOT, class_names)

    # Stratify and copy for each class independently
    for class_name, images in class_to_images.items():
        train_files, val_files, test_files = stratified_split(
            images, TRAIN_RATIO, VAL_RATIO
        )

        copy_split("train", train_files, TARGET_ROOT, class_name)
        copy_split("val", val_files, TARGET_ROOT, class_name)
        copy_split("test", test_files, TARGET_ROOT, class_name)


if __name__ == "__main__":
    main()

