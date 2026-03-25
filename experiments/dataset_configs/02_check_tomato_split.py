from pathlib import Path

DATASET_ROOT = Path("data/processed/tomato_split")
SPLITS = ["train", "val", "test"]


def count_images_per_class():
    summary = {}

    for split in SPLITS:
        split_path = DATASET_ROOT / split
        if not split_path.exists():
            raise FileNotFoundError(f"{split_path} not found")

        summary[split] = {}

        for class_dir in sorted(p for p in split_path.iterdir() if p.is_dir()):
            images = [f for f in class_dir.iterdir() if f.is_file()]
            summary[split][class_dir.name] = len(images)

    return summary


def print_summary(summary: dict):
    print("\n=== Dataset Split Summary ===\n")

    for split, classes in summary.items():
        print(f"[{split.upper()}]")
        total = 0
        for class_name, count in classes.items():
            print(f"  {class_name}: {count}")
            total += count
        print(f"  TOTAL: {total}\n")


def main():
    summary = count_images_per_class()
    print_summary(summary)


if __name__ == "__main__":
    main()
