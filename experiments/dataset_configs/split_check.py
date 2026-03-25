from pathlib import Path
from collections import defaultdict

BASE_DIR = Path("/Users/duhanaydin/cursor/plant disease study/data/processed/full_split")

SPLITS = ["train", "val", "test"]


def collect_files():
    data = defaultdict(set)

    for split in SPLITS:
        for cls in (BASE_DIR / split).iterdir():
            if cls.is_dir():
                for img in cls.iterdir():
                    data[(split, cls.name)].add(img.name)
    return data

def check_duplicates(data):
    for cls in set(c for _, c in data.keys()):
        train = data.get(("train", cls), set())
        val = data.get(("val", cls), set())
        test = data.get(("test", cls), set())

        if train & val or train & test or val & test:
            print(f"[WARNING] Duplicate images found in class: {cls}")
        else:
            print(f"[OK] No duplicates for class: {cls}")

def print_counts(data):
    print("\nImage counts per split:")
    for (split, cls), files in sorted(data.items()):
        print(f"{split:5s} | {cls:25s}: {len(files)}")

if __name__ == "__main__":
    data = collect_files()
    print_counts(data)
    check_duplicates(data)
