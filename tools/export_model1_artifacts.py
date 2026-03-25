

# - model1_baseline_cnn.pth içinden class_to_idx çıkarır
# - model1_class_to_idx.json üretir
# - model1_config.json üretir


import json
from pathlib import Path
import torch


CKPT_PATH = Path(
    "/Users/duhanaydin/cursor/plant disease study/experiments/cnn/results/05_model1/model1_baseline_cnn.pth"
)

OUT_DIR = CKPT_PATH.parent
CLASS_JSON_PATH = OUT_DIR / "model1_class_to_idx.json"
CONFIG_JSON_PATH = OUT_DIR / "model1_config.json"


# CONFIG

IMAGE_SIZE = 224
MODEL_NAME = "SimpleCNN"
NORMALIZE = None  # Train'de normalize YOK
NOTES = "Resize(224) + ToTensor, no normalization"


def main():
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    print(f" Loading checkpoint:\n{CKPT_PATH}\n")

    ckpt = torch.load(CKPT_PATH, map_location="cpu")

    if "class_to_idx" not in ckpt:
        raise KeyError(
            "Checkpoint içinde 'class_to_idx' yok.\n"
    
        )

    class_to_idx = ckpt["class_to_idx"]

  
    # class_to_idx.json dosyasını kayıt etme
   
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CLASS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

    print(f" Saved class mapping:\n{CLASS_JSON_PATH}\n")

   
    # config.json dosyasını kayıt etme
    config = {
        "model_name": MODEL_NAME,
        "image_size": IMAGE_SIZE,
        "normalize": NORMALIZE,
        "num_classes": len(class_to_idx),
        "notes": NOTES
    }

    with open(CONFIG_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f" Saved config:\n{CONFIG_JSON_PATH}\n")

  
    print(" Classes:")
    for k, v in sorted(class_to_idx.items(), key=lambda x: x[1]):
        print(f"  {v:02d} → {k}")

    print("\n Export completed successfully.")



if __name__ == "__main__":
    main()
