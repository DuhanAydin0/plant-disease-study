# experiments/cnn/runs/06_model2/train_model2_single.py

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# ----------------------------
# CONFIG
# ----------------------------
BATCH_SIZE = 32
IMAGE_SIZE = 224


NUM_WORKERS = 4

RANDOM_SEED = 42

# threshold seçimi (VAL quantile)
THRESHOLD_Q95 = 0.95
THRESHOLD_Q99 = 0.99

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(RANDOM_SEED)


# ----------------------------
# MODEL: Pretrained feature extractor
# ----------------------------
def build_feature_extractor():
    """
    ResNet18 pretrained -> 512-d embedding.
    classification head (fc) kaldırıldı.
    """
    weights = models.ResNet18_Weights.DEFAULT  # ilk seferde otomatik indirir
    model = models.resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    return model


@torch.no_grad()
def extract_embeddings(model, loader):
    model.eval()
    embs = []
    for x, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        z = model(x)                       # (B,512)
        z = nn.functional.normalize(z, dim=1)  # cosine distance için iyi
        embs.append(z.cpu())
    return torch.cat(embs, dim=0)


def cosine_distance(emb, prototype):
    """
    emb: (N, D), prototype: (1, D) normalized
    dist = 1 - cosine_similarity
    """
    return 1.0 - (emb @ prototype.T).squeeze(1)


def main(plant: str):
    # ---- PATHS (deterministik yapı) ----
    data_dir = Path(f"/Users/duhanaydin/cursor/plant disease study/data/processed/model2_{plant}")
    save_dir = Path(f"/Users/duhanaydin/cursor/plant disease study/experiments/cnn/results/06_model2/{plant}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- FEATURE EXTRACTOR ----
    model = build_feature_extractor().to(DEVICE)

    # ---- TRANSFORMS (deterministik) ----
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    # ---- DATASETS ----
    train_ds = datasets.ImageFolder(data_dir / "train", transform=transform)
    val_ds   = datasets.ImageFolder(data_dir / "val", transform=transform)
    test_ds  = datasets.ImageFolder(data_dir / "test", transform=transform)

    # single-class sanity check
    if len(train_ds.classes) != 1:
        raise ValueError(
            f"[{plant}] single-class bekleniyordu ama classes={train_ds.classes}. "
            f"Bu script sadece tek sınıflı bitkiler için."
        )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print("\n==============================")
    print("MODEL-2 SINGLE-CLASS OOD TRAIN STARTED")
    print(f"Plant: {plant}")
    print(f"Class (single): {train_ds.classes[0]}")
    print(f"Device: {DEVICE}")
    print("==============================\n")

    # ---- 1) TRAIN embeddings -> prototype ----
    train_emb = extract_embeddings(model, train_loader)           # (N,512)
    prototype = train_emb.mean(dim=0, keepdim=True)               # (1,512)
    prototype = nn.functional.normalize(prototype, dim=1)         # normalize

    # ---- 2) VAL embeddings -> thresholds ----
    val_emb  = extract_embeddings(model, val_loader)
    val_dist = cosine_distance(val_emb, prototype)

    thr_p95 = float(torch.quantile(val_dist, THRESHOLD_Q95).item())
    thr_p99 = float(torch.quantile(val_dist, THRESHOLD_Q99).item())

    # ---- 3) TEST embeddings -> test stats ----
    test_emb  = extract_embeddings(model, test_loader)
    test_dist = cosine_distance(test_emb, prototype)

    # default threshold'u p95 tutuyoruz (geri uyumluluk)
    default_thr = thr_p95

    metrics = {
        "plant": plant,
        "single_class_name": train_ds.classes[0],
        "embedding_dim": int(train_emb.shape[1]),

        "threshold_p95": thr_p95,
        "threshold_p99": thr_p99,
        "threshold_default": default_thr,

        "val_dist_mean": float(val_dist.mean().item()),
        "val_dist_std": float(val_dist.std().item()),
        "test_dist_mean": float(test_dist.mean().item()),
        "test_dist_std": float(test_dist.std().item()),

        "test_flag_rate_over_p95": float((test_dist > thr_p95).float().mean().item()),
        "test_flag_rate_over_p99": float((test_dist > thr_p99).float().mean().item()),

        "note": "p95 daha agresif, p99 daha konservatif. Deploy'da genelde p99 daha güvenli oluyormuş."
    }

    # ---- SAVE ARTIFACTS ----
    # Bu extractor pretrained olduğu için her plantte aynı; ama deploy kolaylığı için burada da kaydediyoruz.
    torch.save(model.state_dict(), save_dir / "feature_extractor_resnet18.pth")
    torch.save(prototype, save_dir / "prototype.pt")

    with open(save_dir / "single_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Saved:", save_dir / "feature_extractor_resnet18.pth")
    print("Saved:", save_dir / "prototype.pt")
    print("Saved:", save_dir / "single_metrics.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plant",
        required=True,
        help="single-class plant (e.g., blueberry, raspberry, soybean, squash, orange)"
    )
    args = parser.parse_args()
    main(args.plant)
