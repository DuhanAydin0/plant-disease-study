

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from PIL import Image


# ----------------------------
# CONFIG
# ----------------------------
BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_WORKERS = 4

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ImageNet normalize (ResNet18 pretrained)
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    NORMALIZE,
])


# ----------------------------
# Simple dataset for flat folders (no class subfolders needed)
# ----------------------------
class FlatImageFolder(Dataset):
    def __init__(self, folder: Path, transform=None):
        self.folder = Path(folder)
        self.transform = transform
        exts = {".jpg", ".jpeg"}
        self.paths = sorted([p for p in self.folder.rglob("*") if p.suffix.lower() in exts])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # label yok -> -1
        return img, -1, str(p)


# ----------------------------
# MODEL loader: ResNet18 feature extractor
# ----------------------------
def build_feature_extractor():
    # aynı mimari: resnet18 + fc identity
    model = models.resnet18(weights=None)
    model.fc = nn.Identity()
    return model


@torch.no_grad()
def extract_embeddings(model, loader):
    model.eval()
    embs = []
    meta = []  # path veya label vb.

    for batch in loader:
        # loader'dan gelen format farklı olabilir
        if len(batch) == 2:
            x, y = batch
            paths = None
        else:
            x, y, paths = batch

        x = x.to(DEVICE)
        z = model(x)  # (B,512)
        z = nn.functional.normalize(z, dim=1)
        embs.append(z.cpu())

        if paths is None:
            meta.extend([None] * x.size(0))
        else:
            meta.extend(list(paths))

    return torch.cat(embs, dim=0), meta


def cosine_distance(emb, prototype):
    # emb: (N,D), prototype: (1,D) normalized
    return 1.0 - (emb @ prototype.T).squeeze(1)


def summarize_dist(dist: torch.Tensor):
    return {
        "n": int(dist.numel()),
        "mean": float(dist.mean().item()) if dist.numel() else None,
        "std": float(dist.std().item()) if dist.numel() > 1 else 0.0,
        "p50": float(torch.quantile(dist, 0.50).item()) if dist.numel() else None,
        "p90": float(torch.quantile(dist, 0.90).item()) if dist.numel() else None,
        "p95": float(torch.quantile(dist, 0.95).item()) if dist.numel() else None,
        "p99": float(torch.quantile(dist, 0.99).item()) if dist.numel() else None,
        "min": float(dist.min().item()) if dist.numel() else None,
        "max": float(dist.max().item()) if dist.numel() else None,
    }


def topk_paths_by_dist(dist: torch.Tensor, paths, k=5):
    if dist.numel() == 0:
        return []
    k = min(k, dist.numel())
    vals, idxs = torch.topk(dist, k=k, largest=True)
    out = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        out.append({"dist": float(v), "path": paths[i]})
    return out


def main(plant: str, threshold: str):
    # ---- PATHS ----
    data_dir = Path(f"/Users/duhanaydin/cursor/plant disease study/data/processed/model2_{plant}")
    save_dir = Path(f"/Users/duhanaydin/cursor/plant disease study/experiments/cnn/results/06_model2/{plant}")

    model_path = save_dir / "feature_extractor_resnet18.pth"
    proto_path = save_dir / "prototype.pt"
    metrics_path = save_dir / "single_metrics.json"

    if not model_path.exists():
        raise FileNotFoundError(f"feature extractor bulunamadı: {model_path}")
    if not proto_path.exists():
        raise FileNotFoundError(f"prototype bulunamadı: {proto_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"single_metrics bulunamadı: {metrics_path}")

    # ---- LOAD threshold ----
    with open(metrics_path, "r") as f:
        single_metrics = json.load(f)

    if threshold == "p95":
        thr = float(single_metrics["threshold_p95"])
    elif threshold == "p99":
        thr = float(single_metrics["threshold_p99"])
    else:
        # default: json'da threshold_default varsa onu al, yoksa p95
        thr = float(single_metrics.get("threshold_default", single_metrics["threshold_p95"]))

    # ---- LOAD model + prototype ----
    model = build_feature_extractor().to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    prototype = torch.load(proto_path, map_location="cpu")  # (1,512) normalized
    prototype = prototype.float()

    # ---- ID TEST (ImageFolder) ----
    test_dir = data_dir / "test"
    test_ds = datasets.ImageFolder(test_dir, transform=TRANSFORM)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    test_emb, _ = extract_embeddings(model, test_loader)
    test_dist = cosine_distance(test_emb, prototype)
    test_flag = (test_dist > thr).float().mean().item() if test_dist.numel() else None

    # ---- OOD TEST image folder kullanmadım çünkü test resimlerini direkt ood_test klasörüne ekledim. ----
    ood_dir = data_dir / "ood_test"
    ood_summary = None
    ood_top = []
    if ood_dir.exists():
        ood_ds = FlatImageFolder(ood_dir, transform=TRANSFORM)
        if len(ood_ds) > 0:
            ood_loader = DataLoader(ood_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
            ood_emb, ood_paths = extract_embeddings(model, ood_loader)
            ood_dist = cosine_distance(ood_emb, prototype)
            ood_flag = (ood_dist > thr).float().mean().item()
            ood_summary = summarize_dist(ood_dist)
            ood_summary["flag_rate_over_threshold"] = float(ood_flag)
            ood_top = topk_paths_by_dist(ood_dist, ood_paths, k=5)
        else:
            ood_summary = {"n": 0, "note": "ood_test klasörü boş"}
    else:
        ood_summary = {"note": "ood_test klasörü yok"}

    # ---- PRINT + SAVE ----
    out = {
        "plant": plant,
        "threshold_choice": threshold,
        "threshold_value": float(thr),
        "id_test": {
            "summary": summarize_dist(test_dist),
            "flag_rate_over_threshold": float(test_flag),
            "n_classes": len(test_ds.classes),
            "classes": test_ds.classes,
        },
        "ood_test": ood_summary,
        "ood_top_examples": ood_top,
        
    }

    print("=== SINGLE-CLASS EVALUATION (ID vs OOD) ===")
    print(json.dumps(out, indent=2, ensure_ascii=False))

    with open(save_dir / "single_evaluation.json", "w") as f:
        json.dump(out, f, indent=4, ensure_ascii=False)

    print("\nSaved:", save_dir / "single_evaluation.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant", required=True, help="e.g. blueberry, raspberry, soybean, squash, orange")
    parser.add_argument(
        "--threshold",
        default="default",
        choices=["default", "p95", "p99"],
        help="Which threshold to use from single_metrics.json",
    )
    args = parser.parse_args()
    main(args.plant, args.threshold)
