# experiments/cnn/runs/08_transfer_learning/train.py
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import (
    build_resnet18_classifier,
    freeze_all_backbone,
    unfreeze_layer4_and_fc,
    get_param_groups_stage_b,
)


DATA_DIR = Path(r"/Users/duhanaydin/cursor/plant disease study/data/processed/full_split")
BASE_RESULTS_DIR = Path(r"/Users/duhanaydin/cursor/plant disease study/experiments/cnn/results/08_transfer_learning")
STAGE_A_DIR = BASE_RESULTS_DIR / "stageA"
STAGE_B_DIR = BASE_RESULTS_DIR / "stageB"

# ==========================
# Transforms 
# ==========================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

# ==========================
# Utils
# ==========================
def seed_everything(seed: int = 42) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ✅ UPDATED: CUDA-only calls should be guarded (Mac MPS/CPU'da gereksiz warning/overhead olmasın)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ✅ UPDATED: cudnn flags only matter on CUDA
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# UPDATED: accuracy-only yerine loss+acc dönen eval fonksiyonu
@torch.no_grad()
def eval_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float]:
    """
    Returns: (val_loss, val_acc)
    """
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        bs = y.size(0)
        loss_sum += loss.item() * bs
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()

    val_loss = loss_sum / max(total, 1)
    val_acc = correct / max(total, 1)
    return val_loss, val_acc


def get_stage_dirs(stage: str) -> Tuple[Path, Optional[Path]]:
    stage = stage.upper().strip()
    if stage == "A":
        return STAGE_A_DIR, None
    if stage == "B":
        # Stage B initializes from Stage A best checkpoint
        init_ckpt = STAGE_A_DIR / "model_best.pth"
        return STAGE_B_DIR, init_ckpt
    raise ValueError("stage must be 'A' or 'B'")


def save_class_maps(out_dir: Path, dataset: datasets.ImageFolder) -> None:
    class_to_idx = dataset.class_to_idx
    idx_to_class = {str(v): k for k, v in class_to_idx.items()}
    classes = dataset.classes

    (out_dir / "class_to_idx.json").write_text(json.dumps(class_to_idx, indent=2, ensure_ascii=False))
    (out_dir / "idx_to_class.json").write_text(json.dumps(idx_to_class, indent=2, ensure_ascii=False))
    (out_dir / "classes.json").write_text(json.dumps(classes, indent=2, ensure_ascii=False))


#  UPDATED: LR loglamak için helper (Stage A / Stage B uyumlu)
def get_current_lrs(optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    """
    Stage A: {"lr": ...}
    Stage B: {"lr_layer4": ..., "lr_fc": ...}
    """
    if len(optimizer.param_groups) == 1:
        return {"lr": float(optimizer.param_groups[0]["lr"])}
    # Stage B: [0]=layer4, [1]=fc (biz böyle kurduk)
    return {
        "lr_layer4": float(optimizer.param_groups[0]["lr"]),
        "lr_fc": float(optimizer.param_groups[1]["lr"]),
    }


def train_one_experiment(stage: str) -> None:
    # -----------------------
    # Fixed hyperparams by stage
    # -----------------------
    # Stage A: train only fc (frozen backbone)
    # Stage B: unfreeze layer4+fc, very small LR on layer4
    cfg: Dict[str, object]
    if stage.upper() == "A":
        cfg = dict(
            stage="A",
            max_epochs=20,
            patience=4,
            batch_size=64,
            lr=1e-3,
            weight_decay=1e-4,

            # ✅ UPDATED: scheduler ayarları (val_loss plato -> LR düşür)
            scheduler=dict(
                name="ReduceLROnPlateau",
                monitor="val_loss",
                mode="min",
                factor=0.5,
                patience=2,
                min_lr=1e-6,
            ),
        )
    else:
        cfg = dict(
            stage="B",
            max_epochs=10,
            patience=3,
            batch_size=64,
            lr_layer4=1e-5,
            lr_fc=1e-4,
            weight_decay=1e-4,

            # ✅ UPDATED: scheduler ayarları (val_loss plato -> LR düşür)
            scheduler=dict(
                name="ReduceLROnPlateau",
                monitor="val_loss",
                mode="min",
                factor=0.5,
                patience=1,
                min_lr=1e-6,
            ),
        )

    seed_everything(42)

    # device seçimi (M3 için MPS)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    out_dir, init_ckpt = get_stage_dirs(stage)
    ensure_dir(out_dir)

    # -----------------------
    # Datasets / loaders
    # -----------------------
    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Expected split folders under {DATA_DIR}: train/, val/ (got train={train_dir.exists()}, val={val_dir.exists()})."
        )

    train_ds = datasets.ImageFolder(str(train_dir), transform=build_transforms(train=True))
    val_ds = datasets.ImageFolder(str(val_dir), transform=build_transforms(train=False))

    save_class_maps(out_dir, train_ds)

    num_classes = len(train_ds.classes)
    model = build_resnet18_classifier(num_classes=num_classes, pretrained=True)

    # -----------------------
    # Stage-specific setup
    # -----------------------
    if stage.upper() == "A":
        freeze_all_backbone(model)
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
        )
    else:
        if init_ckpt is None or not init_ckpt.exists():
            raise FileNotFoundError(
                f"Stage B requires Stage A checkpoint at {init_ckpt}. Run Stage A first."
            )
        # load Stage A best
        state = torch.load(init_ckpt, map_location="cpu")
        model.load_state_dict(state)

        unfreeze_layer4_and_fc(model)
        groups = get_param_groups_stage_b(model)
        optimizer = torch.optim.AdamW(
            [
                {"params": groups["layer4"], "lr": float(cfg["lr_layer4"])},
                {"params": groups["fc"], "lr": float(cfg["lr_fc"])},
            ],
            weight_decay=float(cfg["weight_decay"]),
        )

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ✅ UPDATED: ReduceLROnPlateau scheduler (val_loss izleyecek)
    sched_cfg = cfg.get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=str(sched_cfg.get("mode", "min")),
        factor=float(sched_cfg.get("factor", 0.5)),
        patience=int(sched_cfg.get("patience", 2)),
        min_lr=float(sched_cfg.get("min_lr", 1e-6)),

    )

    # -----------------------
    # Train loop + Early stopping
    # -----------------------
    # UPDATED: Early stopping ve best checkpoint kriterini val_loss yaptık.
    # Neden: accuracy plateau iken model confidence overfit yapabilir; loss bunu yakalar.
    best_val_loss = float("inf")
    best_val_acc = -1.0  # config raporunda dursun
    best_epoch = -1
    bad_epochs = 0
    history_rows = []

    best_path = out_dir / "model_best.pth"
    last_path = out_dir / "model_last.pth" # I saved last bcs some times val_loss can be best but val acc worst

    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        #  UPDATED: val_loss + val_acc aynı anda hesaplanıyor
        val_loss, val_acc = eval_metrics(model, val_loader, device, criterion)

        #  UPDATED: scheduler step (val_loss üzerinden)
        scheduler.step(val_loss)

        #  UPDATED: LR logla (Stage A/B uyumlu)
        lrs = get_current_lrs(optimizer)

        #  UPDATED: early stopping criterion = val_loss
        is_best = False
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            bad_epochs = 0
            is_best = True
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1

        # checkpoint last
        torch.save(model.state_dict(), last_path)

        #  UPDATED: daha açıklayıcı log (val_loss + lr + bad_epochs)
        if "lr" in lrs:
            lr_str = f"lr={lrs['lr']:.2e}"
        else:
            lr_str = f"lr_layer4={lrs['lr_layer4']:.2e} lr_fc={lrs['lr_fc']:.2e}"

        print(
            f"[{stage.upper()}] epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"{lr_str} | bad_epochs={bad_epochs}"
        )

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,          # UPDATED
            "val_acc": val_acc,
            "bad_epochs": bad_epochs,      # UPDATED
            "is_best": int(is_best),       # UPDATED
        }
        #  UPDATED: LR kolonlarını history'ye ekle
        history_row.update(lrs)
        history_rows.append(history_row)

        # early stop
        if bad_epochs >= int(cfg["patience"]):
            print(
                f"[{stage.upper()}] Early stopping at epoch {epoch} "
                f"(best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f}, best_val_acc={best_val_acc:.4f})."
            )
            break

    # -----------------------
    # Save training artifacts
    # -----------------------
    config_out = {
        "run_name": "08_transfer_learning",
        "stage": stage.upper(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(DATA_DIR),
        "out_dir": str(out_dir),
        "device": str(device),
        "num_classes": num_classes,
        "hyperparams": cfg,
        "best": {
            "epoch": best_epoch,                #  UPDATED
            "val_loss": best_val_loss,          #  UPDATED
            "val_acc": best_val_acc,            #  UPDATED
        },
        "ckpt_best": str(best_path),
        "ckpt_last": str(last_path),
        "init_ckpt": str(init_ckpt) if init_ckpt else None,
        "transforms": {
            "train": "Resize(224,224)+HFlip+Rot(10)+ToTensor+ImageNetNorm",
            "eval": "Resize(224,224)+ToTensor+ImageNetNorm",
        },
        #  UPDATED: scheduler bilgisi config'e yazılsın ki GitHub story’de net olsun
        "scheduler": cfg.get("scheduler", None),
    }
    (out_dir / "train_config.json").write_text(json.dumps(config_out, indent=2, ensure_ascii=False))

    #  UPDATED: CSV kolonları genişledi (val_loss + lr + is_best + bad_epochs)
    # fieldnames'i dinamik kuruyoruz (Stage A/B LR farkı sorun olmasın)
    fieldnames = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "bad_epochs", "is_best"]
    # LR kolonları
    if stage.upper() == "A":
        fieldnames += ["lr"]
    else:
        fieldnames += ["lr_layer4", "lr_fc"]

    with (out_dir / "train_history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in history_rows:
            # CSV’de olmayan extra key varsa sorun olmasın diye sadece fieldnames'i al
            w.writerow({k: r.get(k) for k in fieldnames})

    print(f"[{stage.upper()}] Saved: {best_path}")
    print(f"[{stage.upper()}] Saved: {last_path}")
    print(f"[{stage.upper()}] Saved: {out_dir / 'train_config.json'}")
    print(f"[{stage.upper()}] Saved: {out_dir / 'train_history.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["A", "B", "a", "b"], required=True)
    args = parser.parse_args()

    train_one_experiment(args.stage)


if __name__ == "__main__":
    main()