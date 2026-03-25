import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ..config import MODEL1_CKPT, MODEL2_DIR, IMAGE_SIZE, SINGLE_PLANTS, SINGLE_THRESHOLD_KEY
from ..labels import plant_id_from_model1_label, clean_display_name
from ..preprocess import TFM_SIMPLE, TFM_IMAGENET
from ..model_defs import SimpleCNN, build_resnet18_feature_extractor


def _invert_class_to_idx(class_to_idx: Dict[str, int]) -> Dict[int, str]:
    return {int(v): k for k, v in class_to_idx.items()}

def _load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def _softmax_top1(logits: torch.Tensor) -> Tuple[int, float]:
    probs = F.softmax(logits, dim=1)
    conf, idx = probs.max(dim=1)
    return int(idx.item()), float(conf.item())


class Model1Model2Backend:
    """
    Hierarchical inference:
      image -> Model1 (plant) -> route:
        - single plants: ResNet18 embedding + prototype + threshold => ID/OOD
        - others: per-plant SimpleCNN classifier
    """

    def __init__(self, device: Optional[str] = None):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)

        # ---------- Load Model-1 ----------
        if not MODEL1_CKPT.exists():
            raise FileNotFoundError(f"Model-1 checkpoint not found: {MODEL1_CKPT}")

        ckpt1 = torch.load(MODEL1_CKPT, map_location="cpu")
        if "model_state_dict" not in ckpt1 or "class_to_idx" not in ckpt1:
            raise KeyError("Model-1 checkpoint must contain 'model_state_dict' and 'class_to_idx'.")

        self.model1_idx_to_class = _invert_class_to_idx(ckpt1["class_to_idx"])
        self.model1 = SimpleCNN(num_classes=len(ckpt1["class_to_idx"]), image_size=IMAGE_SIZE)
        self.model1.load_state_dict(ckpt1["model_state_dict"])
        self.model1.to(self.device).eval()

        # ---------- Load shared ResNet18 extractor (for all single plants) ----------
        self.feat_extractor = build_resnet18_feature_extractor().to(self.device).eval()

        # Load extractor state_dict from any single plant folder (they should be identical)
        extractor_sd = None
        for p in SINGLE_PLANTS:
            cand = MODEL2_DIR / p / "feature_extractor_resnet18.pth"
            if cand.exists():
                extractor_sd = torch.load(cand, map_location="cpu")
                break
        if extractor_sd is None:
            raise FileNotFoundError(
                "Could not find any 'feature_extractor_resnet18.pth' under single plant folders."
            )
        self.feat_extractor.load_state_dict(extractor_sd)
        self.feat_extractor.eval()

        # Caches
        self._model2_multi_cache: Dict[str, Tuple[nn.Module, Dict[int, str]]] = {}
        self._single_cache: Dict[str, Dict[str, Any]] = {}

    def _load_model2_multi(self, plant_id: str) -> Tuple[nn.Module, Dict[int, str]]:
        if plant_id in self._model2_multi_cache:
            return self._model2_multi_cache[plant_id]

        ckpt_path = MODEL2_DIR / plant_id / "best_model.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Model-2 best_model.pth not found for '{plant_id}': {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" not in ckpt or "class_to_idx" not in ckpt:
            raise KeyError(f"Model-2 checkpoint for '{plant_id}' must contain 'model_state_dict' and 'class_to_idx'.")

        idx_to_class = _invert_class_to_idx(ckpt["class_to_idx"])
        model = SimpleCNN(num_classes=len(ckpt["class_to_idx"]), image_size=IMAGE_SIZE)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device).eval()

        self._model2_multi_cache[plant_id] = (model, idx_to_class)
        return model, idx_to_class

    def _load_single_artifacts(self, plant_id: str) -> Dict[str, Any]:
        if plant_id in self._single_cache:
            return self._single_cache[plant_id]

        plant_dir = MODEL2_DIR / plant_id
        proto_path = plant_dir / "prototype.pt"
        metrics_path = plant_dir / "single_metrics.json"

        if not proto_path.exists() or not metrics_path.exists():
            raise FileNotFoundError(f"Single-class artifacts missing for '{plant_id}' in: {plant_dir}")

        prototype = torch.load(proto_path, map_location="cpu")  # (1, 512)
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)

        if SINGLE_THRESHOLD_KEY not in m:
            raise KeyError(f"single_metrics.json missing '{SINGLE_THRESHOLD_KEY}' for '{plant_id}'.")

        info = {
            "prototype": prototype,
            "threshold": float(m[SINGLE_THRESHOLD_KEY]),
            "single_class_name": m["single_class_name"],
        }
        self._single_cache[plant_id] = info
        return info

    @torch.inference_mode()
    def predict_one(self, image_path: str) -> Dict[str, Any]:
        img = _load_image(image_path)

        # ---------- Model-1 plant ----------
        x1 = TFM_SIMPLE(img).unsqueeze(0).to(self.device)
        logits1 = self.model1(x1)
        plant_idx, plant_conf = _softmax_top1(logits1)

        raw_plant_label = self.model1_idx_to_class[plant_idx]   # e.g., "Pepper,_bell"
        plant_id = plant_id_from_model1_label(raw_plant_label)  # e.g., "pepper_bell"
        plant_display = clean_display_name(raw_plant_label)     # e.g., "Pepper Bell"

        # ---------- Route ----------
        if plant_id in SINGLE_PLANTS:
            # Single-class OOD
            info = self._load_single_artifacts(plant_id)
            proto = info["prototype"].to(self.device)
            thr = float(info["threshold"])

            x2 = TFM_IMAGENET(img).unsqueeze(0).to(self.device)
            emb = self.feat_extractor(x2)            # (1, 512)
            emb = F.normalize(emb, dim=1)
            proto = F.normalize(proto, dim=1)

            dist = 1.0 - torch.sum(emb * proto, dim=1)  # cosine distance
            dist_val = float(dist.item())
            is_ood = dist_val > thr
            id_margin = thr - dist_val
            id_score = max(0.0, min(1.0, 1.0 - (dist_val / thr))) if thr > 0 else 0.0


            return {
                "plant": plant_display,
                "plant_id": plant_id,
                "plant_conf": plant_conf,
                "mode": "singleclass_ood",
                "disease": None if is_ood else info["single_class_name"],
                "disease_conf": None,
                "ood": bool(is_ood),
                "distance": dist_val,
                "threshold": thr,
                "id_margin": id_margin,
                "id_score": id_score,
            }

        # Multi/Binary CNN
        model2, idx_to_class = self._load_model2_multi(plant_id)
        logits2 = model2(x1)
        dis_idx, dis_conf = _softmax_top1(logits2)
        disease = idx_to_class[dis_idx]

        return {
            "plant": plant_display,
            "plant_id": plant_id,
            "plant_conf": plant_conf,
            "mode": "multiclass_or_binary",
            "disease": disease,
            "disease_conf": dis_conf,
            "ood": False,
            "distance": None,
            "threshold": None,
        }
