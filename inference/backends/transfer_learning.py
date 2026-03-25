import json
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

from inference.config import TL08_CKPT, TL08_CLASSES, TL08_IDX_TO_CLASS
from inference.preprocess import TFM_IMAGENET


def _get_device(device: str) -> torch.device:
    if device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_classes() -> List[str]:
    # Preferred: classes.json is ["classA", "classB", ...]
    if Path(TL08_CLASSES).exists():
        return json.loads(Path(TL08_CLASSES).read_text(encoding="utf-8"))

    # Fallback: idx_to_class.json is {"0":"classA", ...}
    if Path(TL08_IDX_TO_CLASS).exists():
        raw = json.loads(Path(TL08_IDX_TO_CLASS).read_text(encoding="utf-8"))
        idx_to_class = {int(k): v for k, v in raw.items()}
        return [idx_to_class[i] for i in range(len(idx_to_class))]

    raise FileNotFoundError(
        f"Neither classes.json nor idx_to_class.json found under: {Path(TL08_CKPT).parent}"
    )


class TransferLearning08Backend:
    """
    08_transfer_learning StageA backend.
    Assumes: ResNet18 architecture with final fc = num_classes, trained with ImageNet normalization.
    Loads: model_best.pth + classes mapping.
    """

    def __init__(self, device: str = "cpu"):
        self.device = _get_device(device)

        if not Path(TL08_CKPT).exists():
            raise FileNotFoundError(f"08 checkpoint not found: {TL08_CKPT}")

        self.classes = _load_classes()
        num_classes = len(self.classes)

        # Build arch (must match training)
        self.model: nn.Module = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        state = torch.load(TL08_CKPT, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict_pil(self, img: Image.Image, topk: int = 3) -> Dict[str, Any]:
        x = TFM_IMAGENET(img.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

        k = min(int(topk), probs.numel())
        top_probs, top_idx = torch.topk(probs, k=k)

        preds = []
        for p, i in zip(top_probs.tolist(), top_idx.tolist()):
            preds.append({
                "class_id": int(i),
                "class_name": self.classes[int(i)],
                "prob": float(p),
            })

        return {"backend": "transfer_learning_08", "top1": preds[0], "preds": preds}

    def predict_path(self, image_path: str, topk: int = 3) -> Dict[str, Any]:
        return self.predict_pil(Image.open(image_path), topk=topk)