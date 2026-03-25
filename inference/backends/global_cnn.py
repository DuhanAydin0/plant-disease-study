import json
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from inference.config import GLOBAL_CNN03_CKPT, GLOBAL_CNN03_IDX_TO_CLASS
from inference.model_defs import CNN_03_All_Dataset


def _get_device(device: str) -> torch.device:
    if device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class GlobalCNNBackend:
    """
    Global CNN (03_all_dataset) inference backend.

    Loads:
      - state_dict: GLOBAL_CNN03_CKPT
      - idx_to_class: GLOBAL_CNN03_IDX_TO_CLASS
    """

    def __init__(self, device: str = "cpu", image_size: int = 224):
        self.device = _get_device(device)
        self.image_size = image_size

        # class map
        idx_to_class_raw = json.loads(Path(GLOBAL_CNN03_IDX_TO_CLASS).read_text(encoding="utf-8"))
        self.idx_to_class = {int(k): v for k, v in idx_to_class_raw.items()}
        num_classes = len(self.idx_to_class)

        # model (architecture is now in inference/model_defs.py)
        self.model: nn.Module = CNN_03_All_Dataset(num_classes=num_classes)
        state = torch.load(GLOBAL_CNN03_CKPT, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        # NOTE: must match training/eval normalization
        self.tfm = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def predict_pil(self, img: Image.Image, topk: int = 3) -> Dict[str, Any]:
        x = self.tfm(img.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(x)  # (1, C)
        probs = torch.softmax(logits, dim=1).squeeze(0)  # (C,)

        k = min(int(topk), probs.numel())
        top_probs, top_idx = torch.topk(probs, k=k)

        preds = []
        for p, i in zip(top_probs.tolist(), top_idx.tolist()):
            preds.append({
                "class_id": int(i),
                "class_name": self.idx_to_class.get(int(i), str(i)),
                "prob": float(p),
            })

        return {"backend": "global_cnn", "top1": preds[0], "preds": preds}

    def predict_path(self, image_path: str, topk: int = 3) -> Dict[str, Any]:
        return self.predict_pil(Image.open(image_path), topk=topk)
