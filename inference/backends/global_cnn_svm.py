import json
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from inference.config import CNN_SVM07_EXTRACTOR_CKPT, CNN_SVM07_SVM_JOBLIB, CNN_SVM07_IDX_TO_CLASS
from inference.model_defs import CNN_03_All_Dataset


def _get_device(device: str) -> torch.device:
    if device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class CNNFeatureExtractor(nn.Module):
    """
    Matches your train_cnn_svm.py:
    embedding = fc1 + ReLU output (pre-dropout), i.e. classifier[1] then classifier[2]

    Assumes CNN_03_All_Dataset:
      conv1, conv2, conv3
      classifier = [Flatten, Linear(...->128), ReLU, Dropout, Linear(128->num_classes)]
    """

    def __init__(self, base: nn.Module):
        super().__init__()
        self.conv1 = base.conv1
        self.conv2 = base.conv2
        self.conv3 = base.conv3
        self.flatten = base.classifier[0]
        self.fc1 = base.classifier[1]
        self.relu = base.classifier[2]

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        return x  # (B, 128)


class GlobalCNNSVMBackend:
    """
    Global CNN + SVM (07_cnn_svm) inference backend.

    Loads:
      - base CNN state_dict: CNN_SVM07_EXTRACTOR_CKPT  (saved via torch.save(base.state_dict()))
      - svm_model.joblib: CNN_SVM07_SVM_JOBLIB
      - idx_to_class.json: CNN_SVM07_IDX_TO_CLASS
    """

    def __init__(self, device: str = "cpu", image_size: int = 224, l2_normalize: bool = True):
        self.device = _get_device(device)
        self.image_size = image_size
        self.l2_normalize = l2_normalize

        # class map
        idx_to_class_raw = json.loads(Path(CNN_SVM07_IDX_TO_CLASS).read_text(encoding="utf-8"))
        self.idx_to_class = {int(k): v for k, v in idx_to_class_raw.items()}
        num_classes = len(self.idx_to_class)

        # base CNN (same arch as CNN_03_All_Dataset)
        base = CNN_03_All_Dataset(num_classes=num_classes)
        state = torch.load(CNN_SVM07_EXTRACTOR_CKPT, map_location="cpu")
        base.load_state_dict(state)
        base.to(self.device)
        base.eval()

        # wrapper for embedding extraction
        self.extractor = CNNFeatureExtractor(base).to(self.device)
        self.extractor.eval()

        # svm classifier
        self.svm = joblib.load(CNN_SVM07_SVM_JOBLIB)

        # transform (must match embedding extraction preprocessing used in training)
        self.tfm = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def _embed(self, img: Image.Image) -> np.ndarray:
        x = self.tfm(img.convert("RGB")).unsqueeze(0).to(self.device)
        emb = self.extractor(x).detach().cpu().numpy().astype(np.float32)  # (1, 128)

        if self.l2_normalize:
            norm = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
            emb = emb / norm

        return emb

    def predict_pil(self, img: Image.Image, topk: int = 3) -> Dict[str, Any]:
        emb = self._embed(img)

        if hasattr(self.svm, "predict_proba"):
            probs = self.svm.predict_proba(emb)[0]  # (C,)
        else:
            scores = np.asarray(self.svm.decision_function(emb)).reshape(-1)
            exps = np.exp(scores - scores.max())
            probs = exps / exps.sum()

        probs = np.asarray(probs).reshape(-1)
        k = min(int(topk), probs.size)
        top_idx = np.argsort(-probs)[:k]

        preds = []
        for i in top_idx:
            preds.append({
                "class_id": int(i),
                "class_name": self.idx_to_class.get(int(i), str(i)),
                "prob": float(probs[i]),
            })

        return {"backend": "global_cnn_svm", "top1": preds[0], "preds": preds}

    def predict_path(self, image_path: str, topk: int = 3) -> Dict[str, Any]:
        return self.predict_pil(Image.open(image_path), topk=topk)
