# experiments/cnn/runs/08_transfer_learning/model.py
from __future__ import annotations

from typing import Dict, List

import torch.nn as nn
from torchvision import models


def build_resnet18_classifier(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    ResNet18 for multi-class classification.

    - If pretrained=True: loads ImageNet pretrained weights.
    - We always replace the final fc layer to match num_classes.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    m = models.resnet18(weights=weights)

    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    return m


def freeze_all_backbone(model: nn.Module) -> None:
    """Freeze everything except the classification head (fc)."""
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("fc.")


def unfreeze_layer4_and_fc(model: nn.Module) -> None:
    """
    Stage B helper:
    - Unfreeze the last residual block group (layer4)
    - Keep earlier layers frozen
    - Keep fc trainable
    """
    for name, p in model.named_parameters():
        if name.startswith("layer4.") or name.startswith("fc."):
            p.requires_grad = True
        else:
            p.requires_grad = False


def get_param_groups_stage_b(model: nn.Module) -> Dict[str, List[nn.Parameter]]:
    """
    Returns parameter groups for Stage B.
    Used to assign different learning rates (layer4 vs fc).
    """
    layer4_params: List[nn.Parameter] = []
    fc_params: List[nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("layer4."):
            layer4_params.append(p)
        elif name.startswith("fc."):
            fc_params.append(p)

    if not layer4_params:
        raise RuntimeError("No trainable layer4 params found. Did you call unfreeze_layer4_and_fc()?")

    if not fc_params:
        raise RuntimeError("No trainable fc params found.")

    return {"layer4": layer4_params, "fc": fc_params}
