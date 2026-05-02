"""Model factory for Pokemon classification experiments.

Supports several backbones with two transfer-learning strategies:
- ``feature_extract``: freeze the backbone, train only the classifier head.
- ``full_finetune``: train all weights.
- ``from_scratch``: ignore pretrained weights and train the entire network.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models


SUPPORTED = {"resnet18", "resnet50", "mobilenet_v2", "efficientnet_b0"}


def _replace_classifier(model: nn.Module, name: str, num_classes: int) -> nn.Module:
    if name in {"resnet18", "resnet50"}:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "mobilenet_v2":
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "efficientnet_b0":
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def _freeze_backbone(model: nn.Module, name: str) -> None:
    """Freeze every parameter, then unfreeze the classifier head."""
    for p in model.parameters():
        p.requires_grad = False
    if name in {"resnet18", "resnet50"}:
        for p in model.fc.parameters():
            p.requires_grad = True
    elif name in {"mobilenet_v2", "efficientnet_b0"}:
        for p in model.classifier.parameters():
            p.requires_grad = True


def build_model(
    name: str,
    num_classes: int,
    mode: str = "full_finetune",
) -> nn.Module:
    """Create a model by name with the requested training mode.

    Args:
        name: One of ``SUPPORTED``.
        num_classes: Number of output classes.
        mode: ``feature_extract`` | ``full_finetune`` | ``from_scratch``.
    """
    if name not in SUPPORTED:
        raise ValueError(f"Unknown backbone {name!r}. Supported: {sorted(SUPPORTED)}")
    if mode not in {"feature_extract", "full_finetune", "from_scratch"}:
        raise ValueError(f"Unknown mode {mode!r}")

    use_pretrained = mode != "from_scratch"

    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
        model = models.resnet18(weights=weights)
    elif name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if use_pretrained else None
        model = models.resnet50(weights=weights)
    elif name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if use_pretrained else None
        model = models.mobilenet_v2(weights=weights)
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if use_pretrained else None
        model = models.efficientnet_b0(weights=weights)

    model = _replace_classifier(model, name, num_classes)
    if mode == "feature_extract":
        _freeze_backbone(model, name)
    return model


def trainable_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
