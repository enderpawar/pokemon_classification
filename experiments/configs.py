"""Five experiment configurations covering the three transfer-learning axes.

The first four cover the rubric's required four settings; the fifth (MobileNetV2)
adds an architecture-family comparison.

Axes under study
----------------
- Pretraining:           E1, E2, E3, E5 use ImageNet weights;  E4 trains from scratch.
- Fine-tuning scope:     E1 freezes the backbone (head only);  E2-E5 train all weights.
- Backbone capacity:     E2 (ResNet18) vs E3 (ResNet50) vs E5 (MobileNetV2).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    tag: str
    backbone: str
    mode: str          # "feature_extract" | "full_finetune" | "from_scratch"
    epochs: int
    batch_size: int
    lr: float
    description: str


EXPERIMENTS: list[ExperimentConfig] = [
    ExperimentConfig(
        tag="exp1_resnet18_frozen",
        backbone="resnet18",
        mode="feature_extract",
        epochs=15,
        batch_size=32,
        lr=1e-3,
        description="ResNet18 (ImageNet) — backbone frozen, classifier head only.",
    ),
    ExperimentConfig(
        tag="exp2_resnet18_full",
        backbone="resnet18",
        mode="full_finetune",
        epochs=15,
        batch_size=32,
        lr=3e-4,
        description="ResNet18 (ImageNet) — full fine-tuning of all layers.",
    ),
    ExperimentConfig(
        tag="exp3_resnet50_full",
        backbone="resnet50",
        mode="full_finetune",
        epochs=15,
        batch_size=32,
        lr=2e-4,
        description="ResNet50 (ImageNet) — full fine-tuning. Larger capacity.",
    ),
    ExperimentConfig(
        tag="exp4_resnet18_scratch",
        backbone="resnet18",
        mode="from_scratch",
        epochs=15,
        batch_size=32,
        lr=3e-4,
        description="ResNet18 trained from scratch — no pretrained weights.",
    ),
    ExperimentConfig(
        tag="exp5_mobilenetv2_full",
        backbone="mobilenet_v2",
        mode="full_finetune",
        epochs=15,
        batch_size=32,
        lr=3e-4,
        description="MobileNetV2 (ImageNet) — full fine-tuning. Lightweight backbone.",
    ),
]
