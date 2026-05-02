"""Plotting and IO helpers shared by training and evaluation."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_learning_curves(
    history: dict,
    out_path: str | Path,
    title: str = "Learning Curves",
) -> None:
    """Save a 2-panel figure: loss and accuracy over epochs."""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], label="train", marker="o")
    axes[0].plot(epochs, history["val_loss"], label="val", marker="o")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="train", marker="o")
    axes[1].plot(epochs, history["val_acc"], label="val", marker="o")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    out_path: str | Path,
    title: str = "Confusion Matrix",
    max_labels: int = 30,
) -> None:
    """Render a confusion-matrix heatmap; tick labels are dropped if too dense."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    if len(class_names) <= max_labels:
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90, fontsize=7)
        ax.set_yticklabels(class_names, fontsize=7)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_json(data: dict, out_path: str | Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
