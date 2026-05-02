"""Generate learning-curve and comparison figures for the README.

If ``results/summary.json`` exists (i.e. ``experiments/run_all.py`` has been
run), ``make_comparison_bar`` uses the real test metrics. Otherwise it falls
back to illustrative numbers so the README isn't visually empty.

    python scripts/make_demo_assets.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS = PROJECT_ROOT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)


def _curve(start, end, epochs, noise=0.01, rng=None):
    rng = rng or np.random.default_rng(0)
    base = np.linspace(start, end, epochs)
    return base + rng.normal(0, noise, epochs)


def _save_curves(name: str, train_acc, val_acc, train_loss, val_loss, title: str):
    epochs = list(range(1, len(train_acc) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, train_loss, label="train", marker="o")
    axes[0].plot(epochs, val_loss, label="val", marker="o")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, train_acc, label="train", marker="o")
    axes[1].plot(epochs, val_acc, label="val", marker="o")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(ASSETS / name, dpi=120)
    plt.close(fig)


def make_curves():
    epochs = 15
    rng = np.random.default_rng(7)

    # Exp1 — frozen ResNet18: fast plateau, modest val acc
    _save_curves(
        "curves_exp1_resnet18_frozen.png",
        train_acc=_curve(0.10, 0.78, epochs, 0.01, rng),
        val_acc=_curve(0.15, 0.74, epochs, 0.015, rng),
        train_loss=_curve(4.2, 0.95, epochs, 0.05, rng),
        val_loss=_curve(4.0, 1.10, epochs, 0.05, rng),
        title="Learning Curves — exp1_resnet18_frozen (illustrative)",
    )
    # Exp2 — full fine-tune ResNet18: strongest small-model result
    _save_curves(
        "curves_exp2_resnet18_full.png",
        train_acc=_curve(0.20, 0.97, epochs, 0.01, rng),
        val_acc=_curve(0.25, 0.93, epochs, 0.015, rng),
        train_loss=_curve(3.9, 0.18, epochs, 0.04, rng),
        val_loss=_curve(3.6, 0.32, epochs, 0.05, rng),
        title="Learning Curves — exp2_resnet18_full (illustrative)",
    )
    # Exp3 — ResNet50 full: highest val acc
    _save_curves(
        "curves_exp3_resnet50_full.png",
        train_acc=_curve(0.22, 0.98, epochs, 0.01, rng),
        val_acc=_curve(0.27, 0.95, epochs, 0.015, rng),
        train_loss=_curve(3.7, 0.12, epochs, 0.04, rng),
        val_loss=_curve(3.3, 0.25, epochs, 0.05, rng),
        title="Learning Curves — exp3_resnet50_full (illustrative)",
    )
    # Exp4 — from scratch: slow, plateaus low, big train/val gap (overfit)
    _save_curves(
        "curves_exp4_resnet18_scratch.png",
        train_acc=_curve(0.02, 0.55, epochs, 0.02, rng),
        val_acc=_curve(0.02, 0.32, epochs, 0.02, rng),
        train_loss=_curve(5.0, 1.80, epochs, 0.05, rng),
        val_loss=_curve(5.0, 2.90, epochs, 0.06, rng),
        title="Learning Curves — exp4_resnet18_scratch (illustrative)",
    )
    # Exp5 — MobileNetV2 full: comparable to Exp2
    _save_curves(
        "curves_exp5_mobilenetv2_full.png",
        train_acc=_curve(0.18, 0.96, epochs, 0.01, rng),
        val_acc=_curve(0.22, 0.91, epochs, 0.015, rng),
        train_loss=_curve(4.0, 0.20, epochs, 0.04, rng),
        val_loss=_curve(3.7, 0.36, epochs, 0.05, rng),
        title="Learning Curves — exp5_mobilenetv2_full (illustrative)",
    )


def make_comparison_bar():
    """Side-by-side bar chart of test accuracy across the experiments.

    Reads real numbers from ``results/summary.json`` when available; falls back
    to illustrative values if training hasn't been run yet.
    """
    summary_path = PROJECT_ROOT / "results" / "summary.json"
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as f:
            payload = json.load(f)
        experiments = payload["experiments"]
        tags = ["\n".join(e["tag"].split("_")) for e in experiments]
        acc = [e["test"]["accuracy"] for e in experiments]
        f1 = [e["test"]["f1_macro"] for e in experiments]
        title = "Test performance across experiment settings"
    else:
        tags = [
            "exp1\nresnet18\nfrozen",
            "exp2\nresnet18\nfull",
            "exp3\nresnet50\nfull",
            "exp4\nresnet18\nscratch",
            "exp5\nmobilenetv2\nfull",
        ]
        acc = [0.74, 0.93, 0.95, 0.32, 0.91]
        f1 = [0.71, 0.92, 0.94, 0.28, 0.90]
        title = "Test performance across experiment settings (illustrative)"

    x = np.arange(len(tags))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.5))
    b1 = ax.bar(x - width / 2, acc, width, label="Test accuracy")
    b2 = ax.bar(x + width / 2, f1, width, label="Test F1 (macro)")
    ax.set_xticks(x); ax.set_xticklabels(tags, fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{b.get_height():.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "comparison_bar.png", dpi=120)
    plt.close(fig)


def make_demo_screenshot():
    """A simple mock-up of the Streamlit app for the README header."""
    fig = plt.figure(figsize=(11, 5.5))
    fig.patch.set_facecolor("#fafafa")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()

    ax.add_patch(FancyBboxPatch((0.02, 0.85), 0.96, 0.10,
                                boxstyle="round,pad=0.01", linewidth=0,
                                facecolor="#1f2937"))
    ax.text(0.05, 0.90, "Pokemon Classifier — Transfer Learning Demo",
            color="white", fontsize=16, weight="bold")

    # Sidebar
    ax.add_patch(FancyBboxPatch((0.02, 0.05), 0.22, 0.78,
                                boxstyle="round,pad=0.01", linewidth=0,
                                facecolor="#e5e7eb"))
    ax.text(0.04, 0.78, "Model", fontsize=12, weight="bold")
    ax.text(0.04, 0.73, "Checkpoint:", fontsize=10)
    ax.add_patch(FancyBboxPatch((0.04, 0.66), 0.18, 0.045,
                                boxstyle="round,pad=0.005", linewidth=0,
                                facecolor="white"))
    ax.text(0.05, 0.675, "exp3_resnet50_full.pth", fontsize=9)
    ax.text(0.04, 0.60, "Top-K = 5", fontsize=10)
    ax.add_patch(FancyBboxPatch((0.04, 0.18), 0.18, 0.18,
                                boxstyle="round,pad=0.005", linewidth=0,
                                facecolor="#d1fae5"))
    ax.text(0.05, 0.31, "Loaded\nexp3_resnet50_full", fontsize=9, color="#065f46")
    ax.text(0.05, 0.23, "Device: cuda:0\nClasses: 150", fontsize=8, color="#065f46")

    # Input image area
    ax.add_patch(FancyBboxPatch((0.26, 0.10), 0.34, 0.70,
                                boxstyle="round,pad=0.01", linewidth=0,
                                facecolor="white"))
    ax.text(0.43, 0.74, "Input image", ha="center", fontsize=12, weight="bold")
    ax.add_patch(FancyBboxPatch((0.29, 0.18), 0.28, 0.50,
                                boxstyle="round,pad=0.01", linewidth=1,
                                edgecolor="#9ca3af", facecolor="#f3f4f6"))
    ax.text(0.43, 0.43, "[ Pokemon image ]", ha="center", fontsize=11, color="#6b7280")
    ax.text(0.43, 0.13, "rockruff.png", ha="center", fontsize=9, color="#374151")

    # Predictions panel
    ax.add_patch(FancyBboxPatch((0.62, 0.10), 0.36, 0.70,
                                boxstyle="round,pad=0.01", linewidth=0,
                                facecolor="white"))
    ax.text(0.80, 0.74, "Predictions", ha="center", fontsize=12, weight="bold")
    ax.text(0.65, 0.66, "Top-1: Rockruff (96.41%)", fontsize=11, weight="bold", color="#1f2937")

    preds = [("Rockruff", 0.9641),
             ("Lillipup", 0.0182),
             ("Yamper", 0.0094),
             ("Growlithe", 0.0042),
             ("Snubbull", 0.0021)]
    base_y = 0.55
    for i, (name, conf) in enumerate(preds):
        y = base_y - i * 0.07
        ax.text(0.65, y, f"{i+1}. {name}", fontsize=10)
        ax.add_patch(FancyBboxPatch((0.78, y - 0.005), 0.17 * conf, 0.025,
                                    boxstyle="round,pad=0.0", linewidth=0,
                                    facecolor="#3b82f6"))
        ax.text(0.96, y, f"{conf*100:.2f}%", fontsize=9, ha="right")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.savefig(ASSETS / "demo_screenshot.png", dpi=140)
    plt.close(fig)


def main():
    make_curves()
    make_comparison_bar()
    make_demo_screenshot()
    print(f"Wrote demo assets to {ASSETS}")


if __name__ == "__main__":
    main()
