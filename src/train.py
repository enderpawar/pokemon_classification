"""Training entrypoint for a single experiment configuration.

Usage
-----
    python -m src.train --data-dir data/PokemonData --backbone resnet18 \
        --mode full_finetune --epochs 15 --tag resnet18_full

Each run writes:
    checkpoints/{tag}.pth            best model (by val accuracy)
    results/learning_curves/{tag}.png
    results/metrics/{tag}.json       train/val history + test metrics
    results/confusion_matrices/{tag}.png
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .dataset import make_loaders, split_dataset
from .evaluate import evaluate_model
from .model import build_model, trainable_param_count
from .utils import (
    get_device,
    save_confusion_matrix,
    save_json,
    save_learning_curves,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_one_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in tqdm(loader, leave=False, desc="train" if train else "val"):
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    return loss_sum / total, correct / total


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = get_device()
    print(f"[{args.tag}] device={device}, backbone={args.backbone}, mode={args.mode}")

    train_set, val_set, test_set, class_names = split_dataset(
        args.data_dir, image_size=args.image_size, seed=args.seed,
    )
    train_loader, val_loader, test_loader = make_loaders(
        train_set, val_set, test_set,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    print(f"[{args.tag}] classes={len(class_names)}, "
          f"train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    model = build_model(args.backbone, num_classes=len(class_names), mode=args.mode).to(device)
    print(f"[{args.tag}] trainable params = {trainable_param_count(model):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, best_state = 0.0, None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc = run_one_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss); history["val_acc"].append(va_acc)
        print(f"[{args.tag}] epoch {epoch:02d} | "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val loss={va_loss:.4f} acc={va_acc:.4f}")
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    elapsed = time.time() - t0
    print(f"[{args.tag}] training done in {elapsed/60:.1f} min, best val acc={best_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_dir = PROJECT_ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{args.tag}.pth"
    torch.save({
        "state_dict": model.state_dict(),
        "class_names": class_names,
        "backbone": args.backbone,
        "mode": args.mode,
        "image_size": args.image_size,
    }, ckpt_path)
    print(f"[{args.tag}] saved checkpoint -> {ckpt_path}")

    # Test evaluation
    test_metrics, cm = evaluate_model(model, test_loader, device, class_names)
    print(f"[{args.tag}] test: acc={test_metrics['accuracy']:.4f} "
          f"precision={test_metrics['precision_macro']:.4f} "
          f"recall={test_metrics['recall_macro']:.4f} "
          f"f1={test_metrics['f1_macro']:.4f}")

    # Save artifacts
    curves_path = PROJECT_ROOT / "results" / "learning_curves" / f"{args.tag}.png"
    save_learning_curves(history, curves_path, title=f"Learning Curves — {args.tag}")
    cm_path = PROJECT_ROOT / "results" / "confusion_matrices" / f"{args.tag}.png"
    save_confusion_matrix(cm, class_names, cm_path, title=f"Confusion Matrix — {args.tag}")
    metrics_path = PROJECT_ROOT / "results" / "metrics" / f"{args.tag}.json"
    save_json({
        "tag": args.tag,
        "backbone": args.backbone,
        "mode": args.mode,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "best_val_acc": best_acc,
        "elapsed_min": round(elapsed / 60, 2),
        "history": history,
        "test": test_metrics,
        "trainable_params": trainable_param_count(model),
    }, metrics_path)

    return {"tag": args.tag, "best_val_acc": best_acc, "test": test_metrics}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Pokemon classifier.")
    p.add_argument("--data-dir", required=True, help="Path to ImageFolder root.")
    p.add_argument("--backbone", default="resnet18",
                   choices=["resnet18", "resnet50", "mobilenet_v2", "efficientnet_b0"])
    p.add_argument("--mode", default="full_finetune",
                   choices=["feature_extract", "full_finetune", "from_scratch"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", required=True, help="Run identifier used for saved files.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
