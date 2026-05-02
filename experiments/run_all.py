"""Run every configuration in experiments.configs and write a summary table.

    python -m experiments.run_all --data-dir data/PokemonData

The summary lands at ``results/summary.json`` and ``results/summary.md``.
"""
from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

from experiments.configs import EXPERIMENTS
from src.train import train
from src.utils import save_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run all experiments end-to-end.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--only", nargs="*", default=None,
                   help="Optional subset of experiment tags to run.")
    return p.parse_args()


def to_train_args(cfg, common: argparse.Namespace) -> Namespace:
    return Namespace(
        data_dir=common.data_dir,
        backbone=cfg.backbone,
        mode=cfg.mode,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        weight_decay=1e-4,
        image_size=common.image_size,
        num_workers=common.num_workers,
        seed=common.seed,
        tag=cfg.tag,
    )


def write_markdown_table(rows: list[dict], path: Path) -> None:
    header = (
        "| Tag | Backbone | Mode | Epochs | Best Val Acc | Test Acc | Top-5 | "
        "Precision (macro) | Recall (macro) | F1 (macro) |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for r in rows:
        t = r["test"]
        lines.append(
            f"| {r['tag']} | {r['backbone']} | {r['mode']} | {r['epochs']} | "
            f"{r['best_val_acc']:.4f} | {t['accuracy']:.4f} | {t['top5_accuracy']:.4f} | "
            f"{t['precision_macro']:.4f} | {t['recall_macro']:.4f} | {t['f1_macro']:.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    common = parse_args()
    targets = EXPERIMENTS
    if common.only:
        wanted = set(common.only)
        targets = [c for c in EXPERIMENTS if c.tag in wanted]
        if not targets:
            raise SystemExit(f"No experiments matched --only {common.only}")

    rows = []
    for cfg in targets:
        print("=" * 80)
        print(f"Running {cfg.tag}: {cfg.description}")
        print("=" * 80)
        result = train(to_train_args(cfg, common))
        rows.append({
            "tag": cfg.tag,
            "backbone": cfg.backbone,
            "mode": cfg.mode,
            "epochs": cfg.epochs,
            "best_val_acc": result["best_val_acc"],
            "test": result["test"],
            "description": cfg.description,
        })

    save_json({"experiments": rows}, PROJECT_ROOT / "results" / "summary.json")
    write_markdown_table(rows, PROJECT_ROOT / "results" / "summary.md")
    print("\nSummary written to results/summary.json and results/summary.md")


if __name__ == "__main__":
    main()
