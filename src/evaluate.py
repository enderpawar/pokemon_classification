"""Test-set evaluation: accuracy, macro/weighted precision/recall/F1, top-5, confusion matrix."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: Sequence[str],
) -> Tuple[dict, np.ndarray]:
    """Return (metrics_dict, confusion_matrix)."""
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[np.ndarray] = []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())
        all_probs.append(probs)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.concatenate(all_probs, axis=0)
    labels_idx = list(range(len(class_names)))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "top5_accuracy": float(top_k_accuracy_score(y_true, y_prob, k=5, labels=labels_idx)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n_test": int(len(y_true)),
        "n_classes": int(len(class_names)),
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    return metrics, cm
