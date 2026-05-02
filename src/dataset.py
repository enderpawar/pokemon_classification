"""Pokemon dataset loading and splitting utilities.

The dataset is the "7,000 Labeled Pokemon" set from Kaggle, organised as one
sub-directory per class (ImageFolder layout).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int = 224, augment: bool = True):
    """Return (train_transform, eval_transform).

    Eval transform is deterministic; train transform adds light augmentation.
    """
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    if not augment:
        return eval_tf, eval_tf

    train_tf = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def split_dataset(
    data_dir: str | Path,
    image_size: int = 224,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    augment: bool = True,
) -> Tuple[Subset, Subset, Subset, list[str]]:
    """Load an ImageFolder dataset and split it into train/val/test.

    Train uses augmentation; val/test use deterministic transforms.
    Returns (train_set, val_set, test_set, class_names).
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}. "
            "Download 7,000 Labeled Pokemon from Kaggle and extract it."
        )

    train_tf, eval_tf = build_transforms(image_size=image_size, augment=augment)

    full_train_view = datasets.ImageFolder(str(data_dir), transform=train_tf)
    full_eval_view = datasets.ImageFolder(str(data_dir), transform=eval_tf)
    class_names = full_train_view.classes

    n_total = len(full_train_view)
    n_test = int(round(n_total * test_ratio))
    n_val = int(round(n_total * val_ratio))
    n_train = n_total - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx, test_idx = random_split(
        range(n_total), [n_train, n_val, n_test], generator=generator
    )

    train_set = Subset(full_train_view, list(train_idx))
    val_set = Subset(full_eval_view, list(val_idx))
    test_set = Subset(full_eval_view, list(test_idx))
    return train_set, val_set, test_set, class_names


def make_loaders(
    train_set, val_set, test_set,
    batch_size: int = 32,
    num_workers: int = 0,
):
    """Create DataLoaders. num_workers=0 by default for Windows safety."""
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader
