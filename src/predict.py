"""Single-image prediction helper used by the Streamlit GUI."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torchvision import transforms

from .dataset import IMAGENET_MEAN, IMAGENET_STD
from .model import build_model


def load_checkpoint(ckpt_path: str | Path, device: torch.device | None = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    model = build_model(ckpt["backbone"], num_classes=len(class_names), mode="full_finetune")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    image_size = ckpt.get("image_size", 224)
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return model, class_names, tf, device


@torch.no_grad()
def predict_topk(
    image: Image.Image,
    model,
    class_names: List[str],
    transform,
    device: torch.device,
    k: int = 5,
) -> List[Tuple[str, float]]:
    image = image.convert("RGB")
    x = transform(image).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    top_idx = probs.argsort()[::-1][:k]
    return [(class_names[i], float(probs[i])) for i in top_idx]
