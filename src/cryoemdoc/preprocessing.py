"""Inference preprocessing copied from the prototype notebooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .constants import ANALYZER_IMAGE_SIZE, IMAGE_CLASSIFIER_SIZE
from .io import read_cryo_pil

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SquarePadResize:
    """Resize while preserving aspect ratio, then pad to a square."""

    def __init__(self, size: int = IMAGE_CLASSIFIER_SIZE, fill: int = 0):
        self.size = int(size)
        self.fill = int(fill)

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.copy()
        img.thumbnail((self.size, self.size), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), (self.fill, self.fill, self.fill))
        x = (self.size - img.width) // 2
        y = (self.size - img.height) // 2
        canvas.paste(img, (x, y))
        return canvas


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("cryoEMdoc inference requires torch. Install with `pip install cryoemdoc`.") from exc
    return torch


def pil_to_normalized_tensor(image: Image.Image):
    """Convert a PIL RGB image to an ImageNet-normalized ``torch.Tensor``."""

    torch = _require_torch()
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN[None, None, :]) / IMAGENET_STD[None, None, :]
    arr = np.transpose(arr, (2, 0, 1)).copy()
    return torch.from_numpy(arr)


def classifier_tensor(path: str | Path, image_size: int = IMAGE_CLASSIFIER_SIZE):
    """Preprocess one image exactly like the classifier eval transform."""

    image = read_cryo_pil(path)
    image = SquarePadResize(image_size, fill=0)(image)
    return pil_to_normalized_tensor(image)


def analyzer_tensor(path: str | Path, image_size: int = ANALYZER_IMAGE_SIZE):
    """Preprocess one image like the square/atlas analyzer eval transform."""

    image = Image.open(path).convert("L")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR).convert("RGB")
    return pil_to_normalized_tensor(image)
