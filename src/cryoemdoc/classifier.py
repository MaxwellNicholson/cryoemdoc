"""Step 1 image classifier inference."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .assets import artifact_path
from .constants import CLASSIFIER_ROUTE_MAP, IMAGE_CLASS_NAMES
from .io import find_images
from .models import load_classifier_checkpoint, require_torch, resolve_device
from .preprocessing import classifier_tensor
from ._standard_output import save_standard_output


@lru_cache(maxsize=8)
def _load_classifier_cached(model_path: str, device_name: str):
    device = resolve_device(device_name)
    return load_classifier_checkpoint(model_path, device, IMAGE_CLASS_NAMES)


def _classify_image_result(
    image_path: str | Path,
    *,
    confidence_threshold: float = 0.60,
    device: str = "auto",
    model_dir: str | Path | None = None,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Classify one image as Atlas, Square, Protein, or Other."""

    torch = require_torch()
    path = Path(image_path)
    resolved_model_path = Path(model_path) if model_path is not None else artifact_path(
        "image_classifier",
        "model",
        model_dir=model_dir,
    )
    resolved_device = resolve_device(device)
    model, class_names = _load_classifier_cached(str(resolved_model_path), str(resolved_device))

    tensor = classifier_tensor(path).unsqueeze(0).to(resolved_device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]
    confidence = float(probs[pred_idx])
    route = CLASSIFIER_ROUTE_MAP.get(pred_label.strip().lower(), "unknown_route")

    return {
        "path": str(path),
        "image_path": str(path),
        "predicted_label": pred_label,
        "confidence": confidence,
        "low_confidence_warning": bool(confidence < confidence_threshold),
        "route": route,
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
    }


def classify_image(
    image_path: str | Path,
    *,
    output: str | Path | None = "",
    confidence_threshold: float = 0.60,
    device: str = "auto",
    model_dir: str | Path | None = None,
    model_path: str | Path | None = None,
) -> str:
    """Classify one image, save standardized JSON, and return a summary string."""

    result = _classify_image_result(
        image_path,
        confidence_threshold=confidence_threshold,
        device=device,
        model_dir=model_dir,
        model_path=model_path,
    )
    return save_standard_output(result, task="classify_image", input_path=image_path, output=output)


def _classify_images_result(
    input_path: str | Path,
    *,
    recursive: bool = True,
    confidence_threshold: float = 0.60,
    device: str = "auto",
    model_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Classify one image or every image in a folder."""

    return [
        _classify_image_result(
            path,
            confidence_threshold=confidence_threshold,
            device=device,
            model_dir=model_dir,
        )
        for path in find_images(input_path, recursive=recursive)
    ]


def classify_images(
    input_path: str | Path,
    *,
    output: str | Path | None = "",
    recursive: bool = True,
    confidence_threshold: float = 0.60,
    device: str = "auto",
    model_dir: str | Path | None = None,
) -> str:
    """Classify one image or every image in a folder, save JSON, and return a summary string."""

    result = _classify_images_result(
        input_path,
        recursive=recursive,
        confidence_threshold=confidence_threshold,
        device=device,
        model_dir=model_dir,
    )
    return save_standard_output(result, task="classify_images", input_path=input_path, output=output)
