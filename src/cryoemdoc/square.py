"""Square analyzer inference."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .assets import artifact_path, read_json
from .constants import SQUARE_NO_TAG_LABEL
from .io import find_images
from .models import build_multitask_resnet18, load_state_dict_model, require_torch, resolve_device
from .postprocessing import (
    apply_square_rating_threshold,
    apply_tag_thresholds,
    postprocess_square_tags,
    square_recommendations_from_tags,
    tags_from_binary,
)
from .preprocessing import analyzer_tensor
from ._standard_output import save_standard_output


def _square_paths(model_dir: str | Path | None = None) -> dict[str, Path]:
    return {
        "model": artifact_path("square_analyzer", "model", model_dir=model_dir),
        "labels": artifact_path("square_analyzer", "labels", model_dir=model_dir),
        "thresholds": artifact_path("square_analyzer", "thresholds", model_dir=model_dir),
    }


@lru_cache(maxsize=8)
def _load_square_cached(model_path: str, labels_path: str, thresholds_path: str, device_name: str):
    device = resolve_device(device_name)
    labels = read_json(labels_path)
    thresholds = read_json(thresholds_path)
    tag_classes = list(labels["tags"])
    rating_classes = list(labels["rating_classes"])
    model = build_multitask_resnet18(len(tag_classes), len(rating_classes))
    model = load_state_dict_model(model, model_path, device)
    return model, labels, thresholds


def _analyze_square_result(
    image_path: str | Path,
    *,
    device: str = "auto",
    model_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run square analyzer inference on one image."""

    torch = require_torch()
    paths = _square_paths(model_dir)
    resolved_device = resolve_device(device)
    model, labels, thresholds = _load_square_cached(
        str(paths["model"]),
        str(paths["labels"]),
        str(paths["thresholds"]),
        str(resolved_device),
    )
    tag_classes = list(labels["tags"])
    rating_classes = list(labels["rating_classes"])
    rating_to_idx = dict(labels["rating_to_idx"])

    tensor = analyzer_tensor(image_path).unsqueeze(0).to(resolved_device)
    with torch.no_grad():
        outputs = model(tensor)
        tag_probs = torch.sigmoid(outputs["tag_logits"])[0].detach().cpu().numpy()
        rating_probs = torch.softmax(outputs["rating_logits"], dim=1)[0].detach().cpu().numpy()

    tag_pred = apply_tag_thresholds(
        tag_probs,
        thresholds.get("tag_thresholds", {}),
        tag_classes,
        default=0.5,
    )[0]
    predicted_tags = postprocess_square_tags(tags_from_binary(tag_pred, tag_classes, no_issue_label=None) or [SQUARE_NO_TAG_LABEL])
    rating_pred = apply_square_rating_threshold(
        rating_probs,
        thresholds.get("rating_threshold", 0.5),
        rating_to_idx,
        rating_classes,
    )[0]
    predicted_rating = rating_classes[int(rating_pred)]

    return {
        "image_path": str(Path(image_path)),
        "analyzer": "square",
        "predicted_tags": predicted_tags,
        "recommendations": square_recommendations_from_tags(predicted_tags),
        "tag_probabilities": {tag: float(tag_probs[i]) for i, tag in enumerate(tag_classes)},
        "predicted_rating": predicted_rating,
        "rating_probabilities": {rating: float(rating_probs[i]) for i, rating in enumerate(rating_classes)},
        "thresholds": {
            "tag_thresholds": thresholds.get("tag_thresholds", {}),
            "rating_threshold": thresholds.get("rating_threshold", 0.5),
        },
    }


def analyze_square(
    image_path: str | Path,
    *,
    output: str | Path | None = "",
    device: str = "auto",
    model_dir: str | Path | None = None,
) -> str:
    """Run square analyzer inference, save standardized JSON, and return a summary string."""

    result = _analyze_square_result(image_path, device=device, model_dir=model_dir)
    return save_standard_output(result, task="analyze_square", input_path=image_path, output=output)


def _analyze_square_images_result(
    input_path: str | Path,
    *,
    recursive: bool = True,
    device: str = "auto",
    model_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run square analyzer inference on one image or a folder."""

    return [
        _analyze_square_result(path, device=device, model_dir=model_dir)
        for path in find_images(input_path, recursive=recursive)
    ]


def analyze_square_images(
    input_path: str | Path,
    *,
    output: str | Path | None = "",
    recursive: bool = True,
    device: str = "auto",
    model_dir: str | Path | None = None,
) -> str:
    """Run square analyzer inference, save standardized JSON, and return a summary string."""

    result = _analyze_square_images_result(
        input_path,
        recursive=recursive,
        device=device,
        model_dir=model_dir,
    )
    return save_standard_output(result, task="analyze_square_images", input_path=input_path, output=output)
