"""Stable Python API wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .atlas import analyze_atlas, analyze_atlas_images, analyze_atlas_with_csv, analyze_atlas_without_csv
from .atlas_prerecognition import atlas_prerecognize
from .classifier import classify_image, classify_images
from .pipeline import analyze_image, analyze_images
from .square import analyze_square, analyze_square_images

__all__ = [
    "analyze_atlas",
    "analyze_atlas_images",
    "analyze_atlas_with_csv",
    "analyze_atlas_without_csv",
    "analyze_image",
    "analyze_images",
    "analyze_square",
    "analyze_square_images",
    "atlas_prerecognize",
    "classify_image",
    "classify_images",
]


def atlas_prerecognition(
    input_path: str | Path,
    output: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Backward-friendly alias for ``atlas_prerecognize``."""

    return atlas_prerecognize(input_path, output=output, **kwargs)
