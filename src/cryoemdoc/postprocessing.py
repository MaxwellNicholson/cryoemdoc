"""Thresholding and recommendation logic from the notebooks."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .constants import SQUARE_NO_TAG_LABEL


def apply_tag_thresholds(
    tag_probs: np.ndarray,
    thresholds: dict[str, float],
    tag_classes: list[str],
    default: float = 0.5,
) -> np.ndarray:
    """Apply per-tag probability thresholds."""

    probs = np.asarray(tag_probs)
    if probs.ndim == 1:
        probs = probs[None, :]
    threshold_array = np.array([thresholds.get(tag, default) for tag in tag_classes], dtype=np.float32)
    return (probs >= threshold_array[None, :]).astype(int)


def tags_from_binary(row: Iterable[int], tag_classes: list[str], no_issue_label: Any = None):
    """Convert a binary tag row to tag labels."""

    labels = [tag_classes[i] for i, value in enumerate(row) if value]
    if labels:
        return labels
    return no_issue_label if no_issue_label is not None else []


def apply_square_rating_threshold(
    rating_probs: np.ndarray,
    threshold: float,
    rating_to_idx: dict[str, int],
    rating_classes: list[str],
) -> np.ndarray:
    """Gate unacceptable, then choose good versus acceptable by argmax."""

    probs = np.asarray(rating_probs)
    if probs.ndim == 1:
        probs = probs[None, :]
    unacceptable_idx = rating_to_idx["unacceptable"]
    non_unacceptable_idx = np.array(
        [idx for idx in range(len(rating_classes)) if idx != unacceptable_idx],
        dtype=int,
    )
    pred = non_unacceptable_idx[probs[:, non_unacceptable_idx].argmax(axis=1)]
    pred = pred.astype(int)
    pred[probs[:, unacceptable_idx] >= float(threshold)] = unacceptable_idx
    return pred


def apply_binary_unacceptable_threshold(
    rating_probs: np.ndarray,
    threshold: float,
    rating_to_idx: dict[str, int],
) -> np.ndarray:
    """Apply the binary atlas unacceptable probability threshold."""

    probs = np.asarray(rating_probs)
    if probs.ndim == 1:
        probs = probs[None, :]
    unacceptable_idx = rating_to_idx["unacceptable"]
    return (probs[:, unacceptable_idx] >= float(threshold)).astype(int)


def postprocess_square_tags(tags) -> list[str]:
    """Match the square notebook's no-tag convention."""

    if isinstance(tags, str):
        tag_list = [tag.strip() for tag in tags.split(";") if tag.strip()]
    else:
        tag_list = list(tags)
    issue_tags = [tag for tag in tag_list if tag != SQUARE_NO_TAG_LABEL]
    return issue_tags if issue_tags else [SQUARE_NO_TAG_LABEL]


def square_recommendations_from_tags(tags) -> list[str]:
    """Return square-collection recommendations from predicted tags."""

    tag_set = set(postprocess_square_tags(tags))
    recommendations: list[str] = []
    if "thick ice" in tag_set:
        recommendations.append("increase blotting force/time")
    if "non-uniform ice" in tag_set:
        recommendations.append("increase glow discharge time")
    if "ice contamination" in tag_set:
        recommendations.append("careful handling")
    return recommendations or ["no recommendation"]


def atlas_recommendations_from_tags(tags) -> list[str]:
    """Return atlas-collection recommendations from predicted tags."""

    if isinstance(tags, str):
        if not tags or tags == "no issues":
            tag_set: set[str] = set()
        else:
            tag_set = {tag.strip() for tag in tags.split(";") if tag.strip()}
    else:
        tag_set = set(tags)

    recommendations: list[str] = []
    has_cracks = "cracks" in tag_set
    has_non_uniform = "non-uniform ice" in tag_set

    if "thick ice" in tag_set:
        recommendations.append("decrease blotting force/time")
    if has_cracks and has_non_uniform:
        recommendations.append("decrease glow-discharge time/current")
        recommendations.append("use grid within 20 min")
    else:
        if has_cracks:
            recommendations.append("decrease glow discharge time/current")
        if has_non_uniform:
            recommendations.append("increase glow discharge time/current")

    return recommendations or ["no recommendation"]
