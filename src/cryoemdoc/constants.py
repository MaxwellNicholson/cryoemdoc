"""Shared constants for inference."""

from __future__ import annotations

IMAGE_CLASS_NAMES = ["Atlas", "Square", "Protein", "Other"]
IMAGE_CLASSIFIER_SIZE = 224
ANALYZER_IMAGE_SIZE = 384

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
    ".mrc",
}

CLASSIFIER_ROUTE_MAP = {
    "atlas": "send_to_step2_atlas_analyzer",
    "square": "send_to_step3_square_hole_analyzer",
    "protein": "send_to_step4_protein_micrograph_analyzer",
    "other": "return_error_or_request_different_image",
}

SQUARE_NO_TAG_LABEL = "no tags"
ATLAS_NO_ISSUE_LABEL = "no issues"
