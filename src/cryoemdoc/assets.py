"""Artifact path helpers.

The package can use bundled artifacts under ``src/cryoemdoc/artifacts``, models
downloaded from GitHub Releases, or an external model root passed by the user or
through ``CRYOEMDOC_MODEL_DIR``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
BUNDLED_ARTIFACT_ROOT = PACKAGE_ROOT / "artifacts"
MODEL_DIR_ENV = "CRYOEMDOC_MODEL_DIR"
MODEL_CACHE_ENV = "CRYOEMDOC_CACHE_DIR"
DEFAULT_MODEL_VERSION = "v0.1.0"
DEFAULT_RELEASE_REPOSITORY = "MaxwellNicholson/cryoemdoc"
DEFAULT_MODEL_ARCHIVE = f"cryoemdoc-models-{DEFAULT_MODEL_VERSION}.zip"

ARTIFACT_FILENAMES = {
    "image_classifier": {
        "model": "best_resnet18_step1_classifier.pt",
    },
    "square_analyzer": {
        "model": "best_model_state.pt",
        "metadata": "best_model_metadata.json",
        "labels": "label_mappings.json",
        "thresholds": "thresholds.json",
    },
    "atlas_analyzer_without_csv": {
        "model": "best_model_state.pt",
        "metadata": "best_model_metadata.json",
        "labels": "label_mappings.json",
        "thresholds": "thresholds.json",
    },
    "atlas_analyzer_with_csv": {
        "model": "best_model_state.pt",
        "metadata": "best_model_metadata.json",
        "labels": "label_mappings.json",
        "thresholds": "thresholds.json",
        "tabular": "tabular_preprocessing.json",
    },
}


class ArtifactNotFoundError(FileNotFoundError):
    """Raised when a required model artifact cannot be resolved."""


def default_model_cache_root() -> Path:
    """Return the root directory used for downloaded model releases."""

    env_dir = os.getenv(MODEL_CACHE_ENV)
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return Path.home().joinpath(".cache", "cryoemdoc", "models").resolve()


def default_model_root(version: str = DEFAULT_MODEL_VERSION) -> Path:
    """Return the model root for a downloaded cryoEMdoc model release."""

    return default_model_cache_root() / version


def iter_required_artifacts(root: str | Path) -> list[Path]:
    """Return required artifact paths under an artifact root."""

    root = Path(root)
    return [
        root / group / filename
        for group, filenames in ARTIFACT_FILENAMES.items()
        for filename in filenames.values()
    ]


def missing_artifacts(root: str | Path) -> list[Path]:
    """Return required artifact files missing from an artifact root."""

    return [path for path in iter_required_artifacts(root) if not path.exists()]


def artifacts_complete(root: str | Path) -> bool:
    """Return whether a root contains all required cryoEMdoc artifacts."""

    return not missing_artifacts(root)


def artifact_root(model_dir: str | Path | None = None) -> Path:
    """Return the root directory that contains model artifact subdirectories."""

    if model_dir is not None:
        return Path(model_dir).expanduser().resolve()
    env_dir = os.getenv(MODEL_DIR_ENV)
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    if artifacts_complete(BUNDLED_ARTIFACT_ROOT):
        return BUNDLED_ARTIFACT_ROOT
    cached_root = default_model_root()
    if artifacts_complete(cached_root):
        return cached_root
    return BUNDLED_ARTIFACT_ROOT


def artifact_path(
    artifact_group: str,
    artifact_key: str,
    model_dir: str | Path | None = None,
    required: bool = True,
) -> Path:
    """Resolve one artifact path from a group/key pair."""

    try:
        filename = ARTIFACT_FILENAMES[artifact_group][artifact_key]
    except KeyError as exc:
        raise KeyError(f"Unknown artifact {artifact_group!r}/{artifact_key!r}") from exc

    path = artifact_root(model_dir) / artifact_group / filename
    if required and not path.exists():
        raise ArtifactNotFoundError(
            f"Missing artifact: {path}. Run `cryoemdoc download-models`, provide a "
            f"model directory with the same subfolder layout, or set {MODEL_DIR_ENV}."
        )
    return path


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a UTF-8 JSON file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
