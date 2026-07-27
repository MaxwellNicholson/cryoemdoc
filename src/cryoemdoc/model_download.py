"""Download and install cryoEMdoc model artifacts from GitHub Releases."""

from __future__ import annotations

import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from .assets import (
    DEFAULT_MODEL_VERSION,
    DEFAULT_RELEASE_REPOSITORY,
    artifacts_complete,
    default_model_root,
    missing_artifacts,
)


def release_asset_url(
    version: str = DEFAULT_MODEL_VERSION,
    repository: str = DEFAULT_RELEASE_REPOSITORY,
    archive_name: str | None = None,
) -> str:
    """Return the GitHub Release asset URL for a model archive."""

    archive = archive_name or f"cryoemdoc-models-{version}.zip"
    return f"https://github.com/{repository}/releases/download/{version}/{archive}"


def _safe_extract(zip_path: Path, destination: Path) -> None:
    """Extract a zip file while rejecting paths that escape the destination."""

    destination = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"Unsafe path in model archive: {member.filename}")
        archive.extractall(destination)


def _find_artifact_root(search_root: Path) -> Path:
    """Find the extracted directory containing the expected artifact layout."""

    candidates = [search_root, *[path for path in search_root.rglob("*") if path.is_dir()]]
    for candidate in candidates:
        if artifacts_complete(candidate):
            return candidate
    missing = missing_artifacts(search_root)
    preview = ", ".join(str(path.relative_to(search_root)) for path in missing[:4])
    raise FileNotFoundError(f"Downloaded archive does not contain a complete model layout. Missing: {preview}")


def _copy_missing_tree(source: Path, destination: Path) -> None:
    """Copy files from source to destination without overwriting existing files."""

    for item in source.rglob("*"):
        relative = item.relative_to(source)
        target = destination / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def download_models(
    *,
    version: str = DEFAULT_MODEL_VERSION,
    repository: str = DEFAULT_RELEASE_REPOSITORY,
    archive_name: str | None = None,
    destination: str | Path | None = None,
) -> Path:
    """Download, extract, and install a GitHub Release model archive."""

    target_root = Path(destination).expanduser().resolve() if destination else default_model_root(version)
    if artifacts_complete(target_root):
        return target_root

    url = release_asset_url(version=version, repository=repository, archive_name=archive_name)
    with tempfile.TemporaryDirectory(prefix="cryoemdoc-models-") as temp_name:
        temp_dir = Path(temp_name)
        archive_path = temp_dir / (archive_name or f"cryoemdoc-models-{version}.zip")
        urllib.request.urlretrieve(url, archive_path)

        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir()
        _safe_extract(archive_path, extract_dir)
        artifact_root = _find_artifact_root(extract_dir)

        target_root.mkdir(parents=True, exist_ok=True)
        _copy_missing_tree(artifact_root, target_root)

    if not artifacts_complete(target_root):
        missing = ", ".join(str(path) for path in missing_artifacts(target_root)[:4])
        raise FileNotFoundError(f"Model installation is incomplete. Missing: {missing}")
    return target_root
