"""Build the GitHub Release zip containing cryoEMdoc model artifacts.

This script copies model artifacts into a zip file without modifying the source
artifact directory. Run it from a checkout that has the local ``.pt`` weights.
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path


DEFAULT_VERSION = "v0.1.0"

REQUIRED_FILES = {
    "image_classifier": [
        "best_resnet18_step1_classifier.pt",
    ],
    "square_analyzer": [
        "best_model_state.pt",
        "best_model_metadata.json",
        "label_mappings.json",
        "thresholds.json",
    ],
    "atlas_analyzer_without_csv": [
        "best_model_state.pt",
        "best_model_metadata.json",
        "label_mappings.json",
        "thresholds.json",
    ],
    "atlas_analyzer_with_csv": [
        "best_model_state.pt",
        "best_model_metadata.json",
        "label_mappings.json",
        "thresholds.json",
        "tabular_preprocessing.json",
    ],
}


def required_paths(source_artifacts: Path) -> list[Path]:
    return [
        source_artifacts / group / filename
        for group, filenames in REQUIRED_FILES.items()
        for filename in filenames
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_archive(source_artifacts: Path, output_dir: Path, version: str) -> tuple[Path, Path]:
    missing = [path for path in required_paths(source_artifacts) if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Cannot build model release; missing files:\n{formatted}")

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_name = f"cryoemdoc-models-{version}.zip"
    archive_path = output_dir / archive_name
    root_name = f"cryoemdoc-models-{version}"

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in required_paths(source_artifacts):
            relative = path.relative_to(source_artifacts)
            archive.write(path, Path(root_name) / relative)

    checksum_path = output_dir / f"{archive_name}.sha256"
    checksum_path.write_text(f"{sha256_file(archive_path)}  {archive_name}\n", encoding="utf-8")
    return archive_path, checksum_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the cryoEMdoc model GitHub Release archive.")
    parser.add_argument(
        "--source-artifacts",
        default="src/cryoemdoc/artifacts",
        help="Artifact root containing model subfolders and .pt weights.",
    )
    parser.add_argument("--output-dir", default="dist_release", help="Directory for release assets.")
    parser.add_argument("--version", default=DEFAULT_VERSION, help="Release tag, for example v0.1.0.")
    args = parser.parse_args()

    archive_path, checksum_path = build_archive(
        Path(args.source_artifacts).expanduser().resolve(),
        Path(args.output_dir).expanduser().resolve(),
        args.version,
    )
    print(f"Wrote {archive_path}")
    print(f"Wrote {checksum_path}")


if __name__ == "__main__":
    main()
