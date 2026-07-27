"""Image and file IO helpers."""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageFile

from .constants import IMAGE_EXTENSIONS

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_raw_image(path: str | Path) -> np.ndarray:
    """Read PNG/JPG/TIFF/MRC-like inputs into a 2D ``float32`` array."""

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".mrc":
        try:
            import mrcfile
        except ImportError as exc:
            raise ImportError("Install cryoemdoc[mrc] or mrcfile to read MRC files.") from exc
        with mrcfile.open(path, permissive=True) as mrc:
            arr = np.asarray(mrc.data)
        arr = np.squeeze(arr)
        if arr.ndim == 3:
            arr = arr[0]
    elif suffix in {".tif", ".tiff"}:
        try:
            import tifffile
        except ImportError as exc:
            raise ImportError("Install tifffile to read TIFF files.") from exc
        arr = np.asarray(tifffile.imread(path))
        arr = np.squeeze(arr)
        if arr.ndim == 3:
            if arr.shape[-1] in (3, 4):
                arr = np.asarray(Image.fromarray(arr).convert("L"))
            else:
                arr = arr[0]
    else:
        arr = np.asarray(Image.open(path).convert("L"))

    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D image after loading, got shape {arr.shape} for {path}")
    return arr


def normalize_to_uint8(arr: np.ndarray, low_pct: float = 1, high_pct: float = 99) -> np.ndarray:
    """Percentile clip and scale an image array to uint8."""

    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        raise ValueError("Image contains no finite pixel values.")

    arr = arr.copy()
    arr[~finite] = np.nanmedian(arr[finite])
    lo, hi = np.percentile(arr, [low_pct, high_pct])
    if hi <= lo:
        hi = lo + 1.0

    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    return (arr * 255).astype(np.uint8)


def read_cryo_pil(path: str | Path) -> Image.Image:
    """Return a normalized grayscale image duplicated into RGB channels."""

    raw = read_raw_image(path)
    img_u8 = normalize_to_uint8(raw)
    return Image.fromarray(img_u8, mode="L").convert("RGB")


def is_hidden_or_checkpoint(path: Path) -> bool:
    """Return true for hidden paths such as ``.ipynb_checkpoints``."""

    for part in path.parts:
        if part in {".", ".."}:
            continue
        if part.startswith("."):
            return True
    return False


def find_images(input_path: str | Path, recursive: bool = True) -> list[Path]:
    """Return sorted image files from a file or directory."""

    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in IMAGE_EXTENSIONS else []
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    iterator: Iterable[Path] = input_path.rglob("*") if recursive else input_path.glob("*")
    return sorted(
        p
        for p in iterator
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not is_hidden_or_checkpoint(p)
    )


def write_json(data: Any, path: str | Path) -> Path:
    """Write JSON data with stable indentation."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
    return path


def flatten_record_for_csv(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten a nested result dictionary for CSV output."""

    flat: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (dict, list)):
                    flat[f"{key}.{sub_key}"] = json.dumps(sub_value, default=str)
                else:
                    flat[f"{key}.{sub_key}"] = sub_value
        elif isinstance(value, list):
            flat[key] = "; ".join(str(item) for item in value)
        else:
            flat[key] = value
    return flat


def write_records_csv(records: list[dict[str, Any]] | dict[str, Any], path: str | Path) -> Path:
    """Write one or more result dictionaries to a flat CSV file."""

    if isinstance(records, dict):
        normalized_records = [records]
    else:
        normalized_records = list(records)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [flatten_record_for_csv(record) for record in normalized_records]
    fieldnames = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
