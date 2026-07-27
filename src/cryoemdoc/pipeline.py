"""High-level cryoEMdoc routing pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .atlas import _analyze_atlas_result, resolve_atlas_features
from .classifier import _classify_image_result
from .io import find_images, write_records_csv
from .square import _analyze_square_result
from ._standard_output import save_standard_output


def _analyze_image_result(
    image_path: str | Path,
    *,
    atlas_features: str | Path = "none",
    use_atlas_csv: bool | None = None,
    recursive: bool = True,
    confidence_threshold: float = 0.60,
    device: str = "auto",
    model_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    results_output: str | Path | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Classify one image, or every image in a folder, and route to analyzers."""

    atlas_features = resolve_atlas_features(atlas_features, use_atlas_csv)
    image_path = Path(image_path)
    if image_path.is_dir():
        return _analyze_images_result(
            image_path,
            atlas_features=atlas_features,
            recursive=recursive,
            confidence_threshold=confidence_threshold,
            device=device,
            model_dir=model_dir,
            output_dir=output_dir,
            results_output=results_output,
        )

    classifier_result = _classify_image_result(
        image_path,
        confidence_threshold=confidence_threshold,
        device=device,
        model_dir=model_dir,
    )
    predicted_label = str(classifier_result["predicted_label"])
    label_key = predicted_label.strip().lower()

    result: dict[str, Any] = {
        "image_path": str(Path(image_path)),
        "image_type": predicted_label,
        "classifier": classifier_result,
    }

    if label_key == "atlas":
        result["analysis"] = _analyze_atlas_result(
            image_path,
            atlas_features=atlas_features,
            use_atlas_csv=None,
            recursive=recursive,
            device=device,
            model_dir=model_dir,
            output_dir=output_dir,
        )
        result["status"] = "ok"
    elif label_key == "square":
        result["analysis"] = _analyze_square_result(image_path, device=device, model_dir=model_dir)
        result["status"] = "ok"
    elif label_key == "protein":
        result.update({
            "status": "not_implemented",
            "message": "Protein analyzer is not implemented yet.",
        })
    elif label_key == "other":
        result.update({
            "status": "unsupported",
            "message": "Unsupported or unknown image type.",
        })
    else:
        result.update({
            "status": "unknown",
            "message": f"No analyzer route is available for classifier label {predicted_label!r}.",
        })

    if results_output is not None:
        write_records_csv(result, results_output)
        result["results_output"] = str(Path(results_output))
    return result


def analyze_image(
    image_path: str | Path,
    *,
    output: str | Path | None = "",
    atlas_features: str | Path = "none",
    use_atlas_csv: bool | None = None,
    recursive: bool = True,
    confidence_threshold: float = 0.60,
    device: str = "auto",
    model_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    results_output: str | Path | None = None,
) -> str:
    """Classify and route, save standardized JSON, and return a summary string."""

    result = _analyze_image_result(
        image_path,
        atlas_features=atlas_features,
        use_atlas_csv=use_atlas_csv,
        recursive=recursive,
        confidence_threshold=confidence_threshold,
        device=device,
        model_dir=model_dir,
        output_dir=output_dir,
        results_output=results_output,
    )
    return save_standard_output(result, task="analyze_image", input_path=image_path, output=output)


def _analyze_images_result(
    input_path: str | Path,
    *,
    atlas_features: str | Path = "none",
    use_atlas_csv: bool | None = None,
    recursive: bool = True,
    confidence_threshold: float = 0.60,
    device: str = "auto",
    model_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    results_output: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Classify and route every image found in a folder or one image path.

    When ``atlas_features="generate"``, atlas prerecognition output is written
    under one per-image subdirectory so batch runs do not overwrite CSV files.
    """

    atlas_features = resolve_atlas_features(atlas_features, use_atlas_csv)
    paths = find_images(input_path, recursive=recursive)
    output_root = Path(output_dir) if output_dir is not None else None
    results: list[dict[str, Any]] = []

    for index, path in enumerate(paths, start=1):
        image_output_dir = output_root
        if str(atlas_features).lower() == "generate":
            if output_root is None:
                if Path(input_path).is_dir():
                    image_output_dir = Path(input_path) / "cryoemdoc_batch_outputs" / "atlas_features" / f"{index:04d}_{path.stem}"
                else:
                    image_output_dir = path.parent / "cryoemdoc_atlas_features"
            else:
                image_output_dir = output_root / "atlas_features" / f"{index:04d}_{path.stem}"

        results.append(
            _analyze_image_result(
                path,
                atlas_features=atlas_features,
                use_atlas_csv=None,
                confidence_threshold=confidence_threshold,
                device=device,
                model_dir=model_dir,
                output_dir=image_output_dir,
                results_output=None,
            )
        )
    if results_output is not None:
        write_records_csv(results, results_output)
        for result in results:
            result["results_output"] = str(Path(results_output))
    return results


def analyze_images(
    input_path: str | Path,
    *,
    output: str | Path | None = "",
    atlas_features: str | Path = "none",
    use_atlas_csv: bool | None = None,
    recursive: bool = True,
    confidence_threshold: float = 0.60,
    device: str = "auto",
    model_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    results_output: str | Path | None = None,
) -> str:
    """Classify and route images, save standardized JSON, and return a summary string."""

    result = _analyze_images_result(
        input_path,
        atlas_features=atlas_features,
        use_atlas_csv=use_atlas_csv,
        recursive=recursive,
        confidence_threshold=confidence_threshold,
        device=device,
        model_dir=model_dir,
        output_dir=output_dir,
        results_output=results_output,
    )
    return save_standard_output(result, task="analyze_images", input_path=input_path, output=output)
