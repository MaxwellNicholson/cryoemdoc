"""Atlas analyzer inference, with and without generated CSV features."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .assets import artifact_path, read_json
from .constants import ATLAS_NO_ISSUE_LABEL
from .io import find_images, write_records_csv
from .models import (
    build_atlas_csv_resnet18,
    build_multitask_resnet18,
    load_state_dict_model,
    require_torch,
    resolve_device,
)
from .postprocessing import (
    apply_binary_unacceptable_threshold,
    apply_tag_thresholds,
    atlas_recommendations_from_tags,
    tags_from_binary,
)
from .preprocessing import analyzer_tensor
from ._standard_output import save_standard_output

ATLAS_SCORE_KEY_CANDIDATES = ["image_name", "combined_key", "filename", "Image Name", "image_path", "path"]


def resolve_atlas_features(
    atlas_features: str | Path = "none",
    use_atlas_csv: bool | None = None,
) -> str | Path:
    """Normalize atlas CSV mode aliases.

    ``use_atlas_csv=True`` means generate prerecognition CSV features unless a
    concrete CSV path was also supplied through ``atlas_features``.
    """

    value = str(atlas_features).strip()
    lowered = value.lower()
    off_values = {"", "none", "off", "false", "0", "no"}
    generate_values = {"generate", "on", "true", "1", "yes"}

    if use_atlas_csv is True:
        if lowered in off_values:
            return "generate"
        return atlas_features
    if use_atlas_csv is False:
        if lowered in off_values:
            return "none"
        raise ValueError("use_atlas_csv=False conflicts with an atlas_features CSV/generate value.")
    if lowered in off_values:
        return "none"
    if lowered in generate_values:
        return "generate"
    return atlas_features


def normalize_atlas_score_key(value: Any) -> str:
    """Normalize image identifiers to the basename-lowercase key used in training."""

    if pd.isna(value):
        return ""
    key = str(value).strip().replace("\\", "/")
    if key == "" or key.lower() == "nan":
        return ""
    return Path(key).name.lower()


def _choose_atlas_score_key_column(score_df: pd.DataFrame) -> str:
    for column in ATLAS_SCORE_KEY_CANDIDATES:
        if column in score_df.columns:
            return column
    raise ValueError(
        "Atlas-score CSV needs one image identifier column. Tried: "
        f"{ATLAS_SCORE_KEY_CANDIDATES}. Found: {list(score_df.columns)}"
    )


def _add_atlas_score_fraction_features(
    score_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    score_df = score_df.copy()
    if "visible_square_count" not in score_df.columns:
        return score_df, feature_columns

    visible = pd.to_numeric(score_df["visible_square_count"], errors="coerce").replace(0, np.nan)
    ratio_specs = {
        "good_square_fraction": "good_square_count",
        "non_uniform_square_fraction": "non_uniform_square_count",
        "cracked_square_fraction": "cracked_square_count",
        "bad_size_square_fraction": "bad_size_square_count",
    }
    for new_column, count_column in ratio_specs.items():
        if count_column in score_df.columns:
            score_df[new_column] = pd.to_numeric(score_df[count_column], errors="coerce") / visible
            if new_column not in feature_columns:
                feature_columns.append(new_column)
    return score_df, feature_columns


def load_atlas_score_table(
    atlas_score_csv_path: str | Path,
    configured_feature_columns: Sequence[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Load and aggregate an atlas prerecognition CSV for inference."""

    atlas_score_csv_path = Path(atlas_score_csv_path)
    if not atlas_score_csv_path.exists():
        raise FileNotFoundError(f"Atlas-score CSV not found: {atlas_score_csv_path}")

    score_df = pd.read_csv(atlas_score_csv_path)
    key_column = _choose_atlas_score_key_column(score_df)
    score_df["atlas_score_key"] = score_df[key_column].apply(normalize_atlas_score_key)
    score_df = score_df[score_df["atlas_score_key"].ne("")].copy()

    feature_columns = [column for column in configured_feature_columns if column in score_df.columns]
    score_df, feature_columns = _add_atlas_score_fraction_features(score_df, feature_columns)
    feature_columns = [column for column in feature_columns if column in score_df.columns]
    if not feature_columns:
        return pd.DataFrame(columns=["atlas_score_key"]), []

    for column in feature_columns:
        score_df[column] = pd.to_numeric(score_df[column], errors="coerce")

    return (
        score_df[["atlas_score_key"] + feature_columns].groupby("atlas_score_key", as_index=False).mean(),
        feature_columns,
    )


def _numeric_feature_frame(frame: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    values = frame.reindex(columns=feature_columns).copy()
    for column in feature_columns:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    return values


def build_tabular_feature_array(frame: pd.DataFrame, preprocessor: dict[str, Any]) -> np.ndarray:
    """Normalize atlas CSV features using saved training fill/mean/std values."""

    feature_columns = list(preprocessor.get("feature_columns", []))
    if not feature_columns:
        return np.zeros((len(frame), 0), dtype=np.float32)

    values = _numeric_feature_frame(frame, feature_columns)
    fill_values = pd.Series(preprocessor.get("fill_values", {}), dtype=float).reindex(feature_columns).fillna(0.0)
    means = pd.Series(preprocessor.get("means", {}), dtype=float).reindex(feature_columns).fillna(0.0)
    stds = pd.Series(preprocessor.get("stds", {}), dtype=float).reindex(feature_columns).replace(0, 1.0).fillna(1.0)
    filled = values.fillna(fill_values)
    normalized = ((filled - means) / stds).to_numpy(dtype=np.float32)
    return normalized.astype(np.float32)


def build_inference_frame(
    image_paths: Sequence[str | Path],
    atlas_score_csv: str | Path | None,
    preprocessor: dict[str, Any],
) -> pd.DataFrame:
    """Build the single-row/multi-row feature frame used by the CSV model."""

    frame = pd.DataFrame({
        "combined_key": [Path(path).name for path in image_paths],
        "image_path": [Path(path) for path in image_paths],
    })
    frame["atlas_score_key"] = frame["combined_key"].apply(normalize_atlas_score_key)
    feature_columns = list(preprocessor.get("feature_columns", []))
    configured = [column for column in feature_columns if column != "atlas_score_features_missing"]

    if atlas_score_csv is not None:
        score_feature_df, _ = load_atlas_score_table(atlas_score_csv, configured)
        merge_columns = ["atlas_score_key"] + [column for column in configured if column in score_feature_df.columns]
        if len(merge_columns) > 1:
            frame = frame.merge(score_feature_df[merge_columns], on="atlas_score_key", how="left")
            frame["atlas_score_features_found"] = frame["atlas_score_key"].isin(set(score_feature_df["atlas_score_key"]))
        else:
            frame["atlas_score_features_found"] = False
    else:
        frame["atlas_score_features_found"] = False

    if "atlas_score_features_missing" in feature_columns:
        frame["atlas_score_features_missing"] = (~frame["atlas_score_features_found"]).astype(float)
    return frame


def _atlas_paths(with_csv: bool, model_dir: str | Path | None = None) -> dict[str, Path]:
    group = "atlas_analyzer_with_csv" if with_csv else "atlas_analyzer_without_csv"
    paths = {
        "model": artifact_path(group, "model", model_dir=model_dir),
        "labels": artifact_path(group, "labels", model_dir=model_dir),
        "thresholds": artifact_path(group, "thresholds", model_dir=model_dir),
    }
    if with_csv:
        paths["tabular"] = artifact_path(group, "tabular", model_dir=model_dir)
        paths["metadata"] = artifact_path(group, "metadata", model_dir=model_dir)
    return paths


@lru_cache(maxsize=8)
def _load_atlas_image_cached(model_path: str, labels_path: str, thresholds_path: str, device_name: str):
    device = resolve_device(device_name)
    labels = read_json(labels_path)
    thresholds = read_json(thresholds_path)
    tag_classes = list(labels["tags"])
    rating_classes = list(labels["rating_classes"])
    model = build_multitask_resnet18(len(tag_classes), len(rating_classes))
    model = load_state_dict_model(model, model_path, device)
    return model, labels, thresholds


@lru_cache(maxsize=8)
def _load_atlas_csv_cached(
    model_path: str,
    labels_path: str,
    thresholds_path: str,
    tabular_path: str,
    metadata_path: str,
    device_name: str,
):
    device = resolve_device(device_name)
    labels = read_json(labels_path)
    thresholds = read_json(thresholds_path)
    preprocessor = read_json(tabular_path)
    metadata = read_json(metadata_path)
    config = metadata.get("config", {})
    tag_classes = list(labels["tags"])
    rating_classes = list(labels["rating_classes"])
    model = build_atlas_csv_resnet18(
        len(tag_classes),
        len(rating_classes),
        tabular_dim=len(preprocessor.get("feature_columns", [])),
        tabular_hidden_dim=int(config.get("tabular_hidden_dim", 32)),
        tabular_dropout=float(config.get("tabular_dropout", 0.15)),
    )
    model = load_state_dict_model(model, model_path, device)
    return model, labels, thresholds, preprocessor


def _postprocess_atlas_outputs(
    image_path: str | Path,
    tag_probs: np.ndarray,
    rating_probs: np.ndarray,
    labels: dict[str, Any],
    thresholds: dict[str, Any],
    analyzer_name: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tag_classes = list(labels["tags"])
    rating_classes = list(labels["rating_classes"])
    rating_to_idx = dict(labels["rating_to_idx"])

    tag_pred = apply_tag_thresholds(
        tag_probs,
        thresholds.get("tag_thresholds", {}),
        tag_classes,
        default=0.5,
    )[0]
    predicted_tags = tags_from_binary(tag_pred, tag_classes, no_issue_label=None) or ATLAS_NO_ISSUE_LABEL
    predicted_rating_idx = apply_binary_unacceptable_threshold(
        rating_probs,
        thresholds.get("rating_threshold", 0.5),
        rating_to_idx,
    )[0]
    predicted_rating = rating_classes[int(predicted_rating_idx)]

    result: dict[str, Any] = {
        "image_path": str(Path(image_path)),
        "analyzer": analyzer_name,
        "predicted_tags": predicted_tags,
        "recommendations": atlas_recommendations_from_tags(predicted_tags),
        "tag_probabilities": {tag: float(tag_probs[i]) for i, tag in enumerate(tag_classes)},
        "predicted_rating": predicted_rating,
        "rating_probabilities": {rating: float(rating_probs[i]) for i, rating in enumerate(rating_classes)},
        "thresholds": {
            "tag_thresholds": thresholds.get("tag_thresholds", {}),
            "rating_threshold": thresholds.get("rating_threshold", 0.5),
        },
    }
    if extra:
        result.update(extra)
    return result


def _analyze_atlas_without_csv_result(
    image_path: str | Path,
    *,
    device: str = "auto",
    model_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the image-only atlas analyzer."""

    torch = require_torch()
    paths = _atlas_paths(False, model_dir)
    resolved_device = resolve_device(device)
    model, labels, thresholds = _load_atlas_image_cached(
        str(paths["model"]),
        str(paths["labels"]),
        str(paths["thresholds"]),
        str(resolved_device),
    )
    tensor = analyzer_tensor(image_path).unsqueeze(0).to(resolved_device)
    with torch.no_grad():
        outputs = model(tensor)
        tag_probs = torch.sigmoid(outputs["tag_logits"])[0].detach().cpu().numpy()
        rating_probs = torch.softmax(outputs["rating_logits"], dim=1)[0].detach().cpu().numpy()
    return _postprocess_atlas_outputs(
        image_path,
        tag_probs,
        rating_probs,
        labels,
        thresholds,
        analyzer_name="atlas_without_csv",
    )


def analyze_atlas_without_csv(
    image_path: str | Path,
    *,
    output: str | Path | None = "",
    device: str = "auto",
    model_dir: str | Path | None = None,
) -> str:
    """Run image-only atlas inference, save standardized JSON, and return a summary string."""

    result = _analyze_atlas_without_csv_result(image_path, device=device, model_dir=model_dir)
    return save_standard_output(result, task="analyze_atlas_without_csv", input_path=image_path, output=output)


def _analyze_atlas_with_csv_result(
    image_path: str | Path,
    atlas_score_csv: str | Path,
    *,
    device: str = "auto",
    model_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the atlas analyzer with CSV-derived atlas features."""

    torch = require_torch()
    paths = _atlas_paths(True, model_dir)
    resolved_device = resolve_device(device)
    model, labels, thresholds, preprocessor = _load_atlas_csv_cached(
        str(paths["model"]),
        str(paths["labels"]),
        str(paths["thresholds"]),
        str(paths["tabular"]),
        str(paths["metadata"]),
        str(resolved_device),
    )
    tensor = analyzer_tensor(image_path).unsqueeze(0).to(resolved_device)
    feature_frame = build_inference_frame([image_path], atlas_score_csv, preprocessor)
    tabular_values = build_tabular_feature_array(feature_frame, preprocessor)
    tabular = torch.tensor(tabular_values, dtype=torch.float32, device=resolved_device)
    with torch.no_grad():
        outputs = model(tensor, tabular)
        tag_probs = torch.sigmoid(outputs["tag_logits"])[0].detach().cpu().numpy()
        rating_probs = torch.softmax(outputs["rating_logits"], dim=1)[0].detach().cpu().numpy()

    raw_features = {
        column: (
            None
            if column not in feature_frame.columns or pd.isna(feature_frame.iloc[0][column])
            else float(feature_frame.iloc[0][column])
        )
        for column in preprocessor.get("feature_columns", [])
    }
    return _postprocess_atlas_outputs(
        image_path,
        tag_probs,
        rating_probs,
        labels,
        thresholds,
        analyzer_name="atlas_with_csv",
        extra={
            "atlas_score_csv": str(Path(atlas_score_csv)),
            "atlas_score_features_found": bool(feature_frame.iloc[0].get("atlas_score_features_found", False)),
            "atlas_score_features": raw_features,
        },
    )


def analyze_atlas_with_csv(
    image_path: str | Path,
    atlas_score_csv: str | Path,
    *,
    output: str | Path | None = "",
    device: str = "auto",
    model_dir: str | Path | None = None,
) -> str:
    """Run atlas CSV inference, save standardized JSON, and return a summary string."""

    result = _analyze_atlas_with_csv_result(
        image_path,
        atlas_score_csv,
        device=device,
        model_dir=model_dir,
    )
    return save_standard_output(result, task="analyze_atlas_with_csv", input_path=image_path, output=output)


def _analyze_atlas_result(
    image_path: str | Path,
    *,
    atlas_features: str | Path = "none",
    use_atlas_csv: bool | None = None,
    recursive: bool = True,
    device: str = "auto",
    model_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    results_output: str | Path | None = None,
    save_annotated_image: bool = False,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Run atlas analysis for one atlas image or a folder of atlas images."""

    atlas_features = resolve_atlas_features(atlas_features, use_atlas_csv)
    image_path = Path(image_path)
    if image_path.is_dir():
        return _analyze_atlas_images_result(
            image_path,
            atlas_features=atlas_features,
            recursive=recursive,
            device=device,
            model_dir=model_dir,
            output_dir=output_dir,
            results_output=results_output,
            save_annotated_images=save_annotated_image,
        )

    if str(atlas_features).lower() == "none":
        result = _analyze_atlas_without_csv_result(image_path, device=device, model_dir=model_dir)
        if results_output is not None:
            write_records_csv(result, results_output)
            result["results_output"] = str(Path(results_output))
        return result
    if str(atlas_features).lower() == "generate":
        from .atlas_prerecognition import atlas_prerecognize

        output_root = Path(output_dir) if output_dir is not None else Path(image_path).parent / "cryoemdoc_atlas_features"
        generated_results_output = Path(results_output) if results_output is not None else output_root / "atlas_csv_analyzer_predictions.csv"
        prerecognition = atlas_prerecognize(
            image_path,
            output=output_root / "atlas_summary_scores.csv",
            save_annotated_images=save_annotated_image,
            recursive=False,
        )
        result = _analyze_atlas_with_csv_result(
            image_path,
            prerecognition["csv_path"],
            device=device,
            model_dir=model_dir,
        )
        result["atlas_prerecognition"] = prerecognition
        result["atlas_quality_scores_csv"] = prerecognition["csv_path"]
        write_records_csv(result, generated_results_output)
        result["results_output"] = str(generated_results_output)
        result["atlas_csv_analyzer_predictions_csv"] = str(generated_results_output)
        print(f"Saved atlas CSV analyzer predictions: {generated_results_output}")
        return result
    result = _analyze_atlas_with_csv_result(image_path, atlas_features, device=device, model_dir=model_dir)
    if results_output is not None:
        write_records_csv(result, results_output)
        result["results_output"] = str(Path(results_output))
    return result


def analyze_atlas(
    image_path: str | Path,
    *,
    output: str | Path | None = "",
    atlas_features: str | Path = "none",
    use_atlas_csv: bool | None = None,
    recursive: bool = True,
    device: str = "auto",
    model_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    results_output: str | Path | None = None,
    save_annotated_image: bool = False,
) -> str:
    """Run atlas analysis, save standardized JSON, and return a summary string."""

    result = _analyze_atlas_result(
        image_path,
        atlas_features=atlas_features,
        use_atlas_csv=use_atlas_csv,
        recursive=recursive,
        device=device,
        model_dir=model_dir,
        output_dir=output_dir,
        results_output=results_output,
        save_annotated_image=save_annotated_image,
    )
    return save_standard_output(result, task="analyze_atlas", input_path=image_path, output=output)


def _analyze_atlas_images_result(
    input_path: str | Path,
    *,
    atlas_features: str | Path = "none",
    use_atlas_csv: bool | None = None,
    recursive: bool = True,
    device: str = "auto",
    model_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    results_output: str | Path | None = None,
    save_annotated_images: bool = False,
) -> list[dict[str, Any]]:
    """Run atlas analyzer inference on one image or a folder."""

    atlas_features = resolve_atlas_features(atlas_features, use_atlas_csv)
    paths = find_images(input_path, recursive=recursive)
    if str(atlas_features).lower() == "generate":
        from .atlas_prerecognition import atlas_prerecognize

        input_path_obj = Path(input_path)
        if output_dir is not None:
            output_root = Path(output_dir)
        elif input_path_obj.is_dir():
            output_root = input_path_obj / "cryoemdoc_atlas_features"
        else:
            output_root = input_path_obj.parent / "cryoemdoc_atlas_features"
        generated_results_output = Path(results_output) if results_output is not None else output_root / "atlas_csv_analyzer_predictions.csv"

        prerecognition = atlas_prerecognize(
            input_path,
            output=output_root / "atlas_summary_scores.csv",
            recursive=recursive,
            save_annotated_images=save_annotated_images,
        )
        prerecognition_summary = {
            "csv_path": prerecognition["csv_path"],
            "summary_csv_path": prerecognition.get("summary_csv_path", prerecognition["csv_path"]),
            "output_format": prerecognition.get("output_format", "summary"),
            "row_count": prerecognition["row_count"],
        }
        results = [
            _analyze_atlas_with_csv_result(path, prerecognition["csv_path"], device=device, model_dir=model_dir)
            for path in paths
        ]
        for result in results:
            result["atlas_prerecognition"] = prerecognition_summary
            result["atlas_quality_scores_csv"] = prerecognition["csv_path"]
        write_records_csv(results, generated_results_output)
        for result in results:
            result["results_output"] = str(generated_results_output)
            result["atlas_csv_analyzer_predictions_csv"] = str(generated_results_output)
        print(f"Saved atlas CSV analyzer predictions: {generated_results_output}")
        return results

    results = [
        _analyze_atlas_result(
            path,
            atlas_features=atlas_features,
            use_atlas_csv=None,
            device=device,
            model_dir=model_dir,
            output_dir=output_dir,
            results_output=None,
        )
        for path in paths
    ]
    if results_output is not None:
        write_records_csv(results, results_output)
        for result in results:
            result["results_output"] = str(Path(results_output))
    return results


def analyze_atlas_images(
    input_path: str | Path,
    *,
    output: str | Path | None = "",
    atlas_features: str | Path = "none",
    use_atlas_csv: bool | None = None,
    recursive: bool = True,
    device: str = "auto",
    model_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    results_output: str | Path | None = None,
    save_annotated_images: bool = False,
) -> str:
    """Run atlas analyzer inference, save standardized JSON, and return a summary string."""

    result = _analyze_atlas_images_result(
        input_path,
        atlas_features=atlas_features,
        use_atlas_csv=use_atlas_csv,
        recursive=recursive,
        device=device,
        model_dir=model_dir,
        output_dir=output_dir,
        results_output=results_output,
        save_annotated_images=save_annotated_images,
    )
    return save_standard_output(result, task="analyze_atlas_images", input_path=input_path, output=output)
