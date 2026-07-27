"""Private standardized output formatting for public API functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import write_json

SCHEMA_VERSION = "1.0"
NO_ISSUES_LABEL = "no issues"
NO_ISSUE_VALUES = {"", "no issue", "no issues", "no tag", "no tags", "none", "nan"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _normalize_tags(tags: Any) -> list[str]:
    if tags is None:
        return [NO_ISSUES_LABEL]
    if isinstance(tags, str):
        raw_tags = [tag.strip() for tag in tags.split(";")]
    else:
        raw_tags = [str(tag).strip() for tag in tags]
    normalized = [tag for tag in raw_tags if tag.lower() not in NO_ISSUE_VALUES]
    return normalized or [NO_ISSUES_LABEL]


def _result_source(raw_result: dict[str, Any]) -> dict[str, Any]:
    analysis = raw_result.get("analysis")
    if isinstance(analysis, dict):
        return analysis
    return raw_result


def _image_type_label(raw_result: dict[str, Any], source: dict[str, Any]) -> str | None:
    if raw_result.get("image_type") is not None:
        return str(raw_result["image_type"])
    if raw_result.get("predicted_label") is not None:
        return str(raw_result["predicted_label"])
    classifier = raw_result.get("classifier")
    if isinstance(classifier, dict) and classifier.get("predicted_label") is not None:
        return str(classifier["predicted_label"])
    analyzer = str(source.get("analyzer", "")).lower()
    if analyzer.startswith("square"):
        return "Square"
    if analyzer.startswith("atlas"):
        return "Atlas"
    return None


def _classifier_source(raw_result: dict[str, Any]) -> dict[str, Any]:
    classifier = raw_result.get("classifier")
    if isinstance(classifier, dict):
        return classifier
    return raw_result


def _artifact_values(raw_result: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for container in (source, raw_result):
        for key in (
            "results_output",
            "atlas_score_csv",
            "atlas_quality_scores_csv",
            "atlas_csv_analyzer_predictions_csv",
        ):
            if container.get(key) is not None:
                artifacts[key] = _json_safe(container[key])
    prerecognition = source.get("atlas_prerecognition") or raw_result.get("atlas_prerecognition")
    if isinstance(prerecognition, dict):
        artifacts["atlas_prerecognition"] = {
            key: _json_safe(value)
            for key, value in prerecognition.items()
            if key != "rows"
        }
    return artifacts


def _standard_item(
    raw_result: dict[str, Any],
    *,
    task: str,
    input_path: str | Path | None,
) -> dict[str, Any]:
    source = _result_source(raw_result)
    classifier = _classifier_source(raw_result)
    thresholds = source.get("thresholds", {}) if isinstance(source.get("thresholds"), dict) else {}
    image_path = raw_result.get("image_path") or source.get("image_path") or raw_result.get("path") or input_path
    status = raw_result.get("status", source.get("status", "ok"))

    item = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "image_path": str(Path(image_path)) if image_path is not None else "",
        "analyzer": source.get("analyzer") or raw_result.get("analyzer") or "classifier",
        "status": status,
        "predictions": {
            "image_type": {
                "label": _image_type_label(raw_result, source),
                "confidence": classifier.get("confidence") if isinstance(classifier, dict) else None,
                "probabilities": _json_safe(classifier.get("probabilities", {})) if isinstance(classifier, dict) else {},
            },
            "tags": {
                "labels": _normalize_tags(source.get("predicted_tags")),
                "probabilities": _json_safe(source.get("tag_probabilities", {})),
                "thresholds": _json_safe(thresholds.get("tag_thresholds", {})),
            },
            "rating": {
                "label": source.get("predicted_rating"),
                "probabilities": _json_safe(source.get("rating_probabilities", {})),
                "threshold": thresholds.get("rating_threshold"),
            },
        },
        "recommendations": _json_safe(source.get("recommendations", [])),
        "artifacts": _artifact_values(raw_result, source),
        "warnings": [],
        "errors": [],
    }

    if isinstance(classifier, dict) and classifier.get("low_confidence_warning") is True:
        item["warnings"].append("Classifier confidence is below the configured threshold.")
    if raw_result.get("message"):
        target = "errors" if status not in {"ok", None} else "warnings"
        item[target].append(str(raw_result["message"]))
    return item


def standardized_result(
    raw_result: dict[str, Any] | list[dict[str, Any]],
    *,
    task: str,
    input_path: str | Path | None,
) -> dict[str, Any]:
    if isinstance(raw_result, list):
        items = [
            _standard_item(item, task=task, input_path=item.get("image_path", input_path))
            for item in raw_result
        ]
        statuses = {str(item.get("status", "ok")) for item in items}
        status = "ok" if statuses <= {"ok"} else "partial" if "ok" in statuses else sorted(statuses)[0]
        return {
            "schema_version": SCHEMA_VERSION,
            "task": task,
            "status": status,
            "input": {
                "path": str(Path(input_path)) if input_path is not None else "",
                "type": "folder" if input_path is not None and Path(input_path).is_dir() else "image",
            },
            "items": items,
            "artifacts": {},
            "warnings": [],
            "errors": [],
        }
    return _standard_item(raw_result, task=task, input_path=input_path)


def default_output_path(
    output: str | Path | None,
    *,
    task: str,
    input_path: str | Path | None,
    is_batch: bool,
) -> Path:
    suffix = "standardized_predictions" if is_batch else "standardized_prediction"
    stem = Path(input_path).stem if input_path is not None else task
    stem = stem or task or "cryoemdoc"
    default_name = f"{stem}_{suffix}.json"
    if output is None or str(output).strip() == "":
        return Path.cwd() / default_name
    output_path = Path(output)
    if output_path.suffix:
        return output_path
    return output_path / default_name


def summary_text(result: dict[str, Any], output_path: str | Path) -> str:
    output_path = Path(output_path)
    if "items" in result:
        items = result.get("items", [])
        count = len(items)
        noun = "prediction" if count == 1 else "predictions"
        lines = [f"Saved standardized {noun} to {output_path}."]
        if not items:
            lines.append("No images were found.")
            return "\n".join(lines)
        lines.extend(_item_summary(items[0], first=True))
        if count > 1:
            lines.append(f"Showing 1 of {count} results; the rest are in the saved file.")
        return "\n".join(lines)

    lines = [f"Saved standardized prediction to {output_path}."]
    lines.extend(_item_summary(result, first=False))
    return "\n".join(lines)


def _item_summary(item: dict[str, Any], *, first: bool) -> list[str]:
    predictions = item.get("predictions", {})
    image_type = predictions.get("image_type", {}) if isinstance(predictions.get("image_type"), dict) else {}
    tags = predictions.get("tags", {}) if isinstance(predictions.get("tags"), dict) else {}
    rating = predictions.get("rating", {}) if isinstance(predictions.get("rating"), dict) else {}
    tag_labels = _normalize_tags(tags.get("labels"))
    label = rating.get("label") or "unavailable"
    prefix = "First image" if first else "Image"
    lines = [f"{prefix}: {item.get('image_path', '')}"]
    if image_type.get("label"):
        lines.append(f"Image type: {image_type['label']}")
    lines.append(f"Predicted rating: {label}")
    lines.append(f"Predicted tags: {', '.join(tag_labels)}")
    return lines


def save_standard_output(
    raw_result: dict[str, Any] | list[dict[str, Any]],
    *,
    task: str,
    input_path: str | Path | None,
    output: str | Path | None,
) -> str:
    result = standardized_result(raw_result, task=task, input_path=input_path)
    output_path = default_output_path(output, task=task, input_path=input_path, is_batch=isinstance(raw_result, list))
    if "items" in result:
        result["artifacts"]["standardized_output"] = str(output_path)
    else:
        result["artifacts"]["standardized_output"] = str(output_path)
    write_json(result, output_path)
    return summary_text(result, output_path)
