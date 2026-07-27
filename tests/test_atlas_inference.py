from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from cryoemdoc.atlas import _analyze_atlas_with_csv_result, _analyze_atlas_without_csv_result
import cryoemdoc.atlas as atlas_module
import cryoemdoc.atlas_prerecognition as prerec_module

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_atlas_without_csv_model_loads_and_predicts_sample():
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torchvision") is None:
        pytest.skip("torch/torchvision are not installed")

    sample = ROOT / "data" / "Atlas_All" / "Atlas_2.jpg"
    if not sample.exists():
        pytest.skip("sample atlas image is not available")

    result = _analyze_atlas_without_csv_result(sample, device="cpu")
    assert result["analyzer"] == "atlas_without_csv"
    assert result["predicted_rating"] in {"acceptable", "unacceptable"}
    assert set(result["tag_probabilities"]) == {"cracks", "non-uniform ice", "thick ice"}


@pytest.mark.slow
def test_atlas_with_csv_model_loads_and_predicts_sample():
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torchvision") is None:
        pytest.skip("torch/torchvision are not installed")

    sample = ROOT / "data" / "Atlas_All" / "Atlas_2.jpg"
    csv_path = ROOT / "data" / "atlas_summary_scores.csv"
    if not sample.exists() or not csv_path.exists():
        pytest.skip("sample atlas image or CSV is not available")

    result = _analyze_atlas_with_csv_result(sample, csv_path, device="cpu")
    assert result["analyzer"] == "atlas_with_csv"
    assert result["predicted_rating"] in {"acceptable", "unacceptable"}
    assert "atlas_score_features" in result


def test_analyze_atlas_folder_delegates(monkeypatch, tmp_path):
    image_dir = tmp_path / "Atlas_All"
    image_dir.mkdir()
    seen = {}

    def fake_analyze_atlas_images(input_path, **kwargs):
        seen["input_path"] = input_path
        seen["kwargs"] = kwargs
        return [{"status": "ok"}]

    monkeypatch.setattr(atlas_module, "_analyze_atlas_images_result", fake_analyze_atlas_images)
    result = atlas_module._analyze_atlas_result(image_dir, use_atlas_csv=True)

    assert result == [{"status": "ok"}]
    assert seen["input_path"] == image_dir
    assert seen["kwargs"]["atlas_features"] == "generate"


def test_generated_atlas_csv_mode_auto_saves_predictions(monkeypatch, tmp_path):
    image_dir = tmp_path / "Atlas_All"
    image_dir.mkdir()
    image_path = image_dir / "atlas.png"
    image_path.write_bytes(b"placeholder")

    def fake_prerecognize(input_path, output, **kwargs):
        return {
            "csv_path": str(output),
            "summary_csv_path": str(output),
            "output_format": "summary",
            "row_count": 1,
            "rows": [],
        }

    def fake_analyze_atlas_with_csv(path, csv_path, **kwargs):
        return {
            "image_path": str(path),
            "analyzer": "atlas_with_csv",
            "predicted_rating": "acceptable",
            "predicted_tags": "no issues",
            "recommendations": ["no recommendation"],
        }

    monkeypatch.setattr(prerec_module, "atlas_prerecognize", fake_prerecognize)
    monkeypatch.setattr(atlas_module, "_analyze_atlas_with_csv_result", fake_analyze_atlas_with_csv)

    results = atlas_module._analyze_atlas_images_result(image_dir, use_atlas_csv=True)

    prediction_csv = image_dir / "cryoemdoc_atlas_features" / "atlas_csv_analyzer_predictions.csv"
    assert prediction_csv.exists()
    assert results[0]["analyzer"] == "atlas_with_csv"
    assert results[0]["atlas_csv_analyzer_predictions_csv"] == str(prediction_csv)
