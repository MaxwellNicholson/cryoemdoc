from __future__ import annotations

import json
from pathlib import Path

import cryoemdoc.pipeline as pipeline
import cryoemdoc.square as square


def test_public_square_saves_standardized_json_and_returns_summary(monkeypatch, tmp_path):
    def fake_square_result(*args, **kwargs):
        return {
            "image_path": "square.jpg",
            "analyzer": "square",
            "predicted_tags": ["no tags"],
            "recommendations": ["no recommendation"],
            "tag_probabilities": {"thick ice": 0.1},
            "predicted_rating": "acceptable",
            "rating_probabilities": {"acceptable": 0.9},
            "thresholds": {
                "tag_thresholds": {"thick ice": 0.5},
                "rating_threshold": 0.6,
            },
        }

    monkeypatch.setattr(square, "_analyze_square_result", fake_square_result)
    output_path = tmp_path / "prediction.json"
    summary = square.analyze_square("square.jpg", output=output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved["predictions"]["tags"]["labels"] == ["no issues"]
    assert saved["predictions"]["rating"]["label"] == "acceptable"
    assert saved["artifacts"]["standardized_output"] == str(output_path)
    assert "Predicted rating: acceptable" in summary
    assert "Predicted tags: no issues" in summary


def test_public_batch_summary_shows_first_and_saves_all(monkeypatch, tmp_path):
    def fake_results(*args, **kwargs):
        return [
            {
                "image_path": "first.jpg",
                "status": "ok",
                "analysis": {
                    "analyzer": "atlas_without_csv",
                    "predicted_tags": ["cracks"],
                    "predicted_rating": "unacceptable",
                },
            },
            {
                "image_path": "second.jpg",
                "status": "ok",
                "analysis": {
                    "analyzer": "square",
                    "predicted_tags": ["no tags"],
                    "predicted_rating": "acceptable",
                },
            },
        ]

    monkeypatch.setattr(pipeline, "_analyze_images_result", fake_results)
    output_path = tmp_path / "batch.json"
    summary = pipeline.analyze_images(tmp_path, output=output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert len(saved["items"]) == 2
    assert saved["items"][0]["image_path"] == "first.jpg"
    assert "First image: first.jpg" in summary
    assert "Showing 1 of 2 results; the rest are in the saved file." in summary


def test_blank_output_defaults_to_current_directory(monkeypatch, tmp_path):
    def fake_square_result(*args, **kwargs):
        return {
            "image_path": str(Path("atlas.png")),
            "analyzer": "atlas_without_csv",
            "predicted_tags": "no issues",
            "predicted_rating": "acceptable",
        }

    monkeypatch.setattr(square, "_analyze_square_result", fake_square_result)
    monkeypatch.chdir(tmp_path)
    summary = square.analyze_square("atlas.png", output="")
    expected_path = tmp_path / "atlas_standardized_prediction.json"

    assert expected_path.exists()
    assert str(expected_path) in summary
