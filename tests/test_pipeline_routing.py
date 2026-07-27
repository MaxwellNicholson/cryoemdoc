from __future__ import annotations

from pathlib import Path

import cryoemdoc.pipeline as pipeline


def _classifier(label: str):
    return {
        "path": "image.png",
        "image_path": "image.png",
        "predicted_label": label,
        "confidence": 0.99,
        "low_confidence_warning": False,
        "route": "test",
        "probabilities": {label: 0.99},
    }


def test_pipeline_routes_atlas_case_insensitively(monkeypatch):
    monkeypatch.setattr(pipeline, "_classify_image_result", lambda *args, **kwargs: _classifier("Atlas"))
    seen = {}

    def fake_analyze_atlas(*args, **kwargs):
        seen["kwargs"] = kwargs
        return {"analyzer": "atlas_without_csv"}

    monkeypatch.setattr(pipeline, "_analyze_atlas_result", fake_analyze_atlas)

    result = pipeline._analyze_image_result(Path("image.png"))
    assert result["status"] == "ok"
    assert result["analysis"]["analyzer"] == "atlas_without_csv"
    assert seen["kwargs"]["atlas_features"] == "none"


def test_pipeline_can_turn_generated_atlas_csv_on(monkeypatch):
    monkeypatch.setattr(pipeline, "_classify_image_result", lambda *args, **kwargs: _classifier("Atlas"))
    seen = {}

    def fake_analyze_atlas(*args, **kwargs):
        seen["kwargs"] = kwargs
        return {"analyzer": "atlas_with_csv"}

    monkeypatch.setattr(pipeline, "_analyze_atlas_result", fake_analyze_atlas)

    result = pipeline._analyze_image_result(Path("image.png"), use_atlas_csv=True)
    assert result["status"] == "ok"
    assert result["analysis"]["analyzer"] == "atlas_with_csv"
    assert seen["kwargs"]["atlas_features"] == "generate"


def test_pipeline_routes_square(monkeypatch):
    monkeypatch.setattr(pipeline, "_classify_image_result", lambda *args, **kwargs: _classifier("square"))
    monkeypatch.setattr(pipeline, "_analyze_square_result", lambda *args, **kwargs: {"analyzer": "square"})

    result = pipeline._analyze_image_result(Path("image.png"))
    assert result["status"] == "ok"
    assert result["analysis"]["analyzer"] == "square"


def test_pipeline_reports_protein_not_implemented(monkeypatch):
    monkeypatch.setattr(pipeline, "_classify_image_result", lambda *args, **kwargs: _classifier("Protein"))

    result = pipeline._analyze_image_result(Path("image.png"))
    assert result["status"] == "not_implemented"
    assert "Protein analyzer" in result["message"]


def test_pipeline_reports_other_unsupported(monkeypatch):
    monkeypatch.setattr(pipeline, "_classify_image_result", lambda *args, **kwargs: _classifier("Other"))

    result = pipeline._analyze_image_result(Path("image.png"))
    assert result["status"] == "unsupported"


def test_analyze_images_routes_folder(monkeypatch, tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    first = image_dir / "first.png"
    second = image_dir / "second.jpg"
    first.write_bytes(b"placeholder")
    second.write_bytes(b"placeholder")

    seen = []

    def fake_analyze_image(path, **kwargs):
        seen.append((Path(path).name, kwargs.get("output_dir")))
        return {"image_path": str(path), "status": "ok"}

    monkeypatch.setattr(pipeline, "_analyze_image_result", fake_analyze_image)
    results = pipeline._analyze_images_result(image_dir, atlas_features="generate", output_dir=tmp_path / "out")

    assert len(results) == 2
    assert {name for name, _ in seen} == {"first.png", "second.jpg"}
    assert all(output_dir is not None for _, output_dir in seen)
    assert all("atlas_features" in str(output_dir) for _, output_dir in seen)
