from __future__ import annotations

from cryoemdoc import (
    analyze_atlas,
    analyze_atlas_images,
    analyze_image,
    analyze_images,
    analyze_square,
    analyze_square_images,
    classify_image,
    classify_images,
)
from cryoemdoc.assets import ARTIFACT_FILENAMES, artifact_path


def test_public_api_imports():
    assert callable(analyze_image)
    assert callable(analyze_images)
    assert callable(classify_image)
    assert callable(classify_images)
    assert callable(analyze_square)
    assert callable(analyze_square_images)
    assert callable(analyze_atlas)
    assert callable(analyze_atlas_images)


def test_bundled_metadata_paths_resolve():
    assert artifact_path("square_analyzer", "labels").exists()
    assert artifact_path("square_analyzer", "thresholds").exists()
    assert artifact_path("atlas_analyzer_without_csv", "labels").exists()
    assert artifact_path("atlas_analyzer_with_csv", "tabular").exists()


def test_model_artifact_paths_can_be_optional():
    path = artifact_path("image_classifier", "model", required=False)
    assert path.name == ARTIFACT_FILENAMES["image_classifier"]["model"]
