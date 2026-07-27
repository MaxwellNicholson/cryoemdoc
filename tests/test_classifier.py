from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from cryoemdoc.classifier import _classify_image_result

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_classifier_model_loads_and_predicts_sample():
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torchvision") is None:
        pytest.skip("torch/torchvision are not installed")

    sample = ROOT / "data" / "Atlas_All" / "Atlas_2.jpg"
    if not sample.exists():
        pytest.skip("sample atlas image is not available")

    result = _classify_image_result(sample, device="cpu")
    assert result["predicted_label"] in {"Atlas", "Square", "Protein", "Other"}
    assert set(result["probabilities"]) == {"Atlas", "Square", "Protein", "Other"}
    assert result["route"] != "unknown_route"
