from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from cryoemdoc.square import _analyze_square_result

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_square_analyzer_model_loads_and_predicts_sample():
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torchvision") is None:
        pytest.skip("torch/torchvision are not installed")

    sample = next((ROOT / "data" / "Square_testing").glob("*.jpg"), None)
    if sample is None:
        pytest.skip("sample square image is not available")

    result = _analyze_square_result(sample, device="cpu")
    assert result["analyzer"] == "square"
    assert result["predicted_rating"] in {"good", "acceptable", "unacceptable"}
    assert set(result["tag_probabilities"]) == {"thick ice", "non-uniform ice", "ice contamination"}
