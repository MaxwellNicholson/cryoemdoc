from __future__ import annotations

import importlib.util

import pytest
from PIL import Image, ImageDraw

from cryoemdoc.atlas_prerecognition import atlas_prerecognize


def test_atlas_prerecognition_writes_csv_on_synthetic_image(tmp_path):
    if importlib.util.find_spec("cv2") is None:
        pytest.skip("opencv-python-headless is not installed")

    image_path = tmp_path / "synthetic_atlas.png"
    image = Image.new("L", (256, 256), 0)
    draw = ImageDraw.Draw(image)
    for y in range(32, 210, 44):
        for x in range(32, 210, 44):
            draw.rectangle([x, y, x + 24, y + 24], fill=220)
    image.save(image_path)

    output_csv = tmp_path / "atlas_summary_scores.csv"
    result = atlas_prerecognize(image_path, output=output_csv, save_annotated_images=False)
    assert output_csv.exists()
    assert result["csv_path"] == str(output_csv)
    assert result["row_count"] >= 1
    assert set(result["rows"][0]).issuperset({
        "image_name",
        "visible_square_count",
        "good_square_count",
        "non_uniform_square_count",
        "cracked_square_count",
        "bad_size_square_count",
        "atlas_quality_score",
        "error",
    })
    assert "rank" not in result["rows"][0]
