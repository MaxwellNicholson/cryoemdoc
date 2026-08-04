from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import cryoemdoc.cli as cli
import cryoemdoc.square as square


def test_cli_help_runs():
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "cryoemdoc", "--help"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "atlas-prerecognize" in completed.stdout
    assert "download-models" in completed.stdout


def test_cli_prints_recommendations(monkeypatch, tmp_path, capsys):
    def fake_square_result(*args, **kwargs):
        return {
            "image_path": "square.jpg",
            "analyzer": "square",
            "predicted_tags": ["thick ice"],
            "recommendations": ["increase blotting force/time"],
            "predicted_rating": "unacceptable",
        }

    monkeypatch.setattr(square, "_analyze_square_result", fake_square_result)

    cli.main(["square", "square.jpg", "--output", str(tmp_path / "prediction.json")])

    captured = capsys.readouterr()
    assert "Predicted tags: thick ice" in captured.out
    assert "Recommendations: increase blotting force/time" in captured.out
