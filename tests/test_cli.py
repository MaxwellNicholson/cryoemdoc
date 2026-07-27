from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


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
