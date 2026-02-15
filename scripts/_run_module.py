"""Small helper used by the public command entry points."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(relative_file: str) -> None:
    sys.path.insert(0, str(ROOT))
    runpy.run_path(str(ROOT / relative_file), run_name="__main__")
