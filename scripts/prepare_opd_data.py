"""Prepare AllenAI RLVR-IFEval plus bilingual replay data."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    for script in (
        ROOT / "tools" / "prepare_opd_ifeval.py",
        ROOT / "tools" / "prepare_shared_general_replay.py",
    ):
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
