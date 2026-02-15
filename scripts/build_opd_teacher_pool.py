"""Build the verifier-filtered MiniCPM3-4B IFEval teacher pool."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "opd_ifeval_shared_65_20_15.json"


if __name__ == "__main__":
    subprocess.run(
        [sys.executable, str(ROOT / "tools" / "build_opd_ifeval_teacher_pool.py"),
         "--config", str(CONFIG), *sys.argv[1:]],
        cwd=ROOT,
        check=True,
    )
