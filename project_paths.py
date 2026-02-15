"""Portable path helpers shared by training, evaluation, and demo scripts."""
from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(os.environ.get("TINYLLM_ROOT", Path(__file__).resolve().parent)).resolve()


def path(relative: str | os.PathLike[str]) -> Path:
    """Resolve a repository-relative path under ``TINYLLM_ROOT``."""
    candidate = Path(relative)
    return candidate if candidate.is_absolute() else ROOT / candidate


def legacy_path(value: str) -> str:
    """Map paths from the original Windows/Linux experiments into this checkout.

    Existing absolute paths are kept when they are available. Missing historical
    paths are mapped by their known project suffix, so old training modules remain
    usable without embedding one contributor's drive letter or cloud home.
    """
    candidate = Path(value)
    if candidate.exists():
        return str(candidate)

    normalized = value.replace("\\", "/")
    known_roots = (
        "/root/autodl-tmp/tinyLLM/",
        "/root/autodl-tmp/llm/",
        "E:/tinyLLM/",
        "E:/llm/",
    )
    for prefix in known_roots:
        if normalized.startswith(prefix):
            return str(ROOT / normalized[len(prefix):])
    return str(candidate)
