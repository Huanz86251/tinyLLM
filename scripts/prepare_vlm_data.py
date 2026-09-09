#!/usr/bin/env python3
"""Capacity-gated, resumable VLM v5 preprocessing. Never starts training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def run(items: list[object], *, env=None) -> None:
    subprocess.run([str(x) for x in items], cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "vlm_continual_zh_en_v5.json",
    )
    parser.add_argument("--start-preparation", action="store_true")
    parser.add_argument("--skip-pack", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if cfg.get("training_enabled") is not False:
        raise RuntimeError("preparation requires training_enabled=false")

    image_root = project_path(cfg["paths"]["image_root"])
    image_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(image_root).free
    estimated = int(cfg["storage_plan"]["estimated_five_view_bytes"])
    minimum = int(cfg["storage_plan"]["minimum_free_bytes"])
    try:
        existing = int(subprocess.check_output(["du", "-sb", str(image_root)], text=True).split()[0])
    except Exception:
        existing = sum(path.stat().st_size for path in image_root.rglob("*") if path.is_file())
    remaining_estimate = max(0, estimated - existing)
    preflight = {
        "mode": "prepare" if args.start_preparation else "preflight_only",
        "free_gib": round(free / 2**30, 2),
        "existing_five_view_gib": round(existing / 2**30, 2),
        "remaining_estimated_gib": round(remaining_estimate / 2**30, 2),
        "predicted_free_after_gib": round((free - remaining_estimate) / 2**30, 2),
        "minimum_free_gib": round(minimum / 2**30, 2),
        "training_will_start": False,
    }
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    if free - remaining_estimate < minimum:
        raise RuntimeError("capacity gate failed; preprocessing was not started")
    if not args.start_preparation:
        return

    py = sys.executable
    for source, spec in cfg["sources"].items():
        if not spec.get("enabled", False):
            continue
        completed = project_path(cfg["paths"]["prepared_root"]) / "parts" / f"{source}.manifest.json"
        if completed.is_file():
            print(f"[skip completed] {source}", flush=True)
            continue
        run([
            py,
            ROOT / "data_preprocess" / "VLLM" / "build_vlm_source_v5.py",
            "--config", args.config,
            "--source", source,
        ])

    run([
        py,
        ROOT / "data_preprocess" / "VLLM" / "assemble_vlm_continual_v5.py",
        "--config", args.config,
    ])
    if args.skip_pack:
        print("VLM v5 preprocessing completed; packing skipped; training was not started.")
        return
    prepared = project_path(cfg["paths"]["prepared_root"])
    mixed = prepared / "vlm_continual_v5.mixed.jsonl"
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    compat_transformers = ROOT / "runtime" / "minicpm_transformers_449"
    python_paths = [str(compat_transformers), str(ROOT)]
    if existing_pythonpath:
        python_paths.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env.update({
        "VLM_PACK_MAX_LEN": str(cfg["max_total_tokens"]),
        "VLM_PACK_TOKENIZER_DIR": str(project_path(cfg["paths"]["tokenizer"])),
        "VLM_PACK_JSONL": str(mixed),
        "VLM_PACK_IMAGE_DIR": str(project_path(cfg["paths"]["image_root"])),
        "VLM_PACK_OUT_DIR": str(project_path(cfg["paths"]["packed_dataset"])),
    })
    run([py, ROOT / "train" / "vlm" / "pack_continual.py"], env=env)
    print("VLM v5 preprocessing and packing completed; model training was not started.")


if __name__ == "__main__":
    main()




