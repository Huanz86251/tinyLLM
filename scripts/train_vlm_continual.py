#!/usr/bin/env python3
"""Preflight and launch the audited v5 tinyLLM VLM continuation run.

The default invocation is read-only. Training requires both
``training_enabled=true`` in the v5 JSON config and ``--start-training``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def gpu_free_gib() -> float | None:
    tool = shutil.which("nvidia-smi")
    if not tool:
        return None
    result = subprocess.run(
        [tool, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return float(result.stdout.splitlines()[0].strip()) / 1024.0


def require_dir(path: str, label: str) -> None:
    if not Path(path).is_dir():
        raise FileNotFoundError(f"{label} directory is missing: {path}")


def estimate_cache_gib(train: dict) -> float:
    # Measured v5 mean dynamic micro-batch is about 7.2 examples. Each example
    # stores five 1024x1024 BF16 feature grids (roughly 10 MiB) plus metadata.
    train_rows = float(train["vit_cache_chunk_steps"]) * 2.0 * 7.2
    rows = train_rows + float(train["explicit_eval_max_rows"])
    return rows * 10.0 / 1024.0 * 1.08


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "vlm_continual_zh_en_v5.json",
    )
    parser.add_argument("--start-training", action="store_true")
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=0,
        help="Run a bounded real-GPU smoke test in a separate output directory.",
    )
    args = parser.parse_args()
    if args.smoke_steps < 0:
        parser.error("--smoke-steps must be >= 0")

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    paths = cfg["paths"]
    train = cfg["continuation"]

    packed = str(project_path(paths["packed_dataset"]))
    image_root = str(project_path(paths["image_root"]))
    resume = str(project_path(paths["resume_vlm"]))
    tokenizer = str(project_path(paths["tokenizer"]))
    vision = str(project_path(paths["vision_tower"]))
    output = str(project_path(paths["output"]))
    if args.smoke_steps:
        output = str(
            Path(output).parent
            / f"{Path(output).name}_smoke_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
    text_replay = str(project_path(paths["text_replay"])) if paths.get("text_replay") else ""

    summary = {
        "mode": "train" if args.start_training else "preflight_only",
        "training_enabled": bool(cfg.get("training_enabled")),
        "dataset": packed,
        "images": image_root,
        "resume": resume,
        "output": output,
        "epochs": train["epochs"],
        "smoke_steps": args.smoke_steps,
        "lora": {
            "name": train["adapter_name"],
            "rank": train["lora_rank"],
            "alpha": train["lora_alpha"],
        },
        "learning_rates": {
            "language_lora": train["learning_rate_language_lora"],
            "qformer": train["learning_rate_qformer"],
            "bridge": train["learning_rate_bridge"],
        },
        "loss": {
            "reduction": train["loss_reduction"],
            "sample_mean_alpha": train["sample_mean_alpha"],
        },
        "tile_shuffle_probability": train["tile_shuffle_probability"],
        "rolling_cache": {
            "eval_rows": train["explicit_eval_max_rows"],
            "chunk_steps": train["vit_cache_chunk_steps"],
            "map_size_gb": train["vit_cache_map_size_gb"],
        },
        "vision_tower_trainable": bool(train["train_vision_tower"]),
        "text_replay": {
            "dataset": text_replay,
            "ratio": train.get("text_replay_ratio", 0.0),
            "zh_fraction": train.get("text_replay_zh_fraction", 0.0),
            "max_repeats_per_epoch": train.get("text_replay_max_repeats", 16),
            "eval_rows": train.get("text_replay_eval_max_rows", 0),
            "forward": "text_only_without_visual_prefix",
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if not args.start_training:
        print("Preflight only: no model was loaded and no training was started.")
        return
    if not cfg.get("training_enabled"):
        raise RuntimeError(
            "Refusing to train: set training_enabled=true only after OPD has "
            "released the GPU and the preflight has passed."
        )
    if train.get("train_vision_tower"):
        raise RuntimeError(
            "The offline-feature trainer cannot update InternViT. Keep it frozen "
            "for this checkpoint-compatible continuation run."
        )

    for path, label in [
        (packed, "packed dataset"),
        (image_root, "five-view images"),
        (resume, "resume VLM"),
        (tokenizer, "tokenizer"),
        (vision, "vision tower"),
    ]:
        require_dir(path, label)
    if float(train.get("text_replay_ratio", 0.0)) > 0.0:
        require_dir(text_replay, "text replay dataset")

    cache_gib = estimate_cache_gib(train)
    disk_free_gib = shutil.disk_usage(Path(output).parent).free / (1024.0 ** 3)
    if disk_free_gib < cache_gib + 8.0:
        raise RuntimeError(
            f"Only {disk_free_gib:.2f} GiB disk is free, but the rolling cache "
            f"needs about {cache_gib:.2f} GiB plus an 8 GiB checkpoint margin."
        )

    free = gpu_free_gib()
    if free is not None and free < 20.0:
        raise RuntimeError(
            f"Only {free:.2f} GiB GPU memory is free; wait for OPD to release the GPU."
        )

    env = os.environ.copy()
    compat_transformers = ROOT / "runtime" / "minicpm_transformers_449"
    python_paths = [str(ROOT)]
    if compat_transformers.is_dir():
        python_paths.insert(0, str(compat_transformers))
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    qformer_ratio = (
        float(train["learning_rate_qformer"])
        / float(train["learning_rate_language_lora"])
    )
    bridge_ratio = (
        float(train["learning_rate_bridge"])
        / float(train["learning_rate_language_lora"])
    )
    env.update(
        {
            "SFT_STAGE": "2",
            "SFT_RUN_NAME": "tinyllm_vlm_continual_zh_en_v5",
            "SFT_OUT_DIR": output,
            "SFT_RESUME_ROOT": resume,
            "SFT_CACHE_SHORT": packed,
            "SFT_TOKENIZER_DIR": tokenizer,
            "VIT_DIR": vision,
            "IMAGE_DIR": image_root,
            "SFT_SEED": str(cfg["seed"]),
            "SFT_EPOCHS": str(train["epochs"]),
            "SFT_LEARNING_RATE": str(train["learning_rate_language_lora"]),
            "SFT_STAGE2_QFORMER_LR_RATIO": str(qformer_ratio),
            "SFT_STAGE2_BRIDGE_LR_RATIO": str(bridge_ratio),
            "SFT_WARMUP_STEPS": "0",
            "SFT_WARMUP_RATIO": str(train["warmup_ratio"]),
            "SFT_LR_SCHEDULER": str(train["scheduler"]),
            "SFT_MIN_LR_RATIO": str(train["min_lr_ratio"]),
            "SFT_MAX_GRAD_NORM": str(train["max_grad_norm"]),
            "SFT_WEIGHT_DECAY": str(train["weight_decay"]),
            "SFT_LORA_NAME": str(train["adapter_name"]),
            "SFT_LORA_RANK": str(train["lora_rank"]),
            "SFT_LORA_ALPHA": str(train["lora_alpha"]),
            "SFT_USE_BF16": "1" if train["bf16"] else "0",
            "SFT_LOSS_REDUCTION": str(train["loss_reduction"]),
            "SFT_SAMPLE_MEAN_ALPHA": str(train["sample_mean_alpha"]),
            "SFT_TILE_SHUFFLE_P": str(train["tile_shuffle_probability"]),
            "SFT_EXPLICIT_EVAL_MAX": str(train["explicit_eval_max_rows"]),
            "SFT_TEXT_REPLAY_CACHE": str(text_replay),
            "SFT_TEXT_REPLAY_RATIO": str(train.get("text_replay_ratio", 0.0)),
            "SFT_TEXT_REPLAY_MAX_REPEATS": str(train.get("text_replay_max_repeats", 16)),
            "SFT_TEXT_REPLAY_ZH_FRACTION": str(train.get("text_replay_zh_fraction", 2/3)),
            "SFT_TEXT_REPLAY_EVAL_MAX": str(train.get("text_replay_eval_max_rows", 256)),
            "SFT_SAVE_INTERVAL_STEPS": str(train["save_interval_steps"]),
            "VIT_CHUNK_STEPS": str(train["vit_cache_chunk_steps"]),
            "VIT_CLEAR_BETWEEN_CHUNKS": "1",
            "VIT_LMDB_MAP_SIZE_GB": str(train["vit_cache_map_size_gb"]),
            "CLEAR_VIT_CACHE_ON_START": (
                "1" if train["clear_vit_cache_on_start"] else "0"
            ),
        }
    )
    if args.smoke_steps:
        env["SFT_MAX_STEPS"] = str(args.smoke_steps)
        env["SFT_SAVE_INTERVAL_STEPS"] = str(args.smoke_steps)
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / "launch_config.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [sys.executable, str(ROOT / "train" / "vllm_sft_continual.py")],
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
