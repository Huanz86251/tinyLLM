#!/usr/bin/env python3
"""Export only the trainable non-LoRA vision modules from a VLM checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file, save_file


VISION_TRAINABLE_PATTERNS = (
    "query_embed",
    "qformer_blocks",
    "qformer_final_norm",
    "vision_projector",
    "vision_type_embed",
    "vision_view_embed",
    "vision_pos_embed",
    "vision_film_mlp",
    "vision_gate",
    "vision_kv_sep",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_file = args.checkpoint / "base" / "model.safetensors"
    config_file = args.checkpoint / "base" / "config.json"
    if not base_file.is_file() or not config_file.is_file():
        raise FileNotFoundError(f"Incomplete checkpoint: {args.checkpoint}")

    state = load_file(str(base_file), device="cpu")
    selected = {
        name: tensor
        for name, tensor in state.items()
        if any(pattern in name for pattern in VISION_TRAINABLE_PATTERNS)
    }
    if not selected:
        raise RuntimeError("No Q-Former or visual bridge tensors matched")
    unexpected_lora = [name for name in selected if ".adapters." in name]
    if unexpected_lora:
        raise RuntimeError(f"LoRA leaked into visual delta: {unexpected_lora[:3]}")

    args.output.mkdir(parents=True, exist_ok=True)
    delta_file = args.output / "vision_delta.safetensors"
    save_file(selected, str(delta_file))
    (args.output / "config.json").write_text(
        config_file.read_text(encoding="utf-8"), encoding="utf-8"
    )
    manifest = {
        "format": "tinyllm_vlm_trainable_overlay_v1",
        "source_checkpoint": str(args.checkpoint),
        "tensor_count": len(selected),
        "parameter_count": sum(t.numel() for t in selected.values()),
        "patterns": list(VISION_TRAINABLE_PATTERNS),
        "base_required": "local VLM stage 2 checkpoint 145458",
        "vision_tower_included": False,
        "language_base_included": False,
        "lora_included": False,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({**manifest, "bytes": delta_file.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
