#!/usr/bin/env python3
"""Build Hugging Face model and adapter repositories from tinyLLM checkpoints."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "added_tokens.json",
)


def write_remote_code(output: Path) -> None:
    config_source = (ROOT / "model" / "config.py").read_text(encoding="utf-8")
    model_source = (ROOT / "model" / "model.py").read_text(encoding="utf-8")
    model_source = model_source.replace(
        "from model.config import Config",
        "from .configuration_tinyllm import Config",
        1,
    )
    (output / "configuration_tinyllm.py").write_text(config_source, encoding="utf-8")
    (output / "modeling_tinyllm.py").write_text(model_source, encoding="utf-8")
    (output / "__init__.py").write_text(
        "from .configuration_tinyllm import Config\n"
        "from .modeling_tinyllm import TinyLLM\n",
        encoding="utf-8",
    )


def copy_tokenizer(source: Path, output: Path) -> None:
    copied = 0
    for name in TOKENIZER_FILES:
        src = source / name
        if src.is_file():
            shutil.copy2(src, output / name)
            copied += 1
    if not copied:
        raise FileNotFoundError(f"No tokenizer files found in {source}")


def normalize_checkpoint(source: Path, destination: Path) -> dict:
    state = load_file(str(source), device="cpu")
    normalized = {}
    renamed = 0
    for key, value in state.items():
        new_key = key.replace(".base.weight", ".weight").replace(".base.bias", ".bias")
        if new_key in normalized:
            raise RuntimeError(f"Checkpoint normalization produced duplicate key {new_key}")
        normalized[new_key] = value.contiguous()
        renamed += int(new_key != key)
    save_file(normalized, str(destination), metadata={"format": "pt"})
    return {"tensor_count": len(normalized), "renamed_lora_base_tensors": renamed}


def model_card(*, model_kind: str, repo_id: str | None, base_model: str | None,
               vision_tower: str | None) -> str:
    tags = ["tinyllm", "custom_code"]
    if model_kind == "vlm":
        tags += ["vision-language", "image-text-to-text"]
    metadata = ["---", "library_name: transformers", "license: other"]
    if base_model:
        metadata.append(f"base_model: {base_model}")
    if model_kind == "vlm":
        metadata.append("pipeline_tag: image-text-to-text")
    else:
        metadata.append("pipeline_tag: text-generation")
    metadata += ["tags:"] + [f"- {tag}" for tag in tags] + ["---", ""]
    title = "tinyLLM VLM 0.51B" if model_kind == "vlm" else "tinyLLM SFT 0.51B"
    published_id = repo_id or "REPLACE_WITH_REPO_ID"
    usage = f"""```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = \"{published_id}\"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype=\"auto\",
)
```
"""
    body = [
        f"# {title}",
        "",
        "tinyLLM is a custom 0.51B decoder-only model trained from scratch for Chinese and English.",
        "Loading requires `trust_remote_code=True` because the architecture is implemented in this repository.",
        "",
        "## Load",
        "",
        usage,
    ]
    if model_kind == "vlm":
        body += [
            "## Vision encoder",
            "",
            f"The frozen vision tower is `{vision_tower}` and is loaded separately. This repository contains the language model, Q-Former, projector and the visual LoRA adapter.",
            "Call `model.load_lora_pretrained(model_id, subfolder=\"adapter\")` after loading the model.",
            "The model forward accepts precomputed `vision_feats`, `vision_mask`, `global_pos` and `global_off` tensors.",
            "",
        ]
    body += [
        "## Source",
        "",
        "Training code and full project documentation: https://github.com/Huanz86251/tinyLLM",
        "",
    ]
    return "\n".join(metadata + body)


def export_model(args: argparse.Namespace) -> None:
    checkpoint = args.checkpoint.resolve()
    tokenizer = (args.tokenizer or checkpoint).resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    source_config = checkpoint / "config.json"
    source_weights = checkpoint / "model.safetensors"
    if not source_config.is_file() or not source_weights.is_file():
        raise FileNotFoundError(f"Incomplete checkpoint: {checkpoint}")

    config = json.loads(source_config.read_text(encoding="utf-8"))
    config["architectures"] = ["TinyLLM"]
    config["auto_map"] = {
        "AutoConfig": "configuration_tinyllm.Config",
        "AutoModelForCausalLM": "modeling_tinyllm.TinyLLM",
    }
    config["is_decoder"] = True
    config["is_encoder_decoder"] = False
    if args.vision_tower:
        config["vision_tower_name_or_path"] = args.vision_tower
    (output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    copy_tokenizer(tokenizer, output)
    write_remote_code(output)

    if args.normalize_lora_wrappers:
        stats = normalize_checkpoint(source_weights, output / "model.safetensors")
    else:
        shutil.copy2(source_weights, output / "model.safetensors")
        stats = {"tensor_count": None, "renamed_lora_base_tensors": 0}

    generation_config = {
        "_from_model_config": True,
        "bos_token_id": config.get("bos_token_id"),
        "eos_token_id": config.get("eos_token_id"),
        "pad_token_id": config.get("pad_token_id"),
        "do_sample": False,
        "max_new_tokens": 2048,
        "repetition_penalty": 1.08,
        "no_repeat_ngram_size": 16,
    }
    (output / "generation_config.json").write_text(
        json.dumps(generation_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    kind = "vlm" if bool(config.get("use_vision")) else "text"
    (output / "README.md").write_text(
        model_card(model_kind=kind, repo_id=getattr(args, "repo_id", None),
                   base_model=args.base_model, vision_tower=args.vision_tower),
        encoding="utf-8",
    )
    manifest = {
        "format": "tinyllm_huggingface_model_v1",
        "kind": kind,
        "source_checkpoint": str(checkpoint),
        "vision_tower": args.vision_tower,
        **stats,
    }
    (output / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def load_adapter_weights(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        return load_file(str(path), device="cpu")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if isinstance(value, dict) and "state_dict" in value and isinstance(value["state_dict"], dict):
        value = value["state_dict"]
    if not isinstance(value, dict):
        raise TypeError(f"Adapter file is not a state dict: {path}")
    return value


def export_adapter(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = load_adapter_weights(args.weights.resolve())
    state = {key: value.detach().cpu().contiguous() for key, value in state.items()
             if ".adapters." in key}
    if not state:
        raise RuntimeError("No native tinyLLM LoRA tensors found")
    save_file(state, str(output / "adapter_model.safetensors"))
    metadata = {
        "format": "tinyllm_lora_v1",
        "adapter_name": args.name,
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "target": args.target,
        "base_model_name_or_path": args.base_model,
    }
    (output / "adapter_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "---\nlibrary_name: transformers\nlicense: other\ntags:\n- tinyllm\n- lora\n---\n\n"
        f"# tinyLLM adapter: {args.name}\n\n"
        f"Base model: `{args.base_model}`\n\n"
        "```python\n"
        "from transformers import AutoModelForCausalLM\n"
        f"model = AutoModelForCausalLM.from_pretrained(\"{args.base_model}\", trust_remote_code=True)\n"
        f"model.load_lora_pretrained(\"{args.repo_id or 'REPLACE_WITH_ADAPTER_REPO_ID'}\")\n"
        "```\n",
        encoding="utf-8",
    )
    print(json.dumps({**metadata, "tensor_count": len(state)}, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    model = commands.add_parser("model", help="export a base or VLM model repository")
    model.add_argument("--checkpoint", type=Path, required=True)
    model.add_argument("--tokenizer", type=Path)
    model.add_argument("--output", type=Path, required=True)
    model.add_argument("--repo-id")
    model.add_argument("--base-model")
    model.add_argument("--vision-tower", default=None)
    model.add_argument("--normalize-lora-wrappers", action="store_true")
    model.set_defaults(func=export_model)

    adapter = commands.add_parser("adapter", help="export a native LoRA repository")
    adapter.add_argument("--weights", type=Path, required=True)
    adapter.add_argument("--output", type=Path, required=True)
    adapter.add_argument("--repo-id")
    adapter.add_argument("--name", required=True)
    adapter.add_argument("--rank", type=int, required=True)
    adapter.add_argument("--alpha", type=float, required=True)
    adapter.add_argument("--dropout", type=float, default=0.0)
    adapter.add_argument("--target", required=True)
    adapter.add_argument("--base-model", required=True)
    adapter.set_defaults(func=export_adapter)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
