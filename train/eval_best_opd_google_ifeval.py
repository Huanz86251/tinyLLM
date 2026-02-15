"""Evaluate the best eligible OPD adapter on Google's official 541 IFEval prompts."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.eval_ifeval_baselines import (
    atomic_json,
    dataset_fingerprint,
    evaluate_model,
    generate_student,
)
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from project_paths import path
from train import GRPO as g


def file_sha256(file: Path) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    training_report_path = Path(args.training_report).resolve()
    training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
    if training_report.get("status") != "completed":
        raise RuntimeError(f"OPD training is not complete: {training_report.get('status')!r}")
    adapter_path = Path(training_report["best_eligible_adapter"]).resolve()
    if not adapter_path.is_file():
        raise FileNotFoundError(adapter_path)
    config = training_report["config"]
    student_config = config["student"]

    random.seed(20260904)
    torch.manual_seed(20260904)
    torch.cuda.manual_seed_all(20260904)
    dataset = [dict(item) for item in load_dataset("google/IFEval", split="train")]
    if len(dataset) != 541:
        raise RuntimeError(f"expected 541 official IFEval prompts, found {len(dataset)}")
    if args.limit:
        dataset = dataset[:args.limit]

    output_dir = path("runs/evaluation/ifeval_opd_best") / (
        "full" if not args.limit else f"first_{args.limit}"
    )
    state_path = output_dir / "report.json"
    adapter_sha = file_sha256(adapter_path)
    fingerprint = dataset_fingerprint(dataset)
    decode = {"method": "greedy", "system_prompt": None,
              "max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size}
    state = {
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmark": "google/IFEval official 541 evaluation prompts",
        "split_name_in_huggingface": "train (evaluation-only by benchmark semantics)",
        "dataset_rows": len(dataset),
        "dataset_sha256": fingerprint,
        "decode": decode,
        "verifier": "Google instruction_following_eval strict and loose",
        "training_report": str(training_report_path),
        "best_dev_accuracy": training_report.get("best_eligible_accuracy"),
        "adapter_path": str(adapter_path),
        "adapter_sha256": adapter_sha,
        "models": {},
    }
    if state_path.is_file():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        compatible = (
            previous.get("dataset_sha256") == fingerprint
            and previous.get("decode") == decode
            and previous.get("adapter_sha256") == adapter_sha
        )
        if compatible:
            state = previous
            state["status"] = "running"
    atomic_json(state_path, state)

    teacher_dir = path("models/teacher/minicpm3_4b_recovered")
    tokenizer = AutoTokenizer.from_pretrained(
        str(teacher_dir), trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = g.load_tinyllm_from_ckpt(
        str(path(student_config["path"])), tokenizer, strict=True
    )
    model.attach_lora_adapter(
        student_config["adapter_name"],
        rank=int(student_config["lora_rank"]),
        dropout=0.0,
        alpha=float(student_config["lora_alpha"]),
        target=student_config["lora_target"],
    )
    adapter_state = torch.load(adapter_path, map_location="cpu")
    model.load_lora_state_dict(
        adapter_state, student_config["adapter_name"], strict=False
    )
    model.activate_single_lora(student_config["adapter_name"])
    label = "student_opd_best_0_51b"
    if "score" not in state["models"].get(label, {}):
        evaluate_model(
            label, model, tokenizer, dataset, state, state_path,
            args.batch_size, args.max_new_tokens, generate_student,
        )

    baseline_path = path("runs/evaluation/ifeval_baselines/full/report.json")
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_score = baseline.get("models", {}).get("student_sft_0_51b", {}).get("score")
        score = state["models"][label]["score"]
        state["sft_baseline_score"] = baseline_score
        if baseline_score:
            state["delta_vs_sft"] = {
                mode: {
                    metric: score[mode][metric] - baseline_score[mode][metric]
                    for metric in ("prompt_accuracy", "instruction_accuracy")
                }
                for mode in ("strict", "loose")
            }
    state["status"] = "completed"
    state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    atomic_json(state_path, state)
    print(json.dumps({
        "status": state["status"],
        "score": state["models"][label]["score"],
        "delta_vs_sft": state.get("delta_vs_sft"),
        "report": str(state_path),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
