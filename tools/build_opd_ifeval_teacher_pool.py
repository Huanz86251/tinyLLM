"""Build a resumable verified MiniCPM3-4B teacher pool for RLVR-IFeval."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPAT = ROOT / "runtime" / "minicpm_transformers_449"
if not (COMPAT / "transformers" / "__init__.py").is_file():
    raise FileNotFoundError(COMPAT)
sys.path.insert(0, str(COMPAT))
sys.path.insert(0, str(ROOT))

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from project_paths import path
from train import GRPO as g
from train.opd_gsm8k_stage import is_loop
from train.opd_hybrid_v3 import file_sha256


def read_jsonl(file: Path) -> list[dict]:
    if not file.is_file():
        return []
    with file.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(file: Path, value: dict) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(file) + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(file)


def load_verifiers(file: Path) -> dict:
    spec = importlib.util.spec_from_file_location("tinyllm_ifeval_functions", file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import verifier file: {file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.IF_FUNCTIONS_MAP


def verify_response(row: dict, text: str, verifier_map: dict) -> bool:
    function = verifier_map[row["verifier"]]
    try:
        return bool(function(text, **row["verifier_args"]))
    except Exception:
        return False


@torch.inference_mode()
def generate_batch(model, tokenizer, rows: list[dict], attempt: dict, max_input_tokens: int) -> list[dict]:
    prompts = [g.render_prompt_with_tokenizer(tokenizer, row["prompt"], None) for row in rows]
    encoded = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_input_tokens, add_special_tokens=False,
    )
    encoded = {name: value.to("cuda") for name, value in encoded.items()}
    kwargs = {
        "max_new_tokens": int(attempt["max_new_tokens"]),
        "do_sample": bool(attempt["do_sample"]),
        "repetition_penalty": float(attempt.get("repetition_penalty", 1.05)),
        "no_repeat_ngram_size": int(attempt.get("no_repeat_ngram_size", 0)),
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if kwargs["do_sample"]:
        kwargs["temperature"] = float(attempt["temperature"])
        kwargs["top_p"] = float(attempt["top_p"])
    output = model.generate(**encoded, **kwargs)
    completion_ids = output[:, encoded["input_ids"].shape[1]:].cpu().tolist()
    results = []
    stop_ids = {
        int(token) for token in (
            tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>"))
        if token is not None and int(token) >= 0
    }
    for ids in completion_ids:
        for index, token in enumerate(ids):
            if int(token) in stop_ids:
                ids = ids[:index]
                break
        text = tokenizer.decode(ids, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False).strip()
        results.append({"text": text, "token_ids": tokenizer.encode(
            text, add_special_tokens=False), "tokens": len(ids)})
    del encoded, output, completion_ids
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    if transformers.__version__ != "4.49.0":
        raise RuntimeError(f"expected Transformers 4.49.0, got {transformers.__version__}")

    data_cfg = cfg["dataset"]["ifeval"]
    train_file = path(data_cfg["train_file"])
    data_manifest_file = path(data_cfg["manifest_file"])
    data_manifest = json.loads(data_manifest_file.read_text(encoding="utf-8"))
    if file_sha256(train_file) != data_manifest["files"]["train"]["sha256"]:
        raise RuntimeError("RLVR-IFeval train checksum mismatch")
    rows = read_jsonl(train_file)
    verifier_file = path(data_cfg["verifier_file"])
    if file_sha256(verifier_file) != data_manifest["verifier_sha256"]:
        raise RuntimeError("IFEval verifier checksum mismatch")
    verifier_map = load_verifiers(verifier_file)
    missing = sorted({row["verifier"] for row in rows} - set(verifier_map))
    if missing:
        raise RuntimeError(f"missing verifiers: {missing}")

    teacher_cfg = cfg["teacher"]
    teacher_path = path(teacher_cfg["path"])
    weight_file = teacher_path / teacher_cfg["weight_file"]
    if not weight_file.is_file() or file_sha256(weight_file) != teacher_cfg["weight_sha256"]:
        raise RuntimeError("teacher weight missing or checksum mismatch")
    output_dir = path(cfg["teacher_pool"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_file = output_dir / "teacher_candidates.jsonl"
    state_file = output_dir / "scan_state.json"
    manifest_file = output_dir / "manifest.json"

    # Similar prompt/constraint shapes share a batch, avoiding padding waste.
    order = sorted(
        range(len(rows)),
        key=lambda index: (rows[index]["constraint_type"], rows[index]["input_tokens"],
                           rows[index]["source_index"]),
    )
    signature_payload = {
        "schema_version": 1,
        "dataset_sha256": data_manifest["files"]["train"]["sha256"],
        "verifier_sha256": data_manifest["verifier_sha256"],
        "teacher_revision": teacher_cfg["revision"],
        "teacher_weight_sha256": teacher_cfg["weight_sha256"],
        "attempts": cfg["teacher_pool"]["attempts"],
        "max_input_tokens": data_cfg["max_input_tokens"],
        "order_sha256": hashlib.sha256(json.dumps(order).encode()).hexdigest(),
    }
    signature = hashlib.sha256(json.dumps(
        signature_payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    existing = read_jsonl(rows_file)
    if state_file.is_file():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        if state.get("scan_signature") != signature:
            raise RuntimeError("existing teacher pool belongs to a different configuration")
    elif existing:
        raise RuntimeError("teacher rows exist without scan state")
    else:
        write_json(state_file, {"scan_signature": signature,
                                "signature_payload": signature_payload})
    existing_indices = {int(row["source_index"]) for row in existing}
    remaining = [index for index in order if rows[index]["source_index"] not in existing_indices]
    if args.limit:
        remaining = remaining[:args.limit]
    ready = {"status": "ready", "train_rows": len(rows), "already_scanned": len(existing),
             "remaining_this_run": len(remaining), "batch_size": cfg["teacher_pool"]["batch_size"]}
    if args.dry_run:
        print(json.dumps(ready, ensure_ascii=False, indent=2))
        return

    qualified_total = sum(bool(row.get("qualified")) for row in existing)
    if remaining:
        tokenizer = AutoTokenizer.from_pretrained(
            str(teacher_path), trust_remote_code=True, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            str(teacher_path), trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda").eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        maximum_batch_size = int(cfg["teacher_pool"]["batch_size"])
        batch_size = maximum_batch_size
        began = time.perf_counter()
        processed = 0
        cursor = 0
        with rows_file.open("a", encoding="utf-8", newline="\n") as handle:
            while cursor < len(remaining):
                prompt_tokens = int(rows[remaining[cursor]]["input_tokens"])
                length_cap = (maximum_batch_size if prompt_tokens <= 512 else
                              12 if prompt_tokens <= 1024 else
                              8 if prompt_tokens <= 1536 else 4)
                batch_size = min(batch_size, length_cap)
                indices = remaining[cursor:cursor + batch_size]
                batch_rows = [rows[index] for index in indices]
                accepted: list[dict | None] = [None] * len(batch_rows)
                attempt_records: list[list[dict]] = [[] for _ in batch_rows]
                pending = list(range(len(batch_rows)))
                try:
                    for attempt_no, attempt in enumerate(cfg["teacher_pool"]["attempts"], start=1):
                        if not pending:
                            break
                        generated = generate_batch(
                            model, tokenizer, [batch_rows[i] for i in pending], attempt,
                            int(data_cfg["max_input_tokens"]))
                        next_pending = []
                        for local_index, generated_row in zip(pending, generated):
                            text = generated_row["text"]
                            hit_cap = generated_row["tokens"] >= int(attempt["max_new_tokens"])
                            loop = is_loop(text)
                            passed = verify_response(batch_rows[local_index], text, verifier_map)
                            valid = bool(passed and text and not hit_cap and not loop)
                            attempt_records[local_index].append({
                                "attempt": attempt_no, "name": attempt["name"],
                                "verifier_passed": bool(passed), "accepted": valid,
                                "tokens": generated_row["tokens"], "hit_token_cap": hit_cap,
                                "loop": loop,
                            })
                            if valid:
                                accepted[local_index] = generated_row
                            else:
                                next_pending.append(local_index)
                        pending = next_pending
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache(); gc.collect()
                    if batch_size == 1:
                        raise
                    batch_size = max(1, batch_size // 2)
                    print(f"teacher pool OOM; reducing batch_size to {batch_size}", flush=True)
                    continue

                output_rows = []
                for row, chosen, attempts in zip(batch_rows, accepted, attempt_records):
                    qualified = chosen is not None
                    output_rows.append({
                        "source_index": row["source_index"], "key": row["key"],
                        "constraint_type": row["constraint_type"], "verifier": row["verifier"],
                        "verifier_args": row["verifier_args"], "prompt": row["prompt"],
                        "qualified": qualified, "accepted_text": chosen["text"] if chosen else None,
                        "accepted_token_ids": chosen["token_ids"] if chosen else None,
                        "attempts": attempts,
                    })
                for row in output_rows:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush(); os.fsync(handle.fileno())
                cursor += len(indices); processed += len(indices)
                batch_size = maximum_batch_size
                qualified_total += sum(bool(row["qualified"]) for row in output_rows)
                total_done = len(existing) + processed
                elapsed = time.perf_counter() - began
                eta = elapsed / max(processed, 1) * (len(rows) - total_done)
                print(f"IFEval teacher pool {total_done}/{len(rows)} qualified={qualified_total} "
                      f"pass_rate={qualified_total/total_done:.4f} batch={batch_size} "
                      f"peak_gpu_gb={torch.cuda.max_memory_allocated()/2**30:.2f} "
                      f"eta_hours={max(0.0, eta)/3600:.2f}", flush=True)
        del model
        torch.cuda.empty_cache(); gc.collect()

    completed_rows = read_jsonl(rows_file)
    if len(completed_rows) != len(rows):
        print(json.dumps({"status": "partial_teacher_pool_saved", "scanned": len(completed_rows),
                          "expected": len(rows)}, ensure_ascii=False, indent=2))
        return
    qualified = [row for row in completed_rows if row["qualified"]]
    manifest = {
        "schema_version": 1, "completed": True, "scan_signature": signature,
        "dataset": data_manifest["dataset"], "dataset_revision": data_manifest["dataset_revision"],
        "train_rows": len(rows), "total_scanned": len(completed_rows),
        "qualified": len(qualified), "failed_discarded": len(completed_rows) - len(qualified),
        "pass_rate": len(qualified) / len(completed_rows),
        "teacher_model": teacher_cfg["model"], "teacher_revision": teacher_cfg["revision"],
        "teacher_weight_sha256": teacher_cfg["weight_sha256"],
        "generation_backend": "Transformers 4.49.0 BF16 batched generation",
        "qualification": "AllenAI official function passes; nonempty; no loop; no token-cap truncation",
        "answer_leakage": False, "rows_file": str(rows_file),
        "rows_sha256": file_sha256(rows_file), "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_json(manifest_file, manifest)
    print(json.dumps({"status": "teacher_pool_completed", **manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
