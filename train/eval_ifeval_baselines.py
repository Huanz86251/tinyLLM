"""Reproducible full IFEval baseline for tinyLLM-0.51B and MiniCPM3-4B.

The 541 google/IFEval prompts are evaluation-only.  Responses are generated
without a reasoning/system prompt and scored with Google's official strict and
loose programmatic verifiers.  Progress is written after every batch so the
job can resume after interruption.
"""
from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPAT_RUNTIME = ROOT / "runtime" / "minicpm_transformers_449"
GOOGLE_IFEVAL = ROOT / "third_party" / "google_ifeval"
for required in (COMPAT_RUNTIME / "transformers" / "__init__.py",
                 GOOGLE_IFEVAL / "instruction_following_eval" / "evaluation_lib.py"):
    if not required.is_file():
        raise FileNotFoundError(required)
sys.path.insert(0, str(COMPAT_RUNTIME))
sys.path.insert(0, str(GOOGLE_IFEVAL))
sys.path.insert(0, str(ROOT))

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from instruction_following_eval import evaluation_lib
from project_paths import path
from train import GRPO as g


def atomic_json(file: Path, value: dict) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_suffix(file.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(file)


def dataset_fingerprint(rows: list[dict]) -> str:
    payload = "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def render_prompts(tokenizer, prompts: list[str]) -> list[list[int]]:
    rendered = [g.render_prompt_with_tokenizer(tokenizer, prompt, None) for prompt in prompts]
    return [tokenizer.encode(text, add_special_tokens=False) for text in rendered]


@torch.inference_mode()
def generate_student(model, tokenizer, prompts: list[str], max_new_tokens: int) -> tuple[list[str], list[int]]:
    encoded = [torch.tensor(ids, dtype=torch.long) for ids in render_prompts(tokenizer, prompts)]
    batch = len(encoded)
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = eos_id
    width = max(item.numel() for item in encoded)
    input_ids = torch.full((batch, width), int(pad_id), dtype=torch.long, device="cuda")
    attention = torch.zeros_like(input_ids)
    for row, ids in enumerate(encoded):
        input_ids[row, -ids.numel():] = ids.to("cuda")
        attention[row, -ids.numel():] = 1

    output = model(input_ids=input_ids, attention_mask=attention, labels=None,
                   use_cache=True, past_states=None)
    past_states = output["past_states"]
    logits = output["logits"][:, -1, :]
    generated: list[list[int]] = [[] for _ in range(batch)]
    finished = [False] * batch
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    valid_stops = {int(x) for x in (eos_id, im_end_id) if x is not None and int(x) >= 0}

    for _ in range(max_new_tokens):
        next_ids = torch.argmax(logits.float(), dim=-1, keepdim=True)
        for row in range(batch):
            if finished[row]:
                next_ids[row, 0] = int(eos_id)
                continue
            token = int(next_ids[row, 0].item())
            if token in valid_stops:
                finished[row] = True
            else:
                generated[row].append(token)
        if all(finished):
            break
        one_attention = torch.ones_like(next_ids, dtype=torch.long, device="cuda")
        output = model(input_ids=next_ids, attention_mask=one_attention, labels=None,
                       use_cache=True, past_states=past_states)
        past_states = output["past_states"]
        logits = output["logits"][:, -1, :]

    texts = [tokenizer.decode(ids, skip_special_tokens=True,
                              clean_up_tokenization_spaces=False).strip()
             for ids in generated]
    return texts, [len(ids) for ids in generated]


@torch.inference_mode()
def generate_teacher(model, tokenizer, prompts: list[str], max_new_tokens: int) -> tuple[list[str], list[int]]:
    rendered = [g.render_prompt_with_tokenizer(tokenizer, prompt, None) for prompt in prompts]
    encoded = tokenizer(rendered, return_tensors="pt", padding=True,
                        add_special_tokens=False).to("cuda")
    input_width = encoded["input_ids"].shape[1]
    outputs = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    completion_ids = outputs[:, input_width:]
    texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)
    lengths = []
    stop_ids = {int(x) for x in (tokenizer.eos_token_id,
                                  tokenizer.convert_tokens_to_ids("<|im_end|>"))
                if x is not None and int(x) >= 0}
    for row in completion_ids.tolist():
        length = len(row)
        for index, token in enumerate(row):
            if int(token) in stop_ids:
                length = index
                break
        lengths.append(length)
    return [text.strip() for text in texts], lengths


def input_example(row: dict) -> evaluation_lib.InputExample:
    # Hugging Face parquet unionizes constraint kwargs and fills unrelated
    # fields with null. Google's JSONL contains only applicable keys.
    kwargs = [{key: value for key, value in item.items() if value is not None}
              for item in row["kwargs"]]
    return evaluation_lib.InputExample(
        key=int(row["key"]), prompt=row["prompt"],
        instruction_id_list=list(row["instruction_id_list"]),
        kwargs=kwargs,
    )


def score(rows: list[dict], dataset: list[dict]) -> dict:
    responses = {row["prompt"]: row["response"] for row in rows}
    result = {}
    for mode, evaluator in (("strict", evaluation_lib.test_instruction_following_strict),
                            ("loose", evaluation_lib.test_instruction_following_loose)):
        outputs = [evaluator(input_example(item), responses) for item in dataset[:len(rows)]]
        prompt_correct = sum(bool(item.follow_all_instructions) for item in outputs)
        instruction_total = sum(len(item.follow_instruction_list) for item in outputs)
        instruction_correct = sum(sum(item.follow_instruction_list) for item in outputs)
        by_type = collections.defaultdict(lambda: [0, 0])
        for item in outputs:
            for instruction_id, passed in zip(item.instruction_id_list,
                                               item.follow_instruction_list):
                by_type[instruction_id][1] += 1
                by_type[instruction_id][0] += int(bool(passed))
        result[mode] = {
            "prompt_correct": prompt_correct,
            "prompt_total": len(outputs),
            "prompt_accuracy": prompt_correct / len(outputs) if outputs else 0.0,
            "instruction_correct": instruction_correct,
            "instruction_total": instruction_total,
            "instruction_accuracy": instruction_correct / instruction_total if instruction_total else 0.0,
            "by_instruction": {
                name: {"correct": values[0], "total": values[1],
                       "accuracy": values[0] / values[1]}
                for name, values in sorted(by_type.items())
            },
        }
    return result


def evaluate_model(label: str, model, tokenizer, dataset: list[dict], state: dict,
                   state_path: Path, batch_size: int, max_new_tokens: int,
                   generator) -> None:
    rows = state["models"].setdefault(label, {}).setdefault("rows", [])
    if [int(item["index"]) for item in rows] != list(range(len(rows))):
        raise RuntimeError(f"{label} resume rows are not contiguous")
    model.eval()
    started = time.perf_counter()
    while len(rows) < len(dataset):
        begin = len(rows)
        end = min(len(dataset), begin + batch_size)
        prompts = [item["prompt"] for item in dataset[begin:end]]
        texts, lengths = generator(model, tokenizer, prompts, max_new_tokens)
        for index, item, response, tokens in zip(range(begin, end), dataset[begin:end], texts, lengths):
            rows.append({"index": index, "key": int(item["key"]),
                         "prompt": item["prompt"], "response": response,
                         "tokens": int(tokens), "hit_token_cap": tokens >= max_new_tokens})
        state["models"][label]["rows"] = rows
        state["models"][label]["partial_score"] = score(rows, dataset)
        state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        atomic_json(state_path, state)
        strict = state["models"][label]["partial_score"]["strict"]
        elapsed = time.perf_counter() - started
        rate = max((len(rows) - begin) / max(elapsed, 1e-6), 1e-6)
        print(f"{label} {len(rows)}/{len(dataset)} strict={strict['prompt_correct']}/"
              f"{strict['prompt_total']} tokens={sum(x['tokens'] for x in rows)}", flush=True)
    state["models"][label]["score"] = score(rows, dataset)
    state["models"][label]["average_tokens"] = sum(x["tokens"] for x in rows) / len(rows)
    state["models"][label]["token_cap_count"] = sum(x["hit_token_cap"] for x in rows)
    state["models"][label].pop("partial_score", None)
    atomic_json(state_path, state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--models", choices=("both", "student", "teacher"), default="both")
    args = parser.parse_args()

    random.seed(20260904)
    torch.manual_seed(20260904)
    torch.cuda.manual_seed_all(20260904)
    dataset = [dict(item) for item in load_dataset("google/IFEval", split="train")]
    if len(dataset) != 541:
        raise RuntimeError(f"expected 541 official IFEval prompts, found {len(dataset)}")
    if args.limit:
        dataset = dataset[:args.limit]

    output_dir = path("runs/evaluation/ifeval_baselines") / ("full" if not args.limit else f"first_{args.limit}")
    state_path = output_dir / "report.json"
    fingerprint = dataset_fingerprint(dataset)
    state = {
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmark": "google/IFEval official 541 evaluation prompts",
        "split_name_in_huggingface": "train (evaluation-only by benchmark semantics)",
        "dataset_rows": len(dataset),
        "dataset_sha256": fingerprint,
        "decode": {"method": "greedy", "system_prompt": None,
                   "max_new_tokens": args.max_new_tokens,
                   "batch_size": args.batch_size},
        "verifier": "Google instruction_following_eval strict and loose",
        "models": {},
    }
    if state_path.is_file():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous.get("dataset_sha256") == fingerprint and previous.get("decode") == state["decode"]:
            state = previous
            state["status"] = "running"
    atomic_json(state_path, state)

    teacher_dir = path("models/teacher/minicpm3_4b_recovered")
    tokenizer = AutoTokenizer.from_pretrained(str(teacher_dir), trust_remote_code=True,
                                              local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if args.models in ("both", "student") and "score" not in state["models"].get("student_sft_0_51b", {}):
        student = g.load_tinyllm_from_ckpt(str(path("models/text/sft_base_50000")),
                                          tokenizer, strict=True)
        evaluate_model("student_sft_0_51b", student, tokenizer, dataset, state,
                       state_path, args.batch_size, args.max_new_tokens, generate_student)
        del student
        gc.collect()
        torch.cuda.empty_cache()

    if args.models in ("both", "teacher") and "score" not in state["models"].get("teacher_minicpm3_4b", {}):
        teacher = AutoModelForCausalLM.from_pretrained(
            str(teacher_dir), trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda").eval()
        evaluate_model("teacher_minicpm3_4b", teacher, tokenizer, dataset, state,
                       state_path, args.batch_size, args.max_new_tokens, generate_teacher)
        del teacher
        gc.collect()
        torch.cuda.empty_cache()

    requested = (["student_sft_0_51b", "teacher_minicpm3_4b"] if args.models == "both"
                 else ["student_sft_0_51b" if args.models == "student" else "teacher_minicpm3_4b"])
    if all("score" in state["models"].get(label, {}) for label in requested):
        state["status"] = "completed"
        state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    atomic_json(state_path, state)
    compact = {label: {"score": value.get("score"),
                       "average_tokens": value.get("average_tokens"),
                       "token_cap_count": value.get("token_cap_count")}
               for label, value in state["models"].items()}
    print(json.dumps({"status": state["status"], "models": compact},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
