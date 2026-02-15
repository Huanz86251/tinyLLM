"""Prepare a deterministic, leakage-checked RLVR-IFeval split for tinyLLM OPD."""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
COMPAT_RUNTIME = ROOT / "runtime" / "minicpm_transformers_449"
if not (COMPAT_RUNTIME / "transformers" / "__init__.py").is_file():
    raise FileNotFoundError(COMPAT_RUNTIME)
sys.path.insert(0, str(COMPAT_RUNTIME))

from transformers import AutoTokenizer
DATASET_NAME = "allenai/RLVR-IFeval"
DATASET_REVISION = "47c03c73621c4aab2b824b7818681117d662770e"
GOOGLE_IFEVAL_NAME = "google/IFEval"
SPLIT_SEED = 20260904
DEV_PER_CONSTRAINT = 20


def canonical_prompt(text: str) -> str:
    return " ".join(text.strip().split()).casefold()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def load_verifier_map(file: Path) -> dict:
    spec = importlib.util.spec_from_file_location("tinyllm_ifeval_functions", file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import verifier file: {file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.IF_FUNCTIONS_MAP


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "datasets" / "opd_ifeval"))
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    args = parser.parse_args()

    verifier_file = ROOT / "third_party" / "open_instruct_ifeval" / "if_functions.py"
    if not verifier_file.is_file():
        raise FileNotFoundError(verifier_file)
    verifier_map = load_verifier_map(verifier_file)
    tokenizer = AutoTokenizer.from_pretrained(
        str(ROOT / "models" / "teacher" / "minicpm3_4b_recovered"),
        trust_remote_code=True, local_files_only=True,
    )

    source = [dict(row) for row in load_dataset(
        DATASET_NAME, revision=DATASET_REVISION, split="train")]
    official = [dict(row) for row in load_dataset(GOOGLE_IFEVAL_NAME, split="train")]
    official_prompts = {canonical_prompt(row["prompt"]) for row in official}

    retained: list[dict] = []
    dropped_too_long: list[dict] = []
    exact_overlap: list[str] = []
    for source_index, row in enumerate(source):
        messages = row["messages"]
        if len(messages) != 1 or messages[0]["role"] != "user":
            raise RuntimeError(f"unexpected messages schema at source index {source_index}")
        prompt = messages[0]["content"].strip()
        if canonical_prompt(prompt) in official_prompts:
            exact_overlap.append(row.get("key", str(source_index)))
            continue
        ground_truth = json.loads(row["ground_truth"])
        function_name = ground_truth.pop("func_name")
        verifier_args = {key: value for key, value in ground_truth.items() if value is not None}
        if function_name not in verifier_map:
            raise RuntimeError(f"missing verifier {function_name!r} at source index {source_index}")
        input_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        prepared = {
            "source_index": source_index,
            "key": row.get("key", f"rlvr_ifeval_{source_index}"),
            "prompt": prompt,
            "constraint_type": row["constraint_type"],
            "constraint": row["constraint"],
            "verifier": function_name,
            "verifier_args": verifier_args,
            "input_tokens": input_tokens,
        }
        if input_tokens > args.max_input_tokens:
            dropped_too_long.append(prepared)
        else:
            retained.append(prepared)

    if exact_overlap:
        raise RuntimeError(f"normalized Google IFEval leakage detected: {exact_overlap[:5]}")

    by_constraint: dict[str, list[dict]] = collections.defaultdict(list)
    for row in retained:
        by_constraint[row["constraint_type"]].append(row)
    dev_keys: set[str] = set()
    for constraint_type, rows in sorted(by_constraint.items()):
        ordered = list(rows)
        random.Random(f"{SPLIT_SEED}:{constraint_type}").shuffle(ordered)
        if len(ordered) <= DEV_PER_CONSTRAINT:
            raise RuntimeError(f"not enough rows for {constraint_type}")
        dev_keys.update(row[list(row)[1]] for row in ordered[:DEV_PER_CONSTRAINT])

    train_rows = [row for row in retained if row["key"] not in dev_keys]
    dev_rows = [row for row in retained if row["key"] in dev_keys]
    random.Random(SPLIT_SEED).shuffle(train_rows)
    random.Random(SPLIT_SEED + 1).shuffle(dev_rows)

    output_dir = Path(args.output_dir)
    train_file = output_dir / "train.jsonl"
    dev_file = output_dir / "dev.jsonl"
    dropped_file = output_dir / "dropped_too_long.jsonl"
    write_jsonl(train_file, train_rows)
    write_jsonl(dev_file, dev_rows)
    write_jsonl(dropped_file, dropped_too_long)

    def distribution(rows: list[dict]) -> dict:
        return dict(sorted(collections.Counter(row["constraint_type"] for row in rows).items()))

    manifest = {
        "schema_version": 1,
        "dataset": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "license": "ODC-BY-1.0; research and educational use under AI2 Responsible Use Guidelines",
        "google_ifeval_role": "evaluation only",
        "google_ifeval_rows": len(official),
        "normalized_exact_overlap": 0,
        "split_seed": SPLIT_SEED,
        "dev_per_constraint": DEV_PER_CONSTRAINT,
        "max_input_tokens": args.max_input_tokens,
        "source_rows": len(source),
        "retained_rows": len(retained),
        "dropped_too_long": len(dropped_too_long),
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "constraint_types": len(by_constraint),
        "train_distribution": distribution(train_rows),
        "dev_distribution": distribution(dev_rows),
        "verifier_file": str(verifier_file),
        "verifier_sha256": sha256_file(verifier_file),
        "verifier_functions": sorted(verifier_map),
        "files": {
            "train": {"path": str(train_file), "sha256": sha256_file(train_file)},
            "dev": {"path": str(dev_file), "sha256": sha256_file(dev_file)},
            "dropped": {"path": str(dropped_file), "sha256": sha256_file(dropped_file)},
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
