#!/usr/bin/env python3
"""Build unique Chinese/English text-only replay for VLM continual SFT.

The resulting Hugging Face dataset deliberately uses the same Arrow schema as
``packed_vlm_v5``. Text rows carry ``image_file=[]`` and regular ChatML without
``<img>`` or synthetic chain-of-thought. The continual trainer therefore sends
them through a true text-only forward pass.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import heapq
import json
import os
from pathlib import Path
import re
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "vlm_continual_zh_en_v5.json"
FORBIDDEN = ("<img>", "<|im_start|>", "<|im_end|>", "<|thought_start|>", "<|thought_end|>")
SPACE_RE = re.compile(r"[ \t]+")


def project_path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate

SOURCES = {
    "zh": {
        "repository": "opencsg/Fineweb-Edu-Chinese-V2.3",
        "revision": "56d146373c19865268fdb3b6696403d0411b5583",
        "file": "sft/train_messages_no_sys.jsonl",
        "size": 339982894,
        "sha256": "a48f56eb6c8e072a3245329a897194139270fc99072c2372b505c0d82a5de35d",
        "license": "Fineweb-Edu-Chinese-V2.3 dataset license agreement",
    },
    "en": {
        "repository": "allenai/tulu-3-sft-personas-instruction-following",
        "revision": "fe0c7d350c9b4542b8d829a6f1daa1c259f0ba0e",
        "file": "data/train-00000-of-00001.parquet",
        "size": 39171921,
        "sha256": "19a16c5f1649d367f69899b3cfadbbeb5ffef91f24e20c6617588bdd87cd3e60",
        "license": "ODC-BY-1.0",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(source: dict, target: Path) -> None:
    import requests

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == source["size"]:
        if sha256(target) == source["sha256"]:
            print(f"[DOWNLOAD] verified existing {target}", flush=True)
            return
        target.unlink()

    part = target.with_suffix(target.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    bases = [
        os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/"),
        "https://hf-mirror.com",
        "https://huggingface.co",
    ]
    errors = []
    for base in dict.fromkeys(bases):
        url = (
            f"{base}/datasets/{source['repository']}/resolve/"
            f"{source['revision']}/{source['file']}"
        )
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                if offset and not append:
                    offset = 0
                mode = "ab" if append else "wb"
                with part.open(mode) as handle:
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if part.stat().st_size != source["size"]:
                raise RuntimeError(
                    f"size mismatch: {part.stat().st_size} != {source['size']}"
                )
            actual = sha256(part)
            if actual != source["sha256"]:
                raise RuntimeError(f"sha256 mismatch: {actual}")
            part.replace(target)
            print(f"[DOWNLOAD] completed {target} ({target.stat().st_size} bytes)", flush=True)
            return
        except Exception as exc:
            errors.append(f"{base}: {exc!r}")
            offset = part.stat().st_size if part.exists() else 0
    raise RuntimeError("download failed: " + " | ".join(errors))


def normalize_messages(raw) -> list[dict] | None:
    if not isinstance(raw, list):
        return None
    out = []
    for message in raw:
        if not isinstance(message, dict):
            return None
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).replace("\r\n", "\n").replace("\r", "\n").strip()
        if role == "system":
            continue
        if role not in {"user", "assistant"} or not content:
            return None
        if "\ufffd" in content or "\x00" in content or any(tag in content for tag in FORBIDDEN):
            return None
        out.append({"role": role, "content": content})
    while out and out[0]["role"] != "user":
        out.pop(0)
    while out and out[-1]["role"] != "assistant":
        out.pop()
    if len(out) < 2 or len(out) % 2:
        return None
    if any(row["role"] != ("user" if i % 2 == 0 else "assistant") for i, row in enumerate(out)):
        return None
    return out


def prompt_key(messages: list[dict]) -> str:
    prompt = "\n".join(m["content"] for m in messages if m["role"] == "user")
    return SPACE_RE.sub(" ", prompt).strip().casefold()


def stable_score(seed: int, lang: str, key: str) -> int:
    raw = f"{seed}:{lang}:{key}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=16).digest(), "big")


def iter_zh(path: Path) -> Iterable[tuple[str, list[dict]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except Exception:
                continue
            messages = normalize_messages(row.get("messages"))
            if messages:
                yield f"line-{line_no}", messages


def iter_en(path: Path) -> Iterable[tuple[str, list[dict]]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["id", "messages"], batch_size=2048):
        values = batch.to_pylist()
        for row in values:
            messages = normalize_messages(row.get("messages"))
            if messages:
                yield str(row.get("id", "")), messages


def preselect(
    rows: Iterable[tuple[str, list[dict]]], *, lang: str, seed: int, limit: int
) -> tuple[list[tuple[int, str, list[dict]]], Counter]:
    """Keep the deterministic best candidates without materializing a source."""
    heap: list[tuple[int, str, list[dict]]] = []
    seen = set()
    counts = Counter()
    for source_id, messages in rows:
        counts["structurally_valid"] += 1
        key = prompt_key(messages)
        if not key or key in seen:
            counts["duplicate_or_empty"] += 1
            continue
        seen.add(key)
        score = stable_score(seed, lang, key)
        entry = (-score, source_id, messages)
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    selected = [(-neg, source_id, messages) for neg, source_id, messages in heap]
    selected.sort(key=lambda item: item[0])
    counts["preselected"] = len(selected)
    return selected, counts


def render_chat(messages: list[dict]) -> str:
    return "".join(
        f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        for m in messages
    )


def find_subsequence(values: list[int], needle: list[int], start: int) -> int:
    stop = len(values) - len(needle) + 1
    for index in range(start, max(start, stop)):
        if values[index:index + len(needle)] == needle:
            return index
    return -1


def encode_row(tokenizer, messages: list[dict], max_tokens: int, min_answer_tokens: int):
    text = render_chat(messages)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids or len(ids) > max_tokens:
        return None, "too_long_or_empty"

    header = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not header or end_id is None or end_id == tokenizer.unk_token_id:
        raise RuntimeError("tokenizer does not expose the required ChatML boundary tokens")
    target = [0] * len(ids)
    cursor = 0
    assistant_spans = 0
    assistant_tokens = 0
    while True:
        start = find_subsequence(ids, header, cursor)
        if start < 0:
            break
        content_start = start + len(header)
        try:
            end = ids.index(end_id, content_start)
        except ValueError:
            return None, "missing_assistant_end"
        for position in range(content_start, end + 1):
            target[position] = 1
        assistant_spans += 1
        assistant_tokens += end - content_start
        cursor = end + 1
    if assistant_spans == 0 or assistant_tokens < min_answer_tokens:
        return None, "assistant_too_short"
    return (ids, target), "ok"


def build_language_rows(
    *, tokenizer, candidates, lang: str, train_count: int, eval_count: int,
    max_tokens: int, min_answer_tokens: int, index_start: int,
) -> tuple[list[dict], Counter]:
    source = SOURCES[lang]
    need = train_count + eval_count
    retained = []
    counts = Counter()
    for _, source_id, messages in candidates:
        encoded, reason = encode_row(tokenizer, messages, max_tokens, min_answer_tokens)
        counts[reason] += 1
        if encoded is None:
            continue
        ids, target = encoded
        split = "train" if len(retained) < train_count else "eval"
        retained.append({
            "index": index_start + len(retained),
            "image_file": [],
            "dataset": "text_replay_fineweb_edu_zh_v23" if lang == "zh" else "text_replay_tulu3_if_en",
            "lang": lang,
            "has_cot": False,
            "split": split,
            "vision_meta": "{}",
            "input_ids": ids,
            "input_len": len(ids),
            "target_mask": target,
            "_source_id": source_id,
        })
        if len(retained) == need:
            break
    if len(retained) != need:
        raise RuntimeError(
            f"{lang}: needed {need} rows but only retained {len(retained)}; filters={dict(counts)}"
        )
    return retained, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--min-answer-tokens", type=int, default=8)
    parser.add_argument("--audit-rows", type=int, default=30)
    args = parser.parse_args()
    if args.download_only and args.build_only:
        raise ValueError("choose at most one of --download-only and --build-only")

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    raw_dir = args.raw_dir or project_path(cfg["paths"]["raw_root"]) / "text_replay_v1"
    output = args.output or project_path(cfg["paths"]["text_replay"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_paths = {
        lang: raw_dir / Path(source["file"]).name
        for lang, source in SOURCES.items()
    }
    if not args.build_only:
        for lang, source in SOURCES.items():
            download(source, raw_paths[lang])
    if args.download_only:
        return

    for lang, source in SOURCES.items():
        path = raw_paths[lang]
        if not path.is_file() or path.stat().st_size != source["size"] or sha256(path) != source["sha256"]:
            raise RuntimeError(f"raw source is missing or failed checksum: {path}")

    compat = ROOT / "runtime" / "minicpm_transformers_449"
    if compat.is_dir():
        sys.path.insert(0, str(compat))
    from datasets import Dataset, Features, Sequence, Value, load_from_disk
    from transformers import AutoTokenizer

    seed = int(cfg["seed"])
    packed = load_from_disk(str(project_path(cfg["paths"]["packed_dataset"])))
    if "split" not in packed.column_names:
        raise RuntimeError("packed VLM dataset has no explicit split column")
    vision_rows = sum(str(value).lower() == "train" for value in packed["split"])
    ratio = float(cfg["continuation"]["text_replay_ratio"])
    total_train = int(round(vision_rows * ratio / (1.0 - ratio)))
    zh_fraction = float(cfg["continuation"]["text_replay_zh_fraction"])
    zh_train = int(round(total_train * zh_fraction))
    en_train = total_train - zh_train
    eval_total = int(cfg["continuation"]["text_replay_eval_max_rows"])
    zh_eval = int(round(eval_total * zh_fraction))
    en_eval = eval_total - zh_eval

    tokenizer = AutoTokenizer.from_pretrained(
        str(project_path(cfg["paths"]["tokenizer"])), trust_remote_code=True, local_files_only=True
    )
    targets = {"zh": (zh_train, zh_eval), "en": (en_train, en_eval)}
    all_rows = []
    reports = {}
    for language, iterator in (
        ("zh", iter_zh(raw_paths["zh"])),
        ("en", iter_en(raw_paths["en"])),
    ):
        train_count, eval_count = targets[language]
        # Preselect extra candidates before the more expensive token filter.
        candidate_limit = min(
            200000 if language == "zh" else 29980,
            max(train_count + eval_count + 4096, int((train_count + eval_count) * 1.7)),
        )
        candidates, structural = preselect(
            iterator, lang=language, seed=seed, limit=candidate_limit
        )
        rows, token_filters = build_language_rows(
            tokenizer=tokenizer,
            candidates=candidates,
            lang=language,
            train_count=train_count,
            eval_count=eval_count,
            max_tokens=args.max_tokens,
            min_answer_tokens=args.min_answer_tokens,
            index_start=10_000_000 if language == "zh" else 20_000_000,
        )
        all_rows.extend(rows)
        reports[language] = {
            "source": SOURCES[language],
            "train_rows": train_count,
            "eval_rows": eval_count,
            "structural_counts": dict(structural),
            "token_filter_counts": dict(token_filters),
            "min_tokens": min(r["input_len"] for r in rows),
            "max_tokens": max(r["input_len"] for r in rows),
            "mean_tokens": sum(r["input_len"] for r in rows) / len(rows),
        }

    # Preserve source ids only in the manifest audit, not in the VLM Arrow
    # schema; exact schema equality is enforced before concatenation.
    all_rows.sort(key=lambda r: (r["split"] == "eval", r["lang"], r["index"]))
    audit = []
    stride = max(1, len(all_rows) // max(1, args.audit_rows))
    for row in all_rows[::stride][:args.audit_rows]:
        audit.append({
            "source_id": row["_source_id"], "dataset": row["dataset"],
            "lang": row["lang"], "split": row["split"],
            "tokens": row["input_len"], "target_tokens": sum(row["target_mask"]),
            "preview": tokenizer.decode(row["input_ids"], skip_special_tokens=False)[:400],
        })
    for row in all_rows:
        row.pop("_source_id", None)

    features = Features({
        "index": Value("int64"),
        "image_file": Sequence(Value("string")),
        "dataset": Value("string"),
        "lang": Value("string"),
        "has_cot": Value("bool"),
        "split": Value("string"),
        "vision_meta": Value("string"),
        "input_ids": Sequence(Value("int32")),
        "input_len": Value("int64"),
        "target_mask": Sequence(Value("int64")),
    })
    dataset = Dataset.from_list(all_rows, features=features)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output))

    manifest = {
        "schema_version": 1,
        "purpose": "text-only language replay for VLM continual SFT",
        "output": str(output),
        "seed": seed,
        "vision_train_rows_estimate": vision_rows,
        "requested_text_ratio": ratio,
        "actual_train_rows": sum(r["split"] == "train" for r in all_rows),
        "actual_eval_rows": sum(r["split"] == "eval" for r in all_rows),
        "actual_zh_fraction": sum(r["lang"] == "zh" for r in all_rows) / len(all_rows),
        "max_tokens": args.max_tokens,
        "format": "regular ChatML; no image token; no synthetic thought boundary",
        "reports": reports,
        "audit": audit,
    }
    (output / "text_replay_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    raise SystemExit(
        "This Fineweb/Tulu preparation entrypoint was retired on 2026-09-05. "
        "Run tools/prepare_shared_general_replay.py for COIG-CQIA + SmolTalk2 everyday."
    )
