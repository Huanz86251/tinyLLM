#!/usr/bin/env python3
"""Deterministic structural audit for a packed tinyLLM VLM dataset."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re

from datasets import load_from_disk
from transformers import AutoTokenizer


ALLOWED_CONTROL_TAGS = {"<|im_start|>", "<|im_end|>"}
CONTROL_TAG_RE = re.compile(r"<\|[^<>\r\n]+\|>")
SOURCE_IMAGE_MARKER_RE = re.compile(
    r"</?image>|<image[_-]?\d*>|\[image(?:[_ -]?\d+)?\]|"
    r"<\/?img[_-]?\d+>|<img[_-]?\d+>|IMAGE_TOKEN|DEFAULT_IMAGE_TOKEN",
    flags=re.IGNORECASE,
)


def select_rows(dataset, count: int, seed: int) -> list[int]:
    by_source: dict[str, list[int]] = defaultdict(list)
    for i, (source, split) in enumerate(zip(dataset["dataset"], dataset["split"])):
        if str(split).lower() == "train":
            by_source[str(source)].append(i)
    rng = random.Random(seed)
    selected: list[int] = []
    for source in sorted(by_source):
        selected.extend(rng.sample(by_source[source], min(2, len(by_source[source]))))
    remaining = [
        i for source in sorted(by_source) for i in by_source[source]
        if i not in set(selected)
    ]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, count - len(selected))])
    rng.shuffle(selected)
    return selected[:count]


def role_spans(ids: list[int], tokenizer) -> tuple[list[tuple[str, int, int]], list[str]]:
    start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    spans: list[tuple[str, int, int]] = []
    errors: list[str] = []
    i = 0
    while i < len(ids):
        if ids[i] != start_id:
            try:
                next_start = ids.index(start_id, i + 1)
            except ValueError:
                next_start = len(ids)
            gap = tokenizer.decode(ids[i:next_start], skip_special_tokens=False)
            if gap.strip():
                errors.append(f"non_whitespace_outside_chatml@{i}")
            i = next_start
            continue
        try:
            end = ids.index(end_id, i + 1)
        except ValueError:
            errors.append(f"missing_im_end_after@{i}")
            break
        body = tokenizer.decode(ids[i + 1:end], skip_special_tokens=False)
        role = body.split("\n", 1)[0].strip()
        spans.append((role, i, end))
        i = end + 1
    return spans, errors


def audit_one(row, tokenizer, image_root: Path) -> dict:
    ids = [int(x) for x in row["input_ids"]]
    mask = [int(x) for x in row["target_mask"]]
    text = tokenizer.decode(ids, skip_special_tokens=False)
    spans, errors = role_spans(ids, tokenizer)

    if len(ids) != len(mask):
        errors.append("target_mask_length_mismatch")
    if text.count("<img>") != 1:
        errors.append(f"canonical_img_count={text.count('<img>')}")
    if not spans or spans[0][0] != "user":
        errors.append("first_role_is_not_user")
    roles = [role for role, _, _ in spans]
    if any(role not in {"user", "assistant", "system"} for role in roles):
        errors.append("unknown_chatml_role")
    dialogue_roles = [r for r in roles if r != "system"]
    if any(a == b for a, b in zip(dialogue_roles, dialogue_roles[1:])):
        errors.append("non_alternating_roles")

    controls = set(CONTROL_TAG_RE.findall(text))
    unexpected_controls = sorted(controls - ALLOWED_CONTROL_TAGS)
    if unexpected_controls:
        errors.append("unexpected_control_tags")
    residue = sorted(set(SOURCE_IMAGE_MARKER_RE.findall(text)))
    if residue:
        errors.append("source_image_marker_residue")
    if "\ufffd" in text:
        errors.append("unicode_replacement_character")

    # Masked positions must live in assistant content, never in the role header.
    # We allow the closing im_end token because this pack supervises it.
    assistant_positions: set[int] = set()
    for role, start, end in spans:
        if role == "assistant":
            body_ids = ids[start + 1:end]
            prefix_len = None
            for prefix_text in ("assistant\n", " assistant\n"):
                prefix = tokenizer.encode(prefix_text, add_special_tokens=False)
                if body_ids[:len(prefix)] == prefix:
                    prefix_len = len(prefix)
                    break
            if prefix_len is None:
                errors.append("unparsed_assistant_role_header")
                prefix_len = len(body_ids)
            content_start = start + 1 + prefix_len
            if any(mask[start + 1:content_start]):
                errors.append("target_mask_on_assistant_role_header")
            assistant_positions.update(range(content_start, end + 1))
    leaked = [i for i, value in enumerate(mask) if value and i not in assistant_positions]
    if leaked:
        errors.append(f"target_mask_outside_assistant={len(leaked)}")
    if sum(mask) == 0:
        errors.append("empty_target_mask")

    images = row["image_file"]
    if not isinstance(images, list) or len(images) != 5:
        errors.append(f"image_count={len(images) if isinstance(images, list) else 'not_list'}")
        images = images if isinstance(images, list) else []
    missing_images = [name for name in images if not (image_root / name).is_file()]
    if missing_images:
        errors.append(f"missing_images={len(missing_images)}")
    try:
        meta = json.loads(row["vision_meta"])
        views = meta.get("views", [])
        if len(views) != 5:
            errors.append(f"vision_meta_views={len(views)}")
    except Exception:
        errors.append("invalid_vision_meta_json")

    assistant_texts = []
    for role, start, end in spans:
        if role == "assistant":
            assistant_texts.append(
                tokenizer.decode(ids[start + 1:end], skip_special_tokens=False)
            )
    assistant_joined = "\n".join(assistant_texts)
    if "<img>" in assistant_joined or SOURCE_IMAGE_MARKER_RE.search(assistant_joined):
        errors.append("image_marker_in_assistant")

    return {
        "dataset": str(row["dataset"]),
        "index": int(row["index"]),
        "input_tokens": len(ids),
        "target_tokens": sum(mask),
        "roles": roles,
        "image_files": images,
        "unexpected_control_tags": unexpected_controls,
        "marker_residue": residue,
        "errors": errors,
        "preview": text.replace("\r", " ").replace("\n", " ")[:360],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = load_from_disk(str(args.dataset))
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), trust_remote_code=True, local_files_only=True
    )
    chosen = select_rows(dataset, args.count, args.seed)
    audited = [audit_one(dataset[i], tokenizer, args.image_root) for i in chosen]
    error_counts = Counter(error for row in audited for error in row["errors"])
    source_counts = Counter(row["dataset"] for row in audited)
    result = {
        "schema_version": 1,
        "dataset": str(args.dataset),
        "tokenizer": str(args.tokenizer),
        "image_root": str(args.image_root),
        "seed": args.seed,
        "sample_count": len(audited),
        "source_counts": dict(sorted(source_counts.items())),
        "rows_with_errors": sum(bool(row["errors"]) for row in audited),
        "error_counts": dict(sorted(error_counts.items())),
        "samples": audited,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: result[k] for k in (
        "sample_count", "source_counts", "rows_with_errors", "error_counts"
    )}, ensure_ascii=False, indent=2))
    for i, row in enumerate(audited, 1):
        print(
            f"[{i:02d}] {row['dataset']} index={row['index']} "
            f"tokens={row['input_tokens']}/{row['target_tokens']} "
            f"errors={row['errors']} preview={row['preview']}"
        )


if __name__ == "__main__":
    main()
