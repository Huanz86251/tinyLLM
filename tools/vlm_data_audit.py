#!/usr/bin/env python3
"""Audit and preselect Chinese VLM sources without decoding images."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from statistics import median


SPECULATIVE_RE = re.compile(r"可能|也许|似乎|或许|大概|推测|看起来像")
OCR_RE = re.compile(r"表格|图表|文字|字幕|标志|品牌|广告|海报|菜单|屏幕|网页|界面|文档|二维码|公式|坐标轴|矩阵")
LOOP_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")
BAD_GENERATION_RE = re.compile(r"根据你的描述|设计了一个新的问题|作为(?:一个|一名)?AI|无法看到图片")
MMI_ALLOW_DOMAINS = {
    "attribute_recognition",
    "commonsense_reasoning",
    "image_emotion",
    "image_scene",
    "object_localization",
    "object_relation",
    "social_relation",
    "spatial_relationship",
    "species_recognition",
}


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * fraction))]


def normalize_conversations(value) -> list[dict[str, str]]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        return []
    result = []
    for turn in value:
        if hasattr(turn, "as_py"):
            turn = turn.as_py()
        if not isinstance(turn, dict):
            continue
        raw_role = str(turn.get("role") or turn.get("from") or "").lower()
        role = "assistant" if raw_role in {"assistant", "gpt"} else "user"
        content = str(turn.get("content") or turn.get("value") or "").strip()
        if content:
            result.append({"role": role, "content": content})
    return result


def has_loop(text: str) -> bool:
    units = [unit.strip() for unit in LOOP_SPLIT_RE.split(text) if len(unit.strip()) >= 6]
    if any(count >= 3 for count in Counter(units).values()):
        return True
    width = 12
    return any(text[i:i+width] == text[i+width:i+2*width] == text[i+2*width:i+3*width] for i in range(max(0, len(text) - 3*width + 1)))

def conv_quality(messages: list[dict[str, str]]) -> dict[str, object]:
    assistants = [m["content"] for m in messages if m["role"] == "assistant"]
    joined = "\n".join(assistants)
    alternating = bool(messages) and all(
        messages[i]["role"] != messages[i - 1]["role"] for i in range(1, len(messages))
    )
    return {
        "turns": len(assistants),
        "assistant_chars": len(joined),
        "speculative": bool(SPECULATIVE_RE.search(joined)),
        "ocr_doc": bool(OCR_RE.search(joined)),
        "loop": has_loop(joined),
        "bad_generation": bool(BAD_GENERATION_RE.search(joined)),
        "alternating": alternating,
        "image_markers": sum(m["content"].count("<image>") + m["content"].count("<img>") for m in messages),
    }


def summarize(rows: list[dict]) -> dict:
    qualities = [r["quality"] for r in rows]
    lengths = [int(q["assistant_chars"]) for q in qualities]
    return {
        "sampled": len(rows),
        "assistant_chars_median": int(median(lengths)) if lengths else 0,
        "assistant_chars_p90": percentile(lengths, 0.90),
        "turns": dict(Counter(int(q["turns"]) for q in qualities)),
        "speculative": sum(bool(q["speculative"]) for q in qualities),
        "ocr_doc": sum(bool(q["ocr_doc"]) for q in qualities),
        "loop": sum(bool(q["loop"]) for q in qualities),
        "bad_generation": sum(bool(q["bad_generation"]) for q in qualities),
        "non_alternating": sum(not bool(q["alternating"]) for q in qualities),
        "multiple_image_markers": sum(int(q["image_markers"]) > 1 for q in qualities),
    }


def audit_mminstruct(path: Path, sample_size: int, seed: int) -> tuple[dict, list[dict]]:
    rng = random.Random(seed)
    samples: list[dict] = []
    total = 0
    domain_counts: Counter[str] = Counter()
    selected_counts: Counter[str] = Counter()
    eligible = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            image = str(row.get("image") or "")
            domain = image.split("/", 1)[0]
            domain_counts[domain] += 1
            messages = normalize_conversations(row.get("conversations"))
            quality = conv_quality(messages)
            is_eligible = (
                domain in MMI_ALLOW_DOMAINS
                and len(messages) == 2
                and quality["turns"] == 1
                and quality["alternating"]
                and not quality["ocr_doc"]
                and not quality["loop"]
                and not quality["bad_generation"]
                and quality["assistant_chars"] <= 180
                and quality["image_markers"] == 1
            )
            if is_eligible:
                eligible += 1
                selected_counts[domain] += 1
            item = {
                "id": row.get("id"),
                "image": image,
                "domain": domain,
                "messages": messages,
                "quality": quality,
                "eligible": is_eligible,
            }
            if len(samples) < sample_size:
                samples.append(item)
            else:
                slot = rng.randrange(total)
                if slot < sample_size:
                    samples[slot] = item
    return {
        "source": "yuecao0119/MMInstruct-GPT4V/caption_cn",
        "total_rows": total,
        "eligible_rows": eligible,
        "eligible_rate": eligible / max(1, total),
        "domain_counts": dict(domain_counts.most_common()),
        "eligible_domain_counts": dict(selected_counts.most_common()),
        "sample_metrics": summarize(samples),
    }, samples


def audit_opencsg(path: Path, sample_size: int, seed: int) -> tuple[dict, list[dict]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    columns = set(table.column_names)
    type_col = "type" if "type" in columns else "task_type"
    image_col = "image" if "image" in columns else "image_path"
    conv_col = "conversations" if "conversations" in columns else "conversation"
    required = {type_col, image_col, conv_col}
    if not required.issubset(columns):
        raise RuntimeError(f"OpenCSG columns={sorted(columns)}, missing={sorted(required-columns)}")
    type_values = table[type_col].to_pylist()
    image_values = table[image_col].to_pylist()
    conversation_values = table[conv_col].to_pylist()
    type_counts = Counter(str(x) for x in type_values)
    conversation_indices = [
        i for i, value in enumerate(type_values) if str(value).lower() in {"conversation", "conv"}
    ]
    rng = random.Random(seed)
    chosen = rng.sample(conversation_indices, min(sample_size, len(conversation_indices)))
    samples: list[dict] = []
    eligible = 0
    eligible_turn_counts: Counter[int] = Counter()
    # Full eligibility count is cheap once the parquet is in memory.
    for i in conversation_indices:
        messages = normalize_conversations(conversation_values[i])
        q = conv_quality(messages)
        ok = (
            2 <= q["turns"] <= 5
            and q["alternating"]
            and not q["ocr_doc"]
            and not q["loop"]
            and not q["bad_generation"]
            and q["assistant_chars"] <= 900
            and q["image_markers"] <= 1
        )
        if ok:
            eligible += 1
            eligible_turn_counts[int(q["turns"])] += 1
    for i in chosen:
        messages = normalize_conversations(conversation_values[i])
        quality = conv_quality(messages)
        ok = (
            2 <= quality["turns"] <= 5
            and quality["alternating"]
            and not quality["ocr_doc"]
            and not quality["loop"]
            and not quality["bad_generation"]
            and quality["assistant_chars"] <= 900
            and quality["image_markers"] <= 1
        )
        samples.append({
            "row_index": i,
            "image": image_values[i],
            "type": type_values[i],
            "messages": messages,
            "quality": quality,
            "eligible": ok,
        })
    return {
        "source": "opencsg/LLaVA-Instruct-600K-Chinese/conversation",
        "total_rows": table.num_rows,
        "type_counts": dict(type_counts),
        "conversation_rows": len(conversation_indices),
        "eligible_rows": eligible,
        "eligible_rate": eligible / max(1, len(conversation_indices)),
        "eligible_turn_counts": dict(eligible_turn_counts),
        "sample_metrics": summarize(samples),
    }, samples


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reports = {}
    mmi_report, mmi_samples = audit_mminstruct(
        args.raw_root / "mminstruct" / "caption_cn.jsonl", args.sample_size, args.seed
    )
    reports["mminstruct_caption_cn"] = mmi_report
    write_jsonl(args.output_dir / "mminstruct_sample.jsonl", mmi_samples)

    opencsg_path = args.raw_root / "opencsg" / "data.parquet"
    if opencsg_path.exists():
        opencsg_report, opencsg_samples = audit_opencsg(opencsg_path, args.sample_size, args.seed)
        reports["opencsg_conversation"] = opencsg_report
        write_jsonl(args.output_dir / "opencsg_conversation_sample.jsonl", opencsg_samples)

    report_path = args.output_dir / "audit_report.json"
    report_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


