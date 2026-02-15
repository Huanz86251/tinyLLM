#!/usr/bin/env python3
"""Validate and deterministically assemble completed VLM v5 source parts."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def score(seed: int, source: str, source_id: str) -> str:
    return hashlib.blake2b(f"{seed}:{source}:{source_id}".encode(), digest_size=16).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--parts-root", type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if cfg.get("training_enabled") is not False:
        raise RuntimeError("assembly requires training_enabled=false")
    prepared = args.parts_root or resolve_project_path(cfg["paths"]["prepared_root"])
    output = args.output or prepared / "vlm_continual_v5.mixed.jsonl"
    rows = []
    rejected = Counter()
    counts = Counter()
    for source, spec in cfg["sources"].items():
        if not spec.get("enabled", False):
            continue
        path = prepared / "parts" / f"{source}.jsonl"
        manifest = path.with_suffix(".manifest.json")
        if not path.is_file() or not manifest.is_file():
            raise FileNotFoundError(f"source is incomplete: {source}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                images = row.get("image_file")
                text = row.get("text")
                if not isinstance(images, list) or len(images) != 5:
                    rejected["not_five_views"] += 1
                    continue
                if not isinstance(text, str) or text.count("<img>") != 1:
                    rejected["bad_image_marker"] += 1
                    continue
                if "<|im_start|>assistant\n<img>" in text:
                    rejected["assistant_image_marker"] += 1
                    continue
                if row.get("split") not in {"train", "eval"}:
                    rejected["bad_split"] += 1
                    continue
                counts[(source, row["split"])] += 1
                rows.append(row)
    rows.sort(key=lambda r: (r["split"] == "eval", score(int(cfg["seed"]), r["dataset"], r["source_id"])))
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(output)
    report = {
        "schema_version": 5,
        "output": str(output),
        "rows": len(rows),
        "train_rows": sum(1 for row in rows if row["split"] == "train"),
        "eval_rows": sum(1 for row in rows if row["split"] == "eval"),
        "counts": {f"{k[0]}:{k[1]}": v for k, v in sorted(counts.items())},
        "rejected": dict(rejected),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


