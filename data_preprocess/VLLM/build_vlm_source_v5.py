#!/usr/bin/env python3
"""Build one audited tinyLLM VLM v5 source with resumable five-view output."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
COMPAT_TRANSFORMERS = ROOT / "runtime" / "minicpm_transformers_449"
if COMPAT_TRANSFORMERS.is_dir():
    sys.path.insert(0, str(COMPAT_TRANSFORMERS))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from data_preprocess.VLLM.vllm import save_image_2x2_plus_thumb
from data_preprocess.VLLM.vlm_cleaning_v5 import (
    caption_messages,
    normalize_messages,
    qa_messages,
    render_and_validate,
)


def resolve_project_path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def stable_score(seed: int, value: Any) -> int:
    return int.from_bytes(
        hashlib.blake2b(f"{seed}:{value}".encode("utf-8"), digest_size=8).digest(), "big"
    )


def decode_base64(value: str) -> bytes:
    if value.startswith("data:") and "," in value:
        value = value.split(",", 1)[1]
    return base64.b64decode(value, validate=True)


def select_lowest(items: Iterable[Any], count: int, score_fn) -> list[Any]:
    """Streaming deterministic sample retaining the smallest hash values."""
    heap: list[tuple[int, int, Any]] = []
    for serial, item in enumerate(items):
        score = int(score_fn(item))
        entry = (-score, -serial, item)
        if len(heap) < count:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    return [entry[2] for entry in sorted(heap, key=lambda x: (-x[0], -x[1]))]


class Writer:
    def __init__(self, cfg: dict, source: str, output_root: Path, image_root: Path, overwrite: bool):
        self.cfg = cfg
        self.source = source
        self.spec = cfg["sources"][source]
        self.output_root = output_root
        self.image_root = image_root
        self.output = output_root / "parts" / f"{source}.jsonl"
        self.partial = self.output.with_suffix(".jsonl.partial")
        self.progress = self.output.with_suffix(".progress.json")
        self.manifest = self.output.with_suffix(".manifest.json")
        self.image_dir = image_root / source
        self.minimum_free = int(cfg["storage_plan"]["minimum_free_bytes"])
        self.quality = int(cfg["storage_plan"]["five_view_jpeg_quality"])
        self.max_tokens = int(cfg["max_total_tokens"])
        self.rejected: Counter[str] = Counter()
        self.completed_ids: set[str] = set()
        self.kept = 0
        self.started = time.time()

        if overwrite:
            self.output.unlink(missing_ok=True)
            self.partial.unlink(missing_ok=True)
            self.progress.unlink(missing_ok=True)
            self.manifest.unlink(missing_ok=True)
            if self.image_dir.exists():
                if self.image_dir.resolve().parent != self.image_root.resolve():
                    raise RuntimeError("refusing to remove image directory outside image root")
                shutil.rmtree(self.image_dir)
        if self.output.exists():
            raise FileExistsError(f"completed source exists: {self.output}")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        if self.partial.exists():
            with self.partial.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    self.completed_ids.add(str(row["source_id"]))
                    self.kept = max(self.kept, int(row["index"]) + 1)
            print(f"[resume] {source}: {len(self.completed_ids)} rows", flush=True)

        self.handle = self.partial.open("a", encoding="utf-8")

    def enough_disk(self) -> None:
        free = shutil.disk_usage(self.image_root).free
        if free < self.minimum_free:
            raise RuntimeError(
                f"disk safety stop: free={free / 2**30:.2f} GiB, "
                f"minimum={self.minimum_free / 2**30:.2f} GiB"
            )

    def add(self, *, source_id: Any, image_value: Any, messages: list[dict[str, str]], lang: str,
            task_role: str) -> bool:
        source_key = str(source_id)
        if source_key in self.completed_ids:
            return True
        rendered, token_count, reason = render_and_validate(
            self.tokenizer, messages, self.max_tokens
        )
        if reason:
            self.rejected[reason] += 1
            return False
        if self.kept % 200 == 0:
            self.enough_disk()
        try:
            if isinstance(image_value, Path):
                image_value = str(image_value)
            saved = save_image_2x2_plus_thumb(
                image_value, self.image_dir, self.kept, jpeg_quality=self.quality
            )
        except Exception:
            saved = None
        if saved is None:
            self.rejected["image_decode"] += 1
            return False
        names, vision_meta = saved
        relative = [f"{self.source}/{name}" for name in names]
        for view in vision_meta.get("views", []):
            name = view.get("file") or view.get("image_file")
            if name:
                view["file"] = f"{self.source}/{name}"
        holdout = int(self.spec.get("holdout_rows", 0))
        requested = max(1, int(self.spec["target_rows"]))
        split = "eval" if stable_score(int(self.cfg["seed"]) + 701, source_key) % requested < holdout else "train"
        record = {
            "text": rendered,
            "index": self.kept,
            "image_file": relative,
            "dataset": self.source,
            "lang": lang,
            "has_cot": False,
            "split": split,
            "vision_meta": vision_meta,
            "source_id": source_key,
            "task_role": task_role,
            "token_count": token_count,
        }
        self.handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.completed_ids.add(source_key)
        self.kept += 1
        if self.kept % 200 == 0:
            self.handle.flush()
            elapsed = max(0.001, time.time() - self.started)
            self.progress.write_text(
                json.dumps({
                    "source": self.source,
                    "kept": self.kept,
                    "rejected": dict(self.rejected),
                    "rows_per_second": self.kept / elapsed,
                    "free_gib": shutil.disk_usage(self.image_root).free / 2**30,
                }, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                f"{self.source} kept={self.kept} rejected={sum(self.rejected.values())} "
                f"rows_per_s={self.kept / elapsed:.2f} "
                f"free_gib={shutil.disk_usage(self.image_root).free / 2**30:.1f}",
                flush=True,
            )
        return True

    def __enter__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(resolve_project_path(self.cfg["paths"]["tokenizer"])),
            trust_remote_code=True,
            local_files_only=True,
        )
        if self.tokenizer.chat_template is None:
            raise RuntimeError("tokenizer chat_template is missing")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.handle.flush()
        self.handle.close()
        if exc_type is not None:
            return False
        if self.kept <= 0:
            raise RuntimeError(f"{self.source}: no valid records were produced")
        self.enough_disk()
        self.partial.replace(self.output)
        report = {
            "schema_version": 5,
            "source": self.source,
            "completed": True,
            "kept": self.kept,
            "rejected": dict(self.rejected),
            "five_views_per_record": True,
            "canonical_image_marker": "one <img> at first user turn",
            "semantic_speculation_filter": False,
            "incidental_ocr_filter": False,
            "output": str(self.output),
            "image_dir": str(self.image_dir),
            "free_gib_after": shutil.disk_usage(self.image_root).free / 2**30,
        }
        self.manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.progress.unlink(missing_ok=True)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return False


def iter_fm_groups(path: Path):
    current_id = None
    image_b64 = None
    pairs: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            image_id = row.get("image_id")
            if current_id is not None and image_id != current_id:
                yield current_id, image_b64, pairs
                pairs = []
            current_id = image_id
            image_b64 = row.get("base64")
            pairs.append((row.get("question"), row.get("answer")))
    if current_id is not None:
        yield current_id, image_b64, pairs


def build_fm(cfg: dict, raw: Path, writer: Writer, limit: int | None):
    count = 0
    for image_id, image_b64, pairs in iter_fm_groups(raw / "fm_iqa" / "para_train.jsonl"):
        if limit is not None and count >= limit:
            break
        messages, reason = qa_messages(pairs[: int(writer.spec.get("max_qas_per_image", 4))], language="zh")
        if reason or not messages or not isinstance(image_b64, str):
            writer.rejected[reason or "missing_image"] += 1
            continue
        try:
            image = {"bytes": decode_base64(image_b64)}
        except Exception:
            writer.rejected["image_base64"] += 1
            continue
        writer.add(source_id=image_id, image_value=image, messages=messages, lang="zh", task_role="short_vqa")
        count += 1


def build_m3it(cfg: dict, raw: Path, writer: Writer, limit: int | None):
    name = "coco_cn_train.jsonl" if writer.source == "m3it_coco_cn" else "flickr8k_cn_train.jsonl"
    path = raw / "m3it_caption_cn" / name
    wanted = limit if limit is not None else int(writer.spec["target_rows"])
    with path.open(encoding="utf-8") as handle:
        for ordinal, line in enumerate(handle):
            if writer.kept >= wanted:
                break
            row = json.loads(line)
            caption = row.get("caption")
            image_b64 = row.get("image_base64") or row.get("image_str")
            source_id = row.get("image_id") if "image_id" in row else row.get("img_id", ordinal)
            messages, reason = caption_messages(caption, language="zh")
            if reason or not messages or not isinstance(image_b64, str):
                writer.rejected[reason or "missing_image"] += 1
                continue
            try:
                image = {"bytes": decode_base64(image_b64)}
            except Exception:
                writer.rejected["image_base64"] += 1
                continue
            writer.add(source_id=source_id, image_value=image, messages=messages, lang="zh", task_role="caption")


def build_pangea(cfg: dict, raw: Path, writer: Writer, limit: int | None):
    root = raw / "pangea_llava_en_zh_300k"
    rows = json.loads((root / "data.json").read_text(encoding="utf-8"))
    wanted = limit if limit is not None else int(writer.spec["target_rows"])
    chosen = select_lowest(rows, min(wanted, len(rows)), lambda r: stable_score(int(cfg["seed"]) + 31, r.get("id")))
    for row in chosen:
        rel = str(row.get("image") or "").replace("\\", "/")
        image_path = root / "images" / Path(rel).name
        if not image_path.is_file():
            writer.rejected["image_missing"] += 1
            continue
        messages, reason = normalize_messages(
            row.get("conversations") or [], default_user_prompt="请描述这张图片。"
        )
        if reason or not messages:
            writer.rejected[reason or "messages"] += 1
            continue
        writer.add(source_id=row.get("id"), image_value=image_path, messages=messages, lang="zh", task_role="multi_turn")


def cog_paths(root: Path, source: str) -> tuple[Path, Path, str, str]:
    _, form, lang = source.split("_", 2)
    form_dir = {
        "detail": "llava_details-minigpt4_3500_formate",
        "multi": "llava_instruction_multi_conversations_formate",
        "single": "llava_instruction_single_conversation_formate",
    }[form]
    base = root / "cogvlm_modelscope" / "CogVLM-SFT-311K" / form_dir
    return base / f"labels_{lang}", base / "images", form, lang


def build_cog(cfg: dict, raw: Path, writer: Writer, limit: int | None):
    label_dir, image_dir, form, lang = cog_paths(raw, writer.source)
    # Chinese and English labels share the same image directory. Partition by
    # image stem so language variants do not materialize duplicate five-view JPEGs.
    labels = [
        path for path in sorted(label_dir.glob("*.json"))
        if stable_score(int(cfg["seed"]) + 40, path.stem) % 2 == (0 if lang == "zh" else 1)
    ]
    wanted = limit if limit is not None else int(writer.spec["target_rows"])
    chosen = select_lowest(labels, min(wanted, len(labels)), lambda p: stable_score(int(cfg["seed"]) + 41, p.name))
    for label_path in chosen:
        image_path = None
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            candidate = image_dir / f"{label_path.stem}{ext}"
            if candidate.is_file():
                image_path = candidate
                break
        if image_path is None:
            writer.rejected["image_missing"] += 1
            continue
        row = json.loads(label_path.read_text(encoding="utf-8"))
        if form == "detail":
            captions = row.get("captions") or []
            caption = captions[0].get("content") if captions and isinstance(captions[0], dict) else None
            messages, reason = caption_messages(caption, language=lang)
        else:
            messages, reason = normalize_messages(
                row.get("conversations") or [],
                default_user_prompt="请描述这张图片。" if lang == "zh" else "Describe this image.",
            )
        if reason or not messages:
            writer.rejected[reason or "messages"] += 1
            continue
        writer.add(source_id=label_path.stem, image_value=image_path, messages=messages, lang=lang,
                   task_role="caption" if form == "detail" else f"{form}_turn")


def vqa_shape_compatible(question: Any, answer: Any) -> bool:
    """Drop only obvious question/answer type mismatches."""
    q = " ".join(str(question or "").lower().split())
    a = " ".join(str(answer or "").lower().split()).strip(" .!?")
    if not q or not a:
        return False
    wh = ("what ", "which ", "who ", "where ", "when ", "why ", "how ")
    if q.startswith(wh) and a in {"yes", "no"}:
        return False
    if q.startswith(("how many ", "what number ")) and not any(ch.isdigit() for ch in a):
        number_words = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}
        if a not in number_words:
            return False
    return True


def build_vqav2(cfg: dict, raw: Path, writer: Writer, limit: int | None):
    import pyarrow.parquet as pq
    paths = sorted((raw / "vqav2").glob("train-*.parquet"))
    expected = int(writer.spec.get("shards", 8))
    if len(paths) != expected:
        raise RuntimeError(f"expected {expected} VQAv2 shards, found {len(paths)}")
    wanted = limit if limit is not None else int(writer.spec["target_rows"])
    candidates = []
    for shard_index, path in enumerate(paths):
        texts = pq.read_table(path, columns=["texts"])["texts"].to_pylist()
        for row_index, turns in enumerate(texts):
            valid = []
            for turn in turns or []:
                if (isinstance(turn, dict) and turn.get("user") and turn.get("assistant")
                        and vqa_shape_compatible(turn["user"], turn["assistant"])):
                    valid.append((turn["user"], turn["assistant"]))
            if not valid:
                continue
            key = f"{shard_index}:{row_index}"
            score = stable_score(int(cfg["seed"]) + 51, key)
            qa = valid[score % len(valid)]
            candidates.append((score, shard_index, row_index, qa))
    candidates.sort(key=lambda x: x[0])
    by_shard: dict[int, dict[int, tuple[str, str]]] = {}
    for _, shard_index, row_index, qa in candidates[:wanted]:
        by_shard.setdefault(shard_index, {})[row_index] = qa
    for shard_index, selected in sorted(by_shard.items()):
        images = pq.read_table(paths[shard_index], columns=["images"])["images"].to_pylist()
        for row_index, qa in selected.items():
            image_values = images[row_index]
            if not isinstance(image_values, list) or len(image_values) != 1:
                writer.rejected["not_single_image"] += 1
                continue
            messages, reason = qa_messages([qa], language="en")
            if reason or not messages:
                writer.rejected[reason or "messages"] += 1
                continue
            writer.add(source_id=f"{shard_index}:{row_index}", image_value=image_values[0], messages=messages,
                       lang="en", task_role="short_vqa")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--overwrite-source", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if cfg.get("training_enabled") is not False:
        raise RuntimeError("preprocessing requires training_enabled=false")
    if args.source not in cfg["sources"]:
        raise KeyError(f"unknown source {args.source}")
    raw = resolve_project_path(cfg["paths"]["raw_root"])
    output_root = args.output_root or resolve_project_path(cfg["paths"]["prepared_root"])
    image_root = args.image_root or resolve_project_path(cfg["paths"]["image_root"])
    image_root.mkdir(parents=True, exist_ok=True)
    with Writer(cfg, args.source, output_root, image_root, args.overwrite_source) as writer:
        if args.source == "fm_iqa":
            build_fm(cfg, raw, writer, args.limit)
        elif args.source.startswith("m3it_"):
            build_m3it(cfg, raw, writer, args.limit)
        elif args.source == "pangea_zh_multiturn":
            build_pangea(cfg, raw, writer, args.limit)
        elif args.source.startswith("cog_"):
            build_cog(cfg, raw, writer, args.limit)
        elif args.source == "vqav2":
            build_vqav2(cfg, raw, writer, args.limit)
        else:
            raise KeyError(args.source)


if __name__ == "__main__":
    main()




