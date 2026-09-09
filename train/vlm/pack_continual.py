from project_paths import legacy_path
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, sys
from datetime import datetime
from typing import Iterator, Dict, Any

import numpy as np
from datasets import Dataset, Features, Value
from transformers import AutoTokenizer
from datasets import Sequence, Value
# ================== 配置（按需修改） ==================
# 这里只是一个“建议最大长度”，不在脚本里做强制截断
MAX_LEN        = int(os.environ.get("VLM_PACK_MAX_LEN", "2048"))
APPEND_EOS     = False   # ✅ 不再额外追加 EOS
CAND_KEYS      = ("TXT", "text", "txt", "TEXT")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# ✅ 这里改成你 TinyLLM / VLM 的 tokenizer 路径
TOK_DIR   = os.environ.get("VLM_PACK_TOKENIZER_DIR", legacy_path("/root/autodl-tmp/llm/tiny_05B_sft/checkpoint-50000"))

# ✅ 这里是你刚才那个 VLM 收集脚本的输出 JSONL
JSONL     = os.environ.get("VLM_PACK_JSONL", legacy_path("/root/autodl-tmp/vlm_sft_mix_onevision_xkev_scienceqa.shuf.jsonl"))

# ✅ 这里是图片目录（收集脚本里的 IMAGE_DIR 对应位置）
IMAGE_DIR = os.environ.get("VLM_PACK_IMAGE_DIR", legacy_path("/root/autodl-tmp/vlm_sft_images"))

# ✅ 输出 HF Dataset 目录
OUT_DIR   = os.environ.get("VLM_PACK_OUT_DIR", legacy_path("/root/autodl-tmp/llm/cache_sft/vlm_short_varlen"))

# 强制离线（和你其他脚本保持一致）
os.environ.update({
    "HF_HOME": legacy_path("/root/autodl-tmp/hf_home_strict_offline"),
    "HF_DATASETS_CACHE": legacy_path("/root/autodl-tmp/hf_datasets_cache_force"),
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_ENABLE_ONLINE_MODE": "0",
    "TOKENIZERS_PARALLELISM": "false",
})
# =====================================================

def die(msg: str):
    print(f"[FATAL] {msg}", file=sys.stderr, flush=True)
    sys.exit(1)
def _norm_img_list(v):
    if isinstance(v, str) and v:
        return [v]
    if isinstance(v, (list, tuple)):
        out = [x for x in v if isinstance(x, str) and x]
        return out
    return []

def _norm_split(v):
    value = str(v or "").strip().lower()
    return value if value in {"train", "eval"} else ""
def iter_jsonl_vlm(path: str, cand_keys=CAND_KEYS) -> Iterator[Dict[str, Any]]:
    """
    从 VLM JSONL 逐行读取，只保留：
      - TXT: 文本（从 cand_keys 中择一）
      - image_file: 图片文件名（如 "123.jpg"）
      - index: 整型 index
      - dataset / lang: 字符串，缺省给 ""
      - has_cot: bool（缺省 False）

    一行 JSON → 一条样本，不做合并。
    """
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[WARN] line {ln} JSON decode failed: {e}", file=sys.stderr)
                continue

            # ----- 1) TXT -----
            txt = None
            for k in cand_keys:
                v = obj.get(k)
                if isinstance(v, str) and v:
                    txt = v
                    break
            if not txt:
                continue

            # ----- 2) index -----
            try:
                idx = int(obj.get("index", -1))
            except Exception:
                idx = -1
            if idx < 0:
                print(f"[WARN] line {ln} missing/invalid index, skip.", file=sys.stderr)
                continue

            # ----- 3) image_file -----
            img_list = _norm_img_list(obj.get("image_file", None))
            if len(img_list) != 5:
                # 你现在所有样本理论上都该是 5；不是 5 就说明混进了旧数据/脏数据
                continue
            bad = False
            for fn in img_list:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in IMG_EXTS:
                    bad = True;
                    break
                fp = os.path.join(IMAGE_DIR, fn)
                if not os.path.isfile(fp):
                    bad = True;
                    break
            if bad:
                continue
            vm = obj.get("vision_meta", None)
            # 如果你要求所有样本必须有 meta：没有就跳过
            if not isinstance(vm, dict):
                continue

            vm_json = json.dumps(vm, ensure_ascii=False, separators=(",", ":"))
            dataset_name = obj.get("dataset") or ""
            lang = obj.get("lang") or ""
            has_cot = bool(obj.get("has_cot", False))
            split = _norm_split(obj.get("split"))

            yield {
                "TXT": txt,
                "index": idx,
                "image_file": img_list,
                "dataset": dataset_name,
                "lang": lang,
                "has_cot": has_cot,
                "split": split,
                "vision_meta": vm_json,
            }

def main():
    # 基本检查
    if not os.path.isdir(TOK_DIR):
        die(f"Tokenizer dir not found: {TOK_DIR}")
    if not os.path.isfile(os.path.join(TOK_DIR, "tokenizer.json")):
        die(f"Missing tokenizer.json in {TOK_DIR}")
    if not os.path.isfile(JSONL):
        die(f"JSONL not found: {JSONL}")
    if not os.path.isdir(IMAGE_DIR):
        die(f"IMAGE_DIR not found: {IMAGE_DIR}")
    if os.path.exists(OUT_DIR):
        die(f"OUT_DIR already exists: {OUT_DIR} (rm -rf 后再运行)")

    os.makedirs(OUT_DIR, exist_ok=False)

    print(f"[TOKENIZER] loading from {TOK_DIR}")
    tok = AutoTokenizer.from_pretrained(TOK_DIR, trust_remote_code=False, local_files_only=True, use_fast=True)

    # ChatML 模板片段（和你 text SFT 的逻辑保持一致）
    HEADER_TEXT = "<|im_start|>assistant\n"
    END_TEXT    = "<|im_end|>"

    header_ids = tok(
        HEADER_TEXT,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )["input_ids"]

    end_ids = tok(
        END_TEXT,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )["input_ids"]

    if not header_ids or not end_ids:
        die("[CHECK] header_ids or end_ids is empty, please check tokenizer / template.")

    # ✅ 严格要求 <|im_end|> 只能是 1 个 token，否则直接 FATAL
    if len(end_ids) != 1:
        die(f"[CHECK] '<|im_end|>' must be encoded as exactly 1 token, got len={len(end_ids)}. "
            f"Please fix tokenizer config.")

    end_token_id = end_ids[0]

    # 小工具：在 token 序列里找子序列的位置
    def find_subseq(seq, pattern, start=0):
        n, m = len(seq), len(pattern)
        if m == 0:
            return -1
        for i in range(start, n - m + 1):
            if seq[i:i + m] == pattern:
                return i
        return -1

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eos_id = tok.eos_token_id
    vocab_size = len(tok.get_vocab())
    print(f"[TOKENIZER] eos_id={eos_id} vocab_size={vocab_size}")

    # 1) 读取原始 JSONL → Dataset (TXT + index + image_file + dataset + lang + has_cot)
    print(f"[LOAD] JSONL = {JSONL}")
    raw = Dataset.from_generator(
        lambda: iter_jsonl_vlm(JSONL, CAND_KEYS),
        features=Features({
            "TXT":       Value("string"),
            "index":     Value("int64"),
            "image_file": Sequence(Value("string")),
            "dataset":   Value("string"),
            "lang":      Value("string"),
            "has_cot":   Value("bool"),
            "split":      Value("string"),
            "vision_meta": Value("string"),
        }),
        cache_dir=os.environ.get("HF_DATASETS_CACHE"),
    )
    print(f"[LOAD] raw examples = {len(raw):,}")

    # 2) tokenize：每一条样本独立，不合并，不切块，不 padding
    def tokenize_fn(batch):
        enc = tok(
            batch["TXT"],
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )

        all_ids   = enc["input_ids"]

        new_ids   = []
        lens      = []
        tgt_masks = []
        ok_flags  = []

        for ids in all_ids:
            ids = list(ids)
            n   = len(ids)
            tgt = [0] * n
            ok  = 0

            # 支持多轮对话：一个样本里可能有多个 assistant 段
            pos = 0
            while True:
                # 找 "<|im_start|>assistant\n" 这一段的 token 子序列
                header_pos = find_subseq(ids, header_ids, start=pos)
                if header_pos == -1:
                    break

                content_start = header_pos + len(header_ids)

                # 从 content_start 往后找第一个 "<|im_end|>"（单 token）
                end_pos = find_subseq(ids, end_ids, start=content_start)
                if end_pos == -1:
                    # 这一轮没找到闭合 im_end，认为这个 assistant 块无效，退出循环
                    break

                # 1) 标记 assistant 正文 [content_start, end_pos) 为 1
                for k in range(content_start, min(end_pos, n)):
                    tgt[k] = 1

                # 2) ✅ 把 EOS token（<|im_end|>）本身也标记为 1，监督它
                if 0 <= end_pos < n:
                    tgt[end_pos] = 1

                ok = 1
                # 继续向后找下一轮 assistant
                pos = end_pos + 1   # 因为我们已经确定 end_ids 长度=1

            # ✅ 不再额外 append 新的 EOS；所有 EOS 都来自原始 TXT 里的 <|im_end|>
            # 如果以后你真想强行裁长度，可以在这里对 ids / tgt 做截断

            new_ids.append(ids)
            lens.append(len(ids))
            tgt_masks.append(tgt)
            ok_flags.append(ok)

        return {
            "input_ids":   new_ids,
            "input_len":   lens,
            "target_mask": tgt_masks,
            "ok":          ok_flags,

            # 直接把这些字段透传下去（数量不变）
            "index":      batch["index"],
            "image_file": batch["image_file"],
            "dataset":    batch["dataset"],
            "lang":       batch["lang"],
            "has_cot":    batch["has_cot"],
            "split":       batch["split"],
            "vision_meta": batch["vision_meta"],
        }

    print("[TOKENIZE] start ...")
    tokenized = raw.map(
        tokenize_fn,
        batched=True,
        remove_columns=raw.column_names,    # TXT / index / image_file 等由我们重新给
        num_proc=6,
        desc="Tokenizing VLM SFT (short)",
        load_from_cache_file=False,
        keep_in_memory=False,
    )

    # ⭐ 3) 先在“未过滤”状态下做一次统计：有多少条 ok==1 / ok==0
    total_examples = len(tokenized)
    ok_arr   = np.asarray(tokenized["ok"], dtype=np.int64)
    n_ok     = int(ok_arr.sum())
    n_bad    = int(total_examples - n_ok)
    bad_ratio = n_bad / max(1, total_examples)

    print(
        f"[CHECK] after tokenize_fn (before filter): "
        f"total={total_examples:,}, ok={n_ok:,}, bad={n_bad:,}, bad_ratio={bad_ratio:.6f}"
    )

    if n_ok == 0:
        die("[CHECK] No valid ChatML samples found (<|im_start|>assistant\\n ... <|im_end|> 全部缺失).")

    # ⭐ 4) 过滤掉 bad 样本，只保留 ok==1 的
    tokenized = tokenized.filter(
        lambda ex: ex["ok"] == 1,
        num_proc=4,
        desc="Filter invalid ChatML samples (ok==0)",
    )
    tokenized = tokenized.remove_columns(["ok"])
    tokenized = tokenized.filter(lambda ex: ex["input_len"] <= MAX_LEN, num_proc=6)

    # 5) 正式统计 token 长度
    if len(tokenized) == 0:
        die("[CHECK] tokenized dataset is empty after filter(ok==1).")

    arr_len = np.asarray(tokenized["input_len"], dtype=np.int64)
    total_tokens = int(arr_len.sum())
    max_len_seen = int(arr_len.max())
    print(
        f"[STATS] examples={len(tokenized):,} "
        f"total_tokens={total_tokens:,} max_len={max_len_seen}"
    )

    # Measure the real objective mixture by supervised assistant tokens, not rows.
    mixture = {}
    total_supervised = 0
    for dataset_name, lang, split, target_mask in zip(
        tokenized["dataset"], tokenized["lang"], tokenized["split"], tokenized["target_mask"]
    ):
        key = f"{dataset_name}:{lang}:{split}"
        entry = mixture.setdefault(key, {"rows": 0, "supervised_tokens": 0})
        supervised = int(sum(target_mask))
        entry["rows"] += 1
        entry["supervised_tokens"] += supervised
        if split == "train":
            total_supervised += supervised
    for key, entry in mixture.items():
        entry["train_supervised_token_ratio"] = (
            entry["supervised_tokens"] / total_supervised
            if key.endswith(":train") and total_supervised
            else 0.0
        )
    print("[MIXTURE] " + json.dumps(mixture, ensure_ascii=False, sort_keys=True))

    # 6) 保存为 HF dataset 目录
    tokenized.save_to_disk(OUT_DIR)
    print(f"[SAVE] dataset saved to {OUT_DIR}")

    # 7) 写 manifest
    manifest = {
        "stage": "VLM-SFT-short",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "jsonl": os.path.abspath(JSONL),
        "image_dir": os.path.abspath(IMAGE_DIR),
        "out_dir": os.path.abspath(OUT_DIR),
        "max_len_hint": int(MAX_LEN),
        "num_examples": int(len(tokenized)),
        "total_tokens": int(total_tokens),
        "max_len_seen": int(max_len_seen),
        "tokenizer_dir": os.path.abspath(TOK_DIR),
        "vocab_size": int(vocab_size),
        "append_eos": bool(APPEND_EOS),
        "mixture_by_dataset_language_split": mixture,
        "total_train_supervised_tokens": int(total_supervised),
        "vision_meta_format":"json_string_v1",
        "schema": {
            "input_ids": "List[int]",
            "input_len": "int (token count, no extra EOS appended)",
            "target_mask": "List[int] (0/1; 1=assistant 内容 + <|im_end|> 位置)",
            "index": "int (与你 JSONL 里的 index 对应)",
            "image_file": "list[str] (相对 IMAGE_DIR 的文件名，例如 '123.jpg')",
            "dataset": "str (原数据集标识)",
            "lang": "str (en/zh/...)",
            "has_cot": "bool (是否含 COT，来自你的 JSONL)",
            "vision_meta": "str (json string; per-view geometry/pad info; schema_version=1 recommended)",

        },
        "note": (
            "VLM SFT：一行 JSON → 一条样本，不合并、不 padding；"
            "只保留包含完整 <|im_start|>assistant\\n ... <|im_end|> 的样本；"
            "图片在 image_file 字段里，通过 IMAGE_DIR + image_file 读取。"
        ),
    }
    with open(os.path.join(OUT_DIR, "vlm_sft_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("[DONE] VLM SFT short varlen packing finished.")

if __name__ == "__main__":
    main()
