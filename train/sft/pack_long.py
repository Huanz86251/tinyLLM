from project_paths import legacy_path
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, sys
from datetime import datetime
from typing import Iterator, Dict, Any

import numpy as np
from datasets import Dataset, Features, Value
from transformers import AutoTokenizer

# ================== 配置（按需修改） ==================
MAX_LEN        = 16384        # 目标最大长度（>MAX_LEN 的样本将被丢弃）
APPEND_EOS     = False        # ✅ 不再额外追加 EOS
CAND_KEYS      = ("TXT", "text", "txt", "TEXT")

TOK_DIR   = legacy_path("/root/autodl-tmp/llm/KDmodel/minicpm3_4b/snapshot")
JSONL     = legacy_path("/root/autodl-tmp/sft_merged_with_isMathlong.jsonl")
OUT_DIR   = legacy_path("/root/autodl-tmp/llm/cache_sft/long_varlen")

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

def iter_jsonl_sft(path: str, cand_keys=CAND_KEYS) -> Iterator[Dict[str, Any]]:
    """
    从 JSONL 逐行读取，只保留：
      - TXT: 文本
      - isMath: bool（缺省 False）
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

            txt = None
            for k in cand_keys:
                v = obj.get(k)
                if isinstance(v, str) and v:
                    txt = v
                    break
            if not txt:
                continue

            is_math = bool(obj.get("isMath", False))
            yield {"TXT": txt, "isMath": is_math}

def main():
    # 基本检查
    if not os.path.isdir(TOK_DIR):
        die(f"Tokenizer dir not found: {TOK_DIR}")
    if not os.path.isfile(os.path.join(TOK_DIR, "tokenizer.json")):
        die(f"Missing tokenizer.json in {TOK_DIR}")
    if not os.path.isfile(JSONL):
        die(f"JSONL not found: {JSONL}")
    if os.path.exists(OUT_DIR):
        die(f"OUT_DIR already exists: {OUT_DIR} (rm -rf 后再运行)")

    os.makedirs(OUT_DIR, exist_ok=False)

    print(f"[TOKENIZER] loading from {TOK_DIR}")
    tok = AutoTokenizer.from_pretrained(TOK_DIR, trust_remote_code=True, local_files_only=True)

    # ===== ChatML 模板：用完整字符串编码成子序列 =====
    HEADER_TEXT = "<|im_start|>assistant\n"   # assistant 后紧跟一个换行
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

    # ✅ 强制要求 <|im_end|> 只能是一个 token
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

    print(f"[LOAD] JSONL = {JSONL}")
    raw = Dataset.from_generator(
        lambda: iter_jsonl_sft(JSONL, CAND_KEYS),
        features=Features({
            "TXT":    Value("string"),
            "isMath": Value("bool"),
        }),
        cache_dir=os.environ.get("HF_DATASETS_CACHE"),
    )
    print(f"[LOAD] raw examples = {len(raw):,}")

    # 2) tokenize：每一条样本独立；标记 ChatML 中 assistant 区域
    def tokenize_fn(batch):
        enc = tok(
            batch["TXT"],
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )

        all_ids     = enc["input_ids"]
        all_is_math = batch["isMath"]

        new_ids   = []
        lens      = []
        tgt_masks = []
        ok_flags  = []

        for ids, is_math in zip(all_ids, all_is_math):
            ids = list(ids)
            n   = len(ids)
            tgt = [0] * n
            ok  = 0

            # 支持多轮：一个样本中可能出现多段 <|im_start|>assistant\n ... <|im_end|>
            pos = 0
            while True:
                header_pos = find_subseq(ids, header_ids, start=pos)
                if header_pos == -1:
                    break

                content_start = header_pos + len(header_ids)

                end_pos = find_subseq(ids, end_ids, start=content_start)
                if end_pos == -1:
                    # 理论上不该发生，这一轮没有闭合 im_end，直接 break
                    break

                # 1) assistant 正文 [content_start, end_pos) 标记为 1
                for k in range(content_start, min(end_pos, n)):
                    tgt[k] = 1

                # 2) ✅ 把 <|im_end|> 这个 EOS token 本身也监督
                if 0 <= end_pos < n:
                    tgt[end_pos] = 1

                ok = 1
                pos = end_pos + 1   # end_ids 长度=1

            # ❌ 不再追加额外 EOS；所有 EOS 都是原始 TXT 里的 <|im_end|>
            new_ids.append(ids)
            lens.append(len(ids))
            tgt_masks.append(tgt)
            ok_flags.append(ok)

        return {
            "input_ids":   new_ids,
            "input_len":   lens,
            "isMath":      batch["isMath"],
            "target_mask": tgt_masks,
            "ok":          ok_flags,
        }

    print("[TOKENIZE] start ...")
    tokenized = raw.map(
        tokenize_fn,
        batched=True,
        remove_columns=raw.column_names,
        num_proc=4,
        desc="Tokenizing SFT (long)",
        load_from_cache_file=False,
        keep_in_memory=False,
    )

    # 3) ChatML 匹配情况统计（未过滤前）
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

    # 4) 丢弃所有 ok==0 的样本
    tokenized = tokenized.filter(
        lambda ex: ex["ok"] == 1,
        num_proc=4,
        desc="Filter invalid ChatML samples (ok==0)",
    )
    tokenized = tokenized.remove_columns(["ok"])

    if len(tokenized) == 0:
        die("[CHECK] tokenized dataset is empty after filter(ok==1).")

    # 5) 过滤：严格丢弃超过 MAX_LEN 的样本（此时都是 ok==1 的样本）
    def _len_ok(example):
        return example["input_len"] <= MAX_LEN

    before = len(tokenized)
    tokenized = tokenized.filter(_len_ok, desc=f"Filter >{MAX_LEN} tokens")
    after = len(tokenized)
    if after < before:
        print(f"[FILTER] dropped {before-after} examples longer than {MAX_LEN} tokens")

    if len(tokenized) == 0:
        die(f"[CHECK] tokenized dataset is empty after length filter(>{MAX_LEN}).")

    # 6) 统计最终长度信息
    arr_len = np.asarray(tokenized["input_len"], dtype=np.int64)
    total_tokens = int(arr_len.sum())
    max_len_seen = int(arr_len.max())
    print(f"[STATS] examples={len(tokenized):,} total_tokens={total_tokens:,} max_len={max_len_seen}")

    # 7) 保存为 HF dataset 目录
    tokenized.save_to_disk(OUT_DIR)
    print(f"[SAVE] dataset saved to {OUT_DIR}")

    # 8) 写 manifest
    manifest = {
        "stage": "SFT-long",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "jsonl": os.path.abspath(JSONL),
        "out_dir": os.path.abspath(OUT_DIR),
        "max_len": int(MAX_LEN),
        "num_examples": int(len(tokenized)),
        "total_tokens": int(total_tokens),
        "max_len_seen": int(max_len_seen),
        "tokenizer_dir": os.path.abspath(TOK_DIR),
        "vocab_size": int(vocab_size),
        "append_eos": bool(APPEND_EOS),
        "schema": {
            "input_ids": "List[int]",
            "input_len": "int (no extra EOS appended)",
            "isMath": "bool",
            "target_mask": "List[int] (0/1; 1=assistant 内容 + <|im_end|> 位置)",
        },
        "note": (
            f"严格丢弃 token 数 > {MAX_LEN} 的样本；"
            "每一条样本对应原始 JSONL 的一行（不跨样本，不切块，不合并）；"
            "只保留包含完整 <|im_start|>assistant\\n ... <|im_end|> 的样本；"
            "不再额外追加 EOS，EOS 即 <|im_end|>。"
        ),
    }
    with open(os.path.join(OUT_DIR, "sft_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("[DONE] SFT long varlen packing finished.")

if __name__ == "__main__":
    main()
