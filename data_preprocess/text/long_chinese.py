#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, regex as re
from tqdm import tqdm
import xxhash
from datasets import load_dataset

# =========================
# 配置常量（按需改）
# =========================
OUT_PATH           = r"E:\learn\data\pretrain_long_zh.jsonl"          # 继续写同一个文件（追加）
SAVE_HASHSET_PATH  = r"E:\learn\data\pretrain_long_zh.hashes.txt"     # 复用上轮生成的哈希表
HF_CACHE_DIR       = None
USE_STREAMING      = True
TARGET_GB          = 40.0              # 这轮目标增量（按写出字节计）
MIN_CHARS          = 12000              # 清洗后长度阈值
MIN_CHARS_FAST     = 12000              # 原始长度快速预筛
SCORE_MIN          = 0.6
TOKENIZER_DIR      = ""                # 例如 r"E:\tok\minicpm3_4b"
MIN_TOKENS         = None              # 例如 8000；仅当 TOKENIZER_DIR 非空时生效
DEDUPE             = True
APPEND_OUTPUT      = True              # 关键：改为追加；False 则清空重写

DATASET_NAME       = "opencsg/Fineweb-Edu-Chinese-V2.1"
DATASET_CONFIG     = "default"
DATASET_SPLIT      = "train"

# =========================
# 轻量清洗
# =========================
RE_CONTROL = re.compile(r"[\u200B-\u200F\uFEFF]")
RE_EMOJI   = re.compile(r"\p{Emoji_Presentation}|\p{Extended_Pictographic}", re.UNICODE)

def clean_text(s: str) -> str:
    if not s: return ""
    s = RE_CONTROL.sub("", s)
    s = RE_EMOJI.sub("", s)
    s = s.replace("\u0000", "")
    return s.strip()

def human(sz_bytes: int) -> str:
    return f"{sz_bytes/1024**3:.3f} GB"

def load_tokenizer():
    if not TOKENIZER_DIR or MIN_TOKENS is None:
        return None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=True, trust_remote_code=False)
    except Exception as e:
        print(f"[WARN] 无法加载 tokenizer（{e}），仅按字符阈值筛选。")
        return None

def count_tokens(tok, text: str):
    if tok is None: return None
    return len(tok.encode(text))

def load_hashset(path: str):
    seen = set()
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                h = line.strip()
                if h: seen.add(h)
        print(f"[INFO] 载入历史 HASH {len(seen)} 条用于去重")
    return seen

def save_hashset(path: str, seen: set):
    if not path: return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for h in seen: f.write(h + "\n")
    print(f"[INFO] HASH 集保存完成：{path}（共 {len(seen)} 条）")

def main():
    os.makedirs(os.path.dirname(os.path.abspath(OUT_PATH)), exist_ok=True)
    if not APPEND_OUTPUT:
        # 如需重开新文件（不累计以前内容），把 APPEND_OUTPUT 设为 False
        open(OUT_PATH, "wb").close()

    tok = load_tokenizer()
    target_bytes = int(TARGET_GB * (1024 ** 3))
    wrote_bytes = 0
    wrote_rows  = 0
    drop_len = drop_tok = drop_dup = drop_score = 0

    # 关键：预载历史哈希，确保不与上一轮重复
    seen = load_hashset(SAVE_HASHSET_PATH) if (DEDUPE and SAVE_HASHSET_PATH) else set()

    ds = load_dataset(
        DATASET_NAME,
        data_files={DATASET_SPLIT: "4_5/**/*.parquet"},
        split=DATASET_SPLIT,
        streaming=USE_STREAMING,
        trust_remote_code=False,
        cache_dir=HF_CACHE_DIR,
    )

    # 以追加方式写入
    with open(OUT_PATH, "ab") as w:
        pbar = tqdm(ds, desc=f"SCAN({DATASET_NAME}:{DATASET_CONFIG}/{DATASET_SPLIT})", unit="rows")
        for ex in pbar:
            if wrote_bytes >= target_bytes: break

            # 1) 分数硬门槛（更早丢弃低质）
            score = ex.get("score", None)
            if (score is None) or (score < SCORE_MIN):
                drop_score += 1
                continue

            # 2) 原始长度快速预筛（不清洗、不hash、不卡 tokenizer）
            raw = ex.get("text", "")
            if not raw or len(raw) < MIN_CHARS_FAST:
                drop_len += 1
                continue

            # 3) 清洗后精确长度判定
            txt = clean_text(raw)
            if len(txt) < MIN_CHARS:
                drop_len += 1
                continue

            # 4) （可选）边界带再测 token
            if tok is not None and MIN_TOKENS is not None:
                n_tok = count_tokens(tok, txt)
                if n_tok is None or n_tok < MIN_TOKENS:
                    drop_tok += 1
                    continue

            # 5) 哈希去重：与历史 seen 做交集过滤
            h = xxhash.xxh3_64_hexdigest(txt)
            if DEDUPE and h in seen:
                drop_dup += 1
                continue
            if DEDUPE:
                seen.add(h)

            # 6) 写出
            src = ex.get("source", "Fineweb-Edu-Chinese-V2.1")
            rec = {"TXT": txt, "QUESTION": 0,
                   "SCORE": float(score) if isinstance(score, (int, float)) else None,
                   "SRC": str(src), "HASH": h}
            blob = (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")
            w.write(blob)
            wrote_bytes += len(blob); wrote_rows += 1

            if wrote_rows % 1000 == 0:
                pbar.set_postfix(rows=wrote_rows, size=human(wrote_bytes),
                                 drop_score=drop_score, drop_len=drop_len,
                                 drop_tok=drop_tok, drop_dup=drop_dup)

    # 7) 将 seen 回写到同一个哈希文件，便于下轮续采
    if DEDUPE and SAVE_HASHSET_PATH:
        save_hashset(SAVE_HASHSET_PATH, seen)

    print(f"[FINAL] 输出 {wrote_rows} 行，{human(wrote_bytes)}（本轮新增）→ {OUT_PATH}")
    print(f"        丢弃：score<{SCORE_MIN} → {drop_score}，len<{MIN_CHARS}/{MIN_CHARS_FAST} → {drop_len}，"
          f"tok<{MIN_TOKENS or '-'} → {drop_tok}，dup → {drop_dup}")

if __name__ == "__main__":
    main()
