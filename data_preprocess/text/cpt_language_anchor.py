#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_cosmo_corpora_v2.py

- 中文：opencsg/chinese-cosmopedia → 仅收 score > 0.7，写入 ≈40 GB
- 英文：HuggingFaceTB/cosmopedia   → 仅收 500 ≤ text_token_length ≤ 2000，
       stories / wikihow / automathtext 各写 ≈10 GB（总 ≈30 GB）

去重：
- 复用你主脚本的 .hashes.txt（pretrain_500Mmodel.cpt_stage.jsonl.hashes.txt）
- normalize_light(TXT) → md5；若已存在则跳过
- 写出采用 "ab" 追加；结束回刷哈希表；多次运行不会重复写

依赖：datasets tqdm regex transformers
"""

import os, json, time, regex as re
from hashlib import md5
from typing import Optional, Iterable, List
from tqdm import tqdm
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

# ---------------- 常量、路径 ----------------
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)

DATA_DIR   = os.path.join(REPO_ROOT, "data")
DP_DIR     = os.path.join(REPO_ROOT, "data_preprocess")

# 复用你主脚本那份哈希文件（很关键：跨脚本共享、确保不重复写）
DEDUP_HASH_FILE = os.path.join(DP_DIR, "pretrain_500Mmodel.cpt_stage.jsonl.hashes.txt")

# 输出文件
OUT_JSONL_ZH = os.path.join(DP_DIR, "cosmopedia.zh.jsonl")
OUT_JSONL_EN = os.path.join(DP_DIR, "cosmopedia.en.jsonl")

# 体量
GB = 1024 ** 3
TARGET_ZH_BYTES = int(40.0 * GB)  # 中文 ≈40 GB
TARGET_EN_SPLITS = {              # 英文三分支各 ≈10 GB
    "khanacademy":      int(10.0 * GB),
    "openstax":      int(5.0 * GB),
    "wikihow": int(5.0 * GB),
    "auto_math_text": int(5 * GB),  # 注意：官方无下划线
   "web_samples_v1": int(10.0 * GB),
}
EN_SPLIT_ALIAS = {"auto_math_text": "automathtext"}  # 容错别名

# 过滤阈值
ZH_SCORE_MIN = 0.6
EN_MIN_TOK   = 500
EN_MAX_TOK   = 2000

# ---------------- 清洗/归一化/哈希 ----------------
RE_CONTROL = re.compile(r"[\u200B-\u200F\uFEFF]")
RE_EMOJI   = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u26FF\u2700-\u27BF]")

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = RE_CONTROL.sub("", s)
    s = RE_EMOJI.sub("", s)
    s = s.replace("\u0000", "")
    return s.strip()

def normalize_light(s: str) -> str:
    s = clean_text(s)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"[ \t]*\n[ \t]*", "\n", s).strip()
    return s

def md5_of_text(s: str) -> str:
    return md5(s.encode("utf-8")).hexdigest()

def load_hashes_from_file(path: str) -> set:
    s = set()
    if not os.path.exists(path):
        return s
    with open(path, "r", encoding="utf-8") as r:
        for line in r:
            h = line.strip()
            if h:
                s.add(h)
    return s

def save_hashes_to_file(path: str, hashes: Iterable[str]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as w:
        for h in hashes:
            w.write(h + "\n")
    os.replace(tmp, path)

def absorb_jsonl_texts_into_seen(seen: set, jsonl_path: str, desc: str):
    if not os.path.exists(jsonl_path):
        return
    with open(jsonl_path, "r", encoding="utf-8") as r:
        for line in tqdm(r, desc=f"Absorb existing {desc}", unit="rows"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                txt = obj.get("TXT") or obj.get("text") or ""
            except Exception:
                txt = line
            norm = normalize_light(txt)
            if norm:
                seen.add(md5_of_text(norm))

# ---------------- 安全加载（streaming + 重试） ----------------
def safe_streaming_dataset(dataset: str, name: Optional[str], split: str,
                           max_tries: int = 5, delay_sec: float = 5.0):
    """
    - 对英文：name 为 stories/wikihow/automathtext，split 固定 "train"
    - 对中文：name=None，split="train"
    """
    last_err = None
    for _ in range(max_tries):
        try:
            if name:
                ds = load_dataset(dataset, name=name, split=split, streaming=True)
            else:
                ds = load_dataset(dataset, split=split, streaming=True)
            return ds
        except Exception as e:
            last_err = e
            time.sleep(delay_sec)
    print(f"[WARN] load_dataset(streaming) failed: {dataset} (name={name}, split={split}); last error: {last_err}")
    return None

# ---------------- 写出 ----------------
def write_record(f, txt: str, unik: float = 1.0) -> int:
    rec = {"TXT": txt, "QUESTION": 0, "UNIK": float(unik)}
    blob = (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")
    f.write(blob)
    return len(blob)

# ---------------- 主逻辑：中文 ----------------
def append_cosmo_cn(dst, seen: set, target_bytes: int) -> int:
    ds = safe_streaming_dataset("opencsg/chinese-cosmopedia", None, "train")
    if ds is None:
        return 0

    wrote = 0
    kept = dropped_score = 0
    pbar = tqdm(ds, desc=f"CN-cosmopedia(score>{ZH_SCORE_MIN})", unit="rows")

    for ex in pbar:
        if wrote >= target_bytes:
            break

        # 取字段
        txt_raw = clean_text(ex.get("text", "") if isinstance(ex, dict) else "")
        score   = ex.get("score", None)

        if not txt_raw or score is None:
            continue

        # 分数过滤
        try:
            score_f = float(score)
        except Exception:
            continue
        if not (score_f > ZH_SCORE_MIN):
            dropped_score += 1
            continue

        # 去重
        sig = md5_of_text(normalize_light(txt_raw))
        if sig in seen:
            continue

        wrote += write_record(dst, txt_raw, unik=score_f)  # UNIK=score_f（百科通常 <3.0，后续可走 non-strict）
        seen.add(sig)
        kept += 1

        if wrote % (2 << 20) < 2000:
            pbar.set_postfix(size_GB=f"{wrote/1024**3:.3f}", kept=kept, dropped_score=dropped_score)

    return wrote

# ---------------- 主逻辑：英文（按 token 区间过滤） ----------------
def parse_int(v) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int,)):
        return int(v)
    if isinstance(v, float):
        # 有些数据可能是 float，但本质是整数
        return int(v)
    if isinstance(v, str):
        v = v.strip()
        if not v:
            return None
        # 尝试去掉逗号等分隔
        v = v.replace(",", "")
        try:
            return int(float(v))
        except Exception:
            return None
    return None

def append_cosmo_en_split(dst, seen: set, split: str, target_bytes: int) -> int:
    real_split = EN_SPLIT_ALIAS.get(split, split)
    ds = safe_streaming_dataset("HuggingFaceTB/cosmopedia", real_split, "train")
    if ds is None:
        return 0

    wrote = 0
    kept = dropped_tok = 0
    pbar = tqdm(ds, desc=f"EN-cosmopedia[{real_split}]({EN_MIN_TOK}≤tok≤{EN_MAX_TOK})", unit="rows")

    for ex in pbar:
        if wrote >= target_bytes:
            break

        # 字段
        txt_raw = clean_text(ex.get("text", "") if isinstance(ex, dict) else "")
        tlen    = parse_int(ex.get("text_token_length", None))

        if not txt_raw or tlen is None:
            continue

        # token 区间过滤
        if not (EN_MIN_TOK <= tlen <= EN_MAX_TOK):
            dropped_tok += 1
            continue

        # 去重
        sig = md5_of_text(normalize_light(txt_raw))
        if sig in seen:
            continue

        wrote += write_record(dst, txt_raw, unik=1.0)  # 英文百科统一给 1.0（非 strict）
        seen.add(sig)
        kept += 1

        if wrote % (2 << 20) < 2000:
            pbar.set_postfix(size_GB=f"{wrote/1024**3:.3f}", kept=kept, dropped_tok=dropped_tok)

    return wrote

# ---------------- 入口 ----------------
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DP_DIR, exist_ok=True)

    # 载入共享哈希
    seen = load_hashes_from_file(DEDUP_HASH_FILE)
    print(f"[INFO] Loaded {len(seen)} hashes from {DEDUP_HASH_FILE}")

    # 吸收已有输出，避免上次未刷写导致重复
    absorb_jsonl_texts_into_seen(seen, OUT_JSONL_ZH, "cosmopedia.zh.jsonl")
    absorb_jsonl_texts_into_seen(seen, OUT_JSONL_EN, "cosmopedia.en.jsonl")

    # 1) 中文 ≈40 GB（score>0.7）
    with open(OUT_JSONL_ZH, "ab") as wzh:
        wrote_zh = append_cosmo_cn(wzh, seen, TARGET_ZH_BYTES)
        print(f"[CN] wrote ≈ {wrote_zh/1024**3:.3f} GB → {OUT_JSONL_ZH}")

    # 2) 英文三分支，各 ≈10 GB（500≤tokens≤2000）
    with open(OUT_JSONL_EN, "ab") as wen:
        total_en = 0
        for split, bytes_target in TARGET_EN_SPLITS.items():
            wrote = append_cosmo_en_split(wen, seen, split, bytes_target)
            total_en += wrote
            print(f"[EN:{split}] wrote ≈ {wrote/1024**3:.3f} GB")
        print(f"[EN] total ≈ {total_en/1024**3:.3f} GB → {OUT_JSONL_EN}")

    # 回刷哈希
    save_hashes_to_file(DEDUP_HASH_FILE, seen)
    print(f"[INFO] Dedup cache updated: {DEDUP_HASH_FILE} (size {len(seen)} hashes)")

if __name__ == "__main__":
    main()
