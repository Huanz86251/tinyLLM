# -*- coding: utf-8 -*-
"""
合并三份本地 JSONL，然后从 OpenCSG Chinese-Cosmopedia 追加 score>=0.8 的样本，
直到总文件物理大小约 22.8 GB 停止（硬编码）。
依赖：pip install datasets tqdm regex
"""

import os, json, regex as re
from hashlib import md5
from tqdm import tqdm
from datasets import load_dataset

# ============== 路径全部硬编码，且基于此脚本所在目录计算 ==============
SCRIPT_DIR   = os.path.abspath(os.path.dirname(__file__))          # E:\learn\data_preprocess
REPO_ROOT    = os.path.dirname(SCRIPT_DIR)                         # E:\learn
DATA_DIR     = os.path.join(REPO_ROOT, "data")                     # E:\learn\data
DP_DIR       = os.path.join(REPO_ROOT, "data_preprocess")          # E:\learn\data_preprocess

# 3 个本地源文件（按你截图中的位置硬编码）
LOCAL_CHILDREN = os.path.join(DP_DIR,  "children_stories_2G.jsonl")
LOCAL_CODES    = os.path.join(DP_DIR, "tiny_codes_1GB.jsonl")
LOCAL_MIX      = os.path.join(DP_DIR, "pretrain_mix_ciim.jsonl")

# 输出位置也放在 data 目录下
OUT_PATH    = os.path.join(DP_DIR, "pretrain_supermix_45G.jsonl")

# ============== Cosmopedia 过滤参数（硬编码） ==============
HF_NAME     = "opencsg/chinese-cosmopedia"
HF_SPLIT    = "train"
MIN_SCORE   = 0.84

TARGET_GB   = 45
MIN_CHARS   = 1
MAX_CHARS   = 8000

QUESTION_VALUE = 0          # 你说 QUESTION 固定一个值即可
LOCAL_UNIK_DEFAULT = 0.0    # 本地三份没有 score，用这个缺省值填 UNIK

# ============== 简单清洗 ==============
RE_CONTROL = re.compile(r"[\u200B-\u200F\uFEFF]")
RE_EMOJI   = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u26FF\u2700-\u27BF]")

def clean_text(s: str) -> str:
    if not s: return ""
    s = RE_CONTROL.sub("", s)
    s = RE_EMOJI.sub("", s)
    s = s.replace("\u0000", "")
    return s.strip()

def _write_jsonl(dst_bin, txt: str, unik: float, question: int) -> int:
    rec = {"TXT": txt, "QUESTION": question, "UNIK": float(unik)}
    blob = (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")
    dst_bin.write(blob)
    return len(blob)

# ============== 合并本地 JSONL（转三字段） ==============
def append_local_jsonl(dst_bin, path: str) -> int:
    wrote_bytes = 0
    wrote_rows  = 0
    if not os.path.exists(path):
        print(f"[WARN] 本地文件不存在，跳过：{path}")
        return 0

    with open(path, "r", encoding="utf-8") as f:
        pbar = tqdm(f, desc=f"APPEND(local) {os.path.basename(path)}", unit="rows")
        for line in pbar:
            line = line.strip()
            if not line:
                continue
            # 兼容“每行JSON”与“每行纯文本”
            txt, score = None, None
            try:
                obj = json.loads(line)
                txt   = obj.get("text") or obj.get("TXT") or obj.get("content") or obj.get("raw") or ""
                score = obj.get("score") or obj.get("UNIK")  # 若本地行已带score/UNIK则沿用
            except Exception:
                txt = line

            txt = clean_text(txt)
            if not txt:
                continue
            if not (MIN_CHARS <= len(txt) <= MAX_CHARS):
                continue

            unik = float(score) if isinstance(score, (int, float)) else LOCAL_UNIK_DEFAULT
            wrote_bytes += _write_jsonl(dst_bin, txt, unik, QUESTION_VALUE)
            wrote_rows  += 1

            if wrote_rows % 2000 == 0:
                pbar.set_postfix(rows=wrote_rows, size_GB=f"{wrote_bytes/1024**3:.3f}")

    print(f"[LOCAL] 写出 {wrote_rows} 行，约 {wrote_bytes/1024**3:.3f} GB  ← {os.path.basename(path)}")
    return wrote_bytes

# ============== 追加 Cosmopedia（score→UNIK） ==============
def topup_from_cosmopedia(dst_bin, base_bytes: int, target_bytes: int) -> int:
    ds = load_dataset(HF_NAME, split=HF_SPLIT, streaming=True)
    wrote_bytes = 0
    wrote_rows  = 0
    dropped_score = 0
    dropped_len = 0

    pbar = tqdm(ds, desc=f"TOPUP({HF_NAME} score≥{MIN_SCORE})", unit="rows")
    for ex in pbar:
        if base_bytes + wrote_bytes >= target_bytes:
            break
        txt   = clean_text(ex.get("text", ""))
        score = ex.get("score", None)

        if not txt or score is None:
            continue
        if score < MIN_SCORE:
            dropped_score += 1
            continue
        if not (MIN_CHARS <= len(txt) <= MAX_CHARS):
            dropped_len += 1
            continue

        wrote_bytes += _write_jsonl(dst_bin, txt, float(score), QUESTION_VALUE)
        wrote_rows  += 1

        if wrote_rows % 2000 == 0:
            pbar.set_postfix(size_GB=f"{(base_bytes+wrote_bytes)/1024**3:.3f}",
                             remain_GB=f"{max(0, target_bytes - (base_bytes+wrote_bytes))/1024**3:.3f}")

    print(f"[COSMO] 写出 {wrote_rows} 行，约 {wrote_bytes/1024**3:.3f} GB | 丢弃 score<{MIN_SCORE}: {dropped_score} | 长度不合规: {dropped_len}")
    return wrote_bytes

# ============== 主流程 ==============
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    # 清空输出
    open(OUT_PATH, "wb").close()

    target_bytes = int(TARGET_GB * (1024 ** 3))
    total_bytes  = 0

    with open(OUT_PATH, "ab") as w:
        for p in [LOCAL_CHILDREN, LOCAL_CODES, LOCAL_MIX]:
            total_bytes += append_local_jsonl(w, p)
            print(f"[INFO] 合并进度：{total_bytes/1024**3:.3f} GB / 目标 {TARGET_GB} GB")

    if total_bytes >= target_bytes:
        print(f"[FINAL] 本地合并已达目标，无需补充。总计 {total_bytes/1024**3:.3f} GB → {OUT_PATH}")
        return

    with open(OUT_PATH, "ab") as w:
        total_bytes += topup_from_cosmopedia(w, total_bytes, target_bytes)

    print(f"[FINAL] 合并完成：{total_bytes/1024**3:.3f} GB → {OUT_PATH}")

if __name__ == "__main__":
    main()