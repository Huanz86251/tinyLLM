#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, regex as re
from urllib.parse import urlparse
from tqdm import tqdm
import xxhash
from datasets import load_dataset, Features, Value

# =========================
# 配置常量（可按需改）
# =========================
OUT_PATH           = r"E:\learn\data\pretrain_long_EN.jsonl"
SAVE_HASHSET_PATH  = r"E:\learn\data\pretrain_long_EN.hashes.txt"
HF_CACHE_DIR       = None
USE_STREAMING      = True
TARGET_GB          = 40.0           # 目标体量（计算已存在文件大小，避免超过）
SCORE_MIN          = 3.5
LANG_REQUIRE       = "en"           # fineweb-edu 用；finemath 默认跳过语言过滤
LANG_SCORE_MIN     = 0.5
DEDUPE             = True

# 源 1：FineWeb-Edu（维持你原来的逻辑，可一键关闭避免重复扫描）
ENABLE_FINEWEB_EDU = False
DATASET_NAME_FW    = "HuggingFaceFW/fineweb-edu"
DATASET_SPLIT_FW   = "train"
MIN_TOKENS_FW      = 9000

# 源 2：FineMath（新增；默认启用）
ENABLE_FINEMATH    = True
DATASET_NAME_FM    = "HuggingFaceTB/finemath"
# 推荐先扫更高质量的 4plus；infiwebmath-4plus 也很大（token_count 为 int64）
FINEMATH_CONFIGS   = ["finemath-4plus", "infiwebmath-4plus"]
MIN_TOKENS_FM      = 8500

# 显式声明 features（仅用于 fineweb-edu，避免 schema 漂移引发 CastError）
FEATURES_FW = Features({
    "text":           Value("string"),
    "id":             Value("string"),
    "dump":           Value("string"),
    "url":            Value("string"),
    "file_path":      Value("string"),
    "language":       Value("string"),
    "language_score": Value("float64"),
    "token_count":    Value("int64"),
    "score":          Value("float64"),
    "int_score":      Value("int64"),
})

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

def ensure_outfile(path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not os.path.exists(path):
        # 仅首次创建；不会清空已存在文件
        open(path, "wb").close()

def write_records(ds_iter, *, src_prefix: str, min_tokens: int, use_lang_filter: bool,
                  seen: set, w, wrote_bytes: int, target_bytes: int):
    """通用写入循环。返回 (wrote_rows, wrote_bytes, stats_dict)"""
    wrote_rows  = 0
    drop_tok = drop_score = drop_lang = drop_dup = 0

    pbar = tqdm(ds_iter, desc=f"SCAN({src_prefix})", unit="rows")
    for ex in pbar:
        if wrote_bytes >= target_bytes:
            break

        # 语言筛选（可选）
        if use_lang_filter:
            lang = (ex.get("language") or "").lower()
            if not lang.startswith(LANG_REQUIRE):
                drop_lang += 1
                continue
            ls = float(ex.get("language_score") or 0.0)
            if ls < LANG_SCORE_MIN:
                drop_lang += 1
                continue

        # token_count + score（score 仅对 fineweb-edu；finemath 忽略即可）
        tokc  = int(ex.get("token_count") or 0)
        if tokc < min_tokens:
            drop_tok += 1
            continue

        # fineweb-edu 的质量分筛选
        sc = ex.get("score", None)
        if sc is not None:
            sc = float(sc)
            if use_lang_filter and sc < SCORE_MIN:
                drop_score += 1
                continue
        else:
            sc = 0.0  # finemath 若无 score 则记录 0.0

        # 文本
        raw = ex.get("text") or ""
        if not raw:
            continue
        txt = clean_text(raw)
        if not txt:
            continue

        # 去重
        h = xxhash.xxh3_64_hexdigest(txt)
        if DEDUPE and h in seen:
            drop_dup += 1
            continue
        if DEDUPE:
            seen.add(h)

        # 记录来源：优先 url 的域名；其次显式来源前缀
        url  = ex.get("url") or ""
        dom  = urlparse(url).netloc if url else ""
        src  = f"{src_prefix}|{dom}" if dom else src_prefix

        rec = {
            "TXT": txt,
            "QUESTION": 0,
            "SCORE": sc,
            "SRC": src,
            "HASH": h
        }
        blob = (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")
        w.write(blob)
        wrote_bytes += len(blob)
        wrote_rows  += 1

        if wrote_rows % 1000 == 0:
            pbar.set_postfix(rows=wrote_rows,
                             size=human(wrote_bytes),
                             drop_tok=drop_tok,
                             drop_score=drop_score,
                             drop_lang=drop_lang,
                             drop_dup=drop_dup)

    stats = dict(drop_tok=drop_tok, drop_score=drop_score, drop_lang=drop_lang, drop_dup=drop_dup)
    return wrote_rows, wrote_bytes, stats

def main():
    ensure_outfile(OUT_PATH)

    # 总体目标：以“现有文件大小 + 新增”不超过 TARGET_GB 为界
    target_bytes = int(TARGET_GB * (1024 ** 3))
    wrote_bytes  = os.path.getsize(OUT_PATH)
    print(f"[INFO] 已有文件大小：{human(wrote_bytes)}；目标上限：{human(target_bytes)}")

    wrote_rows_total = 0
    stats_total = dict(drop_tok=0, drop_score=0, drop_lang=0, drop_dup=0)

    seen = load_hashset(SAVE_HASHSET_PATH) if (DEDUPE and SAVE_HASHSET_PATH) else set()

    with open(OUT_PATH, "ab") as w:

        # ====== 1) FineWeb-Edu（可选）======
        if ENABLE_FINEWEB_EDU and wrote_bytes < target_bytes:
            ds_fw = load_dataset(
                DATASET_NAME_FW,
                split=DATASET_SPLIT_FW,
                streaming=USE_STREAMING,
                features=FEATURES_FW,          # 仅对 fineweb-edu 施加 features
                trust_remote_code=False,
                cache_dir=HF_CACHE_DIR,
            )
            rows, wrote_bytes, st = write_records(
                ds_fw,
                src_prefix="fineweb-edu",
                min_tokens=MIN_TOKENS_FW,
                use_lang_filter=True,
                seen=seen, w=w,
                wrote_bytes=wrote_bytes, target_bytes=target_bytes
            )
            wrote_rows_total += rows
            for k in stats_total: stats_total[k] += st[k]

        # ====== 2) FineMath（新增，默认启用）======
        if ENABLE_FINEMATH and wrote_bytes < target_bytes:
            for cfg in FINEMATH_CONFIGS:
                if wrote_bytes >= target_bytes:
                    break
                # 注意：finemath 不要强制 features；不同配置的 dtypes 略有差异
                ds_fm = load_dataset(
                    DATASET_NAME_FM,
                    cfg,
                    split="train",
                    streaming=USE_STREAMING,
                    trust_remote_code=False,
                    cache_dir=HF_CACHE_DIR,
                )
                rows, wrote_bytes, st = write_records(
                    ds_fm,
                    src_prefix=f"finemath:{cfg}",
                    min_tokens=MIN_TOKENS_FM,
                    use_lang_filter=False,   # 按你要求：只看 token_count，不做语言过滤
                    seen=seen, w=w,
                    wrote_bytes=wrote_bytes, target_bytes=target_bytes
                )
                wrote_rows_total += rows
                for k in stats_total: stats_total[k] += st[k]

    if DEDUPE and SAVE_HASHSET_PATH:
        save_hashset(SAVE_HASHSET_PATH, seen)

    print(f"[FINAL] 新增 {wrote_rows_total} 行，当前总大小 {human(os.path.getsize(OUT_PATH))} → {OUT_PATH}")
    print(f"        丢弃汇总：tok → {stats_total['drop_tok']}，score → {stats_total['drop_score']}，"
          f"lang → {stats_total['drop_lang']}，dup → {stats_total['drop_dup']}")

if __name__ == "__main__":
    main()
