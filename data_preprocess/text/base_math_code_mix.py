#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 3 个 HF 数据集抽取并合并为统一 JSONL（每行：{"TXT","QUESTION","UNIK"}）：
1) nvidia/OpenCodeInstruct：
   - 过滤：average_test_score == 1
   - TXT = "Instruction:\n{input}\n\nAnswer:\n{output}"
   - UNIK = average_test_score (1.0)

2) nvidia/OpenMathInstruct-2：
   - 目标体量：~4 GiB（可改）
   - 过滤：仅保留 expected_answer 被 generated_solution “包含” 的样本
           （对双方做大小写/空白/LaTeX 等鲁棒归一化后再判断）
   - 长度：<= 2048 tokens（若提供 TOKENIZER_DIR），否则字符近似阈值
   - TXT = "Problem:\n{problem}\n\nAnswer:\n{generated_solution}"
   - UNIK = 0.0（你可自行改为 1.0 表示“已校验”）

3) opencsg/chinese-cosmopedia：
   - 过滤：score >= COSMO_MIN_SCORE（默认 0.82）
   - 长度：同上
   - TXT = text
   - UNIK = score

附：
- 去重：加载一个“已有 JSONL”（通常是你之前合并好的 pretrain_500Mmodel.jsonl），
        对清洗归一化后的 TXT 做 MD5，避免重复。会把已见 MD5 另存到 .hashes.txt，
        下次可直接加载加速。

依赖：
  pip install -U datasets tqdm regex transformers
"""

import os, json, math, regex as re
from hashlib import md5
from typing import Optional, Iterable, Union, List, Dict, Any
from tqdm import tqdm
from datasets import load_dataset

# ===================== 路径（基于脚本位置） =====================
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(REPO_ROOT, "data")
DP_DIR     = os.path.join(REPO_ROOT, "data_preprocess")

# 既有 JSONL（用于“查虫”去重）
DEDUP_BASE_JSONL = os.path.join(DP_DIR, "pretrain_500Mmodel.jsonl")   # 你之前生成的那个
DEDUP_HASH_FILE  = DEDUP_BASE_JSONL + ".hashes.txt"                   # 可选：缓存 MD5
# 输出
OUT_JSONL        = os.path.join(DP_DIR, "pretrain_500Mmodel.from_hf.jsonl")

# ===================== Token 过滤设置 =====================
# 若提供 TOKENIZER_DIR（HF 本地快照或模型名），则用精确 tokens 计数（<=2048）
TOKENIZER_DIR: Optional[str] = None  # 例：r"E:\tok\minicpm3_4b" 或 "/root/.../tokenizer"
MAX_TOKENS = 2048

# 未提供 tokenizer 时，采用字符上限近似 2k tokens（保守）
MAX_CHARS_FOR_2K_TOK = 6000

# ===================== 数据集与预算 =====================
# OpenCodeInstruct
CODE_DATASET    = "nvidia/OpenCodeInstruct"
CODE_SPLIT      = "train"
CODE_SCORE_EQ   = 1.0
CODE_TARGET_GB  = 8.0        # 可调；None 表示不设预算上限
CODE_JOIN_TPL   = "Instruction:\n{inp}\n\nAnswer:\n{out}"

# OpenMathInstruct-2
MATH_DATASET    = "nvidia/OpenMathInstruct-2"
MATH_SPLIT      = "train"
MATH_TARGET_GB  = 8.0
MATH_JOIN_TPL   = "Problem:\n{inp}\n\nAnswer:\n{out}"
MATH_UNIK       = 0.0        # 你想标“已校验”也可改成 1.0

# Chinese-Cosmopedia
COSMO_DATASET   = "opencsg/chinese-cosmopedia"
COSMO_SPLIT     = "train"
COSMO_MIN_SCORE = 0.82
COSMO_TARGET_GB = 40.0       # 你可改到 40.0

QUESTION_VALUE  = 0          # 固定 QUESTION

# ===================== 文本清洗与归一化 =====================
RE_CONTROL = re.compile(r"[\u200B-\u200F\uFEFF]")
RE_EMOJI   = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u26FF\u2700-\u27BF]")

def clean_text(s: str) -> str:
    if not s: return ""
    s = RE_CONTROL.sub("", s)
    s = RE_EMOJI.sub("", s)
    s = s.replace("\u0000", "")
    return s.strip()

def normalize_light(s: str) -> str:
    """轻量统一：去控制符/emoji，折叠空白，统一换行；用于去重签名。"""
    s = clean_text(s)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"[ \t]*\n[ \t]*", "\n", s).strip()
    return s

def md5_of_text(s: str) -> str:
    return md5(s.encode("utf-8")).hexdigest()

# ===================== 精确 token 计数（可选） =====================
_tokenizer = None
def maybe_load_tokenizer():
    global _tokenizer
    if TOKENIZER_DIR and _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, trust_remote_code=True)

def length_ok(txt: str) -> bool:
    if TOKENIZER_DIR:
        maybe_load_tokenizer()
        n_tok = len(_tokenizer.encode(txt, add_special_tokens=False))
        return n_tok <= MAX_TOKENS
    else:
        return len(txt) <= MAX_CHARS_FOR_2K_TOK

# ===================== Math 答案包含性判断 =====================
# 目标：判断 expected_answer “出现在” generated_solution 中。
# 做法：对双方做鲁棒归一化（兼容常见 LaTeX 包裹/命令、空白、大小写），再做子串判断。
RE_LATEX_CMD   = re.compile(r"\\[a-zA-Z]+(\s*\{[^{}]*\})?")
RE_LATEX_BOXED = re.compile(r"\\boxed\s*\{([^{}]*)\}")
RE_LATEX_LEFT_RIGHT = re.compile(r"\\(left|right)[\(\)\[\]\{\}\|]")

def norm_math_string(x: str) -> str:
    """针对数学字符串的鲁棒归一化以提升包含判断的召回：
       - 去 \boxed{...} 外层，仅保留内部
       - 去 \left \right 标记
       - 去常见 LaTeX 命令（保留内容）
       - 去美元符、空白、标点，仅保留字母数字与中文（和常见希腊字母英文名）
       - 全部 lower
    """
    if not x:
        return ""
    s = x
    # 展开 \boxed{...}（保留内部）
    s = RE_LATEX_BOXED.sub(lambda m: m.group(1), s)
    # 去掉 \left \right 括号标记
    s = RE_LATEX_LEFT_RIGHT.sub("", s)
    # 去美元符
    s = s.replace("$", "")
    # 去一般 LaTeX 命令（尽量不吞内容）
    s = RE_LATEX_CMD.sub("", s)
    # 去空白与标点，只留字母数字与中文
    s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", s)
    return s.lower()

def contains_expected_answer(solution: str, expected: Any) -> bool:
    """expected 可能是 str / 数值 / 列表 / 字典；尽力从中提取候选答案串，做包含判断。"""
    sol_n = norm_math_string(str(solution))
    if not sol_n:
        return False

    # 把 expected 规一化为若干候选字符串
    candidates: List[str] = []

    def push_candidate(v: Any):
        if v is None: return
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            candidates.append(norm_math_string(str(v)))
        elif isinstance(v, str):
            candidates.append(norm_math_string(v))
        elif isinstance(v, (list, tuple, set)):
            for item in v:
                push_candidate(item)
        elif isinstance(v, dict):
            # 常见字段尝试
            for k in ["answer", "value", "text", "final", "expected_answer"]:
                if k in v:
                    push_candidate(v[k])
            # 兜底：把所有值都尝试
            for vv in v.values():
                push_candidate(vv)
        else:
            candidates.append(norm_math_string(str(v)))

    push_candidate(expected)
    # 去空去重
    cand_set = {c for c in candidates if c}

    # 多策略：只要有一个候选是 solution 归一化串的子串，就认为“包含”
    for c in cand_set:
        if c and c in sol_n:
            return True
    return False

# ===================== IO 与去重 =====================
def write_record(f, txt: str, unik: float, q: int = QUESTION_VALUE) -> int:
    rec = {"TXT": txt, "QUESTION": q, "UNIK": float(unik)}
    blob = (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")
    f.write(blob)
    return len(blob)

def load_hashes_from_file(path: str) -> set:
    s = set()
    if not os.path.exists(path): return s
    with open(path, "r", encoding="utf-8") as r:
        for line in r:
            h = line.strip()
            if h: s.add(h)
    return s

def save_hashes_to_file(path: str, hashes: Iterable[str]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as w:
        for h in hashes:
            w.write(h + "\n")
    os.replace(tmp, path)

def build_or_load_dedup_set() -> set:
    seen = set()
    # 1) 既有 hash 缓存
    if os.path.exists(DEDUP_HASH_FILE):
        seen |= load_hashes_from_file(DEDUP_HASH_FILE)

    # 2) 既有 JSONL（前次总表）
    if os.path.exists(DEDUP_BASE_JSONL):
        with open(DEDUP_BASE_JSONL, "r", encoding="utf-8") as r:
            for line in tqdm(r, desc=f"Load dedup base: {os.path.basename(DEDUP_BASE_JSONL)}", unit="rows"):
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    txt = obj.get("TXT") or obj.get("text") or ""
                except Exception:
                    txt = line
                norm = normalize_light(txt)
                if norm:
                    seen.add(md5_of_text(norm))

    # 3) 若当前输出文件已存在，也纳入去重
    if os.path.exists(OUT_JSONL):
        with open(OUT_JSONL, "r", encoding="utf-8") as r:
            for line in tqdm(r, desc=f"Load self dedup: {os.path.basename(OUT_JSONL)}", unit="rows"):
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    txt = obj.get("TXT") or obj.get("text") or ""
                except Exception:
                    txt = line
                norm = normalize_light(txt)
                if norm:
                    seen.add(md5_of_text(norm))

    # 更新缓存文件
    save_hashes_to_file(DEDUP_HASH_FILE, seen)
    return seen

# ===================== 各数据集抽取 =====================
def append_opencode(dst, seen: set, target_bytes: Optional[int]) -> int:
    ds = load_dataset(CODE_DATASET, split=CODE_SPLIT, streaming=True)
    wrote = 0
    pbar = tqdm(ds, desc=f"OPENCODE perfect=={CODE_SCORE_EQ}", unit="rows")
    for ex in pbar:
        if target_bytes is not None and wrote >= target_bytes:
            break

        inp   = (ex.get("input")  or "").strip()
        out   = (ex.get("output") or "").strip()
        score = ex.get("average_test_score", None)

        try:
            score_f = float(score) if score is not None else None
        except Exception:
            score_f = None

        if score_f is None or abs(score_f - CODE_SCORE_EQ) > 1e-9:
            continue
        if not inp or not out:
            continue

        txt = clean_text(CODE_JOIN_TPL.format(inp=inp, out=out))
        if not txt or not length_ok(txt):
            continue

        sig = md5_of_text(normalize_light(txt))
        if sig in seen:
            continue

        wrote += write_record(dst, txt, unik=score_f, q=QUESTION_VALUE)
        seen.add(sig)

        if wrote % (2<<20) < 1000:
            pbar.set_postfix(size_GB=f"{wrote/1024**3:.3f}")
    return wrote

def append_openmath2(dst, seen: set, target_bytes: int) -> int:
    ds = load_dataset(MATH_DATASET, split=MATH_SPLIT, streaming=True)
    wrote = 0
    kept, dropped_nohit, dropped_len, dropped_empty = 0, 0, 0, 0

    pbar = tqdm(ds, desc=f"OPENMATH2 target≈{target_bytes/1024**3:.1f}GB", unit="rows")
    for ex in pbar:
        if wrote >= target_bytes:
            break

        problem = (ex.get("problem") or "").strip()
        sol     = (ex.get("generated_solution") or "").strip()
        exp     = ex.get("expected_answer", None)

        if not problem or not sol:
            dropped_empty += 1
            continue

        # 仅当 generated_solution “包含” expected_answer 时保留
        if not contains_expected_answer(sol, exp):
            dropped_nohit += 1
            continue

        txt = clean_text(MATH_JOIN_TPL.format(inp=problem, out=sol))
        if not txt or not length_ok(txt):
            dropped_len += 1
            continue

        sig = md5_of_text(normalize_light(txt))
        if sig in seen:
            continue

        wrote += write_record(dst, txt, unik=MATH_UNIK, q=QUESTION_VALUE)
        seen.add(sig)
        kept += 1

        if wrote % (2<<20) < 1000:
            pbar.set_postfix(size_GB=f"{wrote/1024**3:.3f}",
                             kept=kept, nohit=dropped_nohit, too_long=dropped_len, empty=dropped_empty)
    return wrote

def append_cosmopedia(dst, seen: set, target_bytes: int) -> int:
    ds = load_dataset(COSMO_DATASET, split=COSMO_SPLIT, streaming=True)
    wrote = 0
    dropped_score, dropped_len = 0, 0
    pbar = tqdm(ds, desc=f"COSMO score>={COSMO_MIN_SCORE}", unit="rows")
    for ex in pbar:
        if wrote >= target_bytes:
            break

        txt   = clean_text(ex.get("text", ""))
        score = ex.get("score", None)
        try:
            score_f = float(score) if score is not None else None
        except Exception:
            score_f = None

        if not txt or score_f is None:
            continue
        if score_f < COSMO_MIN_SCORE:
            dropped_score += 1
            continue
        if not length_ok(txt):
            dropped_len += 1
            continue

        sig = md5_of_text(normalize_light(txt))
        if sig in seen:
            continue

        wrote += write_record(dst, txt, unik=score_f, q=QUESTION_VALUE)
        seen.add(sig)

        if wrote % (2<<20) < 1000:
            pbar.set_postfix(size_GB=f"{wrote/1024**3:.3f}",
                             dropped_score=dropped_score, too_long=dropped_len)
    return wrote

# ===================== 主流程 =====================
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DP_DIR, exist_ok=True)

    # 载入/构建去重集
    seen = build_or_load_dedup_set()

    # 追加写出
    with open(OUT_JSONL, "ab") as w:
        # 1) OpenCodeInstruct（perfect only）
        code_bytes_target = None if CODE_TARGET_GB is None else int(CODE_TARGET_GB * (1024 ** 3))
        code_bytes = append_opencode(w, seen, code_bytes_target)
        print(f"[OPENCODE] wrote {code_bytes/1024**3:.3f} GB (perfect==1) → {OUT_JSONL}")

        # 2) OpenMathInstruct-2（仅“包含 expected_answer”的样本，~4GB）
        math_bytes = append_openmath2(w, seen, int(MATH_TARGET_GB * (1024 ** 3)))
        print(f"[OPENMATH2] wrote {math_bytes/1024**3:.3f} GB (with expected_answer containment) → {OUT_JSONL}")

        # 3) Chinese-Cosmopedia（score 过滤，~30GB）
        cosmo_bytes = append_cosmopedia(w, seen, int(COSMO_TARGET_GB * (1024 ** 3)))
        print(f"[COSMO] wrote {cosmo_bytes/1024**3:.3f} GB → {OUT_JSONL}")

    # 保存最新的去重哈希缓存
    save_hashes_to_file(DEDUP_HASH_FILE, seen)

    total = (code_bytes or 0) + (math_bytes or 0) + (cosmo_bytes or 0)
    print(f"[FINAL] total appended ≈ {total/1024**3:.3f} GB → {OUT_JSONL}")

if __name__ == "__main__":
    main()
