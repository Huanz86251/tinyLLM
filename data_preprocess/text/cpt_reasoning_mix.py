#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_cpt_stage_dataset.py

生产一个新的 CPT 训练语料 JSONL，比如 pretrain_500Mmodel.cpt_stage.jsonl
内容按顺序包含六部分（写入顺序已调成先思维链）：

(1) Jackrong/Chinese-Qwen3-235B-Thinking-2507-Distill-100k
    - Input            -> user
    - CoT_content      -> inner_thought
    - Answer_content   -> assistant
    - 渲染成聊天+<|thought_start|>...<|thought_end|>
    - 语言过滤 (仅中/英，允许常规符号/数学/工程单位符号等)
    - 渲染后整条样本的字符数 <= COT_MAX_CHARS (近似单窗口2048)
    - 超长/含其它语种 -> 丢整条 (不截断)
    - 不做去重
    - 给小模型中文为主的“先想->再答”范式
    - UNIK = QWEN_UNIK (strict)

(2) fnlp/Ultra-Innerthought
    - 只保留 data_source in {"qwq_500k_zh","qwq_500k_en"}
    - 对每个 turn:
        user / inner_thought / assistant
      渲染成最终格式
      同样按语言过滤+整条字符数 <= COT_MAX_CHARS
      超长 or 含其它语种 -> 丢整条
    - 不做去重
    - UNIK = INNER_UNIK (strict)

(3) NVIDIA OpenCodeInstruct
    - average_test_score == 1.0 的样本
    - "Instruction:\\n{input}\\n\\nAnswer:\\n{output}"
    - 不做去重
    - UNIK = average_test_score (1.0)

(4) NVIDIA OpenMathInstruct-2
    - 仅保留 generated_solution 确实包含 expected_answer 的样本
    - "Problem:\\n{problem}\\n\\nAnswer:\\n{generated_solution}"
    - contains_expected_answer() 做松弛匹配
    - 高质量数学推理+答案
    - UNIK = MATH_UNIK

(5) Open-Orca/OpenOrca
    - question -> user
    - response -> assistant
    - 主打英文推理/Step-by-step
    - UNIK = ORCA_UNIK (strict)

(6) opencsg/chinese-cosmopedia
    - 中文百科 text，带 score
    - score >= COSMO_MIN_SCORE
    - 长度不过长
    - 去重：normalize_light+md5
    - UNIK = score_f (通常 <3.0, 走 non-strict)

最终输出行形如：
{"TXT": "...", "QUESTION": 0, "UNIK": <float>}

后续训练阶段：
- 你可以用 UNIK >= 3.0 来识别“strict样本”，这些样本就整条打包+padding，不和别的拼接。
- 其他样本(百科/代码)可以高吞吐拼接再切。

依赖:
    pip install -U datasets tqdm regex transformers huggingface_hub
"""

import os, json, time, regex as re, unicodedata
from hashlib import md5
from typing import Optional, Iterable, List, Any
from tqdm import tqdm
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar

disable_progress_bar()  # 我们自己用 tqdm

# ===================== 目录和文件名 =====================
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)

DATA_DIR   = os.path.join(REPO_ROOT, "data")
DP_DIR     = os.path.join(REPO_ROOT, "data_preprocess")

# 历史大 JSONL，用来构建百科去重基线
HIST_JSONLS = [
    os.path.join(DP_DIR, "pretrain_500Mmodel.jsonl"),
    os.path.join(DP_DIR, "pretrain_500Mmodel.from_hf.jsonl"),
]

# 输出文件
OUT_JSONL_NEW = os.path.join(DP_DIR, "pretrain_500Mmodel.cpt_stage.jsonl")

# 百科去重缓存
DEDUP_HASH_FILE = OUT_JSONL_NEW + ".hashes.txt"

# ===================== 长度限制 =====================

# fallback近似：没 tokenizer 时我们用字符限制
TOKENIZER_DIR: Optional[str] = None   # 如果你想用真实tokenizer计数，将其设为模型名或本地路径
MAX_TOKENS = 2048
MAX_CHARS_FOR_2K_TOK = 6000           # ~2048 token 的粗略字符上限
QUESTION_VALUE  = 0

# 对“思维链类样本”(Qwen, Ultra等)的硬长限制（字符计数）
# 整条 (user + thought + answer) 渲染后必须 <= 2048 字符
COT_MAX_CHARS = 2048

# ===================== 数据集配置 & 超参 =====================

# (1) Jackrong/Chinese-Qwen3-235B-Thinking-2507-Distill-100k
QWEN_DATASET    = "Jackrong/Chinese-Qwen3-235B-Thinking-2507-Distill-100k"
QWEN_SPLIT      = "train"
QWEN_TARGET_GB  = 8.0
QWEN_UNIK       = 3.0  # strict

# (2) fnlp/Ultra-Innerthought
ULTRA_DATASET   = "fnlp/Ultra-Innerthought"
ULTRA_SPLIT     = "train"
ULTRA_TARGET_GB = 8.0
INNER_UNIK      = 3.0  # strict
KEEP_ULTRA_SOURCES = {"qwq_500k_zh", "qwq_500k_en"}

# (3) NVIDIA OpenCodeInstruct
CODE_DATASET    = "nvidia/OpenCodeInstruct"
CODE_SPLIT      = "train"
CODE_SCORE_EQ   = 1.0
CODE_TARGET_GB  = 8.0               # None 表示不给上限
CODE_JOIN_TPL   = "Instruction:\n{inp}\n\nAnswer:\n{out}"

# (4) NVIDIA OpenMathInstruct-2
MATH_DATASET    = "nvidia/OpenMathInstruct-2"
MATH_SPLIT      = "train"
MATH_TARGET_GB  = 12.0
MATH_JOIN_TPL   = "Problem:\n{inp}\n\nAnswer:\n{out}"
MATH_UNIK       = 2.0

# (5) Open-Orca/OpenOrca
ORCA_DATASET    = "Open-Orca/OpenOrca"
ORCA_SPLIT      = "train"
ORCA_TARGET_GB  = 8.0
ORCA_UNIK       = 3.0               # strict

# (6) opencsg/chinese-cosmopedia
COSMO_DATASET   = "opencsg/chinese-cosmopedia"
COSMO_SPLIT     = "train"
COSMO_MIN_SCORE = 0.80
COSMO_TARGET_GB = 45.0              # 你想追加多少百科
# 这部分唯一做去重

# ===================== 清洗 / 归一化 / MD5 =====================
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
    """
    百科去重：去零宽/emoji，合并多余空白，压紧换行
    """
    s = clean_text(s)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"[ \t]*\n[ \t]*", "\n", s).strip()
    return s

def md5_of_text(s: str) -> str:
    return md5(s.encode("utf-8")).hexdigest()

# ===================== tokenizer 长度判断（给非思维链类用） =====================
_tokenizer = None
def maybe_load_tokenizer():
    global _tokenizer
    if TOKENIZER_DIR and _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_DIR,
            trust_remote_code=True
        )

def length_ok_general(txt: str) -> bool:
    """
    通用长度过滤：要求整体不超过 ~2048 tokens.
    如果没给 tokenizer，就用字符近似上限。
    用于 code / math / orca / cosmopedia 这些普通样本。
    """
    if TOKENIZER_DIR:
        maybe_load_tokenizer()
        n_tok = len(_tokenizer.encode(txt, add_special_tokens=False))
        return n_tok <= MAX_TOKENS
    else:
        return len(txt) <= MAX_CHARS_FOR_2K_TOK

def length_ok_cot(txt: str) -> bool:
    """
    专门给思维链样本（Qwen, Ultra等）用的长度过滤：
    - 渲染完整(user + thought + answer)后
    - 要求字符数 <= COT_MAX_CHARS
    """
    return len(txt) <= COT_MAX_CHARS

# ===================== 数学答案包含性判断 =====================
RE_LATEX_CMD           = re.compile(r"\\[a-zA-Z]+(\s*\{[^{}]*\})?")
RE_LATEX_BOXED         = re.compile(r"\\boxed\s*\{([^{}]*)\}")
RE_LATEX_LEFT_RIGHT    = re.compile(r"\\(left|right)[\(\)\[\]\{\}\|]")

def norm_math_string(x: str) -> str:
    """
    针对数学解答做鲁棒归一化，去掉 LaTeX 控制符，只保留核心符号。
    """
    if not x:
        return ""
    s = x
    s = RE_LATEX_BOXED.sub(lambda m: m.group(1), s)
    s = RE_LATEX_LEFT_RIGHT.sub("", s)
    s = s.replace("$", "")
    s = RE_LATEX_CMD.sub("", s)
    s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", s)
    return s.lower()

def contains_expected_answer(solution: str, expected: Any) -> bool:
    """
    只要 expected_answer 的任一候选出现在 solution 的归一化串里，就算“正确”
    """
    sol_n = norm_math_string(str(solution))
    if not sol_n:
        return False

    candidates: List[str] = []

    def push_candidate(v: Any):
        if v is None:
            return
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            candidates.append(norm_math_string(str(v)))
        elif isinstance(v, str):
            candidates.append(norm_math_string(v))
        elif isinstance(v, (list, tuple, set)):
            for item in v:
                push_candidate(item)
        elif isinstance(v, dict):
            # 常见可能字段
            for k in ["answer", "value", "text", "final", "expected_answer"]:
                if k in v:
                    push_candidate(v[k])
            for vv in v.values():
                push_candidate(vv)
        else:
            candidates.append(norm_math_string(str(v)))

    push_candidate(expected)
    cand_set = {c for c in candidates if c}

    for c in cand_set:
        if c and c in sol_n:
            return True
    return False

# ===================== 语言过滤（新版） =====================
# 目标：
#   - 允许中文 + 英文
#   - 允许中文标点、全角符号、智能引号、省略号、破折号、圈号数字、上标/下标、数学/物理/电学符号、工程单位符号
#   - Ban 其他语言脚本（阿拉伯文、日文、韩文、印地语/天城文、西里尔文、带重音的西/法/葡/意/德语字母等）
#
# 说明：
#   - 我们显式把一些物理/化学/电学里常用单位也列入安全符号，比如:
#       µ (micro sign U+00B5), Ω/Ω 类符号在希腊/letterlike里已经允许,
#       ℃, Å (埃/Ångström),
#       上下标、箭头、近似号、度数符号等等
#   - 这些不会触发“外语”判定
#
def is_symbol_we_want(ch: str) -> bool:
    code = ord(ch)

    # 空白/换行等
    if ch in ("\n", "\r", "\t", " "):
        return True

    # General Punctuation: “ ” ‘ ’ … — – • ‰ ‱ 等
    if 0x2000 <= code <= 0x206F:
        return True

    # 上标/下标：¹²³⁺⁻ ₀₁₂... (指数, 化学式, 变量下标)
    if 0x2070 <= code <= 0x209F:
        return True

    # 货币符号区: €, ₦, … (不当成外语)
    if 0x20A0 <= code <= 0x20CF:
        return True

    # Letterlike Symbols / 单位：℃ ℓ Ω 等 (U+2100–U+214F)
    if 0x2100 <= code <= 0x214F:
        return True

    # Number Forms: ½ ¼ ⅓ ⅔ ¾ … (U+2150–U+218F)
    if 0x2150 <= code <= 0x218F:
        return True

    # 箭头 / 数学运算符 / 集合符号 / 关系符号 (U+2190–U+22FF)
    # 这块还包括 ≤ ≥ ∑ ∫ ≈ → ↦ ⋅ ⋯ 之类，常见在数学/物理推导
    if 0x2190 <= code <= 0x22FF:
        return True

    # 带圈数字、序号等 (①②③… ⒶⒷ…) U+2460–U+24FF
    if 0x2460 <= code <= 0x24FF:
        return True

    # 几何形状/示意符号 (□ ○ △ ■ ◆ ▶ …) U+25A0–U+25FF
    if 0x25A0 <= code <= 0x25FF:
        return True

    # 变体选择符 (FE0F等)，主要是 emoji 颜色/样式，不当成外语
    if 0xFE00 <= code <= 0xFE0F:
        return True

    # 我们手动白名单一些工程/物理里高频的单位和符号
    # µ (micro sign U+00B5), Å (Angstrom U+00C5),
    # ℃ U+2103, 度数符号°, 正负号±, 乘除×÷, 中点·, 微符号µ, etc.
    if ch in [
        "°", "±", "×", "÷", "·", "℃", "ℓ", "µ", "Å",
        "½", "¼", "⅓", "⅔", "⅛", "¾",
        "○", "△", "□", "●", "■", "◆", "▶", "※",
    ]:
        return True

    return False

def is_basic_chinese_or_english(ch: str) -> bool:
    code = ord(ch)

    # ASCII 可打印字符（英文、数字、英文基础标点）
    if 0x20 <= code <= 0x7E:
        return True

    # CJK 统一汉字区
    if 0x4E00 <= code <= 0x9FFF:
        return True

    # CJK 符号标点（、。！？《》…… 等）
    if 0x3000 <= code <= 0x303F:
        return True

    # 全角 ASCII / 全角标点 / 全角数字 / 全角字母
    if 0xFF00 <= code <= 0xFFEF:
        return True

    # 希腊字母区 (α β γ θ μ Ω ...)，数学/电学/物理里频繁使用
    if 0x0370 <= code <= 0x03FF:
        return True

    return False

def is_foreign_language_char(ch: str) -> bool:
    """
    返回 True 表示这是“我们不想要的(非中英)语言字符”。
    出现任何一个 → 整条样本丢。
    """

    code = ord(ch)

    # ----------- 明确 ban 的脚本 -----------

    # 日文平假名
    if 0x3040 <= code <= 0x309F:
        return True
    # 日文片假名
    if 0x30A0 <= code <= 0x30FF:
        return True
    # 半角片假名 (位于全角块里一部分)
    if 0xFF65 <= code <= 0xFF9F:
        return True

    # 韩文
    if 0x1100 <= code <= 0x11FF:   # Hangul Jamo
        return True
    if 0x3130 <= code <= 0x318F:   # Hangul Compatibility Jamo
        return True
    if 0xAC00 <= code <= 0xD7AF:   # Hangul Syllables
        return True

    # 阿拉伯文 (含基本区、扩展区、连写区)
    if 0x0600 <= code <= 0x06FF:
        return True
    if 0x0750 <= code <= 0x077F:
        return True
    if 0x08A0 <= code <= 0x08FF:
        return True
    if 0xFB50 <= code <= 0xFDFF:
        return True
    if 0xFE70 <= code <= 0xFEFF:
        return True

    # 天城文 / Devanagari (印地语等)
    if 0x0900 <= code <= 0x097F:
        return True

    # 西里尔文 (俄语等)
    if 0x0400 <= code <= 0x04FF:
        return True

    # ----------- 拉丁扩展带重音字母 -----------
    # Latin-1 Supplement: U+00A0–U+00FF
    # Latin Extended-A:   U+0100–U+017F
    # Latin Extended-B:   U+0180–U+024F
    #
    # 这里面既有 é ñ ç ü 这种（我们要 ban，因为它们就是法/西/葡/意/德语字母），
    # 也有一些技术用符号，比如 µ (micro sign)、Å (Angstrom)。
    # 我们在 is_symbol_we_want() 里已经把 µ 和 Å 白名单了。
    # 这里的判定：只有当它是“字母类字符”并且不是我们白名单的工程符号，才 ban。
    if 0x00A0 <= code <= 0x024F:
        # 如果它已经在符号白名单里，比如 µ, Å，就别判成外语
        if is_symbol_we_want(ch):
            return False
        cat = unicodedata.category(ch)
        if cat.startswith("L"):  # Letter (Lu/Ll/Lt/Lm/Lo)
            return True

    # ----------- 如果没命中以上 ban 区，就判断它是不是我们允许的中/英/符号 -----------

    if is_basic_chinese_or_english(ch):
        return False

    if is_symbol_we_want(ch):
        return False

    # 剩下极小概率的奇怪符号：默认不视为外语
    return False

def text_has_foreign_language(txt: str) -> bool:
    for ch in txt:
        if is_foreign_language_char(ch):
            return True
    return False

# ===================== 渲染到 tokenizer 聊天模板 =====================
def render_qwen_turn(inp: str, cot: str, ans: str) -> str:
    """
    Qwen:
    <|im_start|>user
    {Input}
    <|im_end|>
    <|im_start|>assistant
    <|thought_start|>
    {CoT_content}
    <|thought_end|>
    {Answer_content}
    <|im_end|>
    """
    inp = clean_text(inp)
    cot = clean_text(cot)
    ans = clean_text(ans)

    return (
        "<|im_start|>user\n"
        f"{inp}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<|thought_start|>\n"
        f"{cot}\n"
        "<|thought_end|>\n"
        f"{ans}\n"
        "<|im_end|>"
    ).strip()

def render_ultra_turn(user_msg: str, inner_thought: str, assistant_msg: str) -> str:
    """
    Ultra-Innerthought:
    <|im_start|>user
    {user}
    <|im_end|>
    <|im_start|>assistant
    <|thought_start|>
    {inner_thought}
    <|thought_end|>
    {assistant}
    <|im_end|>
    """
    user_msg      = clean_text(user_msg)
    inner_thought = clean_text(inner_thought)
    assistant_msg = clean_text(assistant_msg)

    return (
        "<|im_start|>user\n"
        f"{user_msg}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<|thought_start|>\n"
        f"{inner_thought}\n"
        "<|thought_end|>\n"
        f"{assistant_msg}\n"
        "<|im_end|>"
    ).strip()

def render_code_sample(inp: str, out: str) -> str:
    return clean_text(
        CODE_JOIN_TPL.format(inp=inp, out=out)
    )

def render_math_sample(problem: str, sol: str) -> str:
    return clean_text(
        MATH_JOIN_TPL.format(inp=problem, out=sol)
    )

def render_orca_turn(question: str, response: str) -> str:
    """
    OpenOrca:
    <|im_start|>user
    {question}
    <|im_end|>
    <|im_start|>assistant
    {response}
    <|im_end|>
    """
    question = clean_text(question)
    response = clean_text(response)
    return (
        "<|im_start|>user\n"
        f"{question}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{response}\n"
        "<|im_end|>"
    ).strip()

# ===================== IO & 哈希缓存 =====================
def write_record(f, txt: str, unik: float, q: int = QUESTION_VALUE) -> int:
    rec = {"TXT": txt, "QUESTION": q, "UNIK": float(unik)}
    blob = (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")
    f.write(blob)
    return len(blob)

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
    """
    扫描一个历史 JSONL，把每行 TXT（或 text fallback）做 normalize_light -> md5，
    存进 seen，用作百科去重基线。
    """
    if not os.path.exists(jsonl_path):
        return
    with open(jsonl_path, "r", encoding="utf-8") as r:
        for line in tqdm(r, desc=f"Absorb (dedup baseline) {desc}", unit="rows"):
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

def build_seen_for_cosmo(hist_files: List[str], hash_cache_file: str) -> set:
    """
    构建/加载 百科去重黑名单 seen_cosmo.

    逻辑：
    1) 如果 hash_cache_file 已经存在：
          - 直接读缓存返回
    2) 否则：
          - 扫描 HIST_JSONLS，把所有 TXT 做 md5
          - 保存到 hash_cache_file
    """
    if os.path.exists(hash_cache_file):
        seen = load_hashes_from_file(hash_cache_file)
        print(f"[INFO] Loaded {len(seen)} hashes from cache {hash_cache_file}, "
              f"skipping full rescan of HIST_JSONLS.")
        return seen

    # 首次构建
    seen = set()
    for p in hist_files:
        absorb_jsonl_texts_into_seen(seen, p, desc=os.path.basename(p))

    save_hashes_to_file(hash_cache_file, seen)
    print(f"[INFO] Built {len(seen)} hashes from HIST_JSONLS and cached to {hash_cache_file}")
    return seen

# ===================== Hugging Face 安全加载 (带重试+容错) =====================
def safe_streaming_dataset(dataset_name: str, split: str,
                           max_tries: int = 5,
                           delay_sec: float = 5.0):
    """
    多次尝试 load_dataset(..., streaming=True)。
    成功 -> 返回可迭代数据集。
    失败 -> 打印告警并返回 None
    """
    last_err = None
    for attempt in range(max_tries):
        try:
            ds = load_dataset(dataset_name, split=split, streaming=True)
            return ds
        except Exception as e:
            last_err = e
            time.sleep(delay_sec)
    print(f"[WARN] Could not load {dataset_name}/{split} after retries. "
          f"Skipping this dataset. Last error:\n{last_err}")
    return None

# ===================== 六个 append_* 实现 =====================

def append_qwen_thinking(dst, target_bytes: int) -> int:
    """
    (1) Qwen 思维链蒸馏 (中文高质量思考->回答)
    流程:
      - 拿 Input / CoT_content / Answer_content
      - 渲染整条对话
      - 语言过滤 (仅中英，别的语种全丢；符号/数学/工程符号不过滤)
      - 整条字符数 <= COT_MAX_CHARS
      - 不截断，超长直接丢
    """
    ds = safe_streaming_dataset(QWEN_DATASET, QWEN_SPLIT)
    if ds is None:
        return 0

    wrote = 0
    kept = dropped_empty = dropped_lang = dropped_len = 0
    pbar = tqdm(ds, desc="QWEN-THINKING (full-turn <=2048 chars)", unit="rows")

    for ex in pbar:
        if wrote >= target_bytes:
            break

        inp = (ex.get("Input") or "").strip()
        cot = (ex.get("CoT_content") or "").strip()
        ans = (ex.get("Answer_content") or "").strip()

        if not inp or not cot or not ans:
            dropped_empty += 1
            continue

        txt = render_qwen_turn(inp, cot, ans)

        # 新的语言过滤：出现中文/英文以外的语言 -> 丢
        if text_has_foreign_language(txt):
            dropped_lang += 1
            continue

        # 长度过滤
        if not length_ok_cot(txt):
            dropped_len += 1
            continue

        wrote += write_record(dst, txt, unik=QWEN_UNIK, q=QUESTION_VALUE)
        kept += 1

        if wrote % (2 << 20) < 2000:
            pbar.set_postfix(
                size_GB=f"{wrote/1024**3:.3f}",
                kept=kept,
                dropped_lang=dropped_lang,
                empty=dropped_empty,
                too_long=dropped_len
            )

    return wrote


def append_ultra_innerthought(dst, target_bytes: int) -> int:
    """
    (2) Ultra-Innerthought
    - 只收 data_source in KEEP_ULTRA_SOURCES
    - 对每个 turn 渲染整条
    - 新语言过滤 (同上)
    - 整条字符数 <= COT_MAX_CHARS
    - 丢弃超长/多语种
    """
    ds = safe_streaming_dataset(ULTRA_DATASET, ULTRA_SPLIT)
    if ds is None:
        return 0

    wrote = 0
    kept = dropped_src = dropped_empty = dropped_lang = dropped_len = 0
    pbar = tqdm(ds, desc="ULTRA-INNERTHOUGHT (qwq_* <=2048 chars)", unit="dialogs")

    for dialog in pbar:
        if wrote >= target_bytes:
            break

        data_src = (dialog.get("data_source") or "").strip().lower()
        if data_src not in KEEP_ULTRA_SOURCES:
            dropped_src += 1
            continue

        conv_list = dialog.get("conversations", [])
        if not isinstance(conv_list, list):
            continue

        for turn in conv_list:
            if wrote >= target_bytes:
                break

            user_msg      = (turn.get("user")          or "").strip()
            inner_thought = (turn.get("inner_thought") or "").strip()
            assistant_msg = (turn.get("assistant")     or "").strip()

            if not user_msg or not inner_thought or not assistant_msg:
                dropped_empty += 1
                continue

            txt = render_ultra_turn(user_msg, inner_thought, assistant_msg)

            # 新语言过滤
            if text_has_foreign_language(txt):
                dropped_lang += 1
                continue

            # 长度过滤
            if not length_ok_cot(txt):
                dropped_len += 1
                continue

            wrote += write_record(dst, txt, unik=INNER_UNIK, q=QUESTION_VALUE)
            kept += 1

        if wrote % (2 << 20) < 2000:
            pbar.set_postfix(
                size_GB=f"{wrote/1024**3:.3f}",
                kept=kept,
                dropped_src=dropped_src,
                dropped_lang=dropped_lang,
                dropped_empty=dropped_empty,
                too_long=dropped_len
            )

    return wrote


def append_opencode(dst, target_bytes: Optional[int]) -> int:
    """
    (3) OpenCodeInstruct
    - average_test_score == CODE_SCORE_EQ
    - UNIK = average_test_score (1.0)
    - 不强行设为strict，后续dataloader可以把 UNIK>=3.0 当strict
    """
    ds = safe_streaming_dataset(CODE_DATASET, CODE_SPLIT)
    if ds is None:
        return 0

    wrote = 0
    pbar = tqdm(ds, desc="OPENCODE (score==1.0)", unit="rows")

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

        txt = render_code_sample(inp, out)
        if not txt or not length_ok_general(txt):
            continue

        wrote += write_record(dst, txt, unik=score_f, q=QUESTION_VALUE)

        if wrote % (2 << 20) < 2000:
            pbar.set_postfix(size_GB=f"{wrote/1024**3:.3f}")
    return wrote


def append_openmath2(dst, target_bytes: int) -> int:
    """
    (4) OpenMathInstruct-2
    - solution 必须包含 expected_answer
    - UNIK = MATH_UNIK
    """
    ds = safe_streaming_dataset(MATH_DATASET, MATH_SPLIT)
    if ds is None:
        return 0

    wrote = 0
    kept = dropped_nohit = dropped_len = dropped_empty = 0
    pbar = tqdm(ds, desc="OPENMATH2 (filtered correctness)", unit="rows")

    for ex in pbar:
        if wrote >= target_bytes:
            break

        problem = (ex.get("problem") or "").strip()
        sol     = (ex.get("generated_solution") or "").strip()
        exp     = ex.get("expected_answer", None)

        if not problem or not sol:
            dropped_empty += 1
            continue

        if not contains_expected_answer(sol, exp):
            dropped_nohit += 1
            continue

        txt = render_math_sample(problem, sol)
        if not txt or not length_ok_general(txt):
            dropped_len += 1
            continue

        wrote += write_record(dst, txt, unik=MATH_UNIK, q=QUESTION_VALUE)
        kept += 1

        if wrote % (2 << 20) < 2000:
            pbar.set_postfix(
                size_GB=f"{wrote/1024**3:.3f}",
                kept=kept,
                nohit=dropped_nohit,
                too_long=dropped_len,
                empty=dropped_empty
            )
    return wrote


def append_openorca(dst, target_bytes: int) -> int:
    """
    (5) OpenOrca
    - question / response
    - UNIK = ORCA_UNIK (3.0 -> strict)
    - 不额外语言过滤，这里主要是英文推理
    """
    ds = safe_streaming_dataset(ORCA_DATASET, ORCA_SPLIT)
    if ds is None:
        return 0

    wrote = 0
    kept = dropped_empty = dropped_len = 0
    pbar = tqdm(ds, desc="OPENORCA (question->response)", unit="rows")

    for ex in pbar:
        if wrote >= target_bytes:
            break

        q = (ex.get("question") or "").strip()
        a = (ex.get("response") or "").strip()

        if not q or not a:
            dropped_empty += 1
            continue

        txt = render_orca_turn(q, a)
        if not length_ok_general(txt):
            dropped_len += 1
            continue

        wrote += write_record(dst, txt, unik=ORCA_UNIK, q=QUESTION_VALUE)
        kept += 1

        if wrote % (2 << 20) < 2000:
            pbar.set_postfix(
                size_GB=f"{wrote/1024**3:.3f}",
                kept=kept,
                too_long=dropped_len,
                empty=dropped_empty
            )
    return wrote


def append_cosmopedia(dst, seen_cosmo: set, target_bytes: int) -> int:
    """
    (6) Cosmopedia
    - score >= COSMO_MIN_SCORE
    - 长度过滤
    - 百科去重：normalize_light(txt_raw)->md5 在 seen_cosmo 里就丢
    - 新引入的也写回 seen_cosmo
    - UNIK = score_f  (通常 <3.0, 走 non-strict)
    """
    ds = safe_streaming_dataset(COSMO_DATASET, COSMO_SPLIT)
    if ds is None:
        return 0

    wrote = 0
    dropped_score = dropped_len = 0
    pbar = tqdm(ds, desc=f"COSMO score>={COSMO_MIN_SCORE}", unit="rows")

    for ex in pbar:
        if wrote >= target_bytes:
            break

        txt_raw = clean_text(ex.get("text", ""))
        score   = ex.get("score", None)

        try:
            score_f = float(score) if score is not None else None
        except Exception:
            score_f = None

        if not txt_raw or score_f is None:
            continue
        if score_f < COSMO_MIN_SCORE:
            dropped_score += 1
            continue
        if not length_ok_general(txt_raw):
            dropped_len += 1
            continue

        sig = md5_of_text(normalize_light(txt_raw))
        if sig in seen_cosmo:
            # 已见过 -> 跳
            continue

        wrote += write_record(dst, txt_raw, unik=score_f, q=QUESTION_VALUE)

        # 新百科也推入黑名单
        seen_cosmo.add(sig)

        if wrote % (2 << 20) < 2000:
            pbar.set_postfix(
                size_GB=f"{wrote/1024**3:.3f}",
                dropped_score=dropped_score,
                too_long=dropped_len
            )

    return wrote

# ===================== main =====================
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DP_DIR, exist_ok=True)

    # 1) 构建 / 读取 百科去重黑名单
    seen_cosmo = build_seen_for_cosmo(
        hist_files=HIST_JSONLS,
        hash_cache_file=DEDUP_HASH_FILE,
    )

    # 2) 打开全新输出文件 (覆盖式 "wb")
    with open(OUT_JSONL_NEW, "wb") as w:

        # 先写 COT 类（Qwen + Ultra），方便你提前验证
        qwen_bytes = append_qwen_thinking(
            w,
            int(QWEN_TARGET_GB * (1024 ** 3))
        )
        print(f"[QWEN-THINKING] wrote {qwen_bytes/1024**3:.3f} GB into {OUT_JSONL_NEW}")

        ultra_bytes = append_ultra_innerthought(
            w,
            int(ULTRA_TARGET_GB * (1024 ** 3))
        )
        print(f"[ULTRA-INNERTHOUGHT] wrote {ultra_bytes/1024**3:.3f} GB into {OUT_JSONL_NEW}")

        # 然后写 code / math / orca / cosmo
        code_limit_bytes = (
            None if CODE_TARGET_GB is None
            else int(CODE_TARGET_GB * (1024 ** 3))
        )
        code_bytes = append_opencode(w, code_limit_bytes)
        print(f"[OPENCODE] wrote {code_bytes/1024**3:.3f} GB into {OUT_JSONL_NEW}")

        math_bytes = append_openmath2(
            w,
            int(MATH_TARGET_GB * (1024 ** 3))
        )
        print(f"[OPENMATH2] wrote {math_bytes/1024**3:.3f} GB into {OUT_JSONL_NEW}")

        orca_bytes = append_openorca(
            w,
            int(ORCA_TARGET_GB * (1024 ** 3))
        )
        print(f"[OPENORCA] wrote {orca_bytes/1024**3:.3f} GB into {OUT_JSONL_NEW}")

        cosmo_bytes = append_cosmopedia(
            w,
            seen_cosmo,
            int(COSMO_TARGET_GB * (1024 ** 3))
        )
        print(f"[COSMO] wrote {cosmo_bytes/1024**3:.3f} GB (deduped vs history) into {OUT_JSONL_NEW}")

    # 3) 把更新后的 seen_cosmo（现在包含这轮新百科）写回缓存文件
    save_hashes_to_file(DEDUP_HASH_FILE, seen_cosmo)
    print(f"[INFO] Dedup cache updated at {DEDUP_HASH_FILE} "
          f"(size {len(seen_cosmo)} hashes).")

    total_bytes = (
        (qwen_bytes or 0) +
        (ultra_bytes or 0) +
        (code_bytes or 0) +
        (math_bytes or 0) +
        (orca_bytes or 0) +
        (cosmo_bytes or 0)
    )
    print(f"[FINAL] total new file size ≈ {total_bytes/1024**3:.3f} GB → {OUT_JSONL_NEW}")

if __name__ == "__main__":
    main()
