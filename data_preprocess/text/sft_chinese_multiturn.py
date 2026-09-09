# -*- coding: utf-8 -*-
"""
把 opencsg/smoltalk-chinese 转成 TokenNet-chat-template 渲染后的 TXT，
并按 tokens 长度分成两份：
  - smoltalk_chinese_sft_2k.jsonl   (length <= 2048)
  - smoltalk_chinese_sft_16k.jsonl  (2048 < length <= 16384)

说明：
- 不含 COT、不含 tools，只是普通 user/assistant 对话；
- 只保留 score >= 4 的样本；
- 仅允许中文 / 英文 + 常见数学符号，过滤掉日/韩/阿/印地语/西里尔等；
- 使用 tokenizer.chat_template 渲染，并按 token 长度分桶；
- 对最终 TXT 做 md5 去重。
"""

import os
import json
import re
import unicodedata
import hashlib
from typing import Any, Dict, List, Optional, Set

from huggingface_hub import list_repo_files
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# ====== 配置区 ======
TOKENIZER_PATH = r"E:\learn\tiny_05B_cpt3\checkpoint-48000"

REPO = "opencsg/smoltalk-chinese"

OUTPUT_SHORT_JSONL = "smoltalk_chinese_sft_2k.jsonl"   # length <= 2048
OUTPUT_LONG_JSONL  = "smoltalk_chinese_sft_16k.jsonl"  # 2048 < length <= 16384

MAX_SHORT_TOKENS = 2048
MAX_LONG_TOKENS  = 16384

PRINT_EVERY = 2000

# ====== 轻量清洗 ======
RE_CONTROL = re.compile(r"[\u200B-\u200F\uFEFF]")

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = RE_CONTROL.sub("", s)
    s = s.replace("\u0000", "")
    s = re.sub(r"\n{3,}", "\n\n", s.strip())
    return s

def md5hex(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

# ====== 语言过滤（仅对 smoltalk 样本做） ======
# 允许：中/英 + 常见数学/工程符号；Ban：日/韩/阿/印地语/西里尔等
ALLOW_EXTRA = set("°±×÷·℃ℓµÅ½¼⅓⅔⅛¾○△□●■◆▶※")

def is_symbol_ok(ch: str) -> bool:
    if ch in ("\n", "\r", "\t", " ") or ch in ALLOW_EXTRA:
        return True
    code = ord(ch)
    # ASCII 英文/数字/标点
    if 0x20 <= code <= 0x7E:
        return True
    # CJK 汉字
    if 0x4E00 <= code <= 0x9FFF:
        return True
    # CJK 标点
    if 0x3000 <= code <= 0x303F:
        return True
    # 全角
    if 0xFF00 <= code <= 0xFFEF:
        return True
    # 希腊字母（数学/工程）
    if 0x0370 <= code <= 0x03FF:
        return True
    # 常用标点/上标下标/箭头/数学符号/带圈数字等
    if 0x2000 <= code <= 0x206F:
        return True
    if 0x2070 <= code <= 0x209F:
        return True
    if 0x2190 <= code <= 0x22FF:
        return True
    if 0x2460 <= code <= 0x25FF:
        return True
    return False

def is_foreign_letter(ch: str) -> bool:
    code = ord(ch)
    # 日文（平/片/半角片）
    if 0x3040 <= code <= 0x30FF or 0xFF65 <= code <= 0xFF9F:
        return True
    # 韩文
    if 0x1100 <= code <= 0x11FF or 0x3130 <= code <= 0x318F or 0xAC00 <= code <= 0xD7AF:
        return True
    # 阿拉伯文
    if 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F or 0x08A0 <= code <= 0x08FF or 0xFB50 <= code <= 0xFEFF:
        return True
    # 天城文
    if 0x0900 <= code <= 0x097F:
        return True
    # 西里尔
    if 0x0400 <= code <= 0x04FF:
        return True
    # 拉丁扩展字母（带重音等）——排除 µ / Å，这俩在 ALLOW_EXTRA 里
    if 0x00A0 <= code <= 0x024F:
        if unicodedata.category(ch).startswith("L") and ch not in {"µ", "Å"}:
            return True
    return False

def text_has_forbidden_lang(txt: str) -> bool:
    for ch in txt:
        if is_foreign_letter(ch):
            return True
        if not is_symbol_ok(ch):
            # 不常见符号一律视为“可疑”，直接 Ban
            return True
    return False

# ====== HF parquet 列表 ======
def iter_parquet_urls(repo: str):
    files = list_repo_files(repo, repo_type="dataset")
    base = f"hf://datasets/{repo}"
    for p in files:
        if p.endswith(".parquet"):
            yield f"{base}/{p}"

# ====== 对话 → messages ======
def _norm_role(x: Optional[str]) -> Optional[str]:
    if not isinstance(x, str):
        return None
    r = x.strip().lower()
    if r in ("user", "human"):
        return "user"
    if r in ("assistant", "gpt", "model"):
        return "assistant"
    if r == "system":
        return "system"
    return None

def _get_msg(d: Dict[str, Any]) -> str:
    for k in ("content", "text", "value", "utterance", "message"):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def build_messages_from_convs(convs: Any) -> List[Dict[str, str]]:
    """
    把各种 schema 的 conversations/messages/chat 转成统一的 messages:
    [{role, content}, ...]，忽略空行和未知角色。
    """
    if not isinstance(convs, list):
        return []

    messages: List[Dict[str, str]] = []
    for m in convs:
        if not isinstance(m, dict):
            continue
        role = _norm_role(m.get("role") or m.get("from") or m.get("speaker"))
        if role is None:
            continue
        msg = clean_text(_get_msg(m))
        if not msg:
            continue
        messages.append({"role": role, "content": msg})
    return messages

def has_user_and_assistant(messages: List[Dict[str, str]]) -> bool:
    has_user = any(m.get("role") == "user" for m in messages)
    has_assistant = any(m.get("role") == "assistant" for m in messages)
    return has_user and has_assistant

# ====== 主流程：直接写 2k / 16k SFT 文件 ======
def main():
    print(">>> 加载 tokenizer ...")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    if tok.chat_template is None:
        raise ValueError(
            "当前 tokenizer.chat_template 为空。\n"
            "请先把带工具调用/思维链的 Jinja 模板写进 tokenizer 再跑。"
        )

    out_short = open(OUTPUT_SHORT_JSONL, "w", encoding="utf-8")
    out_long  = open(OUTPUT_LONG_JSONL, "w", encoding="utf-8")

    total = 0
    kept_short = 0
    kept_long = 0
    dropped_score = 0
    dropped_conv_empty = 0
    dropped_lang = 0
    dropped_template = 0
    dropped_too_long = 0

    bytes_short = 0
    bytes_long = 0

    seen: Set[str] = set()

    for url in iter_parquet_urls(REPO):
        try:
            ds = load_dataset("parquet", data_files=url, split="train", streaming=True)
        except Exception as e:
            print(f"[WARN] load_dataset failed: {url} -> {e}")
            continue

        pbar = tqdm(ds, desc=f"scan {os.path.basename(url)}", unit="rows")
        for ex in pbar:
            total += 1

            # 1) score 过滤
            score = ex.get("score")
            try:
                score = float(score)
            except Exception:
                score = None
            if score is None or score < 4:
                dropped_score += 1
                continue

            # 2) 拿会话
            convs = ex.get("conversations") or ex.get("messages") or ex.get("chat")
            messages = build_messages_from_convs(convs)
            if not has_user_and_assistant(messages):
                dropped_conv_empty += 1
                continue

            # 3) 语言过滤：把所有 content 串起来跑一遍
            flat_txt = "\n".join(m["content"] for m in messages)
            if text_has_forbidden_lang(flat_txt):
                dropped_lang += 1
                continue

            # 4) 用 chat_template 渲染 TXT（不带 tools、不带 COT）
            try:
                txt = tok.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    tokenize=False,
                )
            except Exception:
                dropped_template += 1
                continue

            # 5) 去重（基于最终 TXT）
            uk = md5hex(txt)
            if uk in seen:
                continue
            seen.add(uk)

            # 6) 按 token 长度分桶
            ids = tok(txt, add_special_tokens=False)["input_ids"]
            total_len = len(ids)

            rec = {
                "TXT": txt,
                "dataset": REPO,
                "length": total_len,
                "score": score,
            }
            line = json.dumps(rec, ensure_ascii=False) + "\n"

            if total_len <= MAX_SHORT_TOKENS:
                out_short.write(line)
                kept_short += 1
                bytes_short += len(line.encode("utf-8"))
            elif total_len <= MAX_LONG_TOKENS:
                out_long.write(line)
                kept_long += 1
                bytes_long += len(line.encode("utf-8"))
            else:
                dropped_too_long += 1

            if (kept_short + kept_long) % PRINT_EVERY == 0:
                pbar.set_postfix(
                    kept_short=kept_short,
                    kept_long=kept_long,
                    drop_score=dropped_score,
                    drop_conv=dropped_conv_empty,
                    drop_lang=dropped_lang,
                    drop_tpl=dropped_template,
                    drop_long=dropped_too_long,
                )

    out_short.close()
    out_long.close()

    mb_short = bytes_short / (1024 * 1024)
    mb_long  = bytes_long / (1024 * 1024)

    print(">>> 完成！")
    print(f"    原始样本数                        : {total}")
    print(f"    写入短上下文样本数 (≤ {MAX_SHORT_TOKENS})   : {kept_short}")
    print(f"    写入长上下文样本数 (≤ {MAX_LONG_TOKENS})    : {kept_long}")
    print(f"    丢弃（score < 4 或缺失）          : {dropped_score}")
    print(f"    丢弃（会话缺 user/assistant）      : {dropped_conv_empty}")
    print(f"    丢弃（语言含日/韩/阿/西里尔等）   : {dropped_lang}")
    print(f"    丢弃（模板渲染报错）              : {dropped_template}")
    print(f"    丢弃（> {MAX_LONG_TOKENS} tokens）: {dropped_too_long}")
    print()
    print(f"    短上下文文件体积                  : {mb_short:.2f} MB -> {os.path.abspath(OUTPUT_SHORT_JSONL)}")
    print(f"    长上下文文件体积                  : {mb_long:.2f} MB -> {os.path.abspath(OUTPUT_LONG_JSONL)}")


if __name__ == "__main__":
    main()
