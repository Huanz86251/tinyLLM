#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import unicodedata
from pathlib import Path
from statistics import mean

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# ========= 配置区：根据你本地环境改这几项 =========

TOKENIZER_PATH = r"E:\learn\tiny_05B_cpt3\checkpoint-48000"  # 你的 tokenizer / ckpt 路径

# 输出目录 & 文件名（LongAlign 只做 8K / 16K 两个版本）
OUTPUT_DIR = Path("longalign_multi_turn")
OUTPUT_8K   = OUTPUT_DIR / "longalign_8k_ctx.jsonl"
OUTPUT_16K  = OUTPUT_DIR / "longalign_16k_ctx.jsonl"

DATASET_NAME  = "zai-org/LongAlign-10k"
DATASET_SPLIT = "train"

# ========= 长度 / 过滤规则 =========

# 对话总长的最大值
MAX_TOKENS_8K  = 8192
MAX_TOKENS_16K = 16384

# 只有长度够长才分别进入 8K / 16K 文件
MIN_TOKENS_8K  = 5000
MIN_TOKENS_16K = 13000

# 任意一轮 assistant 输出不得超过 1500 token
MAX_ASSISTANT_TOKENS = 1500

# 每个文件物理大小上限（大概控制在 ~1GB，上不去就无视它）
MAX_BYTES_PER_FILE = 1_000_000_000


# ========= 小工具函数 =========

def render_and_count_tokens(tok: AutoTokenizer, messages) -> tuple[str, int]:
    """
    使用 tokenizer.chat_template 渲染一条多轮对话，并返回 (TXT, token 数).
    不额外加 system，只吃数据集自带的 messages。
    """
    txt = tok.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=False,
    )
    ids = tok(txt, add_special_tokens=False)["input_ids"]
    return txt, len(ids)


def truncate_from_end_to_fit(tok, messages, max_tokens):
    """
    和你现在 UltraChat 的逻辑一致：

      - 已保证 messages[0].role == "user", messages[-1].role == "assistant"
      - 先尝试用“从第一个 user 开始到最后一个 assistant 为止”的整段前缀；
      - 如果放不下，就把“最后一对 user+assistant 回合”整对丢掉，
        再试上一个 assistant；
      - 找到第一个 total_len <= max_tokens 的前缀就返回。
    """
    n = len(messages)
    if n < 2:
        return None, None, None

    # 所有 assistant 的下标
    assistant_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    if not assistant_indices:
        return None, None, None

    for a_idx in reversed(assistant_indices):
        sub = messages[: a_idx + 1]

        if sub[0].get("role") != "user" or sub[-1].get("role") != "assistant":
            continue

        txt, total_len = render_and_count_tokens(tok, sub)
        if total_len <= max_tokens:
            return sub, txt, total_len

    # 所有前缀都塞不进 max_tokens，这条对话在该上下文长度下没法用
    return None, None, None


def contains_forbidden_language(text: str) -> bool:
    """
    只保留“纯英文 + 中文 + 数学符号”；
    一旦出现明显其他文字系统，就丢弃整个样本。

    允许：
      - ASCII (英文)
      - CJK 汉字及相关区块（当成中文；里面混一点日文 Kanji 也没办法）
      - 标点、数字、各种符号（含大部分数学符号、emoji）
      - 希腊字母及数学字母符号（常见于数学公式）

    禁止（只要出现一次就判为“含非中英文”）：
      - 日文假名（平假名/片假名 + 半角片假名）
      - 韩文（jamo + 音节）
      - 西里尔文（俄语等）
      - 阿拉伯文
      - 希伯来文
      - 天城文（印地语等）
      - 泰文
      - 其他非 ASCII 的字母（例如很多带重音的拉丁字母 → 粗略当作其它欧洲语言）
    """
    for ch in text:
        code = ord(ch)

        # 直接允许的几类先放行
        if code <= 0x007F:
            # ASCII: 英文、基础标点、数字
            continue

        # ===== 明确禁止的一些脚本块 =====
        # 日文假名
        if 0x3040 <= code <= 0x309F:   # Hiragana
            return True
        if 0x30A0 <= code <= 0x30FF:   # Katakana
            return True
        if 0xFF65 <= code <= 0xFF9F:   # Halfwidth Katakana
            return True

        # 韩文
        if 0x1100 <= code <= 0x11FF:   # Hangul Jamo
            return True
        if 0x3130 <= code <= 0x318F:   # Hangul Compatibility Jamo
            return True
        if 0xAC00 <= code <= 0xD7AF:   # Hangul Syllables
            return True

        # 西里尔文
        if 0x0400 <= code <= 0x052F:
            return True

        # 阿拉伯文
        if 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F:
            return True

        # 希伯来文
        if 0x0590 <= code <= 0x05FF:
            return True

        # 天城文（印地语等）
        if 0x0900 <= code <= 0x097F:
            return True

        # 泰文
        if 0x0E00 <= code <= 0x0E7F:
            return True

        # ===== 允许的中日韩统一表意文字（当作中文） =====
        if (
            0x4E00 <= code <= 0x9FFF or  # CJK Unified Ideographs
            0x3400 <= code <= 0x4DBF or  # CJK Ext A
            0x20000 <= code <= 0x2A6DF or
            0x2A700 <= code <= 0x2B73F or
            0x2B740 <= code <= 0x2B81F or
            0x2B820 <= code <= 0x2CEAF or
            0xF900 <= code <= 0xFAFF      # CJK Compatibility Ideographs
        ):
            continue

        # CJK 标点等
        if 0x3000 <= code <= 0x303F:
            continue

        # 希腊字母（常见于数学）
        if 0x0370 <= code <= 0x03FF or 0x1F00 <= code <= 0x1FFF:
            continue

        # 数学字母符号
        if 0x1D400 <= code <= 0x1D7FF:
            continue

        cat = unicodedata.category(ch)

        # 标点(P)、数字(N)、符号(S) 都当作“中性”，允许
        if cat[0] in ("P", "N", "S"):
            continue

        # 其他情况里，如果是字母(L*)，说明是某种“非 ASCII 非 CJK 非 Greek”字母 → 当成其它语言
        if cat[0] == "L":
            return True

        # 其余类别（比如空白、控制字符）统一忽略
        continue

    return False


def summarize_assistant_lengths(name: str, lengths: list[int]):
    print(f"\n  [{name}] assistant 单轮输出长度统计")
    if not lengths:
        print("    无样本（lengths 为空）")
        return
    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)

    def percentile(p: float) -> int:
        if n == 0:
            return 0
        idx = int(n * p)
        if idx >= n:
            idx = n - 1
        return lengths_sorted[idx]

    avg = mean(lengths_sorted)
    p50 = percentile(0.5)
    p75 = percentile(0.75)
    p90 = percentile(0.9)
    p95 = percentile(0.95)
    mx  = lengths_sorted[-1]

    print(f"    样本数               : {n}")
    print(f"    平均值 mean          : {avg:.2f}")
    print(f"    中位数 P50           : {p50}")
    print(f"    P75 / P90 / P95      : {p75}, {p90}, {p95}")
    print(f"    最大值 max           : {mx}")


# ========= 主流程 =========

def main():
    print(">>> 加载 tokenizer ...")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    if tok.chat_template is None:
        raise ValueError(
            "当前 tokenizer.chat_template 为空。\n"
            "请先把你那段带 <|thought_start|>/<|thought_end|> 的 Jinja 模板写进 tokenizer 再跑。"
        )

    print(f">>> 加载 {DATASET_NAME} ({DATASET_SPLIT} split) ...")
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    f_8k  = OUTPUT_8K.open("w", encoding="utf-8")
    f_16k = OUTPUT_16K.open("w", encoding="utf-8")

    bytes_8k  = 0
    bytes_16k = 0
    full_8k   = False
    full_16k  = False

    total = 0
    used_examples = 0

    dropped_no_messages        = 0
    dropped_no_assistant_tail  = 0
    dropped_first_not_user     = 0
    dropped_assistant_too_long = 0
    dropped_lang_not_en_zh     = 0

    kept_8k  = 0
    kept_16k = 0

    assistant_lengths_8k  = []
    assistant_lengths_16k = []

    for ex in tqdm(ds, desc="LongAlign-10k → long-context TXT"):
        total += 1

        messages = ex.get("messages")
        if not messages or len(messages) < 2:
            dropped_no_messages += 1
            continue

        # 只保留 role/content，确保是字符串
        msgs = []
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if role is None or content is None:
                continue
            msgs.append({"role": str(role), "content": str(content)})

        if len(msgs) < 2:
            dropped_no_messages += 1
            continue

        # 语言过滤：只要整段里出现明显非中/英文脚本就丢弃
        full_text = "".join(m["content"] for m in msgs)
        if contains_forbidden_language(full_text):
            dropped_lang_not_en_zh += 1
            continue

        # 去掉结尾所有非 assistant 消息，保证对话最后是 assistant
        while msgs and msgs[-1]["role"] != "assistant":
            msgs.pop()
        if len(msgs) < 2 or msgs[-1]["role"] != "assistant":
            dropped_no_assistant_tail += 1
            continue

        # 确保开头是 user；不是的话，从前面丢到第一个 user 开始
        while msgs and msgs[0]["role"] != "user":
            msgs.pop(0)
        if len(msgs) < 2 or msgs[0]["role"] != "user":
            dropped_first_not_user += 1
            continue

        # 任一轮 assistant 输出 > 1500 token，整条丢弃
        too_long = False
        for m in msgs:
            if m["role"] != "assistant":
                continue
            aid = tok(m["content"], add_special_tokens=False)["input_ids"]
            if len(aid) > MAX_ASSISTANT_TOKENS:
                too_long = True
                break
        if too_long:
            dropped_assistant_too_long += 1
            continue

        used_flag = False
        ex_id = ex.get("id")

        # ========= 8K 版本 =========
        if not full_8k:
            sub_8k, txt_8k, len_8k = truncate_from_end_to_fit(tok, msgs, MAX_TOKENS_8K)
            if sub_8k is not None and len_8k >= MIN_TOKENS_8K:
                assistant_token_counts = []
                for m in sub_8k:
                    if m["role"] == "assistant":
                        ids_a = tok(m["content"], add_special_tokens=False)["input_ids"]
                        assistant_token_counts.append(len(ids_a))

                last_assist = sub_8k[-1]["content"]
                ans_ids = tok(last_assist, add_special_tokens=False)["input_ids"]
                answer_tokens = len(ans_ids)

                rec = {
                    "TXT": txt_8k,
                    "dataset": DATASET_NAME,
                    "split": DATASET_SPLIT,
                    "length": len_8k,
                    "has_cot": False,
                    "cot_length": 0,
                    "reasoning_len": None,
                    "answer_tokens": answer_tokens,
                    "has_reference_answer": False,
                    "reference_answer": "",
                    "problem_source": ex_id,
                    "assistant_token_counts": assistant_token_counts,
                }

                line = json.dumps(rec, ensure_ascii=False)
                b = len(line.encode("utf-8")) + 1
                if bytes_8k + b <= MAX_BYTES_PER_FILE:
                    f_8k.write(line + "\n")
                    bytes_8k += b
                    kept_8k += 1
                    used_flag = True
                    assistant_lengths_8k.extend(assistant_token_counts)
                else:
                    full_8k = True

        # ========= 16K 版本 =========
        if not full_16k:
            sub_16k, txt_16k, len_16k = truncate_from_end_to_fit(tok, msgs, MAX_TOKENS_16K)
            if sub_16k is not None and len_16k >= MIN_TOKENS_16K:
                assistant_token_counts = []
                for m in sub_16k:
                    if m["role"] == "assistant":
                        ids_a = tok(m["content"], add_special_tokens=False)["input_ids"]
                        assistant_token_counts.append(len(ids_a))

                last_assist = sub_16k[-1]["content"]
                ans_ids = tok(last_assist, add_special_tokens=False)["input_ids"]
                answer_tokens = len(ans_ids)

                rec = {
                    "TXT": txt_16k,
                    "dataset": DATASET_NAME,
                    "split": DATASET_SPLIT,
                    "length": len_16k,
                    "has_cot": False,
                    "cot_length": 0,
                    "reasoning_len": None,
                    "answer_tokens": answer_tokens,
                    "has_reference_answer": False,
                    "reference_answer": "",
                    "problem_source": ex_id,
                    "assistant_token_counts": assistant_token_counts,
                }

                line = json.dumps(rec, ensure_ascii=False)
                b = len(line.encode("utf-8")) + 1
                if bytes_16k + b <= MAX_BYTES_PER_FILE:
                    f_16k.write(line + "\n")
                    bytes_16k += b
                    kept_16k += 1
                    used_flag = True
                    assistant_lengths_16k.extend(assistant_token_counts)
                else:
                    full_16k = True

        if used_flag:
            used_examples += 1

        if full_8k and full_16k:
            break

    f_8k.close()
    f_16k.close()

    # ========= 打印统计 =========
    print(">>> 完成！")
    print(f"    原始样本数: {total}")
    print(f"    至少被用于一个上下文版本的样本数: {used_examples}")
    print()
    print(f"    丢弃（messages 为空/过短）: {dropped_no_messages}")
    print(f"    丢弃（去尾后无 assistant 结尾）: {dropped_no_assistant_tail}")
    print(f"    丢弃（无法以 user 开头）: {dropped_first_not_user}")
    print(f"    丢弃（某轮 assistant > {MAX_ASSISTANT_TOKENS} tokens）: {dropped_assistant_too_long}")
    print(f"    丢弃（检测到非中/英文脚本）: {dropped_lang_not_en_zh}")
    print()
    print("    —— 8K 版本 ——")
    print(f"      写入样本数           : {kept_8k}")
    print(f"      约写入字节数         : {bytes_8k}")
    print("    —— 16K 版本 ——")
    print(f"      写入样本数           : {kept_16k}")
    print(f"      约写入字节数         : {bytes_16k}")
    print()
    print("    输出文件：")
    print(f"      8K  : {OUTPUT_8K.resolve()}")
    print(f"      16K : {OUTPUT_16K.resolve()}")

    # ========= assistant 单轮长度统计 =========
    print("\n=== assistant 单轮长度统计（基于写入样本）===")
    summarize_assistant_lengths("8K ctx", assistant_lengths_8k)
    summarize_assistant_lengths("16K ctx", assistant_lengths_16k)


if __name__ == "__main__":
    main()
