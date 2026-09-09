#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from statistics import mean

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# ========= 配置区：根据你本地环境改这几项 =========

TOKENIZER_PATH = r"E:\learn\tiny_05B_cpt3\checkpoint-48000"  # 你的 tokenizer / ckpt 路径

# 输出目录 & 文件名（一个数据集，三个上下文长度版本）
OUTPUT_DIR = Path("ultrachat_multi_turn")
OUTPUT_2K   = OUTPUT_DIR / "ultrachat_2k_ctx.jsonl"
OUTPUT_8K   = OUTPUT_DIR / "ultrachat_8k_ctx.jsonl"
OUTPUT_16K  = OUTPUT_DIR / "ultrachat_16k_ctx.jsonl"

DATASET_NAME  = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"   # ultrachat_200k 用的 SFT split

# ========= 长度 / 过滤规则 =========

# 对话总长的最大值（只控制“塞得下”，不再看利用率）
MAX_TOKENS_2K  = 2048
MAX_TOKENS_8K  = 8192
MAX_TOKENS_16K = 16384

# 任意一轮 assistant 输出不得超过 1500 token（唯一的强删规则）
MAX_ASSISTANT_TOKENS = 1500

# 每个文件物理大小上限（粗略控制在 ~1GB 左右，你可以改小/改大）
MAX_BYTES_PER_FILE = 2_000_000_000


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
    正确版逻辑（按你的需求）：
      - 已保证 messages[0].role == "user", messages[-1].role == "assistant"
      - 先尝试用“从开头到最后一个 assistant 为止”的整段前缀；
      - 如果放不下，就把“最后一对 user+assistant 回合”整对丢掉，再试上一个 assistant；
      - 其他结构保持不变：开头仍然是最早的那个 user，尽量保留更多轮对话。
    """
    n = len(messages)
    if n < 2:
        return None, None, None

    # 所有 assistant 的下标
    assistant_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    if not assistant_indices:
        return None, None, None

    # 从最后一个 assistant 往前回退
    for a_idx in reversed(assistant_indices):
        # 前缀：[0, a_idx]，即从第一个 user 到当前这个 assistant
        sub = messages[: a_idx + 1]

        # 保险起见，仍然确保首尾是 user / assistant
        if sub[0].get("role") != "user" or sub[-1].get("role") != "assistant":
            continue

        txt, total_len = render_and_count_tokens(tok, sub)
        if total_len <= max_tokens:
            return sub, txt, total_len

        # 如果 total_len > max_tokens，就继续往前，用“上一个 assistant”作为结尾再试

    # 所有前缀都塞不进 max_tokens，这条对话在该上下文档位就不用
    return None, None, None

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
    f_2k  = OUTPUT_2K.open("w", encoding="utf-8")
    f_8k  = OUTPUT_8K.open("w", encoding="utf-8")
    f_16k = OUTPUT_16K.open("w", encoding="utf-8")

    # 每个文件已经写入的字节数（粗略控制 1GB）
    bytes_2k  = 0
    bytes_8k  = 0
    bytes_16k = 0
    full_2k   = False
    full_8k   = False
    full_16k  = False

    # 全局统计
    total = 0
    used_examples = 0

    dropped_no_messages        = 0
    dropped_no_assistant_tail  = 0
    dropped_first_not_user     = 0
    dropped_assistant_too_long = 0

    kept_2k  = 0
    kept_8k  = 0
    kept_16k = 0

    # 用于统计 assistant 单轮长度的全局数组
    assistant_lengths_2k  = []
    assistant_lengths_8k  = []
    assistant_lengths_16k = []

    # —— 主循环 —— #
    for ex in tqdm(ds, desc="UltraChat-200k → multi-context TXT"):
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

        # 任一轮 assistant 输出 > 1500 token，整条丢弃（唯一的强删规则）
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

        # 记录一下这个对话被实际用于生成了多少上下文版本
        used_flag = False

        prompt_id = ex.get("prompt_id")

        # ========= 2K 版本 =========
        if not full_2k:
            sub_2k, txt_2k, len_2k = truncate_from_end_to_fit(tok, msgs, MAX_TOKENS_2K)
            if sub_2k is not None:
                # 所有 assistant 的 token 长度
                assistant_token_counts = []
                for m in sub_2k:
                    if m["role"] == "assistant":
                        ids_a = tok(m["content"], add_special_tokens=False)["input_ids"]
                        assistant_token_counts.append(len(ids_a))
                # 最后一条 assistant
                last_assist = sub_2k[-1]["content"]
                ans_ids = tok(last_assist, add_special_tokens=False)["input_ids"]
                answer_tokens = len(ans_ids)

                rec = {
                    "TXT": txt_2k,
                    "dataset": DATASET_NAME,
                    "split": DATASET_SPLIT,
                    "length": len_2k,
                    "has_cot": False,
                    "cot_length": 0,
                    "reasoning_len": None,
                    "answer_tokens": answer_tokens,
                    "has_reference_answer": False,
                    "reference_answer": "",
                    "problem_source": prompt_id,
                    # 新增：这一条样本里所有 assistant turn 的 token 数
                    "assistant_token_counts": assistant_token_counts,
                }

                line = json.dumps(rec, ensure_ascii=False)
                b = len(line.encode("utf-8")) + 1  # +1 for '\n'
                if bytes_2k + b <= MAX_BYTES_PER_FILE:
                    f_2k.write(line + "\n")
                    bytes_2k += b
                    kept_2k += 1
                    used_flag = True
                    assistant_lengths_2k.extend(assistant_token_counts)
                else:
                    full_2k = True

        # ========= 8K 版本 =========
        if not full_8k:
            sub_8k, txt_8k, len_8k = truncate_from_end_to_fit(tok, msgs, MAX_TOKENS_8K)
            if sub_8k is not None:
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
                    "problem_source": prompt_id,
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
            if sub_16k is not None:
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
                    "problem_source": prompt_id,
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

        # 如果三个文件都已经满 1GB，就可以提前停
        if full_2k and full_8k and full_16k:
            break

    f_2k.close()
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
    print()
    print("    —— 2K 版本 ——")
    print(f"      写入样本数           : {kept_2k}")
    print(f"      约写入字节数         : {bytes_2k}")
    print("    —— 8K 版本 ——")
    print(f"      写入样本数           : {kept_8k}")
    print(f"      约写入字节数         : {bytes_8k}")
    print("    —— 16K 版本 ——")
    print(f"      写入样本数           : {kept_16k}")
    print(f"      约写入字节数         : {bytes_16k}")
    print()
    print("    输出文件：")
    print(f"      2K  : {OUTPUT_2K.resolve()}")
    print(f"      8K  : {OUTPUT_8K.resolve()}")
    print(f"      16K : {OUTPUT_16K.resolve()}")

    # ========= assistant 单轮长度统计 =========
    print("\n=== assistant 单轮长度统计（基于写入样本）===")
    summarize_assistant_lengths("2K ctx", assistant_lengths_2k)
    summarize_assistant_lengths("8K ctx", assistant_lengths_8k)
    summarize_assistant_lengths("16K ctx", assistant_lengths_16k)


if __name__ == "__main__":
    main()
