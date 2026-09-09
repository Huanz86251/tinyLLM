#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# ========= 配置区：根据你本地环境改这几项 =========

TOKENIZER_PATH = r"E:\learn\tiny_05B_cpt3\checkpoint-48000"   # 你的 tokenizer / ckpt 路径
OUTPUT_JSONL = "lingcoder_sft_mixed_600_2048.jsonl"

KEEP_ALL_THRESHOLD = 600       # total_len < 600 → 100% 保留
MAX_TOTAL_TOKENS = 2048       # total_len >= 2048 → 丢弃
SAMPLE_PROB_LONG = 0.05       # 600 ≤ total_len < 2048 → 5% 概率保留

DATASET_TAG = "inclusionAI/Ling-Coder-SFT"


# ========= 小工具函数 =========

def extract_user_and_assistant(messages):
    """
    从 messages 里抽取一对 (user_text, assistant_text)：
    - user：第一个 role 属于 {HUMAN, human, user} 的内容
    - assistant：最后一个 role 属于 {ASSISTANT, assistant} 的内容
    """
    if not messages:
        return None, None

    # 找第一个 HUMAN/user
    user_text = None
    for m in messages:
        role = (m.get("role") or "").upper()
        if role in {"HUMAN", "USER"}:
            user_text = (m.get("content") or "").strip()
            if user_text:
                break

    # 找最后一个 ASSISTANT
    assistant_text = None
    for m in reversed(messages):
        role = (m.get("role") or "").upper()
        if role == "ASSISTANT":
            assistant_text = (m.get("content") or "").strip()
            if assistant_text:
                break

    return user_text, assistant_text


def build_messages(user_text: str, assistant_text: str):
    """
    按你统一的 chat_template 结构构造 messages：
      [{"role": "user", "content": ...},
       {"role": "assistant", "content": ...}]
    """
    user_text = (user_text or "").strip()
    assistant_text = (assistant_text or "").strip()
    if not user_text or not assistant_text:
        return None

    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]


# ========= 主流程 =========

def main():
    random.seed(42)

    print(">>> 加载 tokenizer ...")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    if tok.chat_template is None:
        raise ValueError(
            "当前 tokenizer.chat_template 为空。\n"
            "请先把你那段带 thought 处理的 Jinja 模板写进 tokenizer 再跑。"
        )

    print(">>> 加载 inclusionAI/Ling-Coder-SFT (train split) ...")
    ds = load_dataset("inclusionAI/Ling-Coder-SFT", split="train")

    out_path = Path(OUTPUT_JSONL)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = out_path.open("w", encoding="utf-8")

    total = 0
    extracted_pairs = 0
    kept_total = 0
    kept_short = 0        # < 600 全保留
    kept_sampled = 0      # 600~2047 里被采到的

    dropped_no_pair = 0
    dropped_too_long = 0

    for ex in tqdm(ds, desc="Ling-Coder-SFT → TXT (600 full, 5% long)"):
        total += 1

        # 1) 取 messages：有的版本叫 "messages"，有的叫 "message"，都试一下
        messages = ex.get("messages") or ex.get("message") or ex.get("conversations")
        if messages is None:
            dropped_no_pair += 1
            continue

        user_text, assistant_text = extract_user_and_assistant(messages)
        if not user_text or not assistant_text:
            dropped_no_pair += 1
            continue
        extracted_pairs += 1

        # 2) 用你的 chat_template 渲染整条 TXT
        msgs = build_messages(user_text, assistant_text)
        if msgs is None:
            dropped_no_pair += 1
            continue

        txt = tok.apply_chat_template(
            msgs,
            add_generation_prompt=False,
            tokenize=False,
        )

        # 3) 统计 total_len，并根据长度做采样
        ids = tok(txt, add_special_tokens=False)["input_ids"]
        total_len = len(ids)

        if total_len >= MAX_TOTAL_TOKENS:
            dropped_too_long += 1
            continue

        if total_len < KEEP_ALL_THRESHOLD:
            # <600：全部保留
            keep = True
            kept_short += 1
        else:
            # [600, 2048)：5% 概率保留
            if random.random() < SAMPLE_PROB_LONG:
                keep = True
                kept_sampled += 1
            else:
                keep = False

        if not keep:
            continue

        # 4) 写 JSONL，schema 和之前几份保持一致
        rec = {
            "TXT": txt,
            "dataset": DATASET_TAG,
            "length": total_len,
            "has_cot": False,       # Ling-Coder 本身没显式 CoT 字段
            "cot_length": 0,
            "reasoning_len": None,
        }

        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        kept_total += 1

    fout.close()

    print(">>> 完成！")
    print(f"    原始样本数: {total}")
    print(f"    成功抽取到 (user, assistant) 对的样本数: {extracted_pairs}")
    print(f"    写入样本数: {kept_total}")
    print(f"      其中 < {KEEP_ALL_THRESHOLD} tokens 的样本数 (全保留): {kept_short}")
    print(f"      其中 600~{MAX_TOTAL_TOKENS-1} tokens 被 5% 采样保留的样本数: {kept_sampled}")
    print(f"    丢弃（无法抽取 user/assistant 对）: {dropped_no_pair}")
    print(f"    丢弃（total_len >= {MAX_TOTAL_TOKENS}）: {dropped_too_long}")
    if total:
        kept_ratio = kept_total / total * 100
        print(f"    总保留比例: {kept_ratio:.2f}%")
    print(f"    输出文件: {out_path.resolve()}")


if __name__ == "__main__":
    main()
