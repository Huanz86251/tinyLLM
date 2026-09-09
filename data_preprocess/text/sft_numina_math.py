#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# ========= 配置区：根据你本地环境改这几项 =========

TOKENIZER_PATH = r"E:\learn\tiny_05B_cpt3\checkpoint-48000"  # 你的 tokenizer / ckpt 路径
OUTPUT_JSONL = "numinamath_cot_from_solution_with_system.jsonl"

DATASET_TAG = "AI-MO/NuminaMath-CoT"
DATASET_SPLIT = "train"   # 需要 test 就改成 "test"

# 总长度 & COT 长度控制
MAX_TOTAL_TOKENS = 2048      # 整条对话 TXT 的 token 上限（含 system）
MAX_COT_TOKENS = 450         # COT 段的 token 上限

# 回答长度控制（和 OpenMath 脚本一致）
RESP_SOFT_MAX = 500          # >500 token 就“很长”
RESP_HARD_MAX = 1000         # >1000 token 直接丢弃
KEEP_LONG_RESP_PROB = 0.02   # 500~1000 token 保留 2% 的长样本

# 无 COT 的样本：回答超过 450 token 直接丢弃
NO_COT_ANSWER_MAX = 300

# 无 COT 样本中，额外 1/2 概率丢弃（比 OpenMath 宽松一点）
DROP_NO_COT_PROB = 0.5

# 当存在 COT 时添加的英文 system 提示（和 OpenMath 一致）
SYSTEM_PROMPT_FOR_COT = (
    "Think step by step inside <|thought_start|> and <|thought_end|>. Then provide final answer."
)

# 模板后“尾巴”的最大 token 数（>15 视为定位失败，回退为纯答案）
TAIL_MAX_TOKENS = 15

# 这些模板统一做小写匹配（不区分大小写）
ANSWER_MARKERS = [
    "therefore,",
    "in conclusion,",
    " conclusion,",   # 注意前面有空格，匹配类似“…, conclusion,”
    "in summary,",
    "thus,",
    "the final answer is",
    "the answer is",
    "so,",
]


# ========= 小工具函数 =========

def find_last_marker_span(text: str) -> tuple[int | None, str | None]:
    """
    在 text 里按大小写不敏感的方式，找到所有模板中“最后一次出现”的那一个。
    返回：(marker_start_idx, marker_string)
    找不到则返回 (None, None)
    """
    if not text:
        return None, None

    lower = text.lower()
    last_pos = -1
    last_marker = None

    for marker in ANSWER_MARKERS:
        m = marker.lower()
        start = 0
        while True:
            idx = lower.find(m, start)
            if idx == -1:
                break
            # 不停更新 → 最后一次出现的位置
            last_pos = idx
            last_marker = marker
            start = idx + 1

    if last_marker is None:
        return None, None
    return last_pos, last_marker


def split_solution_with_markers(raw: str, tok: AutoTokenizer) -> tuple[str | None, str | None]:
    """
    按我们约定好的模板，把 solution 拆成 (thought, answer)。

    逻辑：
      1) 所有模板中，只看“最后一次出现”的那个。
      2) 如果模板后面的文本中出现了 >=2 个换行符 → 视为它在正文中，放弃提取 COT，整体当 answer。
      3) 如果模板后面的 token 数 > TAIL_MAX_TOKENS(15) → 视为不干净，整体当 answer。
      4) 从该模板位置往前找最近一个 '\n'，从那个换行符后开始到结尾是 answer，
         之前的所有内容是 thought。
      5) 如果 thought 为空或全是空白 → 视为无 COT，整体当 answer。
    """
    if raw is None:
        return None, None
    text = raw.strip()
    if not text:
        return None, None

    # 1) 找最后一个模板
    marker_start, marker = find_last_marker_span(text)
    if marker_start is None:
        # 找不到模板 → 没法拆 COT
        return None, text

    marker_end = marker_start + len(marker)

    # 2) 模板之后的字符串
    tail = text[marker_end:]
    newline_count = tail.count("\n")
    if newline_count > 1:
        # 模板之后有两个及以上换行 → 大概率还在“中间过程”里
        return None, text

    # 3) 模板之后 token 数不能太多
    tail_stripped = tail.strip()
    if tail_stripped:
        tail_ids = tok(tail_stripped, add_special_tokens=False)["input_ids"]
        if len(tail_ids) > TAIL_MAX_TOKENS:
            # 模板之后内容太长，说明不是“最后一句总结”
            return None, text

    # 4) 往前找最近的换行，分割 thought / answer
    prev_newline = text.rfind("\n", 0, marker_start)
    if prev_newline == -1:
        answer_start = 0
    else:
        answer_start = prev_newline + 1

    thought = text[:answer_start].rstrip()
    answer = text[answer_start:].lstrip()

    # 5) thought 为空就不算 COT
    if not thought.strip():
        return None, text

    return thought, answer


def build_messages(question: str, answer: str, thought: str | None, use_system: bool):
    """
    - 有 thought 且 use_system=True: 在最前添加 system，再用 "thought" 字段
    - 有 thought 且 use_system=False: 仅 assistant 带 "thought"
    - 无 thought: 普通 assistant 回答（不带 "thought"）
    """
    question = (question or "").strip()
    answer = (answer or "").strip()
    if not question or not answer:
        return None

    messages = []
    if use_system:
        messages.append({
            "role": "system",
            "content": SYSTEM_PROMPT_FOR_COT,
        })

    messages.append({"role": "user", "content": question})

    if thought is not None and thought.strip():
        messages.append({
            "role": "assistant",
            "thought": thought.strip(),
            "content": answer,
        })
    else:
        messages.append({
            "role": "assistant",
            "content": answer,
        })
    return messages


# ========= 主流程 =========

def main():
    random.seed(42)

    print(">>> 加载 tokenizer ...")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    if tok.chat_template is None:
        raise ValueError(
            "当前 tokenizer.chat_template 为空。\n"
            "请先把那段带 <|thought_start|>/<|thought_end|> 的 Jinja 模板写进 tokenizer 再跑。"
        )

    print(f">>> 加载 {DATASET_TAG} ({DATASET_SPLIT} split) ...")
    ds = load_dataset(DATASET_TAG, split=DATASET_SPLIT)

    out_path = Path(OUTPUT_JSONL)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = out_path.open("w", encoding="utf-8")

    total = 0
    kept_total = 0
    kept_with_cot = 0
    kept_no_cot = 0

    dropped_no_solution = 0
    dropped_long_resp_hard = 0
    dropped_long_resp_soft = 0
    dropped_long_total = 0
    dropped_long_cot = 0
    dropped_no_cot_long_answer = 0
    dropped_no_cot_random = 0

    for ex in tqdm(ds, desc=f"{DATASET_TAG} → TXT(+COT when possible)"):
        total += 1

        question = (ex.get("problem") or "").strip()
        raw_solution = (ex.get("solution") or "").strip()

        if not question or not raw_solution:
            dropped_no_solution += 1
            continue

        # 1) 整个 solution 的 token 数（统计 + 长度过滤）
        resp_ids = tok(raw_solution, add_special_tokens=False)["input_ids"]
        resp_len = len(resp_ids)

        # 1.1 硬上限：> RESP_HARD_MAX 直接丢弃
        if resp_len > RESP_HARD_MAX:
            dropped_long_resp_hard += 1
            continue

        # 1.2 介于 (RESP_SOFT_MAX, RESP_HARD_MAX] 之间的 2% 保留
        if resp_len > RESP_SOFT_MAX:
            if random.random() >= KEEP_LONG_RESP_PROB:
                dropped_long_resp_soft += 1
                continue
            # 否则就继续保留这条长样本

        # 2) 按模板拆分 solution → (thought, answer)
        thought, answer = split_solution_with_markers(raw_solution, tok)
        has_cot = thought is not None and thought.strip()

        # 3) 如果有 COT，先算一下 COT 的 token 长度；>MAX_COT_TOKENS 就直接丢弃
        cot_length = 0
        if has_cot:
            thought_segment = "<|thought_start|>\n" + thought + "\n<|thought_end|>\n"
            cot_ids = tok(thought_segment, add_special_tokens=False)["input_ids"]
            cot_length = len(cot_ids)
            if cot_length > MAX_COT_TOKENS:
                dropped_long_cot += 1
                continue
        else:
            # 没有 COT 的样本：答案太长直接丢，再随机丢一半
            if resp_len > NO_COT_ANSWER_MAX:
                dropped_no_cot_long_answer += 1
                continue
            if random.random() < DROP_NO_COT_PROB:
                dropped_no_cot_random += 1
                continue

        # 4) 构造 messages（有 COT 样本才加 system）
        use_system = bool(has_cot)
        messages = build_messages(question, answer, thought, use_system)
        if messages is None:
            dropped_no_solution += 1
            continue

        txt = tok.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False,
        )

        # 5) 控制整条对话 ≤ 2048 tokens（此时已包含 system）
        ids = tok(txt, add_special_tokens=False)["input_ids"]
        total_len = len(ids)
        if total_len > MAX_TOTAL_TOKENS:
            dropped_long_total += 1
            continue

        # 6) Numina 没有显式 reference_answer，就留空
        ref_ans = ""
        has_ref = False

        # 7) 写 JSONL —— 字段结构保持和 OpenMath 脚本一致
        rec = {
            "TXT": txt,
            "dataset": DATASET_TAG,
            "split": DATASET_SPLIT,
            "length": total_len,
            "has_cot": bool(has_cot),
            "cot_length": cot_length,
            "reasoning_len": None,      # 这个数据集没有原始 reasoning_len 字段
            "answer_tokens": resp_len,  # 原始 solution 的 token 数
            "has_reference_answer": has_ref,
            "reference_answer": ref_ans,
            "response_model": None,     # Numina 没有 response_model，就填 None
            "source": ex.get("source"), # 保留一下 source 信息（synthetic_math, cn_k12 等）
        }

        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        kept_total += 1
        if has_cot:
            kept_with_cot += 1
        else:
            kept_no_cot += 1

    fout.close()

    print(">>> 完成！")
    print(f"    原始样本数: {total}")
    print(f"    写入样本数: {kept_total}")
    print(f"      其中 解析出 COT 的样本数: {kept_with_cot}")
    print(f"      其中 无显式 COT 的样本数: {kept_no_cot}")
    print(f"    丢弃（无 problem/solution 或空答案）: {dropped_no_solution}")
    print(f"    丢弃（solution token 数 > {RESP_HARD_MAX}）: {dropped_long_resp_hard}")
    print(f"    丢弃（{RESP_SOFT_MAX}< solution token 数 ≤ {RESP_HARD_MAX}，随机丢弃 98%）: {dropped_long_resp_soft}")
    print(f"    丢弃（整条 TXT token 数 > {MAX_TOTAL_TOKENS}）: {dropped_long_total}")
    print(f"    丢弃（COT token 数 > {MAX_COT_TOKENS}）: {dropped_long_cot}")
    print(f"    丢弃（无 COT 且答案 token 数 > {NO_COT_ANSWER_MAX}）: {dropped_no_cot_long_answer}")
    print(f"    丢弃（无 COT 随机 1/2 丢弃）: {dropped_no_cot_random}")
    if total:
        kept_ratio = kept_total / total * 100
        print(f"    保留比例: {kept_ratio:.2f}%")
    print(f"    输出文件: {out_path.resolve()}")


if __name__ == "__main__":
    main()
