#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# ========= 配置区：根据你本地环境改这几项 =========

TOKENIZER_PATH = r"E:\learn\tiny_05B_cpt3\checkpoint-48000"  # 换成你自己的 tokenizer / ckpt 路径
OUTPUT_JSONL = "mathinstruct_cot_from_answer_markers.jsonl"

DATASET_TAG = "TIGER-Lab/MathInstruct"
DATASET_SPLIT = "train"

# ====== 统一的长度 & 采样规则（和 Orca/OpenMath/Numina/MetaMath 保持一致） ======

MAX_TOTAL_TOKENS = 2048      # 整条对话 TXT 的 token 上限（含 system）
MAX_COT_TOKENS = 450         # COT 段的 token 上限（<|thought_start|> 包起来后）

RESP_SOFT_MAX = 500          # >500 token 就“很长”
RESP_HARD_MAX = 1000         # >1000 token 直接丢弃
KEEP_LONG_RESP_PROB = 0.02   # 500~1000 token 保留 2% 的长样本

# 无 COT 的样本：答案 > 450 token 直接丢
NO_COT_ANSWER_MAX = 450
# 无 COT 样本中，额外 1/2 概率丢弃
DROP_NO_COT_PROB = 0.5

# 当存在 COT 时添加的英文 system 提示（和 Orca 版保持一致）
SYSTEM_PROMPT_FOR_COT = (
    "Think step by step inside <|thought_start|> and <|thought_end|>. Then provide final answer."
)

# 模板后“尾巴”的最大 token 数（>15 视为定位失败，回退为纯答案）
TAIL_MAX_TOKENS = 15

# 统一的模板集合（大小写不敏感）
ANSWER_MARKERS = [
    "the final answer is",
    "the answer is",
    "therefore,",
    "in conclusion,",
    " conclusion,",   # 注意前面有空格，匹配类似“…, conclusion,”
    "in summary,",
    "thus,",
    "so,",
]

random.seed(42)


# ========= 小工具函数 =========

def split_solution_with_answer_marker(raw: str, tok: AutoTokenizer):
    """
    按统一规则，从 output 中提取 (thought, answer)。

    规则要点：
      - 统一换行为 '\n'。
      - 大小写不敏感地查找 ANSWER_MARKERS 中所有模板；
        如果有多个匹配，只用“最后一个出现的”。
      - 从这个模板的【结束位置】一直看到整段结尾：
          * 统计 '\n' 数，如果 >= 2 → 直接视为找不到 COT（返回整段为 answer）。
          * 把模板后面的那段文本送进 tokenizer，token 数 > TAIL_MAX_TOKENS
            → 也视为找不到 COT。
      - 若通过上述两个检查：
          * 从模板【起始位置】向前找最近的 '\n'；
          * 若找到，则从该 '\n' 之后到结尾是 answer，之前是 thought；
          * 若找不到 '\n'，则从开头到模板前是 thought，模板到结尾是 answer。
      - 若切出来的 thought 为空 → 仍视为“无 COT”，整段为 answer。
    """

    if raw is None:
        return None, None, "empty"

    raw = raw.replace("\r\n", "\n").strip()
    if not raw:
        return None, None, "empty"

    lower = raw.lower()

    # 1) 找所有模板中“最后一个出现的”位置
    best_pos = -1
    best_marker = None
    for marker in ANSWER_MARKERS:
        idx = lower.rfind(marker)
        if idx != -1 and idx > best_pos:
            best_pos = idx
            best_marker = marker

    if best_marker is None:
        # 没有任何模板 → 无显式 COT
        return None, raw, "no_marker"

    marker_start = best_pos
    marker_end = marker_start + len(best_marker)

    # 2) 模板结束到结尾的子串
    substring_after = raw[marker_end:]
    newline_after = substring_after.count("\n")
    if newline_after >= 2:
        # 模板之后出现了两个或更多换行，视为“中途模板”
        return None, raw, "marker_invalid_many_newlines"

    # 3) 模板之后的 token 数不能超过 TAIL_MAX_TOKENS
    tail_text = substring_after.strip()
    if tail_text:
        tail_ids = tok(tail_text, add_special_tokens=False)["input_ids"]
        if len(tail_ids) > TAIL_MAX_TOKENS:
            return None, raw, "marker_tail_too_long"

    # 4) 认为这是最后的总结答案：从模板起始往前找最近 '\n'
    line_start = raw.rfind("\n", 0, marker_start)
    if line_start == -1:
        line_start = 0
    else:
        line_start = line_start + 1  # 从换行符后开始这一行

    thought = raw[:line_start].strip()
    answer = raw[line_start:].strip()

    if not thought:
        return None, raw, "marker_but_no_prefix"

    return thought, answer, "marker_ok"


def build_messages(instruction: str, answer: str, thought: str | None, use_system: bool):
    """
    - 有 thought 且 use_system=True: 在最前添加 system，再用 "thought" 字段
    - 有 thought 且 use_system=False: 仅 assistant 带 "thought"
    - 无 thought: 普通 assistant 回答（不带 "thought"）
    """
    instruction = (instruction or "").strip()
    answer = (answer or "").strip()
    if not instruction or not answer:
        return None

    messages = []
    if use_system:
        messages.append({
            "role": "system",
            "content": SYSTEM_PROMPT_FOR_COT,
        })

    messages.append({"role": "user", "content": instruction})

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
    with_resp = 0
    kept_total = 0
    kept_with_cot = 0
    kept_no_cot = 0

    dropped_empty = 0
    dropped_resp_hard = 0
    dropped_resp_soft = 0
    dropped_cot_too_long = 0
    dropped_long_total = 0
    dropped_no_cot_long_answer = 0
    dropped_no_cot_random = 0

    marker_ok = 0
    marker_invalid_many_newlines = 0
    marker_tail_too_long = 0
    marker_but_no_prefix = 0
    marker_none = 0

    for ex in tqdm(ds, desc="MathInstruct → TXT(+COT when possible)"):
        total += 1

        # MathInstruct 字段：instruction / output
        instruction = (ex.get("instruction") or "").strip()
        raw_answer = (ex.get("output") or "").strip()

        if not instruction or not raw_answer:
            dropped_empty += 1
            continue

        with_resp += 1

        # 1) 整个 answer 的 token 数（统计 + 长度过滤）
        resp_ids = tok(raw_answer, add_special_tokens=False)["input_ids"]
        resp_len = len(resp_ids)

        # 1.1 硬上限：> RESP_HARD_MAX 直接丢弃
        if resp_len > RESP_HARD_MAX:
            dropped_resp_hard += 1
            continue

        # 1.2 介于 (RESP_SOFT_MAX, RESP_HARD_MAX] 之间的 2% 保留
        if resp_len > RESP_SOFT_MAX:
            if random.random() >= KEEP_LONG_RESP_PROB:
                dropped_resp_soft += 1
                continue
            # 否则就继续保留这条长样本

        # 2) 尝试用模板拆分 → (thought, answer)
        thought, answer, status = split_solution_with_answer_marker(raw_answer, tok)
        if status == "marker_ok":
            marker_ok += 1
        elif status == "marker_invalid_many_newlines":
            marker_invalid_many_newlines += 1
        elif status == "marker_tail_too_long":
            marker_tail_too_long += 1
        elif status == "marker_but_no_prefix":
            marker_but_no_prefix += 1
        elif status in ("no_marker", "empty"):
            marker_none += 1

        has_cot = thought is not None and thought.strip()

        # 3) 有 COT 时检查 COT 长度；无 COT 时对答案更苛刻
        cot_length = 0
        if has_cot:
            thought_segment = "<|thought_start|>\n" + thought + "\n<|thought_end|>\n"
            cot_ids = tok(thought_segment, add_special_tokens=False)["input_ids"]
            cot_length = len(cot_ids)
            if cot_length > MAX_COT_TOKENS:
                dropped_cot_too_long += 1
                continue
        else:
            # 无 COT：答案太长直接丢，然后对剩余样本再丢一半
            if resp_len > NO_COT_ANSWER_MAX:
                dropped_no_cot_long_answer += 1
                continue
            if random.random() < DROP_NO_COT_PROB:
                dropped_no_cot_random += 1
                continue

        # 4) 构造 messages（有 COT 才加 system）
        use_system = bool(has_cot)
        messages = build_messages(instruction, answer, thought, use_system)
        if messages is None:
            dropped_empty += 1
            continue

        txt = tok.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False,
        )

        # 5) 整条对话长度 ≤ MAX_TOTAL_TOKENS
        ids = tok(txt, add_special_tokens=False)["input_ids"]
        total_len = len(ids)
        if total_len > MAX_TOTAL_TOKENS:
            dropped_long_total += 1
            continue

        # 6) 写 JSONL —— schema 跟你 Orca / OpenMath 统一
        problem_source = ex.get("source")

        rec = {
            "TXT": txt,
            "dataset": DATASET_TAG,
            "split": DATASET_SPLIT,
            "length": total_len,
            "has_cot": bool(has_cot),
            "cot_length": cot_length,
            "reasoning_len": None,        # 没有显式 reasoning_len 字段
            "answer_tokens": resp_len,    # 原始 output 的 token 数
            "has_reference_answer": False,
            "reference_answer": "",
            "problem_source": problem_source,
        }

        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        kept_total += 1
        if has_cot:
            kept_with_cot += 1
        else:
            kept_no_cot += 1

    fout.close()

    # ========= 打印统计 =========
    print(">>> 完成！")
    print(f"    原始样本数: {total}")
    print(f"    含 instruction + output 的样本数: {with_resp}")
    print(f"    写入样本数: {kept_total}")
    print(f"      其中 解析出 COT 的样本数: {kept_with_cot}")
    print(f"      其中 无显式 COT 的样本数: {kept_no_cot}")
    print()
    print(f"    丢弃（instruction/output 为空）: {dropped_empty}")
    print(f"    丢弃（response token 数 > {RESP_HARD_MAX}）: {dropped_resp_hard}")
    print(f"    丢弃（{RESP_SOFT_MAX}< response token 数 ≤ {RESP_HARD_MAX}，随机丢弃 98%）: {dropped_resp_soft}")
    print(f"    丢弃（COT token 数 > {MAX_COT_TOKENS}）: {dropped_cot_too_long}")
    print(f"    丢弃（整条 TXT token 数 > {MAX_TOTAL_TOKENS}）: {dropped_long_total}")
    print(f"    丢弃（无 COT 且答案 token 数 > {NO_COT_ANSWER_MAX}）: {dropped_no_cot_long_answer}")
    print(f"    丢弃（无 COT 随机 1/2 丢弃）: {dropped_no_cot_random}")
    if total:
        kept_ratio = kept_total / total * 100
        print(f"    保留比例: {kept_ratio:.2f}%")
    print()
    print("    模板匹配统计：")
    print(f"      marker_ok                     : {marker_ok}")
    print(f"      marker_invalid_many_newlines  : {marker_invalid_many_newlines}")
    print(f"      marker_tail_too_long          : {marker_tail_too_long}")
    print(f"      marker_but_no_prefix          : {marker_but_no_prefix}")
    print(f"      无任何模板 / 空               : {marker_none}")
    print()
    print(f"    输出文件: {out_path.resolve()}")


if __name__ == "__main__":
    main()
