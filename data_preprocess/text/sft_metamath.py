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
OUTPUT_JSONL = "metamathqa_cot_from_answer_markers_350.jsonl"

DATASET_TAG = "meta-math/MetaMathQA"
DATASET_SPLIT = "train"  # 需要 test 就改成 "test"

# 总长度 & COT 长度控制
MAX_TOTAL_TOKENS = 2048      # 整条对话 TXT 的 token 上限（含 system）
MAX_COT_TOKENS = 450         # COT 段的 token 上限（<|thought_start|> 包起来后）

# 回答长度控制（和 OpenMathInstruct/Numina 统一）
RESP_SOFT_MAX = 500          # >500 token 就“很长”
RESP_HARD_MAX = 1000         # >1000 token 直接丢弃
KEEP_LONG_RESP_PROB = 0.02   # 500~1000 token 保留 2% 的长样本

# 无 COT 的样本：答案 > 450 token 直接丢弃
NO_COT_ANSWER_MAX = 450

# 无 COT 样本中，额外 1/2 概率丢弃（比 2/3 宽松一点）
DROP_NO_COT_PROB = 0.5

# 当存在 COT 时添加的英文 system 提示（和你前面那条统一）
SYSTEM_PROMPT_FOR_COT = (
    "Think step by step inside <|thought_start|> and <|thought_end|>. Then provide final answer."
)
# 模板后“尾巴”的最大 token 数（>15 视为定位失败，回退为纯答案）
TAIL_MAX_TOKENS = 15

# 这些模板统一做小写匹配（不区分大小写）
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

random.seed(42)  # 方便复现实验


# ========= 小工具函数 =========

def split_solution_with_answer_marker(raw: str, tok: AutoTokenizer):
    """
    按和 OpenMathInstruct / Numina 同一套规则，从 response 中提取 (thought, answer)。

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

    返回:
      thought, answer, status
    status 取值:
      - "empty"
      - "no_marker"
      - "marker_invalid_many_newlines"
      - "marker_tail_too_long"
      - "marker_but_no_prefix"
      - "marker_ok"
    """

    if raw is None:
        return None, None, "empty"

    # 统一换行符，避免 \r\n 干扰
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

    # 2) 从模板结束位置一直看到结尾，统计换行总数
    substring_after = raw[marker_end:]
    newline_after = substring_after.count("\n")
    if newline_after >= 2:
        # 认为这是中途的模板，不可靠 → 回退无 COT
        return None, raw, "marker_invalid_many_newlines"

    # 3) 模板之后的 token 数不能超过 TAIL_MAX_TOKENS
    tail_text = raw[marker_end:]
    tail_ids = tok(tail_text.strip(), add_special_tokens=False)["input_ids"] if tail_text.strip() else []
    if len(tail_ids) > TAIL_MAX_TOKENS:
        return None, raw, "marker_tail_too_long"

    # 4) 上述两关通过，认为这是最后总结答案：
    #    从模板“起始位置”往前找最近一个 '\n'
    line_start = raw.rfind("\n", 0, marker_start)
    if line_start == -1:
        line_start = 0
    else:
        line_start = line_start + 1  # 从换行符之后开始这一行

    thought = raw[:line_start].strip()
    answer = raw[line_start:].strip()

    if not thought:
        # 例如整段就是 "The answer is: 42" → 无 COT
        return None, raw, "marker_but_no_prefix"

    return thought, answer, "marker_ok"


def build_messages(query: str, answer: str, thought: str | None, use_system: bool):
    """
    - 有 thought 且 use_system=True: 在最前添加 system，再用 "thought" 字段
    - 有 thought 且 use_system=False: 仅 assistant 带 "thought"
    - 无 thought: 普通 assistant 回答（不带 "thought"）
    """
    query = (query or "").strip()
    answer = (answer or "").strip()
    if not query or not answer:
        return None

    messages = []
    if use_system:
        messages.append({
            "role": "system",
            "content": SYSTEM_PROMPT_FOR_COT,
        })

    messages.append({"role": "user", "content": query})

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
            "请先把你那段带 <|thought_start|>/<|thought_end|> 的 Jinja 模板写进 tokenizer 再跑。"
        )

    print(f">>> 加载 {DATASET_TAG} ({DATASET_SPLIT} split) ...")
    ds = load_dataset(DATASET_TAG, split=DATASET_SPLIT)

    out_path = Path(OUTPUT_JSONL)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = out_path.open("w", encoding="utf-8")

    # 统计量
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

    for ex in tqdm(ds, desc="MetaMathQA → TXT(+COT when possible)"):
        total += 1

        # MetaMathQA 字段：type, original_question, query, response
        query = (ex.get("query") or "").strip()
        raw_resp = (ex.get("response") or "").strip()

        if not query or not raw_resp:
            dropped_empty += 1
            continue

        with_resp += 1

        # 1) 整个 response 的 token 数（统计 + 长度过滤）
        resp_ids = tok(raw_resp, add_special_tokens=False)["input_ids"]
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

        # 2) 按模板拆分 solution → (thought, answer)
        thought, answer, status = split_solution_with_answer_marker(raw_resp, tok)
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

        # 3) 如果有 COT，先算一下 COT 的 token 长度，> MAX_COT_TOKENS 就丢弃
        cot_length = 0
        if has_cot:
            thought_segment = "<|thought_start|>\n" + thought + "\n<|thought_end|>\n"
            cot_ids = tok(thought_segment, add_special_tokens=False)["input_ids"]
            cot_length = len(cot_ids)
            if cot_length > MAX_COT_TOKENS:
                dropped_cot_too_long += 1
                continue
        else:
            # 没有 COT 的样本更“苛刻”：答案 > 450 token 直接丢
            if resp_len > NO_COT_ANSWER_MAX:
                dropped_no_cot_long_answer += 1
                continue
            # 剩下的无 COT 样本中，1/2 概率随机丢掉
            if random.random() < DROP_NO_COT_PROB:
                dropped_no_cot_random += 1
                continue

        # 4) 构造 messages（有 COT 样本才加 system）
        use_system = bool(has_cot)
        messages = build_messages(query, answer, thought, use_system)
        if messages is None:
            dropped_empty += 1
            continue

        txt = tok.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False,
        )

        # 5) 控制整条对话 ≤ MAX_TOTAL_TOKENS（此时已包含 system）
        ids = tok(txt, add_special_tokens=False)["input_ids"]
        total_len = len(ids)
        if total_len > MAX_TOTAL_TOKENS:
            dropped_long_total += 1
            continue

        # 6) 写 JSONL —— 字段结构和 OpenMathInstruct 脚本统一，方便合并
        rec = {
            "TXT": txt,
            "dataset": DATASET_TAG,
            "split": DATASET_SPLIT,
            "length": total_len,
            "has_cot": bool(has_cot),
            "cot_length": cot_length,
            "reasoning_len": None,         # MetaMathQA 本身没有这个字段
            "answer_tokens": resp_len,     # 原始 response 的 token 数
            "has_reference_answer": False, # 无标准答案
            "reference_answer": "",
            "problem_source": ex.get("type"),  # 比如 gsm8k / math / k12 等
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
    print(f"    含 query + response 的样本数: {with_resp}")
    print(f"    写入样本数: {kept_total}")
    print(f"      其中 解析出 COT 的样本数: {kept_with_cot}")
    print(f"      其中 无显式 COT 的样本数: {kept_no_cot}")
    print()
    print(f"    丢弃（query/response 为空）: {dropped_empty}")
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
