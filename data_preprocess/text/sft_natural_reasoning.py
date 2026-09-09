#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# ========= 配置区：根据你本地环境改这几项 =========

TOKENIZER_PATH = r"E:\learn\tiny_05B_cpt3\checkpoint-48000"  # 你的 tokenizer / ckpt 路径
OUTPUT_JSONL = "natural_reasoning_cot_from_final_answer_350.jsonl"

MAX_RESPONSE_TOKENS = 550   # 整个 response（含 CoT+final）token 上限
MAX_TOTAL_TOKENS    = 2048  # 整条对话 TXT 的 token 上限
MAX_COT_TOKENS      = 500   # COT 段 token 上限
NO_COT_ANSWER_MAX   = 500   # 无 COT 时，答案 token 上限

DATASET_TAG = "facebook/natural_reasoning"

# 当存在 COT 时添加的英文 system 提示
SYSTEM_PROMPT_FOR_COT = (
    "Think step by step inside <|thought_start|> and <|thought_end|>. Then provide final answer."
)

# COT 解析模板（小写匹配）
ANSWER_MARKERS = [
    "the final answer is",  # natural_reasoning 主模板
    "final answer:",
    "the answer is",
    "therefore,",
    "in conclusion,",
    " conclusion,",
    "in summary,",
    "thus,",
    "so,",
]

TAIL_MAX_TOKENS = 15   # 模板之后的 token 数不能超过 15


# ========= 小工具函数 =========

def split_response_with_answer_marker(raw: str, tok: AutoTokenizer):
    """
    使用统一的模板逻辑从 response 中提取 (thought, answer)。

    规则：
      - 所有 ANSWER_MARKERS 中，找到“最后一次出现”的那个模板；
      - 从该模板的结束位置到文本结尾：
          * 如果出现的 '\n' 数 >= 2 → 认为这是中途模板（不可靠），返回无 COT；
          * 若尾部 token 数 > TAIL_MAX_TOKENS → 也视为中途模板，返回无 COT；
      - 通过上述检查后：
          * 从模板“起始位置”往前找最近的 '\n'；
          * 如果找到，则从该 '\n' 之后到结尾作为 answer，其前面作为 thought；
          * 若找不到 '\n'，则 [0, 模板起始) 为 thought，[模板起始, 结尾] 为 answer；
      - 如果 thought 为空或全空白 → 视为无 COT。
    返回:
      thought, answer, status
    status 用于统计：
      "empty" / "no_marker" / "marker_invalid_many_newlines"
      "marker_tail_too_long" / "marker_but_no_prefix" / "marker_ok"
    """
    if raw is None:
        return None, None, "empty"

    raw = raw.replace("\r\n", "\n").strip()
    if not raw:
        return None, None, "empty"

    lower = raw.lower()

    # 找所有模板中的“最后一次出现”
    best_pos = -1
    best_marker = None
    for marker in ANSWER_MARKERS:
        idx = lower.rfind(marker)
        if idx != -1 and idx > best_pos:
            best_pos = idx
            best_marker = marker

    if best_marker is None:
        return None, raw, "no_marker"

    marker_start = best_pos
    marker_end = marker_start + len(best_marker)

    # 模板之后的子串
    substring_after = raw[marker_end:]
    newline_after = substring_after.count("\n")
    if newline_after >= 2:
        return None, raw, "marker_invalid_many_newlines"

    # 模板之后的 token 数不能太多
    tail_text = substring_after
    tail_ids = tok(tail_text, add_special_tokens=False)["input_ids"]
    if len(tail_ids) > TAIL_MAX_TOKENS:
        return None, raw, "marker_tail_too_long"

    # 往前找最近的换行，分割 thought / answer
    line_start = raw.rfind("\n", 0, marker_start)
    if line_start == -1:
        line_start = 0
    else:
        line_start = line_start + 1

    thought = raw[:line_start].strip()
    answer = raw[line_start:].strip()

    if not thought:
        return None, raw, "marker_but_no_prefix"

    return thought, answer, "marker_ok"


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
    print(">>> 加载 tokenizer ...")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    if tok.chat_template is None:
        raise ValueError(
            "当前 tokenizer.chat_template 为空。\n"
            "请先把你那段带 <|thought_start|>/<|thought_end|> 处理的 Jinja 模板写进 tokenizer 再跑。"
        )

    print(">>> 加载 facebook/natural_reasoning (train split) ...")
    ds = load_dataset(DATASET_TAG, split="train")

    out_path = Path(OUTPUT_JSONL)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = out_path.open("w", encoding="utf-8")

    total = 0
    with_resp = 0
    kept_total = 0
    kept_with_cot = 0
    kept_no_cot = 0

    dropped_no_resp = 0
    dropped_long_resp = 0
    dropped_long_total = 0
    dropped_long_cot = 0
    dropped_no_cot_long_answer = 0

    marker_ok = 0
    marker_invalid_many_newlines = 0
    marker_tail_too_long = 0
    marker_but_no_prefix = 0
    marker_none = 0

    for ex in tqdm(ds, desc="NaturalReasoning → TXT(+COT when possible)"):
        total += 1

        question = (ex.get("question") or "").strip()
        ref_ans = (ex.get("reference_answer") or "").strip()
        resp_list = ex.get("responses") or []

        if not question or not resp_list:
            dropped_no_resp += 1
            continue
        with_resp += 1

        resp0 = resp_list[0] or {}
        raw_response = (resp0.get("response") or "").strip()
        response_model = resp0.get("response_model")

        if not raw_response:
            dropped_no_resp += 1
            continue

        # 1) 整个 response 的 token 数（CoT + final 一起算）
        resp_ids = tok(raw_response, add_special_tokens=False)["input_ids"]
        resp_len = len(resp_ids)
        if resp_len > MAX_RESPONSE_TOKENS:
            dropped_long_resp += 1
            continue

        # 2) 用统一模板逻辑尝试提取 COT
        thought, answer, status = split_response_with_answer_marker(raw_response, tok)
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

        # 3) 如果有 COT，限制 COT 长度
        cot_length = 0
        if has_cot:
            thought_segment = "<|thought_start|>\n" + thought + "\n<|thought_end|>\n"
            cot_ids = tok(thought_segment, add_special_tokens=False)["input_ids"]
            cot_length = len(cot_ids)
            if cot_length > MAX_COT_TOKENS:
                dropped_long_cot += 1
                continue
        else:
            # 无 COT：再稍微严格一点，答案太长的不要
            if resp_len > NO_COT_ANSWER_MAX:
                dropped_no_cot_long_answer += 1
                continue

        # 4) 构造 messages（有 COT 才加 system）
        use_system = bool(has_cot)
        messages = build_messages(question, answer, thought, use_system)
        if messages is None:
            dropped_no_resp += 1
            continue

        txt = tok.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False,
        )

        # 5) 控制整条对话 ≤ 2048 tokens
        ids = tok(txt, add_special_tokens=False)["input_ids"]
        total_len = len(ids)
        if total_len > MAX_TOTAL_TOKENS:
            dropped_long_total += 1
            continue

        # 6) 写 JSONL
        rec = {
            "TXT": txt,
            "dataset": DATASET_TAG,
            "length": total_len,
            "has_cot": bool(has_cot),
            "cot_length": cot_length,
            "reasoning_len": None,        # 本数据集没有该字段
            "answer_tokens": resp_len,    # 整个 response 的 token 数
            "has_reference_answer": bool(ref_ans),
            "reference_answer": ref_ans,
            "response_model": response_model,
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
    print(f"    含 responses 的样本数: {with_resp}")
    print(f"    写入样本数: {kept_total}")
    print(f"      其中 解析出 COT 的样本数: {kept_with_cot}")
    print(f"      其中 无显式 COT 的样本数: {kept_no_cot}")
    print(f"    丢弃（无 response/空答案）: {dropped_no_resp}")
    print(f"    丢弃（response token 数 > {MAX_RESPONSE_TOKENS}）: {dropped_long_resp}")
    print(f"    丢弃（整条 TXT token 数 > {MAX_TOTAL_TOKENS}）: {dropped_long_total}")
    print(f"    丢弃（COT token 数 > {MAX_COT_TOKENS}）: {dropped_long_cot}")
    print(f"    丢弃（无 COT 且答案 token 数 > {NO_COT_ANSWER_MAX}）: {dropped_no_cot_long_answer}")
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
