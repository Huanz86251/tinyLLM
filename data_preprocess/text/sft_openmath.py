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
OUTPUT_JSONL = "openmathinstruct2_cot_from_answer_with_system.jsonl"

DATASET_NAME = "nvidia/OpenMathInstruct-2"
DATASET_SPLIT = "train"  # 可改为 "train_1M" / "train_2M" / "train_5M"

MAX_TOTAL_TOKENS = 2048      # 整条对话 TXT 的 token 上限（含 system）
MAX_COT_TOKENS   = 450       # 思维链 token 上限

RESP_SOFT_MAX       = 500    # > 500 视为“偏长回答”，进入抽样逻辑
RESP_HARD_MAX       = 1000   # > 1000 直接丢弃
KEEP_LONG_RESP_PROB = 0.02   # 500–RESP_HARD_MAX 之间，以 2% 概率保留

# 无 COT 的样本：回答超过 450 token 直接丢弃
NO_COT_ANSWER_MAX = 450

# 无 COT + 有 REF + 答案中包含 REF 的样本：再丢 1/3（保留 2/3）
DROP_NO_COT_PROB_WITH_REF = 1.0 / 3.0

# 当存在 COT 时添加的英文 system 提示
SYSTEM_PROMPT_FOR_COT = (
    "Think step by step inside <|thought_start|> and <|thought_end|>. Then provide final answer."
)

# 扩展后的模板集合（全部转成小写使用）
ANSWER_MARKERS = [
    "the final answer is",
    "the answer is",
    "therefore,",
    "in conclusion,",
    "conclusion,",
    "in summary,",
    "thus,",
    "so,",
]

TAIL_MAX_TOKENS = 15   # 模板之后的 token 数不能超过 15

random.seed(42)  # 方便复现实验


# ========= 小工具函数 =========

def split_solution_with_answer_marker(raw: str, tok: AutoTokenizer):
    """
    从 generated_solution 中提取 (thought, answer)。

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
    tail_ids = tok(tail_text, add_special_tokens=False)["input_ids"]
    if len(tail_ids) > TAIL_MAX_TOKENS:
        return None, raw, "marker_tail_too_long"

    # 4) 上述两关通过，认为这是最后总结答案：
    #    从模板“起始位置”往前找最近的 '\n'
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


def build_messages(question: str, answer: str, thought: str | None, use_system: bool):
    """
    - 有 thought 且 use_system=True: 在最前添加 system，再用 "thought" 字段
    - 有 thought 且 use_system=False: 仅 assistant 带 "thought"
    - 无 thought: 普通 assistant 回答（不带 "thought"）
    """
    question = question.strip()
    answer = answer.strip()
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

    print(f">>> 加载 {DATASET_NAME} ({DATASET_SPLIT} split) ...")
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    out_path = Path(OUTPUT_JSONL)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = out_path.open("w", encoding="utf-8")

    # 统计量
    total = 0
    with_resp = 0
    kept_total = 0
    kept_with_cot = 0
    kept_no_cot = 0

    dropped_no_resp = 0
    dropped_no_ref = 0
    dropped_resp_hard = 0
    dropped_resp_soft = 0
    kept_resp_soft = 0
    dropped_cot_too_long = 0
    dropped_long_total = 0

    dropped_no_cot_long_answer = 0
    dropped_no_cot_random = 0
    dropped_answer_mismatch = 0

    marker_ok = 0
    marker_invalid_many_newlines = 0
    marker_tail_too_long = 0
    marker_but_no_prefix = 0
    marker_none = 0

    for ex in tqdm(ds, desc="OpenMathInstruct-2 → TXT(+COT when possible)"):
        total += 1

        # 字段：problem / generated_solution / expected_answer / problem_source
        question = (ex.get("problem") or "").strip()
        raw_response = (ex.get("generated_solution") or "").strip()
        ref_ans = (ex.get("expected_answer") or "").strip()
        problem_source = ex.get("problem_source")

        if not question or not raw_response:
            dropped_no_resp += 1
            continue

        # 没有 expected_answer 的样本一律丢弃（不管有无 COT）
        if not ref_ans:
            dropped_no_ref += 1
            continue

        with_resp += 1

        # 1) 整个 generated_solution 的 token 数
        resp_ids = tok(raw_response, add_special_tokens=False)["input_ids"]
        resp_len = len(resp_ids)

        # 1.1 硬阈值：> RESP_HARD_MAX 直接丢弃
        if resp_len > RESP_HARD_MAX:
            dropped_resp_hard += 1
            continue

        # 1.2 软阈值：500–RESP_HARD_MAX 之间，以 2% 概率保留
        if resp_len > RESP_SOFT_MAX:
            if random.random() >= KEEP_LONG_RESP_PROB:
                dropped_resp_soft += 1
                continue
            else:
                kept_resp_soft += 1

        # 2) 按模板分成 thought + answer（内部会做换行 + tail token 过滤）
        thought, answer, status = split_solution_with_answer_marker(raw_response, tok)
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

        # 3) 无论有无 COT，只要 reference answer 没出现在 answer 里，直接丢弃
        #   （你宁可少一些，也不要模型学错模板）
        if ref_ans not in answer:
            dropped_answer_mismatch += 1
            continue

        has_cot = thought is not None and thought.strip()

        # 4) COT / 无 COT 的额外约束
        cot_length = 0
        if has_cot:
            # COT 长度限制
            thought_segment = "<|thought_start|>\n" + thought + "\n<|thought_end|>\n"
            cot_ids = tok(thought_segment, add_special_tokens=False)["input_ids"]
            cot_length = len(cot_ids)
            if cot_length > MAX_COT_TOKENS:
                dropped_cot_too_long += 1
                continue
        else:
            # 无 COT：答案太长直接丢
            if resp_len > NO_COT_ANSWER_MAX:
                dropped_no_cot_long_answer += 1
                continue

            # 走到这里说明：
            #   - 有 ref（上面已经过滤掉无 ref）
            #   - ref 已经确认出现在 answer 中
            # 对这类“高置信无 COT 样本”再丢 1/3，保留 2/3
            if random.random() < DROP_NO_COT_PROB_WITH_REF:
                dropped_no_cot_random += 1
                continue

        # 5) 构造 messages（有 COT 才加 system）
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

        # 6) 控制整条对话 ≤ MAX_TOTAL_TOKENS（此时已包含 system）
        ids = tok(txt, add_special_tokens=False)["input_ids"]
        total_len = len(ids)
        if total_len > MAX_TOTAL_TOKENS:
            dropped_long_total += 1
            continue

        # 7) 写 JSONL，结构和之前保持一致
        rec = {
            "TXT": txt,
            "dataset": DATASET_NAME,
            "split": DATASET_SPLIT,
            "length": total_len,
            "has_cot": bool(has_cot),
            "cot_length": cot_length,
            "reasoning_len": None,          # 这里没有额外字段，就占位
            "answer_tokens": resp_len,      # 整个 generated_solution 的 token 数
            "has_reference_answer": True,   # 现在保证有 ref 才会走到这里
            "reference_answer": ref_ans,
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
    print(f"    含 problem + generated_solution 的样本数: {with_resp}")
    print(f"    写入样本数: {kept_total}")
    print(f"      其中 解析出 COT 的样本数: {kept_with_cot}")
    print(f"      其中 无显式 COT 的样本数: {kept_no_cot}")
    print()
    print(f"    丢弃（无 problem/空答案）: {dropped_no_resp}")
    print(f"    丢弃（expected_answer 为空）: {dropped_no_ref}")
    print(f"    丢弃（resp_len > {RESP_HARD_MAX}）: {dropped_resp_hard}")
    print(f"    丢弃（{RESP_SOFT_MAX} < resp_len ≤ {RESP_HARD_MAX}，2% 抽样筛掉的）: {dropped_resp_soft}")
    print(f"    保留（{RESP_SOFT_MAX} < resp_len ≤ {RESP_HARD_MAX}，2% 抽样保留下的）: {kept_resp_soft}")
    print(f"    丢弃（COT token 数 > {MAX_COT_TOKENS}）: {dropped_cot_too_long}")
    print(f"    丢弃（整条 TXT token 数 > {MAX_TOTAL_TOKENS}）: {dropped_long_total}")
    print(f"    丢弃（无 COT 且答案 token 数 > {NO_COT_ANSWER_MAX}）: {dropped_no_cot_long_answer}")
    print(f"    丢弃（无 COT 且随机 1/3 丢弃）: {dropped_no_cot_random}")
    print(f"    丢弃（参考答案未出现在回答中）: {dropped_answer_mismatch}")
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
