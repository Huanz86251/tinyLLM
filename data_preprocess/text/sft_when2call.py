#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# ========= 配置区 =========

TOKENIZER_PATH = r"E:\learn\tiny_05B_cpt3\checkpoint-48000"   # 你的 tokenizer / ckpt 路径
OUTPUT_JSONL   = "when2call_train_sft_chattemplate.jsonl"     # 输出文件
DATASET_NAME   = "nvidia/When2Call"
DATASET_CONFIG = "train_sft"   # SFT 配置
MAX_TOTAL_TOKENS = 2048        # 整条 TXT token 上限


# ========= 小工具 =========
import re

def _map_type_to_jsonschema(t: str) -> str:
    """把 'str, optional' 这类字符串映射到 JSON Schema 的 type。"""
    if not t:
        return "string"
    t = t.strip()
    # 去掉 ", optional" 等后缀
    t = t.split(",")[0].strip().lower()

    if t in ("str", "string", "text"):
        return "string"
    if t in ("int", "integer", "long"):
        return "integer"
    if t in ("float", "double", "number"):
        return "number"
    if t in ("bool", "boolean"):
        return "boolean"
    if t in ("dict", "map", "object"):
        return "object"
    if t in ("list", "array", "sequence"):
        return "array"
    # 兜底
    return "string"


def _normalize_tool_for_template(tool: dict) -> dict:
    """
    把 When2Call 原始 tool schema 规范成模板期望的 JSON Schema 形式。
    支持两种输入:
      - 直接是 function 对象: {name, description, parameters, required, ...}
      - OpenAI 风格: {type:'function', function:{...}}
    """
    # 如果是 OpenAI 风格 {type:'function', function:{...}}，先拿出 function
    if "function" in tool:
        fn = dict(tool["function"])
    else:
        fn = dict(tool)  # 浅拷贝一份，别修改原对象

    params = fn.get("parameters") or {}
    if not isinstance(params, dict):
        params = {}

    # 1) 顶层 parameters.type: dict -> object
    p_type = params.get("type")
    if p_type is None:
        params["type"] = "object"
    else:
        # 有些写 'dict' / 'object' / 'Dict' 之类
        params["type"] = _map_type_to_jsonschema(str(p_type))

    # 2) 把顶层 required 搬到 parameters 里
    if "required" in fn and "required" not in params:
        params["required"] = fn["required"]
    elif "required" not in params:
        params["required"] = []

    # 3) 规范每个字段的 type
    props = params.get("properties") or {}
    if isinstance(props, dict):
        for name, spec in props.items():
            if not isinstance(spec, dict):
                continue
            t = spec.get("type")
            if isinstance(t, str):
                spec["type"] = _map_type_to_jsonschema(t)
            elif isinstance(t, list):
                # 联合类型：每个元素做一次映射
                spec["type"] = [_map_type_to_jsonschema(x) for x in t]
            else:
                # 没 type 的兜底成 string
                spec["type"] = "string"
    else:
        # 没 properties 就给个空 dict，避免模板里 json_spec.properties 报错
        params["properties"] = {}

    fn["parameters"] = params

    # 最终格式既可以直接是 fn，也可以包一层 type='function'
    # 模板里有:
    #   if tool.type is not defined or tool.type == 'function':
    #     if tool.function is defined: tool = tool.function
    # 所以我们直接给一个 {type:'function', function:fn} 最保险。
    return {
        "type": "function",
        "function": fn,
    }


def parse_tools(raw_tools):
    """
    从 When2Call 样本中解析 tools:
      - 原始是 JSON 字符串列表
      - 解析为 dict
      - 再规范成模板能吃的 JSON Schema
    """
    if not raw_tools:
        return []

    parsed = []
    for t in raw_tools:
        if t is None:
            continue

        # 既兼容字符串 JSON，也兼容已经是 dict 的情况
        if isinstance(t, str):
            s = t.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                # 某条 tool 坏掉就跳过这一个
                continue
        elif isinstance(t, dict):
            obj = t
        else:
            # 其他类型直接忽略
            continue

        norm = _normalize_tool_for_template(obj)
        parsed.append(norm)

    return parsed

def has_user_and_assistant(messages):
    """至少得有一个 user 和一个 assistant，才有监督价值。"""
    has_user = False
    has_assistant = False
    for m in messages:
        role = (m.get("role") or "").strip()
        if role == "user":
            has_user = True
        elif role == "assistant":
            has_assistant = True
        if has_user and has_assistant:
            return True
    return False


# ========= 主流程 =========

def main():
    print(">>> 加载 tokenizer ...")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    if tok.chat_template is None:
        raise ValueError(
            "当前 tokenizer.chat_template 为空。\n"
            "请先确认已经把你那段带工具调用/思维链的 Jinja 模板写进 tokenizer 再跑。"
        )

    print(">>> 加载 nvidia/When2Call (train_sft) ...")
    ds_dict = load_dataset(DATASET_NAME, DATASET_CONFIG)
    ds = ds_dict["train"]   # 只有一个 split: train

    out_path = Path(OUTPUT_JSONL)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = out_path.open("w", encoding="utf-8")

    total = 0
    kept_total = 0
    dropped_bad_messages = 0
    dropped_template_error = 0
    dropped_too_long = 0

    total_bytes = 0

    for ex in tqdm(ds, desc="When2Call train_sft → TXT"):
        total += 1

        raw_tools = ex.get("tools", None)
        messages = ex.get("messages", None)

        # 基本结构缺失直接丢
        if not messages or not isinstance(messages, list):
            dropped_bad_messages += 1
            continue
        if not has_user_and_assistant(messages):
            dropped_bad_messages += 1
            continue

        tools = parse_tools(raw_tools or [])

        # 直接把 messages + tools 丢给 chat_template
        try:
            txt = tok.apply_chat_template(
                messages,
                tools=tools,                # 关键：把 tools 传进去，模板会渲染函数列表 + 规则
                add_generation_prompt=False,
                tokenize=False,
            )
        except Exception as e:
            # 某些奇怪样本在模板里炸掉，就跳过
            dropped_template_error += 1
            continue

        # 长度控制：2048 以内
        ids = tok(txt, add_special_tokens=False)["input_ids"]
        total_len = len(ids)
        if total_len > MAX_TOTAL_TOKENS:
            dropped_too_long += 1
            continue

        rec = {
            "TXT": txt,
            "dataset": DATASET_NAME,
            "subset": DATASET_CONFIG,
            "length": total_len,
            "has_tools": bool(tools),
            "num_tools": len(tools),
        }

        line = json.dumps(rec, ensure_ascii=False) + "\n"
        fout.write(line)
        kept_total += 1
        total_bytes += len(line.encode("utf-8"))

    fout.close()

    mb = total_bytes / (1024 * 1024)

    print(">>> 完成！")
    print(f"    原始样本数                    : {total}")
    print(f"    写入样本数                    : {kept_total}")
    print(f"    丢弃（messages 缺失/无 user/assistant）: {dropped_bad_messages}")
    print(f"    丢弃（模板渲染报错）          : {dropped_template_error}")
    print(f"    丢弃（整条 TXT > {MAX_TOTAL_TOKENS} tokens）: {dropped_too_long}")
    print(f"    输出体积估算                  : {mb:.2f} MB")
    print(f"    输出文件                      : {out_path.resolve()}")


if __name__ == "__main__":
    main()
