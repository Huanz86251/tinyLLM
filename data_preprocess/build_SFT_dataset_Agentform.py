#@misc{bai2024coig,
  #   title={COIG-CQIA: Quality is All You Need for Chinese Instruction Fine-tuning},
  #   author={Bai, Yuelin and Du, Xinrun and Liang, Yiming and Jin, Yonggang and Liu, Ziqiang and Zhou, Junting and Zheng, Tianyu and Zhang, Xincheng and Ma, Nuo and Wang, Zekun and others},
  #   year={2024},
  #   eprint={2403.18058},
  #   archivePrefix={arXiv},
  #   primaryClass={cs.CL}
  # }
# Data source: WuDaoCorpora 2.0
# Provider: Beijing Academy of Artificial Intelligence (BAAI)
# Year: 2021
# URL: https://www.scidb.cn/en/detail?dataSetId=c6a3fe684227415a9db8e21bac4a15ab
# License: Please see dataset page for details. Used for research; original authors credited.

import json
from tqdm import tqdm
import os
import regex as re
from hashlib import md5
import random
from collections import deque
random.seed(42)
RESERVED_TOKENS = [f"<|reserved_{i}|>" for i in range(9)]
NO_CALL_TOKEN="<|reserved_9|>"
SYS_DEFAULT = "你是一个乐于助人的中文助手"
# 覆盖：Jamo、兼容Jamo、Jamo扩展A/B、Hangul音节
RE_HANGUL = re.compile(r"[\u1100-\u11FF\u3130-\u318F\uA960-\uA97F\uAC00-\uD7A3\uD7B0-\uD7FF]")

# Japanese Kana（日文假名，不与中文重叠）
# 覆盖：平假名、片假名、片假名音标扩展、小写片假名扩展、半角片假名、Kana扩展A/B、Kana补充等
RE_JP_KANA = re.compile(
    r"[\u3040-\u309F\u30A0-\u30FF\u31F0-\u31FF\uFF65-\uFF9F"
    r"\U0001AFF0-\U0001AFFF\U0001B000-\U0001B16F]"
)

# IPA 国际音标扩展 (ɝ ʊ ᴧ ...)
RE_IPA = re.compile(r"[\u0250-\u02AF]")
# 注音符号（ㄅㄆㄇㄈ…）
RE_BOPOMOFO = re.compile(r"[\u3100-\u312F]")
# 西里尔字母（俄语等）
RE_CYRILLIC = re.compile(r"[\u0400-\u04FF]")
# Emoji (可选)
RE_EMOJI = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u26FF\u2700-\u27BF]")
#控制字符
RE_CONTROL = re.compile(r"[\u200B-\u200F\uFEFF]")
def contains_control(s: str) -> bool:
    return RE_CONTROL.search(s) is not None

def contains_emoji(s: str) -> bool:
    return RE_EMOJI.search(s) is not None
def contains_cyrillic(s: str) -> bool:
    return RE_CYRILLIC.search(s) is not None
def contains_ipa(s: str) -> bool:
    return RE_IPA.search(s) is not None
def contains_bopomofo(s: str) -> bool:
    return RE_BOPOMOFO.search(s) is not None
def contains_hangul(s: str) -> bool:
    return RE_HANGUL.search(s) is not None

def contains_jp_kana(s: str) -> bool:
    return RE_JP_KANA.search(s) is not None
def _shorten(s, max_chars=80):
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= max_chars else (s[:max_chars] + "…")

def extract_tool_select(sample, max_cands=3, keep_correct_first=True):
    """
    支持两种样本：
      - 正例：assistant 有 function_call -> 选择相应 <|reserved_k|>
      - 负例：assistant 无 function_call -> 统一输出 NO_CALL_TOKEN（<|reserved_9|>）
    每条样本仅为“当下选中的 ≤3 个候选”分配 reserved_k；不暴露真实函数名。
    """
    rounds = sample.get("chatrounds") or []
    if not rounds:
        return None, None


    user_txt, tool_name, tool_args = None, None, None
    for m in rounds:
        if user_txt is None and m.get("role") == "user" and m.get("content"):
            user_txt = str(m["content"]).strip()
        if tool_name is None and m.get("role") == "assistant" and m.get("function_call"):
            fc = m.get("function_call") or {}
            tool_name = fc.get("name")
    if not user_txt:
        return None, None

    is_positive = tool_name is not None


    name2desc = {}
    for fn in sample.get("functions", []):
        n = (fn or {}).get("name")
        d = (fn or {}).get("description") or ""
        if n:
            name2desc[n] = d


    cands = []

    if is_positive:
        desc = name2desc[tool_name]
        cands.append((tool_name, desc))

    keys = list(name2desc.keys())
    random.shuffle(keys)
    for k in keys:
        if len(cands) >= max_cands:
            break
        if (not is_positive) or (k != tool_name):
            cands.append((k, name2desc[k]))

    if not cands:
        user_text = f"可用工具：\n（无）\n问题：{user_txt}"
        assist_text = f"使用工具：{NO_CALL_TOKEN}"
        return user_text, assist_text

    k = min(len(RESERVED_TOKENS), len(cands))
    cands = cands[:k]

    tokens = random.sample(RESERVED_TOKENS, k=k)
    assign = {name: tok for (name, _), tok in zip(cands, tokens)}

    lines = [f"- {assign[n]}：{_shorten(d, 80)}" for (n, d) in cands]
    lines.append(f"- {NO_CALL_TOKEN}：如果发现没有工具可以调用，选择此 unknown 选项")
    user_text = "可用工具：\n" + "\n".join(lines) + f"\n问题：{user_txt}"

    if is_positive and (tool_name in assign):
        assist_text = f"使用工具：{assign[tool_name]}"

    else:

        assist_text = f"使用工具：{NO_CALL_TOKEN}"

    return user_text, assist_text
def render_chat_like(sys_txt: str, user_txt: str, assist_txt: str):
    return (
        f"<|im_start|>system\n{sys_txt}<|im_end|>\n"
        f"<|im_start|>user\n{user_txt}<|im_end|>\n"
        f"<|im_start|>assistant\n{assist_txt}<|im_end|>\n"
    )

def filter_dataset(input_path, output_path,target_size=5*1024**3,max_length=2000):
    files=os.listdir(input_path)
    total_size = 0
    os.makedirs(output_path,exist_ok=True)
    out_dir=os.path.join(output_path,"SFT.jsonl")
    print(out_dir)
    seen_hashes = set()
    with open(out_dir,'a',encoding='utf-8') as out:
        for file in tqdm(files,desc="filter_dataset"):
            if total_size > target_size:
                break

            with open(os.path.join(input_path,file),'r',encoding='utf-8') as f:
                sys_text = SYS_DEFAULT
                data= [json.loads(line) for line in f if line.strip()]
                for sample in tqdm(data,desc=f"processing {file}"):

                    user_text, assist_text = extract_tool_select(sample, max_cands=3, keep_correct_first=True)
                    if not user_text:
                        continue

                    tag_list = ["A", "tool-select", "zh"]
                    approx_len = len(render_chat_like(sys_text, user_text, assist_text))

                    if approx_len > max_length:  # 预留48个字符
                        continue

                    text=sys_text+user_text+assist_text
                    if contains_hangul(text) or contains_jp_kana(text) or contains_bopomofo(
                        text) or contains_cyrillic(text) or contains_emoji(text) or contains_ipa(
                        text) or contains_control(text): continue


                    filtered_sample = {
                        "messages": [
                            {"role": "system", "content": sys_text},
                            {"role": "user", "content": user_text},
                            {"role": "assistant", "content": assist_text}
                        ],
                        "tags": tag_list
                    }

                    line = json.dumps(filtered_sample, ensure_ascii=False) + "\n"
                    out.write(line)
                    total_size += len(line.encode("utf-8"))


def main():
    input_path=r"D:\SFT_data"
    output_path="../data/"
    os.makedirs(output_path,exist_ok=True)
    filter_dataset(input_path,output_path)

if __name__ == "__main__":
    main()
