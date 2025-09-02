#使用 晴数智慧高质量大模型多轮对话SFT数据集


import json
from tqdm import tqdm
import os
import regex as re
from hashlib import md5
import random
random.seed(42)
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

def render_chat_like(sys_txt: str, user_txt: str, assist_txt: str):
    return (
        f"<|im_start|>system\n{sys_txt}<|im_end|>\n"
        f"<|im_start|>user\n{user_txt}<|im_end|>\n"
        f"<|im_start|>assistant\n{assist_txt}<|im_end|>\n"
    )

def approx_len_messages(messages):
    """
    粗略估算一条多轮 messages 的字符长度（按你的模板头尾）
    messages: [{"role": "...", "content": "..."}]，含 system/user/assistant 多轮
    """
    total = 0
    for m in messages:
        total += len("<|im_start|>") + len(m["role"]) + 1  # '\n'
        total += len(m["content"])
        total += len("<|im_end|>\n")
    return total

def filter_dataset(input_path, output_path,
                   target_size=5*1024**3,
                   max_length=2000,
                   lines_per_chunk=20,
                   oversample_factor=2):
    files = [f for f in os.listdir(input_path) if f.lower().endswith(".txt")]
    total_size = 0
    os.makedirs(output_path, exist_ok=True)
    out_dir = os.path.join(output_path, "SFT.jsonl")
    seen_hashes = set()
    written = 0

    with open(out_dir, 'a', encoding='utf-8') as out:
        for filename in tqdm(files, desc="filter_dataset"):
            fpath = os.path.join(input_path, filename)
            if total_size > target_size:
                break
            try:


                with open(fpath, "r", encoding="utf-8") as f:
                    raw_lines = [ln.strip() for ln in f if ln.strip()]

                dialog = []
                cur_speaker = None
                buf = []
                for ln in raw_lines:
                    if ln.startswith(("A:", "A：")):
                        if cur_speaker is not None and buf:
                            text = "".join(buf).strip()
                            if text:
                                role = "user" if cur_speaker == 'A' else "assistant"
                                dialog.append((role, text))
                        cur_speaker = 'A'
                        part = ln.split(":", 1)[1] if ":" in ln else ln.split("：", 1)[1]
                        buf = [part.lstrip()]
                    elif ln.startswith(("B:", "B：")):
                        if cur_speaker is not None and buf:
                            text = "".join(buf).strip()
                            if text:
                                role = "user" if cur_speaker == 'A' else "assistant"
                                dialog.append((role, text))
                        cur_speaker = 'B'
                        part = ln.split(":", 1)[1] if ":" in ln else ln.split("：", 1)[1]
                        buf = [part.lstrip()]
                    else:
                        if cur_speaker is not None:
                            buf.append(ln)

                if cur_speaker is not None and buf:
                    text = "".join(buf).strip()
                    if text:
                        role = "user" if cur_speaker == 'A' else "assistant"
                        dialog.append((role, text))

                if not dialog:
                    continue

                merged = []
                for r, t in dialog:
                    if merged and merged[-1][0] == r:
                        merged[-1] = (r, (merged[-1][1] + "\n" + t).strip())
                    else:
                        merged.append((r, t.strip()))
                dialog = [(r, t) for r, t in merged if t]

                while dialog and dialog[0][0] != "user":
                    dialog.pop(0)
                while dialog and dialog[-1][0] != "assistant":
                    dialog.pop()
                if len(dialog) < 2:
                    continue

                for i in range(0, len(dialog), lines_per_chunk):
                    chunk = dialog[i:i + lines_per_chunk]
                    if len(chunk) < 2:
                        continue

                    while chunk and chunk[0][0] != "user":
                        chunk.pop(0)
                    while chunk and chunk[-1][0] != "assistant":
                        chunk.pop()
                    if len(chunk) < 2:
                        continue

                    messages = [{"role": "system", "content": SYS_DEFAULT}]
                    for role, text in chunk:
                        messages.append({"role": role, "content": text})


                    if approx_len_messages(messages) > max_length:
                        continue


                    joined_text = "\n".join([m["content"] for m in messages])
                    if (contains_hangul(joined_text) or contains_jp_kana(joined_text) or
                        contains_bopomofo(joined_text) or contains_cyrillic(joined_text) or
                        contains_emoji(joined_text) or contains_ipa(joined_text) or
                        contains_control(joined_text)):
                        continue


                    h = md5(("||".join([m["role"] + ":" + m["content"] for m in messages])).encode("utf-8")).hexdigest()
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)


                    rec = {"messages": messages, "tags": ["D", "multi"]}
                    line = json.dumps(rec, ensure_ascii=False) + "\n"
                    for _ in range(oversample_factor):
                        out.write(line)
                        total_size += len(line.encode("utf-8"))
                        written += 1

            except Exception as e:
                print(f"[skip] bad file {filename}: {e}")
                continue

    print(f"[done] wrote {written} samples to {out_dir}")


def main():
    input_path=r"D:\SFT_data\mutitune\TXT"
    output_path=r"E:\learn\data"
    os.makedirs(output_path,exist_ok=True)
    filter_dataset(input_path,output_path)

if __name__ == "__main__":
    main()
