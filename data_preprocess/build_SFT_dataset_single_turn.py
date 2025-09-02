#使用数据集https://huggingface.co/datasets/codefuse-ai/CodeExercise-Python-27k
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

def filter_dataset(input_path, output_path,target_size=5*1024**3,max_length=2000):
    files=os.listdir(input_path)
    total_size = 0
    os.makedirs(output_path,exist_ok=True)
    out_dir=os.path.join(output_path,"SFT.jsonl")
    seen_hashes = set()
    with open(out_dir,'a',encoding='utf-8') as out:
        for file in tqdm(files,desc="filter_dataset"):
            if total_size > target_size:
                break
            try:
                with open(os.path.join(input_path,file),'r',encoding='utf-8') as f:

                    data= [json.loads(line) for line in f if line.strip()]
                    for sample in tqdm(data,desc=f"processing {file}"):
                        instr = (sample.get("instruction") or "").strip()
                        inp = (sample.get("input") or "").strip()
                        outp = (sample.get("output") or "").strip()
                        user_text = instr if not inp else f"{instr}\n\n{inp}"
                        sys_text = SYS_DEFAULT
                        assist_text = outp
                        approx_len = len(render_chat_like(sys_text, user_text, assist_text))

                        if approx_len > max_length:  # 预留48个字符
                            target_size += approx_len
                            continue

                        text=sys_text+user_text+assist_text
                        if contains_hangul(text) or contains_jp_kana(text) or contains_bopomofo(
                            text) or contains_cyrillic(text) or contains_emoji(text) or contains_ipa(
                            text) or contains_control(text): continue
                        text_hash = md5((text).encode('utf-8')).hexdigest()
                        if text_hash in seen_hashes:
                            continue
                        seen_hashes.add(text_hash)
                        filtered_sample = {
                            "messages": [
                                {"role": "system", "content": sys_text},
                                {"role": "user", "content": user_text},
                                {"role": "assistant", "content": assist_text}
                            ],
                            "tags": ["D"]
                        }
                        out.write(json.dumps(filtered_sample, ensure_ascii=False) + "\n")
            except Exception as e:
                    print(f"[skip] bad or non-JSON file")
                    continue

def main():
    input_path=r"D:\SFT_data"
    output_path="../data/"
    os.makedirs(output_path,exist_ok=True)
    filter_dataset(input_path,output_path)

if __name__ == "__main__":
    main()
