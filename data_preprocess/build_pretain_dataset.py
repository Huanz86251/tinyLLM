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
random.seed(42)
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
def clean_text(text):
    # 去除隐私信息 电话、身份证
    text = re.sub(r'\d{11}', '', text)
    text = re.sub(r'\d{18}', '', text)
    # 去除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # 确保句子完整
    if not text.strip().endswith(('。', '！', '？', '…')):
        sentences = re.split(r'[。！？…]', text)
        text = ''.join(s + '。' for s in sentences[:-1] if s.strip())
    return text.strip()
def count_digit(txt):
    return len(re.findall(r'[0-9]',txt))
def filter_dataset(input_path, output_path,target_size=5*1024**3,max_length=1024,min_length=20,sample_percentage=1.0):
    files=os.listdir(input_path)
    total_size = 0
    os.makedirs(output_path,exist_ok=True)
    out_dir=os.path.join(output_path,"wudao_filtered_5gb.jsonl")
    question_id=0
    seen_hashes = set()
    with open(out_dir,'w',encoding='utf-8') as out:
        for file in tqdm(files,desc="filter_dataset"):
            if total_size > target_size:
                break
            try:
                with open(os.path.join(input_path,file),'r',encoding='utf-8') as f:

                    data=json.load(f)

                    k = int(len(data) * sample_percentage)
                    data=random.sample(data,k)
                    for sample in tqdm(data,desc=f"processing {file}"):

                        dataType = sample["dataType"]

                        if dataType != "新闻" and dataType != "百科":
                            continue
                        if dataType == "百科":
                            if random.random() > 0.4:
                                continue
                        content=sample["content"]
                        title=sample["title"]
                        text = f"{title} {content}".strip()
                        text = clean_text(text)
                        if contains_hangul(text) or contains_jp_kana(text) or contains_bopomofo(text) or contains_cyrillic(text) or contains_emoji(text) or contains_ipa(text) or contains_control(text):
                            continue
                        if dataType == "百科":
                            if count_digit(text) > 15:
                                continue
                        if len(text)>max_length or len(text)<min_length:
                            continue
                        text_hash = md5(text.encode('utf-8')).hexdigest()
                        if text_hash in seen_hashes:
                            continue
                        seen_hashes.add(text_hash)
                        uniqueKey=sample["uniqueKey"]
                        filterd_sample={"text":text,"question":question_id,"uniqueKey":uniqueKey}
                        filterd_sample=json.dumps(filterd_sample, ensure_ascii=False)
                        line_byte=filterd_sample.encode("utf-8")
                        question_id=question_id+1
                        total_size=total_size+len(line_byte)
                        out.write(filterd_sample + '\n')
            except Exception as e:
                    print(f"[skip] bad or non-JSON file")
                    continue

def main():
    input_path=r"D:\WuDaoCorporaText-2.0-open\WuDaoCorpus2.0_base_200G"
    output_path="../data/"
    os.makedirs(output_path,exist_ok=True)
    filter_dataset(input_path,output_path)

if __name__ == "__main__":
    main()
