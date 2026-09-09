# pip install datasets tqdm
from datasets import load_dataset
from tqdm import tqdm
import json, hashlib, random

# ===== 目标体积（约 200 MB JSONL）=====
BYTES_BUDGET = 1000 * 1024 * 1024   # 200 MiB
OUT_PATH     = "tiny_codes_1GB.jsonl"

# 语言优先级：先 Python -> C/C++ -> Java -> 其他
PHASES = [
    {"names": {"python"},       "label": "python"},
    {"names": {"c", "c++"},     "label": "c_cpp"},
    {"names": {"java"},         "label": "java"},
    {"names": "others",         "label": "others"},   # 除前三类外的所有语言
]

# 质量/噪声简单过滤（可按需要调整/删除）
MAX_FILE_BYTES = 120_000  # 单条最大字符字节数（避免超大文件/粘连）
MIN_LINES      = 2        # 至少两行
MAX_LINES      = 400      # 单文件最多 400 行
# 其他更激进的规则（比如最长行长度/字母数字占比）数据集里未必有相应字段，这里先不加

def get_lang(row):
    # tiny-codes 常见字段是 'lang'，兜底兼容 'language'
    return (row.get("programming_language") or row.get("language") or "").lower().strip()

def get_text(row):
    # 常见字段是 'content'，兜底兼容 'code'/'text'
    return row.get("response") or row.get("code") or row.get("text") or ""

def unique_key(text, row):
    # tiny-codes没有固定唯一ID，就用 md5(text)；如有 'id'/'path' 可优先用
    return row.get("common_sense_topic") or row.get("path") or hashlib.md5(text.encode("utf-8")).hexdigest()

def ok_quality(text):
    if not text:
        return False
    b = len(text.encode("utf-8"))
    if b > MAX_FILE_BYTES:
        return False
    n_lines = text.count("\n") + 1
    if n_lines < MIN_LINES or n_lines > MAX_LINES:
        return False
    return True

def iter_phase(ds_iter, accept_langs, already_bytes, budget_bytes, start_qid):
    """
    从数据流中筛选指定语言；返回新增的字节、行数、最新 question id。
    - accept_langs: set[str] 或 "others"
    """
    bytes_add = 0
    rows_add  = 0
    qid       = start_qid

    for row in ds_iter:
        lang = get_lang(row)
        if not lang:
            continue

        if accept_langs == "others":
            # 除去前三类
            if lang in {"python", "c", "c++", "java"}:
                continue
        else:
            if lang not in accept_langs:
                continue

        text = get_text(row)
        if not ok_quality(text):
            continue

        # 三字段输出
        rec = {
            "text": text,
            "question": lang,
            "uniqueKey": unique_key(text, row),
        }
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        lb = len(line.encode("utf-8"))

        # 不超过总预算：这条会超就跳过，继续找下一条，尽量贴近预算
        if already_bytes + bytes_add + lb > budget_bytes:
            continue

        yield line, lb
        qid       += 1
        rows_add  += 1
        bytes_add += lb

        if already_bytes + bytes_add >= budget_bytes:
            break

def reload_stream():
    # 每个阶段重新构造 streaming 迭代器（因为 streaming 只能单向遍历）
    # tiny-codes 没有多个 split，就直接用 train
    return load_dataset("nampdn-ai/tiny-codes", split="train", streaming=True)

def main():
    total_bytes = 0
    qid = 0

    with open(OUT_PATH, "w", encoding="utf-8") as f, tqdm(total=BYTES_BUDGET, unit="B", desc="Writing ~200MB JSONL (tiny-codes)") as pbar:
        for phase in PHASES:
            if total_bytes >= BYTES_BUDGET:
                break
            ds_iter = reload_stream()

            accept = phase["names"]
            for line, lb in iter_phase(ds_iter, accept, total_bytes, BYTES_BUDGET, qid):
                f.write(line)
                total_bytes += lb
                qid         += 1
                pbar.update(lb)

                if total_bytes >= BYTES_BUDGET:
                    break

    print(f"[OK] wrote {OUT_PATH}")
    print(f"[INFO] bytes≈{total_bytes/1024/1024:.1f} MB  rows={qid:,}")

if __name__ == "__main__":
    main()
