# pip install datasets tqdm
from datasets import load_dataset
from tqdm import tqdm
import json, hashlib

# ====== 参数 ======
DATASET_NAME = "ajibawa-2023/Children-Stories-Collection"
OUT_PATH     = "children_stories_2G.jsonl"
BYTES_BUDGET = 2000 * 1024 * 1024   # 700 MB

LOG_EVERY_N  = 10_000

# ====== 加载 (streaming 模式避免一次性加载到内存) ======
ds = load_dataset(DATASET_NAME, split="train", streaming=True)

bytes_written = 0
wrote_rows    = 0

with open(OUT_PATH, "w", encoding="utf-8") as f, tqdm(total=BYTES_BUDGET, unit="B", desc="Writing ~700MB JSONL") as pbar:
    for doc in ds:
        text = doc.get("text", "").strip()
        if not text:
            continue

        rec = {
            "text": text,
            "question": "Story",
            "uniqueKey": hashlib.md5(text.encode("utf-8")).hexdigest()
        }

        line = json.dumps(rec, ensure_ascii=False) + "\n"
        line_bytes = len(line.encode("utf-8"))

        # 不超过预算
        if bytes_written + line_bytes > BYTES_BUDGET:
            break

        f.write(line)
        wrote_rows   += 1
        bytes_written += line_bytes
        pbar.update(line_bytes)

        if (wrote_rows % LOG_EVERY_N) == 0:
            bpr = bytes_written / max(wrote_rows, 1)
            pbar.set_postfix_str(f"rows={wrote_rows:,}, B/row≈{bpr:.1f}")

print(f"[OK] wrote {OUT_PATH}")
print(f"[INFO] bytes≈{bytes_written/1024/1024:.1f} MB, rows={wrote_rows:,}")
