from project_paths import legacy_path
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, time
import numpy as np
from datetime import datetime
from datasets import load_dataset, Dataset, Features, Sequence, Value
from transformers import AutoTokenizer
import xxhash
import sys

# ================== 全部写死（按需改） ==================
MAX_LEN        = 8192
NUM_PROC_TOK   = 60         # 仅用于 tokenize；分块单线程串流
BATCH_SIZE_TOK = 400
APPEND_EOS     = True

# 固定路径（与你训练脚本一致）
TOK_DIR   = legacy_path("/root/autodl-tmp/llm/KDmodel/minicpm3_4b/snapshot")
OUT_DIR   = legacy_path("/root/autodl-tmp/llm/cache_pretrain_large/packed_len8k_last")

# JSONL：你当前的长文本语料
JSONL     = legacy_path("/root/autodl-tmp/mergedlong.filtered.tokshuf.jsonl")

# 强制离线环境（不触网）
os.environ.update({
    "HF_HOME": legacy_path("/root/autodl-tmp/hf_home_strict_offline"),
    "HF_DATASETS_CACHE": legacy_path("/root/autodl-tmp/hf_datasets_cache_force"),
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_ENABLE_ONLINE_MODE": "0",
    "TOKENIZERS_PARALLELISM": "false",
})
# ======================================================

def die(msg):
    print(f"[FATAL] {msg}", flush=True)
    sys.exit(1)

def xxh3_file(path):
    h = xxhash.xxh3_128()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    dig = h.intdigest()
    return int(dig>>64), int(dig & ((1<<64)-1))

def xxh3_ids_uint32(arr_uint32: np.ndarray):
    h = xxhash.xxh3_128()
    h.update(arr_uint32.tobytes(order="C"))
    dig = h.intdigest()
    return np.uint64(dig>>64), np.uint64(dig & ((1<<64)-1))

def main():
    # 基本校验
    if not os.path.isdir(TOK_DIR): die(f"Tokenizer dir not found: {TOK_DIR}")
    if not os.path.isfile(os.path.join(TOK_DIR, "tokenizer.json")):
        die(f"Missing tokenizer.json in {TOK_DIR}")
    if not os.path.isfile(JSONL):
        die(f"JSONL not found: {JSONL}")

    if os.path.exists(OUT_DIR):
        die(f"OUT_DIR already exists, abort to avoid overwrite: {OUT_DIR}\n"
            f"→ 手动 rm -rf 后再运行以重建。")

    os.makedirs(OUT_DIR, exist_ok=False)

    # 加载 tokenizer（本地）
    tok = AutoTokenizer.from_pretrained(TOK_DIR, trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eos_id = tok.eos_token_id
    vocab_size = len(tok.get_vocab())
    tok_hi, tok_lo = xxh3_file(os.path.join(TOK_DIR, "tokenizer.json"))
    print(f"[TOKENIZER] dir={TOK_DIR}")
    print(f"[TOKENIZER] vocab_size={vocab_size} eos_id={eos_id} xxh3(tokenizer.json) hi={tok_hi} lo={tok_lo}")

    # 读取 JSONL
    print(f"[LOAD] jsonl={JSONL}")
    raw = load_dataset("json", data_files=JSONL, split="train",
                       cache_dir=os.environ["HF_DATASETS_CACHE"],
                       download_mode="force_redownload",
                       keep_in_memory=False)

    # 多进程 tokenize（保序）
    def tokenize_fn(batch):
        ids = tok(batch["TXT"],
                  add_special_tokens=False,
                  truncation=False,
                  padding=False,
                  return_attention_mask=False)["input_ids"]
        if APPEND_EOS:
            ids = [x + [eos_id] for x in ids]
        return {"input_ids": ids, "len": [len(x) for x in ids]}

    print(f"[TOKENIZE] num_proc={NUM_PROC_TOK} batch_size={BATCH_SIZE_TOK}")
    tokenized = raw.map(
        tokenize_fn,
        batched=True,
        remove_columns=raw.column_names,
        num_proc=NUM_PROC_TOK,
        batch_size=BATCH_SIZE_TOK,
        desc="Tokenizing",
        load_from_cache_file=False,
        keep_in_memory=False
    )
    assert "input_ids" in tokenized.features

    # 统计：总 token & 可用长样本（>=8k）的索引
    print("[STATS] counting eligible long samples (>=16k) ...")
    lens = np.asarray(tokenized["len"], dtype=np.int64)

    total_tokens = int(lens.sum())
    eligible_idx = np.flatnonzero(lens >= MAX_LEN).astype(np.int64)  # 长文样本索引

    # L >= MAX_LEN+64 的样本切 3 窗，否则切 1 窗
    long_mask = lens[eligible_idx] >= (MAX_LEN + 64)
    blocks_per_doc = 1 + long_mask.astype(np.int64) * 2
    num_blocks = int(blocks_per_doc.sum())

    if num_blocks == 0:
        die(f"No samples >= {MAX_LEN} tokens. Nothing to build.")
    print(f"[STATS] total_tokens={total_tokens:,}  MAX_LEN={MAX_LEN}  num_blocks={num_blocks:,}")
    # 预分配块级哈希与来源追踪
    hi_path = os.path.join(OUT_DIR, "hash_hi.npy")
    lo_path = os.path.join(OUT_DIR, "hash_lo.npy")
    src_path = os.path.join(OUT_DIR, "block_source.npy")
    off_path = os.path.join(OUT_DIR, "block_offset.npy")

    hash_hi = np.memmap(hi_path, dtype=np.uint64, mode="w+", shape=(num_blocks,))
    hash_lo = np.memmap(lo_path, dtype=np.uint64, mode="w+", shape=(num_blocks,))
    blk_src = np.memmap(src_path, dtype=np.int64,  mode="w+", shape=(num_blocks,))
    blk_off = np.memmap(off_path, dtype=np.int64,  mode="w+", shape=(num_blocks,))
    hash_hi[:] = 0; hash_lo[:] = 0; blk_src[:] = -1; blk_off[:] = -1

    # 生成器：每条样本最多 1 个 8K 窗口；样本足够长时起点 0~64 轻抖动（确定性）
    def block_generator_long():
        """
        - len < MAX_LEN: 排除
        - MAX_LEN <= len < MAX_LEN + 64: 产出 1 个窗口（起点带 ≤64 的确定性抖动）
        - len >= MAX_LEN + 64: 产出 3 个窗口（均匀三锚点 + 每锚点 ±32 抖动，确定性）
        - 不跨样本；同一文档的窗口起点有序递增
        """
        produced = 0
        block_idx = 0
        GLOBAL_SEED = 20251023  # 固定随机性

        for i in eligible_idx:
            ids = np.asarray(tokenized[i]["input_ids"], dtype=np.int32)
            L = ids.size
            max_start = max(0, L - MAX_LEN)  # 起点上界
            rng = np.random.default_rng(xxhash.xxh3_64_intdigest(f"{i}-{GLOBAL_SEED}".encode()))

            starts: list[int]
            if L < (MAX_LEN + 64):
                # 仅 1 窗：起点 ∈ [0, min(64, max_start)]
                if max_start == 0:
                    s0 = 0
                else:
                    s0 = int(rng.integers(0, min(64, max_start) + 1))
                starts = [s0]
            else:
                # 3 窗：均匀三锚点 + 每锚点 ±32 抖动后裁剪
                anchors = np.linspace(0, max_start, num=3, dtype=np.int64)  # 0, ~中点, max_start
                jitter_span = min(64, max_start)  # 保障上界
                half = jitter_span // 2
                cand = []
                for a in anchors:
                    if max_start == 0:
                        s = 0
                    else:
                        j = int(rng.integers(-half, half + 1))  # [-half, half]
                        s = int(np.clip(a + j, 0, max_start))
                    cand.append(s)
                # 去重并排序，若不足 3 个，用随机补齐
                cand = sorted(set(cand))
                while len(cand) < 3 and max_start > 0:
                    s = int(rng.integers(0, max_start + 1))
                    if s not in cand:
                        cand.append(s)
                starts = sorted(cand[:3])

            # 产出窗口 + 记录来源/哈希
            for start in starts:
                end = start + MAX_LEN
                if end > L:
                    # 安全钳制（极端边界）
                    start = max(0, L - MAX_LEN)
                    end = start + MAX_LEN

                window = ids[start:end]
                hi, lo = xxh3_ids_uint32(window.view(np.uint32))
                hash_hi[block_idx] = hi
                hash_lo[block_idx] = lo
                blk_src[block_idx] = i
                blk_off[block_idx] = start

                produced += 1
                block_idx += 1
                yield {"input_ids": window.tolist()}

        if produced != num_blocks:
            raise RuntimeError(f"produced={produced}, expected={num_blocks}")
    # 写 HF dataset
    print(f"[BUILD] writing dataset to {OUT_DIR} ...")
    features = Features({"input_ids": Sequence(Value("int32"))})
    ds = Dataset.from_generator(block_generator_long, features=features)
    ds.save_to_disk(OUT_DIR)

    # 刷盘
    hash_hi.flush(); hash_lo.flush(); blk_src.flush(); blk_off.flush()

    # manifest
    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "jsonl": os.path.abspath(JSONL),
        "out_dir": os.path.abspath(OUT_DIR),
        "max_len": int(MAX_LEN),
        "num_blocks": int(num_blocks),
        "total_tokens": int(total_tokens),
        "tokenizer_dir": os.path.abspath(TOK_DIR),
        "tokenizer_xxh3_hi": int(tok_hi),
        "tokenizer_xxh3_lo": int(tok_lo),
        "vocab_size": int(vocab_size),
        "append_eos": bool(APPEND_EOS),
        "num_proc_tok": int(NUM_PROC_TOK),
        "batch_size_tok": int(BATCH_SIZE_TOK),
        "offline": True,
        "hash_files": {"hi": "hash_hi.npy", "lo": "hash_lo.npy"},
        "provenance": {
            "block_source": "block_source.npy",
            "block_offset": "block_offset.npy",
            "note": "block i 源自原始样本 block_source[i]，在该样本内的起始 token 偏移为 block_offset[i]；块不跨样本。"
        },
        "long_window_cfg": {
            "mode": "per-doc-single-window",
            "jitter_max": 64,
            "jitter_seed": 20251023,
            "min_len": 16384,
            "cross_doc_concat": False
        }
    }
    with open(os.path.join(OUT_DIR, "pack_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 完成提示
    print("="*72)
    print("[DONE] Deterministic 8k-window packing finished.")
    print(f"→ PK_CACHE = {OUT_DIR}")
    print(f"→ num_blocks = {manifest['num_blocks']:,}")
    print("✓ 每样本最多 1 窗口；块不跨样本；有轻微确定性抖动（满足复现）。")
    print("="*72)

if __name__ == "__main__":
    main()
