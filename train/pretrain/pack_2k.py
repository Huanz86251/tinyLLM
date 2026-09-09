from project_paths import legacy_path
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, time
import numpy as np
from datetime import datetime
import shutil
from datasets import load_dataset, Dataset, Features, Sequence, Value
from transformers import AutoTokenizer
import xxhash
import sys
import pyarrow.compute as pac
import pyarrow as pa
from datasets.arrow_writer import ArrowWriter
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ARROW_NUM_THREADS"] = "5"

# ================== 全部写死 ==================
MAX_LEN        = 2048
NUM_PROC_TOK   = 5        # 仅用于 tokenize；分块单线程串流
BATCH_SIZE_TOK = 10_000
APPEND_EOS     = True
CAND_KEYS = ("TXT", "text", "txt", "TEXT")
# 固定路径（与你训练脚本一致）
TOK_DIR   = legacy_path("/root/autodl-tmp/llm/KDmodel/minicpm3_4b/snapshot")
OUT_DIR   = legacy_path("/root/autodl-tmp/llm/cache_pretrain_large/packed_len2048_language")

# JSONL 固定：优先项目根 ../../pretrain_500Mmodel.jsonl，找不到再用 /root 下的
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

JSONL     =  legacy_path("/root/autodl-tmp/cosmopedia.merged.final.jsonl")

# 强制离线环境（不触网）
os.environ.update({
    "HF_HOME": legacy_path("/root/autodl-tmp/hf_home_strict_offline"),
    "HF_DATASETS_CACHE": legacy_path("/root/autodl-tmp/hf_datasets_cache_force"),
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_ENABLE_ONLINE_MODE": "0",

})
# ============================================

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
    mvb = memoryview(arr_uint32).cast('B')
    h = xxhash.xxh3_128(); h.update(mvb); d = h.digest()
    hi = int.from_bytes(d[:8],  'big'); lo = int.from_bytes(d[8:], 'big')
    return np.uint64(hi), np.uint64(lo)

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
    tmp_arrow = os.path.join(OUT_DIR, "_tmp_packed.arrow")
    TMP_SAVE = OUT_DIR + "__new"
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

    def iter_jsonl_text(path, cand_keys=CAND_KEYS):
        with open(path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                try:
                    obj = json.loads(line)
                except Exception:
                    continue  # 也可记录坏行
                for k in cand_keys:
                    v = obj.get(k)
                    if isinstance(v, str) and v:
                        yield {"TXT": v}
                        break


    raw = Dataset.from_generator(
        lambda: iter_jsonl_text(JSONL, CAND_KEYS),
        features=Features({"TXT": Value("string")}),
        cache_dir=os.environ["HF_DATASETS_CACHE"]
    )


    raw = raw.filter(lambda e: len(e["TXT"]) > 0,
                     num_proc=5,
                     desc="Filter empty TXT")



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
        writer_batch_size=50_000,
        batch_size=BATCH_SIZE_TOK,
        desc="Tokenizing",
        load_from_cache_file=False,
        keep_in_memory=False
    )
    assert "input_ids" in tokenized.features

    # 统计总 token（单 pass，低内存）
    print("[STATS] counting total tokens (arrow sum) ...")
    # tokenized.data 是一个 pyarrow.Table，列名 "len" 来自上面 tokenize_fn 的返回
    total_tokens = int(pac.sum(tokenized.data.column("len")).as_py())
    num_blocks = total_tokens // MAX_LEN
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



    # 写 HF dataset
    print(f"[BUILD] writing dataset to {OUT_DIR} ...")

    import math

    features = Features({"input_ids": Sequence(Value("int32"))})


    writer = ArrowWriter(path=tmp_arrow, features=features)

    # 批大小（可按内存调大）：20k blocks ≈ 20_000 * 2048 * 4B ≈ 156 MB
    BATCH_BLOCKS = int(os.environ.get("PACK_BATCH_BLOCKS", "50000"))

    # 预分配一个 batch 缓冲区（行=block，列=MAX_LEN）
    batch_buf = np.empty((BATCH_BLOCKS, MAX_LEN), dtype=np.int32)
    row = 0
    written = 0

    def flush_batch():
        nonlocal row, written
        if row == 0:
            return
        view = batch_buf[:row]  # shape: (row, MAX_LEN)
        values = pa.array(view.ravel(), type=pa.int32())  # 先拍平成一维 values
        offsets_np = np.arange(0, row * MAX_LEN + 1, MAX_LEN, dtype=np.int32)
        offsets = pa.array(offsets_np)
        list_arr = pa.ListArray.from_arrays(offsets, values)  # 组装 List<int32>
        table = pa.Table.from_arrays([list_arr], names=["input_ids"])
        writer.write_table(table)  # ← 无 Python tolist
        written += row
        row = 0
        if written % (BATCH_BLOCKS * 10) == 0 or written == num_blocks:
            print(f"[WRITE] written={written:,}/{num_blocks:,} blocks", flush=True)

    # 用你现有的生成逻辑，但改为写入到 batch_buf，而不是 yield 一行
    buf = np.empty((MAX_LEN,), dtype=np.int32)
    carry = np.empty((0,), dtype=np.int32)
    block_idx = 0
    cur_src = 0
    cur_src_off = 0

    for i in range(len(tokenized)):
        ids = np.asarray(tokenized[i]["input_ids"], dtype=np.int32)

        if carry.size == 0:
            cur_src = i
            cur_src_off = 0
        seq = ids if carry.size == 0 else np.concatenate([carry, ids], axis=0)

        pos = 0;
        L = seq.size
        while pos + MAX_LEN <= L:
            buf[:] = seq[pos:pos + MAX_LEN]

            # 计算块哈希/来源（保持不变）
            hi, lo = xxh3_ids_uint32(buf.view(np.uint32))
            hash_hi[block_idx] = hi
            hash_lo[block_idx] = lo
            blk_src[block_idx] = cur_src
            blk_off[block_idx] = cur_src_off
            cur_src_off += MAX_LEN

            # 写入 batch 缓冲
            batch_buf[row] = buf
            row += 1
            block_idx += 1

            if row == BATCH_BLOCKS:
                flush_batch()

            pos += MAX_LEN

        carry = seq[pos:] if pos < L else np.empty((0,), dtype=np.int32)
        if carry.size == 0:
            cur_src = i + 1
            cur_src_off = 0

    if block_idx != num_blocks:
        raise RuntimeError(f"produced={block_idx}, expected={num_blocks}")

    # 别忘了冲掉最后一批
    flush_batch()

    # 结束写入
    num_examples, num_bytes = writer.finalize()
    print(f"[ARROW] num_examples={num_examples:,}, num_bytes={num_bytes:,}")

    # 用 HF Dataset 读回 arrow，再保存为标准目录结构（与你原流程一致）
    from datasets import Dataset as HFDataset
    ds = HFDataset.from_file(tmp_arrow)
    ds.save_to_disk(TMP_SAVE)
    for name in os.listdir(TMP_SAVE):
        src = os.path.join(TMP_SAVE, name)
        dst = os.path.join(OUT_DIR, name)
        # 若目标已存在（通常不会），可以选择先删除或改名，这里直接覆盖移动：
        if os.path.isdir(src):
            # 把整个子目录搬过去（datasets 会生成多个 data shard 目录/文件）
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.move(src, dst)
        else:
            if os.path.exists(dst):
                os.remove(dst)
            shutil.move(src, dst)
    shutil.rmtree(TMP_SAVE)  # 删掉临时保存目录
    if os.path.exists(tmp_arrow):
        os.remove(tmp_arrow)


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
            "note": "block i 源自原始样本 block_source[i]，起始于该样本内 token 偏移 block_offset[i]；块可能跨多个样本。"
        }
    }
    with open(os.path.join(OUT_DIR, "pack_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 完成提示
    print("="*72)
    print("[DONE] Deterministic packing finished.")
    print(f"→ PK_CACHE = {OUT_DIR}")
    print(f"→ num_blocks = {manifest['num_blocks']:,}")
    print("✓ 分块与并发/批次无关；重跑得到相同块序/哈希（前提：JSONL & tokenizer 不变）")
    print("="*72)

if __name__ == "__main__":
    main()
