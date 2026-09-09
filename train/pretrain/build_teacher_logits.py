from project_paths import legacy_path
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, time, random, struct
import numpy as np
import torch, xxhash, lmdb
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
from collections import deque

# ================== 全部写死（路径 / 配置） ==================
PK_CACHE      = legacy_path("/root/autodl-tmp/llm/cache_pretrain_large/packed_len2048")
OUT_DIR       = legacy_path("/root/autodl-tmp/llm/data/kd_lmdb_minicpm3_4b")
TEACHER_DIR   = legacy_path("/root/autodl-tmp/llm/KDmodel/minicpm3_4b/snapshot")
TOKENIZER_DIR = TEACHER_DIR
MODEL_ID_TAG  = "openbmb/MiniCPM3-4B"

TOPK       = 16
DTYPE      = torch.bfloat16
MAX_LEN    = 2048
BATCH_SZ   = 24
KEEP_RATIO = 1.0    

# 探针（只做“recent”，模拟训练侧真实查回）
PROBE_EVERY         = 40
PROBE_RECENT_SAMPLE = 10
PROBE_TOPK_USE      = TOPK

# LMDB （每 rank 一个库）
LMDB_SUBDIR      = "lmdb_shards"
LMDB_DB_BASENAME = "kd_rank{rank}.lmdb"
LMDB_MAP_SIZE    = 1 << 42  # 4TB address space（仅虚拟地址）

# ================== 加速 & 强制离线 ==================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass

os.environ.update({
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_ENABLE_ONLINE_MODE": "0",
    "HF_HUB_DISABLE_TELEMETRY": "1",
})

# ================== DDP 基础 ==================
RANK  = int(os.environ.get("LOCAL_RANK", 0))
WORLD = int(os.environ.get("WORLD_SIZE", 1))
DEVICE = f"cuda:{RANK}" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.cuda.set_device(RANK)

def ddp_init_if_needed():
    if WORLD > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

def barrier():
    if WORLD > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()

def allreduce_sum_floats(*vals):
    if WORLD <= 1 or not torch.distributed.is_initialized():
        return vals
    tens = torch.tensor(vals, dtype=torch.float64, device=DEVICE)
    torch.distributed.all_reduce(tens, op=torch.distributed.ReduceOp.SUM)
    return tuple(v.item() for v in tens)

def log(*a):
    print(f"[rank{RANK}]", *a, flush=True)

ddp_init_if_needed()

# ================== 目录与路径 ==================
os.makedirs(OUT_DIR, exist_ok=True)
LMDB_DIR = os.path.join(OUT_DIR, LMDB_SUBDIR)
os.makedirs(LMDB_DIR, exist_ok=True)
MY_LMDB_PATH = os.path.join(LMDB_DIR, LMDB_DB_BASENAME.format(rank=RANK))

MANIFEST_PATH = os.path.join(OUT_DIR, "manifest.json")
META_DONE_FLAG= os.path.join(OUT_DIR, f"_rank{RANK}_done")

# ================== 数据集 ==================
if RANK == 0:
    print("[load packed]")
ds = load_from_disk(PK_CACHE)
N = len(ds)
if RANK == 0:
    print(f"packed size: {N}")

# ================== 子采样（确定性 80%） ==================
def keep_this(i: int, ratio=KEEP_RATIO) -> bool:
    if ratio >= 0.999:
        return True
    h = xxhash.xxh3_64_intdigest(str(i))
    return (h % 1000) < int(1000 * ratio)

my_indices = [i for i in range(RANK, N, WORLD) if keep_this(i)]
log(f"assigned {len(my_indices)} / {N} samples (keep_ratio={KEEP_RATIO})")

# rank0 进度条：目标 = 全局应导出的样本数
if RANK == 0:
    target_total = sum(1 for i in range(N) if keep_this(i, KEEP_RATIO))
    print(f"[rank0] target_total={target_total}", flush=True)
    pbar = tqdm(total=target_total, dynamic_ncols=True)
    _last_show = 0
    _last_ts = time.time()

# ================== 指纹 / 哈希 ==================
def xxh3_128_ids(ids_list) -> tuple[int,int,bytes]:
    arr = np.asarray(ids_list, dtype="<u4", order="C")
    h = xxhash.xxh3_128()
    h.update(arr.tobytes())
    dig = h.intdigest()
    lo = int(np.uint64(dig & ((1<<64)-1)))
    hi = int(np.uint64(dig >> 64))
    key = struct.pack("<QQ", hi, lo)  # 16B (hi, lo) little-endian
    return hi, lo, key

def tokenizer_fingerprint(tok_dir: str) -> dict:
    tfile = os.path.join(tok_dir, "tokenizer.json")
    hi = lo = -1
    if os.path.isfile(tfile):
        with open(tfile, "rb") as f:
            data = f.read()
        h = xxhash.xxh3_128(); h.update(data); dig = h.intdigest()
        lo = int(np.uint64(dig & ((1<<64)-1)))
        hi = int(np.uint64(dig >> 64))
    return {"tok_json_xxh3_hi": hi, "tok_json_xxh3_lo": lo}

def pk_cache_info_fingerprint(pk_dir: str) -> dict:
    info = os.path.join(pk_dir, "dataset_info.json")
    hi = lo = -1
    sz = mt = -1
    if os.path.isfile(info):
        with open(info, "rb") as f:
            data = f.read()
        h = xxhash.xxh3_128(); h.update(data); dig = h.intdigest()
        lo = int(np.uint64(dig & ((1<<64)-1)))
        hi = int(np.uint64(dig >> 64))
        st = os.stat(info)
        sz = int(st.st_size); mt = int(st.st_mtime)
    return {"pk_info_xxh3_hi": hi, "pk_info_xxh3_lo": lo, "pk_info_size": sz, "pk_info_mtime": mt}

# ================== LMDB 编解码（不压缩，RAW） ==================
MAGIC = b"KDTOPK01"      # 8B
# header: magic(8) | T(u32) | K(u16) | flags(u16) | len_idx(u32) | len_val(u32)
HDR_STRUCT = struct.Struct("<8sIHHII")
FLAG_VAL_FP16 = 1  # kd_val 为 fp16
FLAG_RAW      = 2  # payload 为原始字节（不压缩）

def encode_record(idx_np: np.ndarray, val_np: np.ndarray) -> bytes:
    assert idx_np.dtype == np.uint32 and val_np.dtype == np.float16
    raw_idx = idx_np.tobytes(order="C")
    raw_val = val_np.tobytes(order="C")
    flags = FLAG_VAL_FP16 | FLAG_RAW
    hdr = HDR_STRUCT.pack(MAGIC, idx_np.shape[0], idx_np.shape[1], flags,
                          len(raw_idx), len(raw_val))
    return hdr + raw_idx + raw_val

def decode_record(blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    magic, T, K, flags, lidx, lval = HDR_STRUCT.unpack_from(blob, 0)
    assert magic == MAGIC, "bad magic"
    off = HDR_STRUCT.size
    raw_idx = memoryview(blob)[off: off + lidx]; off += lidx
    raw_val = memoryview(blob)[off: off + lval]
    # RAW 直接 frombuffer
    idx = np.frombuffer(raw_idx, dtype=np.uint32).reshape(T, K)
    if flags & FLAG_VAL_FP16:
        val = np.frombuffer(raw_val, dtype=np.float16).reshape(T, K)
    else:
        raise ValueError("unsupported kd_val dtype flag")
    return idx, val

# ================== 初始化 Teacher ==================
barrier()
log("load teacher from local snapshot")
t_cfg = AutoConfig.from_pretrained(TEACHER_DIR, trust_remote_code=True, local_files_only=True)
mdl = AutoModelForCausalLM.from_pretrained(
    TEACHER_DIR, config=t_cfg, trust_remote_code=True, local_files_only=True,
    device_map={"": RANK} if torch.cuda.is_available() else None,
    low_cpu_mem_usage=True, torch_dtype=DTYPE,
).eval()

vocab_size = int(getattr(t_cfg, "vocab_size", -1))
tok_fp = tokenizer_fingerprint(TOKENIZER_DIR)
pk_fp  = pk_cache_info_fingerprint(PK_CACHE)

@torch.inference_mode()
def teacher_topk_for_batch(x: torch.Tensor):
    out = mdl(input_ids=x, use_cache=False)
    logits = out.logits.to(torch.float16)       # [B,T,V]
    tv, ti = torch.topk(logits, k=TOPK, dim=-1) # [B,T,K]
    return ti.to(torch.int32).cpu().numpy(), tv.cpu().numpy()

def compute_labels(ids_row):
    arr = np.asarray(ids_row, dtype=np.int64)
    lab = arr.copy()
    lab[:-1] = arr[1:]
    lab[-1] = -100
    return lab

# ================== 打开 LMDB（每 rank 独库） ==================
env = lmdb.open(
    MY_LMDB_PATH,
    map_size=LMDB_MAP_SIZE,
    subdir=True,
    readonly=False,
    lock=True,            # 写时加锁（但每 rank 独库，无跨进程争用）
    readahead=False,      # 随机访问更合适
    max_readers=4096,
    meminit=False,
    sync=False,           # 提高吞吐；commit 后对读事务可见
    map_async=True,
)

def open_ro_env():
    return lmdb.open(
        MY_LMDB_PATH, map_size=LMDB_MAP_SIZE, subdir=True,
        readonly=True, lock=False, readahead=False, max_readers=4096
    )

# ================== 全局条目统计（ETA 用） ==================
def count_entries_in_env(env_path: str) -> int:
    if not os.path.isdir(env_path):
        return 0
    e = lmdb.open(env_path, readonly=True, lock=False, readahead=False, max_readers=1)
    try:
        with e.begin() as txn:
            st = txn.stat()
            return int(st.get("entries", 0))
    finally:
        e.close()

def global_entries_count(world: int) -> int:
    tot = 0
    for r in range(world if world > 0 else 1):
        shard = os.path.join(LMDB_DIR, LMDB_DB_BASENAME.format(rank=r))
        tot += count_entries_in_env(shard)
    return tot

# ================== 主循环 & 探针 ==================
def chunks(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

_batch_tick = 0
done_local = 0

cum_hit_tok = 0.0
cum_tot_tok = 0.0
cum_macro   = 0.0
cum_cnt     = 0.0

recent_pairs = deque(maxlen=PROBE_RECENT_SAMPLE * 8)  # (key, ids_row)

# ETA 状态（rank0）
_eta_last_t = time.time()
_eta_last_cnt = 0
_ema_sps = None
_ema_alpha = 0.2

for batch_idx in chunks(my_indices, BATCH_SZ):
    valid_pairs = []
    for i in batch_idx:
        ids = ds[i]["input_ids"]
        if len(ids) == MAX_LEN:
            valid_pairs.append((i, ids))
    if not valid_pairs:
        continue

    x = torch.tensor([ids for (_, ids) in valid_pairs], dtype=torch.long, device=DEVICE)
    idx_btK, val_btK = teacher_topk_for_batch(x)

    # 在线统计
    b_hit = b_tot = 0
    b_mac_sum = 0.0
    b_mac_cnt = 0

    # 一次事务写多条
    with env.begin(write=True) as wtxn:
        for (i, ids_row), idx_row, val_row in zip(valid_pairs, idx_btK, val_btK):
            # 粗检 token 上界
            vmax = int(np.asarray(ids_row).max()) if len(ids_row) else -1
            if vocab_size > 0 and vmax >= vocab_size:
                log(f"[WARN] token id out of vocab: max_id={vmax} vocab={vocab_size}")

            _, _, key = xxh3_128_ids(ids_row)
            rec = encode_record(
                np.asarray(idx_row, dtype=np.uint32, order="C"),
                np.asarray(val_row, dtype=np.float16, order="C"),
            )
            wtxn.put(key, rec, overwrite=True)
            recent_pairs.append((key, ids_row))

            # 在线 hit（teacher 输出 vs label）
            labels = compute_labels(ids_row)
            valid = labels != -100
            if np.any(valid):
                hit_any = (idx_row[valid] == labels[valid, None]).any(axis=-1)
                b_hit += int(hit_any.sum())
                b_tot += int(valid.sum())
                b_mac_sum += float(hit_any.mean())
                b_mac_cnt += 1

    # 累计
    cum_hit_tok += b_hit
    cum_tot_tok += b_tot
    cum_macro   += b_mac_sum
    cum_cnt     += b_mac_cnt

    done_local += len(valid_pairs)
    _batch_tick += 1

    # 每 PROBE_EVERY 批：只做“recent”探针（真正关心的对齐）
    if (_batch_tick % PROBE_EVERY) == 0:
        ro_env = open_ro_env()
        rec_list = list(recent_pairs)
        random.shuffle(rec_list)
        rec_list = rec_list[:PROBE_RECENT_SAMPLE]

        r_found = 0; r_micro_hit = 0; r_micro_tot = 0
        r_mac_sum = 0.0; r_mac_cnt = 0

        with ro_env.begin(write=False) as rtxn:
            for key, ids_row in rec_list:
                blob = rtxn.get(key)         # ←←← 这里“确实从数据库读”
                if blob is None:
                    continue
                r_found += 1
                kd_idx, _ = decode_record(blob)
                labels = compute_labels(ids_row)
                valid = labels != -100
                if np.any(valid):
                    hit_any = (kd_idx[valid] == labels[valid, None]).any(axis=-1)
                    r_micro_hit += int(hit_any.sum())
                    r_micro_tot += int(valid.sum())
                    r_mac_sum   += float(hit_any.mean())
                    r_mac_cnt   += 1
        ro_env.close()

        r_recall = (r_found / max(1, len(rec_list))) if rec_list else 0.0
        r_micro  = (r_micro_hit / r_micro_tot) if r_micro_tot > 0 else 0.0
        r_macro  = (r_mac_sum / r_mac_cnt) if r_mac_cnt > 0 else 0.0

        # 全局累计（跨 rank）
        G_hit, G_tot, G_msum, G_mcnt = allreduce_sum_floats(cum_hit_tok, cum_tot_tok, cum_macro, cum_cnt)
        G_micro = (G_hit / G_tot) if G_tot > 0 else 0.0
        G_macro = (G_msum / G_mcnt) if G_mcnt > 0 else 0.0

        log(f"[PROBE.db_recent] found={r_found}/{len(rec_list)} recall={r_recall:.3f}  micro@{PROBE_TOPK_USE}={r_micro:.4f}  macro={r_macro:.4f}")
        if RANK == 0:
            print(f"[GLOBAL] cumulative micro@{TOPK}={G_micro:.4f}  macro={G_macro:.4f}", flush=True)

    # rank0：精确完成度 + ETA（每约 3s 刷进度、每约 10s 刷 ETA）
    if RANK == 0 and (time.time() - _last_ts >= 3.0):
        _last_ts = time.time()

        # 统计全局条目数（读 stat，不扫库）
        try:
            done_global = global_entries_count(WORLD)
        except Exception:
            done_global = _last_show

        done_clamped = min(done_global, target_total)
        inc = max(0, done_clamped - _last_show)
        if inc > 0:
            pbar.update(inc)
            _last_show = done_clamped

        now = time.time()
        if now - _eta_last_t >= 10.0:
            g_rate_inst = (done_global - _eta_last_cnt) / max(1e-6, (now - _eta_last_t))
            _eta_last_cnt = done_global
            _eta_last_t = now

            if _ema_sps is None:
                _ema_sps = g_rate_inst
            else:
                _ema_sps = 0.2 * g_rate_inst + 0.8 * _ema_sps

            left = max(0, target_total - done_global)
            eta_sec = left / max(1e-6, _ema_sps)
            eta_min = int(eta_sec // 60)
            eta_s   = int(eta_sec % 60)
            pct = 100.0 * done_global / max(1, target_total)
            print(f"[GLOBAL][ETA] done={done_global}/{target_total} ({pct:.2f}%)  rate≈{_ema_sps:.1f} samp/s  ETA≈{eta_min}m{eta_s:02d}s", flush=True)

# ================== 收尾 ==================
env.sync(); env.close()
open(META_DONE_FLAG, "w").close()

# rank0 写 manifest（列出 shard 列表 + 指纹）
barrier()
if RANK == 0:
    shards = [os.path.join(LMDB_SUBDIR, LMDB_DB_BASENAME.format(rank=r)) for r in range(WORLD)]
    meta = {
        "model_id": MODEL_ID_TAG,
        "topk": int(TOPK),
        "max_len": int(MAX_LEN),
        "keep_ratio": float(KEEP_RATIO),
        "source_pk_cache": os.path.abspath(PK_CACHE),
        "tokenizer_vocab_size": vocab_size,
        "tokenizer_fingerprint": tok_fp,   # 轻量指纹：tokenizer.json 的 xxh3_128
        "pk_cache_info_fingerprint": pk_fp,# 轻量指纹：dataset_info.json 的 xxh3_128
        "num_samples_total": int(N),
        "shards": shards,
        "format": {
            "key": "xxh3_128(ids_u32) -> bytes(hi_lo_le64 each, little-endian)",
            "value": {
                "header": "magic(8='KDTOPK01')|T(u32)|K(u16)|flags(u16:1=val_fp16,2=raw)|len_idx(u32)|len_val(u32)",
                "payload": "RAW idx_bytes + RAW val_bytes",
                "dtypes": {"idx": "uint32", "val": "float16"},
            }
        }
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("="*72)
    print("[GLOBAL] KD->LMDB dump done →", OUT_DIR)
    print("[GLOBAL] shards:", *shards, sep="\n  - ")
    print("="*72)
