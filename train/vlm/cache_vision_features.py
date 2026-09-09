#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
离线把 VLM SFT 里的所有图片（来自 pack 后的 HF Dataset）
跑一遍 Vision Encoder（ViT），把每张图的 token 特征 + padding mask 存到 LMDB 里。

新增：
- 从样本的 vision_meta 里读 views[*].pad.content_box，计算：
  - pending_ratio: 每个 token 对应的像素块内，“非 padding 区域”的面积占比（float16）
  - attn_mask    : pending_ratio>0 -> 1 else 0（uint8）
- LMDB value 格式升级：feat(bf16 bits) + attn_mask(u8) + pending_ratio(fp16)
- PROBE 同步检查 mask/ratio
"""
from project_paths import legacy_path

import os
import json
import time
import random
import struct
import math
from datetime import datetime
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import lmdb
import xxhash
from PIL import Image, ImageOps
from datasets import load_from_disk
from transformers import AutoModel, AutoImageProcessor
from tqdm import tqdm

# ================== 路径 / 配置（按需修改） ==================
PK_CACHE_VLM = legacy_path("/root/autodl-tmp/llm/cache_sft/vlm_short_varlen")
IMAGE_DIR = legacy_path("/root/autodl-tmp/vlm_sft_images")
VIT_DIR = legacy_path("/root/autodl-tmp/llm/vision/VIT/InternViT-300M-448px-V2_5")
VIT_MODEL_ID = "OpenGVLab/InternViT-300M-448px-V2_5"

OUT_DIR = legacy_path("/root/autodl-tmp/llm/data/vlm_vit_lmdb")

DTYPE = torch.bfloat16
BATCH_SZ = 32
KEEP_RATIO = 1.0
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

LMDB_SUBDIR = "lmdb_shards"
LMDB_DB_BASENAME = "vit_rank{rank}.lmdb"
LMDB_MAP_SIZE = 1 << 42

# 探针
PROBE_EVERY = 50
PROBE_RECENT_SAMPLE = 16

# ====== 你提到“不想用 pixel_unshuffle”→ 默认关；要用就改 True ======
USE_PIXEL_UNSHUFFLE = False
UNSHUFFLE_R = 2

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
GLOBAL_RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK  = int(os.environ.get("LOCAL_RANK", 0))
WORLD = int(os.environ.get("WORLD_SIZE", 1))

RANK = GLOBAL_RANK
DEVICE = f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.cuda.set_device(LOCAL_RANK)

def ddp_init_if_needed():
    if WORLD > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

def barrier():
    if WORLD > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()

def log(*a):
    print(f"[rank{RANK}]", *a, flush=True)

ddp_init_if_needed()

# ================== 目录与路径 ==================
os.makedirs(OUT_DIR, exist_ok=True)
LMDB_DIR = os.path.join(OUT_DIR, LMDB_SUBDIR)
os.makedirs(LMDB_DIR, exist_ok=True)
MY_LMDB_PATH = os.path.join(LMDB_DIR, LMDB_DB_BASENAME.format(rank=RANK))

MANIFEST_PATH = os.path.join(OUT_DIR, "manifest.json")
META_DONE_FLAG = os.path.join(OUT_DIR, f"_rank{RANK}_done")

# ================== 数据集 ==================
if RANK == 0:
    print("[load vlm packed dataset]")
ds = load_from_disk(PK_CACHE_VLM)
N = len(ds)
if RANK == 0:
    print(f"[vlm packed] size: {N}")

def keep_this(i: int, ratio=KEEP_RATIO) -> bool:
    if ratio >= 0.999:
        return True
    h = xxhash.xxh3_64_intdigest(str(i))
    return (h % 1000) < int(1000 * ratio)

my_indices = [i for i in range(RANK, N, WORLD) if keep_this(i)]
log(f"assigned {len(my_indices)} / {N} samples (keep_ratio={KEEP_RATIO})")

if RANK == 0:
    target_total = sum(1 for i in range(N) if keep_this(i, KEEP_RATIO))
    print(f"[rank0] target_total={target_total}", flush=True)
    pbar = tqdm(total=target_total, dynamic_ncols=True)
    _last_show = 0
    _last_ts = time.time()
else:
    target_total = 0
    pbar = None
    _last_show = 0
    _last_ts = time.time()

def to_img_list(x):
    if isinstance(x, str):
        return [x] if x else []
    if isinstance(x, list):
        return [s for s in x if isinstance(s, str) and s]
    return []

# ================== key 设计 ==================
def make_vision_key(dataset: str, index: int, image_file: str) -> Tuple[int, int, bytes]:
    s = f"{dataset}||{index}||{image_file}"
    h = xxhash.xxh3_128()
    h.update(s.encode("utf-8"))
    dig = h.intdigest()
    lo = int(np.uint64(dig & ((1 << 64) - 1)))
    hi = int(np.uint64(dig >> 64))
    key = struct.pack("<QQ", hi, lo)
    return hi, lo, key

def vit_config_fingerprint(vit_dir: str) -> dict:
    cfg = os.path.join(vit_dir, "config.json")
    hi = lo = -1
    sz = mt = -1
    if os.path.isfile(cfg):
        with open(cfg, "rb") as f:
            data = f.read()
        h = xxhash.xxh3_128()
        h.update(data)
        dig = h.intdigest()
        lo = int(np.uint64(dig & ((1 << 64) - 1)))
        hi = int(np.uint64(dig >> 64))
        st = os.stat(cfg)
        sz = int(st.st_size)
        mt = int(st.st_mtime)
    return {"vit_cfg_xxh3_hi": hi, "vit_cfg_xxh3_lo": lo,
            "vit_cfg_size": sz, "vit_cfg_mtime": mt}

# ================== vision_meta → pending_ratio / attn_mask ==================
def _safe_json_loads(s: Any) -> Dict[str, Any]:
    if isinstance(s, dict):
        return s
    if isinstance(s, str):
        return json.loads(s)
    return {}

def _clamp_box(box, W, H):
    x0, y0, x1, y1 = [int(v) for v in box]
    x0 = max(0, min(W, x0))
    x1 = max(0, min(W, x1))
    y0 = max(0, min(H, y0))
    y1 = max(0, min(H, y1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1

def _inter_area(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1) -> int:
    x0 = max(ax0, bx0)
    y0 = max(ay0, by0)
    x1 = min(ax1, bx1)
    y1 = min(ay1, by1)
    if x1 <= x0 or y1 <= y0:
        return 0
    return (x1 - x0) * (y1 - y0)

def build_token_mask_from_content_box(
    content_box,          # [x0,y0,x1,y1] on 448x448 canvas
    canvas_size: int,
    patch_size: int,
    grid_hw: Tuple[int, int],
    use_unshuffle: bool,
    unshuffle_r: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回：
      - attn_mask_u8: [Nv] uint8  (ratio>0 -> 1 else 0)
      - pending_ratio_f16: [Nv] float16 in [0,1]
    """
    Hc = Wc = int(canvas_size)
    gx, gy = int(grid_hw[1]), int(grid_hw[0])  # grid_hw=(h,w)
    # token 对应的像素块大小
    if use_unshuffle:
        th = int(gy // unshuffle_r)
        tw = int(gx // unshuffle_r)
        block = int(patch_size * unshuffle_r)  # 2x2 patch => 28x28 pixels
    else:
        th = int(gy)
        tw = int(gx)
        block = int(patch_size)                # 1 patch => 14x14 pixels

    Nv = th * tw
    x0, y0, x1, y1 = _clamp_box(content_box, Wc, Hc)

    # 对每个 token 像素块算与 content_box 的交面积
    ratios = np.empty((Nv,), dtype=np.float16)
    masks = np.empty((Nv,), dtype=np.uint8)

    area = float(block * block)
    k = 0
    for iy in range(th):
        py0 = iy * block
        py1 = py0 + block
        for ix in range(tw):
            px0 = ix * block
            px1 = px0 + block
            ia = _inter_area(px0, py0, px1, py1, x0, y0, x1, y1)
            r = 0.0 if ia <= 0 else (float(ia) / area)
            ratios[k] = np.float16(r)
            masks[k] = 1 if r > 0.0 else 0
            k += 1

    return masks, ratios

def compute_masks_for_image(ex: dict, img_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 ex["vision_meta"] 找到对应 view 的 pad.content_box → 生成 token-level mask/ratio。
    兜底：找不到就认为全有效（mask全1，ratio全1）。
    """
    vm = _safe_json_loads(ex.get("vision_meta", ""))
    enc = vm.get("vision_encoder", {}) if isinstance(vm, dict) else {}
    image_size = int(enc.get("image_size", 448) or 448)
    patch_size = int(enc.get("patch_size", 14) or 14)
    grid_hw = enc.get("grid_hw", [32, 32])
    if not (isinstance(grid_hw, (list, tuple)) and len(grid_hw) == 2):
        grid_hw = [32, 32]
    grid_hw = (int(grid_hw[0]), int(grid_hw[1]))  # (h,w)

    # 建 file->view 的映射（最稳）
    views = vm.get("views", []) if isinstance(vm, dict) else []
    view_map = {}
    if isinstance(views, list):
        for v in views:
            if isinstance(v, dict) and isinstance(v.get("file", None), str):
                view_map[v["file"]] = v

    v = view_map.get(img_name, None)
    content_box = None
    if isinstance(v, dict):
        pad = v.get("pad", None)
        if isinstance(pad, dict):
            cb = pad.get("content_box", None)
            if isinstance(cb, (list, tuple)) and len(cb) == 4:
                content_box = cb

    # 兜底：全有效
    # if content_box is None:
    #     # Nv 取决于是否 unshuffle
    #     gh, gw = grid_hw
    #     if USE_PIXEL_UNSHUFFLE:
    #         Nv = (gh // UNSHUFFLE_R) * (gw // UNSHUFFLE_R)
    #     else:
    #         Nv = gh * gw
    #     masks = np.ones((Nv,), dtype=np.uint8)
    #     ratios = np.ones((Nv,), dtype=np.float16)
    #     return masks, ratios
    if content_box is None:
        raise RuntimeError(
            f"[mask] missing pad.content_box for img_name={img_name!r}; "
            f"vision_meta.views files(head)={list(view_map.keys())[:10]}"
        )
    return build_token_mask_from_content_box(
        content_box=content_box,
        canvas_size=image_size,
        patch_size=patch_size,
        grid_hw=grid_hw,
        use_unshuffle=USE_PIXEL_UNSHUFFLE,
        unshuffle_r=UNSHUFFLE_R,
    )

def _token_grid_hw_block(grid_hw, patch_size: int, use_unshuffle: bool, unshuffle_r: int):
    gh, gw = int(grid_hw[0]), int(grid_hw[1])
    if use_unshuffle:
        th = gh // unshuffle_r
        tw = gw // unshuffle_r
        block = patch_size * unshuffle_r
    else:
        th = gh
        tw = gw
        block = patch_size
    return th, tw, block

def compute_global_pos_for_image(ex: dict, img_name: str, attn_mask_u8: np.ndarray | None = None):
    """
    返回：
      - global_pos_u16: [Nv]  uint16  (token 映射到 thumb 网格上的 token id)
      - global_off_f16: [Nv,2] float16 (dx,dy in [-0.5,0.5], 在该 cell 内的偏移)
    依赖 vision_meta.views 里：
      - tile: crop_box_work + pad(scale/offset/content_box)
      - thumb: pad(scale/offset/...)
    """

    vm = _safe_json_loads(ex.get("vision_meta", ""))
    if not isinstance(vm, dict):
        raise RuntimeError("[gpos] missing vision_meta")

    # 兼容两种 key：vision_encoder（你在 mask 那边用的）/ enc（你这里原本写的）
    enc = vm.get("vision_encoder", None)
    if not isinstance(enc, dict):
        enc = vm.get("enc", {})
    if not isinstance(enc, dict):
        enc = {}

    image_size = int(enc.get("image_size", 448) or 448)
    patch_size = int(enc.get("patch_size", 14) or 14)
    grid_hw = enc.get("grid_hw", (32, 32))
    if not (isinstance(grid_hw, (list, tuple)) and len(grid_hw) == 2):
        grid_hw = (32, 32)

    if not (isinstance(grid_hw, (list, tuple)) and len(grid_hw) == 2):
        grid_hw = (32, 32)

    th, tw, block = _token_grid_hw_block(grid_hw, patch_size, USE_PIXEL_UNSHUFFLE, UNSHUFFLE_R)
    Nv = th * tw

    views = vm.get("views", [])
    if not isinstance(views, list):
        raise RuntimeError("[gpos] vision_meta.views not a list")

    view_map = {}
    for v in views:
        if isinstance(v, dict) and isinstance(v.get("file", None), str):
            view_map[v["file"]] = v

    v = view_map.get(img_name, None)
    if not isinstance(v, dict):
        raise RuntimeError(f"[gpos] cannot find view meta for img={img_name!r}")

    # 找 thumb（推荐按 type=thumb；否则 view_idx==5）
    thumb = None
    for vv in views:
        if not isinstance(vv, dict):
            continue
        if vv.get("type") == "thumb" or vv.get("view_idx") == 5:
            thumb = vv
            break
    if not isinstance(thumb, dict):
        raise RuntimeError("[gpos] cannot find thumb view in vision_meta.views")

    thumb_pad = thumb.get("pad", None)
    if not isinstance(thumb_pad, dict):
        raise RuntimeError("[gpos] missing thumb.pad dict")
    s_thumb = float(thumb_pad.get("scale", 0.0))
    ox_thumb, oy_thumb = thumb_pad.get("offset", (0, 0))
    ox_thumb = float(ox_thumb); oy_thumb = float(oy_thumb)
    if s_thumb <= 0:
        raise RuntimeError(f"[gpos] bad thumb scale: {s_thumb}")

    # 如果自己就是 thumb：gpos=0..Nv-1，偏移=0
    if v.get("type") == "thumb" or v.get("view_idx") == 5:
        gpos = np.arange(Nv, dtype=np.uint16)
        goff = np.zeros((Nv, 2), dtype=np.float16)
        return gpos, goff

    # tile 必备字段
    crop_box = v.get("crop_box_work", None)
    if not (isinstance(crop_box, (list, tuple)) and len(crop_box) == 4):
        raise RuntimeError(f"[gpos] missing tile.crop_box_work for img={img_name!r}")
    x0, y0, x1, y1 = [float(c) for c in crop_box]

    pad = v.get("pad", None)
    if not isinstance(pad, dict):
        raise RuntimeError(f"[gpos] missing tile.pad for img={img_name!r}")
    s_tile = float(pad.get("scale", 0.0))
    ox_tile, oy_tile = pad.get("offset", (0, 0))
    ox_tile = float(ox_tile); oy_tile = float(oy_tile)
    cb = pad.get("content_box", None)
    if not (isinstance(cb, (list, tuple)) and len(cb) == 4):
        raise RuntimeError(f"[gpos] missing tile.pad.content_box for img={img_name!r}")
    cbx0, cby0, cbx1, cby1 = [float(c) for c in cb]
    if s_tile <= 0:
        raise RuntimeError(f"[gpos] bad tile scale: {s_tile}")

    # token centers in tile canvas
    iy = np.arange(th, dtype=np.float32)[:, None]
    ix = np.arange(tw, dtype=np.float32)[None, :]
    xt = (ix + 0.5) * float(block)
    yt = (iy + 0.5) * float(block)

    valid = (xt >= cbx0) & (xt < cbx1) & (yt >= cby0) & (yt < cby1)

    # tile canvas -> crop coords -> work coords
    x_crop = (xt - ox_tile) / s_tile
    y_crop = (yt - oy_tile) / s_tile
    x_work = x0 + x_crop
    y_work = y0 + y_crop

    # work -> thumb canvas
    x_th = ox_thumb + x_work * s_thumb
    y_th = oy_thumb + y_work * s_thumb

    # thumb canvas -> thumb token cell
    block_g = float(block)  # thumb 也是同一套 patch/block
    gx = np.floor(x_th / block_g).astype(np.int32)
    gy = np.floor(y_th / block_g).astype(np.int32)
    gx = np.clip(gx, 0, tw - 1)
    gy = np.clip(gy, 0, th - 1)

    # 偏移：落在 cell 内的 [-0.5,0.5]
    fx = (x_th / block_g) - (gx.astype(np.float32) + 0.5)
    fy = (y_th / block_g) - (gy.astype(np.float32) + 0.5)

    gpos_2d = (gy * tw + gx).astype(np.uint16)
    goff_2d = np.stack([fx, fy], axis=-1).astype(np.float16)

    # padding token 统一置 0（避免训练时被误用）
    if attn_mask_u8 is not None:
        m2d = attn_mask_u8.reshape(th, tw)
        valid = valid & (m2d > 0)

    gpos_2d = np.where(valid, gpos_2d, 0).astype(np.uint16)
    goff_2d = np.where(valid[..., None], goff_2d, 0).astype(np.float16)

    return gpos_2d.reshape(Nv), goff_2d.reshape(Nv, 2)

# ================== LMDB 编解码（feat + mask + ratio） ==================
# ================== LMDB 编解码（feat + mask + ratio + gpos + goff） ==================
"""
header (fixed 40 bytes):
  magic(8='VITREC03')
  Nv(u32) | D(u32) | flags(u32)
  len_feat(u32) | len_mask(u32) | len_ratio(u32) | len_gpos(u32) | len_goff(u32)

payload:
  feat_bytes:  [Nv, D] bfloat16 stored as uint16 bits (len_feat bytes)
  mask_bytes:  [Nv]    uint8 (0/1)                 (len_mask bytes)
  ratio_bytes: [Nv]    float16 in [0,1]            (len_ratio bytes)
  gpos_bytes:  [Nv]    uint16 (0..Nv-1)            (len_gpos bytes)
  goff_bytes:  [Nv,2]  float16 dx,dy               (len_goff bytes)
"""
MAGIC = b"VITREC03"
HDR_STRUCT = struct.Struct("<8sIIIIIIII")  # 8s + 8*uint32 = 40 bytes

FLAG_FEAT_BF16 = 1
FLAG_MASK_U8   = 2
FLAG_RATIO_F16 = 4
FLAG_GPOS_U16  = 8
FLAG_GOFF_F16  = 16

def encode_record(
    feat: torch.Tensor,
    attn_mask_u8: np.ndarray,
    global_pos_u16: np.ndarray,
    global_off_f16: np.ndarray,
) -> bytes:
    if not (torch.is_tensor(feat) and feat.ndim == 2):
        raise ValueError(f"feat must be torch.Tensor [Nv,D], got {type(feat)} {getattr(feat,'shape',None)}")
    Nv, D = feat.shape

    if not (isinstance(attn_mask_u8, np.ndarray) and attn_mask_u8.dtype == np.uint8 and attn_mask_u8.shape == (Nv,)):
        raise ValueError(f"attn_mask_u8 must be np.uint8 shape=({Nv},), got {type(attn_mask_u8)} {getattr(attn_mask_u8,'dtype',None)} {getattr(attn_mask_u8,'shape',None)}")

    if not (isinstance(global_pos_u16, np.ndarray) and global_pos_u16.dtype == np.uint16 and global_pos_u16.shape == (Nv,)):
        raise ValueError(f"global_pos_u16 must be np.uint16 shape=({Nv},), got {type(global_pos_u16)} {getattr(global_pos_u16,'dtype',None)} {getattr(global_pos_u16,'shape',None)}")

    if not (isinstance(global_off_f16, np.ndarray) and global_off_f16.dtype == np.float16 and global_off_f16.shape == (Nv, 2)):
        raise ValueError(f"global_off_f16 must be np.float16 shape=({Nv},2), got {type(global_off_f16)} {getattr(global_off_f16,'dtype',None)} {getattr(global_off_f16,'shape',None)}")

    # feat -> bf16 bits (uint16)
    t_bf16 = feat.to(torch.bfloat16).contiguous()
    t_u16 = t_bf16.view(torch.uint16).cpu().numpy()  # [Nv,D] uint16
    raw_feat = t_u16.tobytes(order="C")

    raw_mask = attn_mask_u8.tobytes(order="C")
    raw_gpos = global_pos_u16.tobytes(order="C")
    raw_goff = global_off_f16.tobytes(order="C")

    # ✅ 不存 ratio：len_ratio=0，flags 去掉 FLAG_RATIO_F16
    flags = (FLAG_FEAT_BF16 | FLAG_MASK_U8 | FLAG_GPOS_U16 | FLAG_GOFF_F16)
    hdr = HDR_STRUCT.pack(
        MAGIC,
        int(Nv),
        int(D),
        int(flags),
        int(len(raw_feat)),
        int(len(raw_mask)),
        0,  # len_ratio
        int(len(raw_gpos)),
        int(len(raw_goff)),
    )
    return hdr + raw_feat + raw_mask + raw_gpos + raw_goff

def decode_record(blob: bytes) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    magic, Nv, D, flags, lfeat, lmask, lratio, lgpos, lgoff = HDR_STRUCT.unpack_from(blob, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic={magic!r}, expect {MAGIC!r}")

    off = HDR_STRUCT.size
    feat_bytes = blob[off: off + lfeat]; off += lfeat
    mask_bytes = blob[off: off + lmask]; off += lmask
    ratio_bytes = blob[off: off + lratio]; off += lratio
    gpos_bytes = blob[off: off + lgpos]; off += lgpos
    goff_bytes = blob[off: off + lgoff]; off += lgoff

    # feat: uint16 bits -> bfloat16 tensor
    feat_u16 = np.frombuffer(feat_bytes, dtype=np.uint16).reshape(Nv, D)
    feat = torch.from_numpy(feat_u16).view(torch.bfloat16).contiguous()

    m = np.frombuffer(mask_bytes, dtype=np.uint8).copy()
    rr = np.frombuffer(ratio_bytes, dtype=np.float16)
    if rr.shape != (Nv,):
        # ✅ 新格式：没存 ratio（len_ratio=0 或 flags 不带）
        if (lratio == 0) or ((flags & FLAG_RATIO_F16) == 0):
            rr = np.zeros((Nv,), dtype=np.float16)
        else:
            raise ValueError(f"bad ratio shape={rr.shape}, expect ({Nv},)")

    gp = np.frombuffer(gpos_bytes, dtype=np.uint16).copy()
    go = np.frombuffer(goff_bytes, dtype=np.float16).reshape(Nv, 2).copy()

    return feat, m, rr, gp, go

# ================== 打开 LMDB（每 rank 独库） ==================
env = lmdb.open(
    MY_LMDB_PATH,
    map_size=LMDB_MAP_SIZE,
    subdir=True,
    readonly=False,
    lock=True,
    readahead=False,
    max_readers=4096,
    meminit=False,
    sync=False,
    map_async=True,
)

def open_ro_env():
    return lmdb.open(
        MY_LMDB_PATH,
        map_size=LMDB_MAP_SIZE,
        subdir=True,
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=4096,
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

def allreduce_sum_floats(*vals):
    if WORLD <= 1 or not torch.distributed.is_initialized():
        return vals
    tens = torch.tensor(vals, dtype=torch.float64, device=DEVICE)
    torch.distributed.all_reduce(tens, op=torch.distributed.ReduceOp.SUM)
    return tuple(v.item() for v in tens)

# ================== 初始化 ViT ==================
barrier()
log("load vision encoder (ViT) from local snapshot")

assert os.path.isdir(VIT_DIR), f"VIT_DIR not found: {VIT_DIR}"
vit_cfg_fp = vit_config_fingerprint(VIT_DIR)

image_processor = AutoImageProcessor.from_pretrained(
    VIT_DIR, trust_remote_code=True, local_files_only=True
)

vit_model = AutoModel.from_pretrained(
    VIT_DIR,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=DTYPE,
    device_map={"": LOCAL_RANK} if torch.cuda.is_available() else None,
    low_cpu_mem_usage=True,
).eval()

def pixel_unshuffle_2d(patch_tokens: torch.Tensor, h: int, w: int, r: int = 2) -> torch.Tensor:
    B, L, C = patch_tokens.shape
    assert L == h * w
    assert h % r == 0 and w % r == 0
    x = patch_tokens.view(B, h, w, C)
    x = x.view(B, h // r, r, w // r, r, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, (h // r) * (w // r), C * r * r)
    return x

@torch.inference_mode()
def vit_features_for_images(pil_list: List[Image.Image]) -> torch.Tensor:
    if len(pil_list) == 0:
        return torch.empty(0, 0, 0, dtype=torch.bfloat16)

    inputs = image_processor(images=pil_list, return_tensors="pt")
    inputs = {k: v.to(DEVICE, dtype=next(vit_model.parameters()).dtype) for k, v in inputs.items()}
    out = vit_model(**inputs)

    feats = getattr(out, "last_hidden_state", None)
    if feats is None:
        if isinstance(out, torch.Tensor):
            feats = out
        else:
            feats = out[0]

    feats = feats[:, 1:, :]  # drop CLS => [B, 1024, 1024] for InternViT-448

    if USE_PIXEL_UNSHUFFLE:
        feats = pixel_unshuffle_2d(feats, h=32, w=32, r=UNSHUFFLE_R)  # [B, 256, 4096] when r=2

    feats = feats.to(torch.bfloat16).cpu()
    return feats

# ========= rank0 probe：同时确定 Nv/D，并验证 mask/ratio length =========
VISION_TOKENS = 0
VISION_DIM = 0

if RANK == 0:
    log("[probe] infer vision feature shape + mask len on rank0 ...")
    probe_ok = False
    for ex in ds:
        img_list = to_img_list(ex.get("image_file", None))
        if not img_list:
            continue

        # 找第一张存在的
        img_name = None
        img_path = None
        for n in img_list:
            p = os.path.join(IMAGE_DIR, n)
            if os.path.isfile(p) and os.path.splitext(p)[1].lower() in IMG_EXTS:
                img_name = n
                img_path = p
                break
        if img_name is None:
            continue

        try:
            with Image.open(img_path) as im:
                im = ImageOps.exif_transpose(im)
                img = im.convert("RGB")
                img.load()
        except Exception:
            continue

        feats = vit_features_for_images([img])
        if feats.ndim != 3:
            raise RuntimeError(f"ViT features ndim != 3, got {feats.shape}")

        _, Nv, D = feats.shape
        m, r = compute_masks_for_image(ex, img_name)
        if m.shape[0] != Nv or r.shape[0] != Nv:
            raise RuntimeError(f"[probe] mask/ratio len mismatch: Nv={Nv}, mask={m.shape}, ratio={r.shape}")

        VISION_TOKENS = int(Nv)
        VISION_DIM = int(D)
        log(f"[probe] ViT feature shape: Nv={VISION_TOKENS}, D={VISION_DIM} ; mask_ok")
        probe_ok = True
        break

    if not probe_ok:
        raise RuntimeError("No valid image found for probing ViT feature shape.")

# 广播 Nv/D
if WORLD > 1 and torch.distributed.is_initialized():
    t = torch.tensor([VISION_TOKENS, VISION_DIM], dtype=torch.int64, device=DEVICE)
    torch.distributed.broadcast(t, src=0)
    VISION_TOKENS = int(t[0].item())
    VISION_DIM = int(t[1].item())

log(f"[vision] feature_shape = [Nv={VISION_TOKENS}, D={VISION_DIM}]  (pixel_unshuffle={USE_PIXEL_UNSHUFFLE})")

# ================== 主循环 & 探针 ==================
def chunks(lst: List[int], size: int):
    for i in range(0, len(lst), size):
        yield lst[i: i + size]

from collections import deque
recent_keys = deque(maxlen=PROBE_RECENT_SAMPLE * 4)
local_feat_bytes = 0

_eta_last_t = time.time()
_eta_last_cnt = 0
_ema_sps = None
_ema_alpha = 0.2

_batch_tick = 0
done_local = 0

for batch_idx in chunks(my_indices, BATCH_SZ):
    imgs: List[Image.Image] = []
    metas: List[Tuple[str, int, str, dict]] = []  # (dataset, index, image_file, ex_dict_for_mask)
    masks_list: List[np.ndarray] = []
    ratios_list: List[np.ndarray] = []
    gpos_list: List[np.ndarray] = []
    goff_list: List[np.ndarray] = []
    for i in batch_idx:
        ex = ds[i]
        img_list = to_img_list(ex.get("image_file", None))
        if not img_list:
            continue

        dataset_name = ex.get("dataset", "") or ""
        idx_val = int(ex.get("index", -1))
        if idx_val < 0:
            continue

        # 对每张图：读图 + 算 mask/ratio
        for img_name in img_list:
            img_path = os.path.join(IMAGE_DIR, img_name)
            if not os.path.isfile(img_path):
                continue
            ext = os.path.splitext(img_path)[1].lower()
            if ext not in IMG_EXTS:
                continue

            try:
                with Image.open(img_path) as im:
                    im = ImageOps.exif_transpose(im)
                    img = im.convert("RGB")
                    img.load()
            except Exception:
                continue

            attn_mask_u8, pending_ratio_f16 = compute_masks_for_image(ex, img_name)

            if attn_mask_u8.shape[0] != VISION_TOKENS or pending_ratio_f16.shape[0] != VISION_TOKENS:
                raise RuntimeError(
                    f"[mask] length mismatch for {img_name}: "
                    f"expect Nv={VISION_TOKENS}, got mask={attn_mask_u8.shape}, ratio={pending_ratio_f16.shape}"
                )
            global_pos_u16, global_off_f16 = compute_global_pos_for_image(ex, img_name, attn_mask_u8)

            if global_pos_u16.shape[0] != VISION_TOKENS or global_off_f16.shape != (VISION_TOKENS, 2):
                raise RuntimeError(
                    f"[gpos] length mismatch for {img_name}: "
                    f"expect Nv={VISION_TOKENS}, got gpos={global_pos_u16.shape}, goff={global_off_f16.shape}"
                )

            gpos_list.append(global_pos_u16)
            goff_list.append(global_off_f16)
            imgs.append(img)
            metas.append((dataset_name, idx_val, img_name, ex))
            masks_list.append(attn_mask_u8)
            ratios_list.append(pending_ratio_f16)

    if not imgs:
        continue

    feats = vit_features_for_images(imgs)  # [Bv, Nv, D] bfloat16 (CPU)

    with env.begin(write=True) as wtxn:
        for (dataset_name, idx_val, img_name, _), feat_row, m_u8, r_f16, gp_u16, go_f16 in zip(
            metas, feats, masks_list, ratios_list, gpos_list, goff_list
        ):
            hi, lo, key = make_vision_key(dataset_name, idx_val, img_name)
            rec = encode_record(feat_row, m_u8, gp_u16, go_f16)

            wtxn.put(key, rec, overwrite=True)
            local_feat_bytes += len(rec)
            recent_keys.append(key)
    done_local += len(metas)
    _batch_tick += 1

    # ========= PROBE：检查 decode + mask/ratio =========
    if (_batch_tick % PROBE_EVERY) == 0:
        ro_env = open_ro_env()
        rec_list = list(recent_keys)
        random.shuffle(rec_list)
        rec_list = rec_list[:PROBE_RECENT_SAMPLE]

        r_found = 0
        r_bad = 0
        with ro_env.begin(write=False) as rtxn:
            for key in rec_list:
                blob = rtxn.get(key)
                if blob is None:
                    continue
                r_found += 1
                feat, m, rr, gp, go = decode_record(blob)
                ok = True
                if feat.shape != (VISION_TOKENS, VISION_DIM):
                    ok = False
                if m.shape != (VISION_TOKENS,) or rr.shape != (VISION_TOKENS,):
                    ok = False
                if gp.shape != (VISION_TOKENS,) or go.shape != (VISION_TOKENS, 2):
                    ok = False

                if ok:
                    if not (np.all((m == 0) | (m == 1))):
                        ok = False
                    if not (np.all(rr >= 0) and np.all(rr <= 1)):
                        ok = False
                    # gpos 范围：0..Nv-1
                    if not (np.all(gp >= 0) and np.all(gp < VISION_TOKENS)):
                        ok = False
                    # goff 轻量范围（别太严格，避免浮点边界）
                    if not (np.all(go >= -1.0) and np.all(go <= 1.0)):
                        ok = False
                    if not ok:
                        r_bad += 1

        ro_env.close()
        log(f"[PROBE.db_recent] found={r_found}/{len(rec_list)} bad={r_bad}")

    # ========= rank0：进度 + ETA =========
    if RANK == 0 and (time.time() - _last_ts >= 3.0):
        _last_ts = time.time()
        try:
            done_global = global_entries_count(WORLD)
        except Exception:
            done_global = _last_show

        done_clamped = min(done_global, target_total)
        inc = max(0, done_clamped - _last_show)
        if inc > 0 and pbar is not None:
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
                _ema_sps = _ema_alpha * g_rate_inst + (1 - _ema_alpha) * _ema_sps

            left = max(0, target_total - done_global)
            eta_sec = left / max(1e-6, _ema_sps)
            eta_min = int(eta_sec // 60)
            eta_s = int(eta_sec % 60)
            pct = 100.0 * done_global / max(1, target_total)
            print(
                f"[GLOBAL][ETA] done={done_global}/{target_total} ({pct:.2f}%)  "
                f"rate≈{_ema_sps:.1f} samp/s  ETA≈{eta_min}m{eta_s:02d}s",
                flush=True,
            )

# ================== 收尾 ==================
env.sync()
env.close()
open(META_DONE_FLAG, "w").close()

barrier()

# ================== rank0 写 manifest ==================
if RANK == 0:
    total_feat_bytes_world, = allreduce_sum_floats(local_feat_bytes)
    shards = [
        os.path.join(LMDB_SUBDIR, LMDB_DB_BASENAME.format(rank=r))
        for r in range(WORLD)
    ]
    gb = total_feat_bytes_world / (1024 ** 3)

    meta = {
        "stage": "VLM-ViT-offline-features+mask",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "vit_model_id": VIT_MODEL_ID,
        "vit_dir": os.path.abspath(VIT_DIR),
        "vit_config_fingerprint": vit_cfg_fp,
        "source_packed_dataset": os.path.abspath(PK_CACHE_VLM),
        "image_root_dir": os.path.abspath(IMAGE_DIR),
        "world_size": int(WORLD),
        "keep_ratio": float(KEEP_RATIO),
        "num_samples_total": int(N),
        "pixel_unshuffle": {"enabled": bool(USE_PIXEL_UNSHUFFLE), "r": int(UNSHUFFLE_R)},
        "feature_shape": {
            "tokens_per_image": int(VISION_TOKENS),
            "feature_dim": int(VISION_DIM),
            "dtype": "bfloat16",
        },
        "mask_shape": {
            "tokens_per_image": int(VISION_TOKENS),
            "attn_mask_dtype": "uint8(0/1)",
            "pending_ratio_dtype": "float16([0,1])",
            "rule": "attn_mask = 1 if pending_ratio>0 else 0",
            "source": "vision_meta.views[*].pad.content_box",
        },
        "format": {
            "key": "xxh3_128(f'{dataset}||{index}||{image_file}') → bytes(hi_lo_le64 little-endian)",
            "value": {
                "header": "magic(8='VITREC03')|Nv(u32)|D(u32)|flags(u32)|len_feat(u32)|len_mask(u32)|len_ratio(u32)",
                "payload": [
                    "feat_bytes:  [Nv,D] bfloat16 stored as uint16 bits",
                    "mask_bytes:  [Nv]   uint8(0/1)",
                    "ratio_bytes: [Nv]   float16([0,1])",
                ],
                "flags": {
                    "1": "feat_bf16",
                    "2": "mask_u8",
                    "4": "ratio_f16",
                },
            },
        },
        "note": (
            "LMDB 存 ViT token 特征 + padding mask（pending_ratio/attn_mask）。"
            "训练时用同样 key 读回 (feat, attn_mask, pending_ratio)，先解决 padding，再做位置编码。"
        ),
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("[GLOBAL] VIT->LMDB dump done →", OUT_DIR)
    print(f"[GLOBAL] approx feature+mask size ≈ {gb:.2f} GiB")
    print("[GLOBAL] shards:", *shards, sep="\n  - ")
    print("=" * 72)
