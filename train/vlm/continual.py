from project_paths import legacy_path, path as project_path
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gc
import os
import re
import math
import json
import time
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import random

import struct
import shutil
from torch.optim.lr_scheduler import LambdaLR
import lmdb
import xxhash
from PIL import Image, ImageOps, ImageFilter
from torch.optim import AdamW
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

import lmdb, struct, numpy as np, torch, os, json
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
RUN_NAME = os.environ.get("SFT_RUN_NAME", "tinyllm_vlm_sft")
RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
LORA_TARGET = "all"
NUM_VISION_VIEWS = 5
SFT_NUM_WORKERS = int(os.environ.get("SFT_NUM_WORKERS", "6"))
SFT_PREFETCH = int(os.environ.get("SFT_PREFETCH", "2"))
SFT_MP_CTX = os.environ.get("SFT_MP_CTX", "forkserver")
VIT_PRECOMPUTE_SKIP_GET = int(os.environ.get("VIT_PRECOMPUTE_SKIP_GET", "1")) == 1
VIP_IMG_LOAD_WORKERS = int(os.environ.get("VIP_IMG_LOAD_WORKERS", "12"))
SFT_EVAL_NUM_WORKERS = int(os.environ.get("SFT_EVAL_NUM_WORKERS", "2"))
AUTO_VIT_ROTATE = int(os.environ.get("AUTO_VIT_ROTATE", "0")) == 1
THUMB_DOWNSAMPLE_P = float(os.environ.get("THUMB_DOWNSAMPLE_P", "0.20"))  # 10%
THUMB_GAUSS_P      = float(os.environ.get("THUMB_GAUSS_P", "0.1"))       # 5%
TILE_SHUFFLE_P     = float(os.environ.get("SFT_TILE_SHUFFLE_P", "0.0"))
assert 0.0 <= THUMB_DOWNSAMPLE_P <= 1.0 and 0.0 <= THUMB_GAUSS_P <= 1.0
assert THUMB_DOWNSAMPLE_P + THUMB_GAUSS_P <= 1.0
assert 0.0 <= TILE_SHUFFLE_P <= 1.0
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
from safetensors.torch import load_file, save_file
from datasets import concatenate_datasets, load_from_disk
from concurrent.futures import ThreadPoolExecutor

import multiprocessing as mp
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    AutoModel,
    AutoImageProcessor,
    AutoConfig,
)
from model.config import Config
from model.model import TinyLLM


# =====================
# 环境 / 全局配置
# =====================

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
GRAD_ACCUM_STEPS = int(os.environ.get("SFT_GRAD_ACCUM_STEPS", "2"))
STAGE2_QFORMER_LR_RATIO = float(os.environ.get("SFT_STAGE2_QFORMER_LR_RATIO", "0.5"))
STAGE2_BRIDGE_LR_RATIO = float(os.environ.get("SFT_STAGE2_BRIDGE_LR_RATIO", "1.0"))

try:
    from torch.backends.cuda import sdp_kernel
    sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True)
except Exception:
    pass

SFT_STAGE=int(os.environ.get("SFT_STAGE", "2"))
SFT_LOSS_REDUCTION = os.environ.get("SFT_LOSS_REDUCTION", "hybrid" if SFT_STAGE == 2 else "token_mean")
SFT_SAMPLE_MEAN_ALPHA = float(os.environ.get("SFT_SAMPLE_MEAN_ALPHA", "0.75"))
assert SFT_LOSS_REDUCTION in {"token_mean", "sample_mean", "hybrid"}
assert 0.0 <= SFT_SAMPLE_MEAN_ALPHA <= 1.0
SFT_STAGE1_STEPS=80000
SFT_STAGE1_STRICT_QFORMER_ONLY=1
VIT_DIR = os.environ.get("VIT_DIR", legacy_path("/root/autodl-tmp/llm/vision/VIT/InternViT-300M-448px-V2_5"))
IMAGE_DIR = os.environ.get("IMAGE_DIR", legacy_path("/root/autodl-tmp/vlm_sft_images"))  # 这里必须能找到 img_name

if SFT_STAGE == 2:
    _default_out_dir = str(project_path("runs/training/vlm_stage2"))
    _default_resume_root = legacy_path("/root/autodl-tmp/llm/tiny_05B_sft_vlm_stage15/checkpoint-80000")
else:
    _default_out_dir = str(project_path("runs/training/vlm_stage1"))
    _default_resume_root = legacy_path("/root/autodl-tmp/llm/tiny_05B_sft/checkpoint-50000")
SFT_OUT_DIR = os.environ.get("SFT_OUT_DIR", _default_out_dir)
RESUME_ROOT = os.environ.get("SFT_RESUME_ROOT", _default_resume_root)
IGNORE_INDEX = -100
VIP_BATCH_IMAGES = int(os.environ.get("VIP_BATCH_IMAGES", "128"))
VIT_LMDB_ROOT = os.environ.get("VIT_LMDB_ROOT", os.path.join(SFT_OUT_DIR, "vit_lmdb_cache"))
VIT_CHUNK_STEPS = int(os.environ.get("VIT_CHUNK_STEPS", "10000"))   # 每个 chunk 的训练步数
VIT_CLEAR_BETWEEN_CHUNKS = int(os.environ.get("VIT_CLEAR_BETWEEN_CHUNKS", "1")) == 1  # 每个chunk后清空LMDB释放磁盘
VIT_PRECOMPUTE_VERBOSE_EVERY = int(os.environ.get("VIT_PRECOMPUTE_VERBOSE_EVERY", "50"))
def _is_mp_child() -> bool:
    return mp.current_process().name != "MainProcess"
CLEAR_VIT_CACHE_ON_START = bool(int(os.environ.get("CLEAR_VIT_CACHE_ON_START", "0"))) and (not _is_mp_child())



# ---- 路径（按自己机器改）----
OUT_DIR = SFT_OUT_DIR
os.makedirs(OUT_DIR, exist_ok=True)

os.environ["VIP_BATCH_IMAGES"] = str(VIP_BATCH_IMAGES)
# 单一 SFT 数据集（pack 脚本产物，只包含短序列 + 图文）
SFT_CACHE_SHORT = os.environ.get("SFT_CACHE_SHORT", legacy_path("/root/autodl-tmp/llm/cache_sft/vlm_short_varlen"))
SFT_TEXT_REPLAY_CACHE = os.environ.get("SFT_TEXT_REPLAY_CACHE", "").strip()
SFT_TEXT_REPLAY_RATIO = float(os.environ.get("SFT_TEXT_REPLAY_RATIO", "0.0"))
SFT_TEXT_REPLAY_EVAL_MAX = int(os.environ.get("SFT_TEXT_REPLAY_EVAL_MAX", "256"))
if not 0.0 <= SFT_TEXT_REPLAY_RATIO < 1.0:
    raise ValueError("SFT_TEXT_REPLAY_RATIO must be in [0, 1)")

SFT_EVAL_FRACTION = float(os.environ.get("SFT_EVAL_FRACTION", "0.0001"))  # 默认 0.01%
SFT_EVAL_MIN = int(os.environ.get("SFT_EVAL_MIN", "1000"))
SFT_EVAL_MAX = int(os.environ.get("SFT_EVAL_MAX", "2000"))
SFT_EXPLICIT_EVAL_MAX = int(os.environ.get("SFT_EXPLICIT_EVAL_MAX", "0"))

# tokenizer 快照
LOCAL_SNAPSHOT = os.environ.get("SFT_TOKENIZER_DIR", legacy_path("/root/autodl-tmp/llm/tiny_05B_sft/checkpoint-50000"))

LOG_DIR = os.path.join(
    legacy_path("/root/autodl-tmp/llm/checkpoints"),
    "tb",
    f"tinyllm-sft-vlm-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# =====================
# SFT 超参 & LoRA & Vision 配置
# =====================
SEED = int(os.environ.get("SFT_SEED", "15"))
USE_BF16 = bool(int(os.environ.get("SFT_USE_BF16", "1")))

LR_BASE = float(os.environ.get("SFT_LEARNING_RATE", "1.7e-5"))
WARMUP_STEPS = int(os.environ.get("SFT_WARMUP_STEPS", "500"))
WARMUP_RATIO = float(os.environ.get("SFT_WARMUP_RATIO", "0.0"))
MAX_GRAD_NORM = float(os.environ.get("SFT_MAX_GRAD_NORM", "1.2"))
WEIGHT_DECAY = float(os.environ.get("SFT_WEIGHT_DECAY", "0.0"))
EPOCHS = int(os.environ.get("SFT_EPOCHS", "2"))
MAX_STEPS_OVERRIDE = int(os.environ.get("SFT_MAX_STEPS", "-1"))

MAX_TOKENS_PER_BATCH_SHORT = int(os.environ.get("SFT_MAX_TOKENS_PER_BATCH_SHORT", "10240"))
SFT_MIN_LR_RATIO = float(os.environ.get("SFT_MIN_LR_RATIO", "0.10"))  # ✅ 10%
assert 0.0 <= SFT_MIN_LR_RATIO <= 1.0
# LoRA 设置
USE_LORA = bool(int(os.environ.get("SFT_USE_LORA", "1")))
LORA_RANK = int(os.environ.get("SFT_LORA_RANK", "128"))
LORA_ALPHA = int(os.environ.get("SFT_LORA_ALPHA", "128"))
LORA_DROPOUT = float(os.environ.get("SFT_LORA_DROPOUT", "0.0"))
LORA_NAME = os.environ.get("SFT_LORA_NAME", "vlm_sft")

# Vision / Q-Former 设置（要和 pack 脚本保持一致）

MAX_VISION_TOKENS = 0

# 日志&保存
LOG_SCALAR_EVERY = 20
SAVE_INTERVAL_STEPS = int(os.environ.get("SFT_SAVE_INTERVAL_STEPS", "2000"))
START_TIME = time.time()


# =====================
# 小工具
# =====================

# =====================
# Vision LMDB (VITREC03)
# =====================

def _pick_first_existing(paths: list[str]) -> str | None:
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None

def _strip_module_prefix(sd: dict) -> dict:
    # 兼容 DataParallel/DDP 产生的 module. 前缀
    if sd and all(k.startswith("module.") for k in sd.keys()):
        return {k[len("module."):]: v for k, v in sd.items()}
    return sd

def _sanity_check_overlap(sd: dict, model: torch.nn.Module, *, tag: str, min_ratio: float = 0.90):
    mk = set(model.state_dict().keys())
    sk = set(sd.keys())
    inter = len(mk & sk)
    ratio = inter / max(1, len(mk))
    print(f"[load][{tag}] key overlap = {inter}/{len(mk)} = {ratio:.2%}", flush=True)
    if ratio < min_ratio:
        # 直接报错，别让 strict=False 静默过去
        sample_m = list(sorted(list(mk - sk)))[:20]
        sample_u = list(sorted(list(sk - mk)))[:20]
        raise RuntimeError(
            f"[load][{tag}] overlap too low -> likely wrong checkpoint file picked.\n"
            f"  missing(sample)={sample_m}\n"
            f"  unexpected(sample)={sample_u}\n"
        )

def load_base_and_optional_lora_into_model(
    *,
    model: torch.nn.Module,
    resume_root: str,
    stage: int,
    lora_name: str,
    map_location: str = "cpu",
    print_keys_limit: int = 50,
):
    # ✅ base：优先选最可靠的 split-base
    base_path = _pick_first_existing([
        os.path.join(resume_root, "base", "model.safetensors"),   # ✅ 最优先
        os.path.join(resume_root, "pytorch_model.bin"),
        os.path.join(resume_root, "model.safetensors"),
    ])
    if base_path is None:
        raise FileNotFoundError(f"[load] base weights not found under resume_root={resume_root}")

    print(f"[load] base <- {base_path}", flush=True)
    if base_path.endswith(".safetensors"):
        base_sd = load_file(base_path, device=map_location)
    else:
        base_sd = torch.load(base_path, map_location=map_location)

    base_sd = _strip_module_prefix(base_sd)
    _sanity_check_overlap(base_sd, model, tag="base", min_ratio=0.90)

    missing, unexpected = model.load_state_dict(base_sd, strict=False)

    # ✅ 你要的：base 阶段把 missing/unexpected 打出来（但别打印 value，本质没意义，打印 key 就够）
    print(f"[load] base done. missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if unexpected:
        print(f"[load][base] unexpected (first {print_keys_limit}) = {unexpected[:print_keys_limit]}", flush=True)
    if missing:
        print(f"[load][base] missing (first {print_keys_limit}) = {missing[:print_keys_limit]}", flush=True)

    # ✅ 不再静默：stage2 的 base load，missing 只允许是 adapters（否则就是你又选错文件/结构不一致）
    if stage == 2:
        bad_missing = [k for k in missing if ".adapters." not in k]
        if bad_missing:
            raise RuntimeError(
                "[load][base] stage2 base-load has non-adapter missing keys -> WRONG LOAD.\n"
                f"bad_missing(first 30) = {bad_missing[:30]}"
            )
        # base 阶段 unexpected 理论上也应该是 0（因为你加载的是 base，不含 adapters）
        if unexpected:
            raise RuntimeError(
                "[load][base] unexpected keys found while loading base -> likely picked wrong file.\n"
                f"unexpected(first 30) = {unexpected[:30]}"
            )

    del base_sd
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- optional lora ----
    lora_path = _pick_first_existing([
        os.path.join(resume_root, "lora", "lora.safetensors"),
        os.path.join(resume_root, "lora.safetensors"),
    ])

    if stage == 2 and lora_path is not None:
        print(f"[load] lora <- {lora_path}", flush=True)
        lora_sd = load_file(lora_path, device=map_location)
        lora_sd = _strip_module_prefix(lora_sd)

        # lora 阶段：按你说的，不刷 missing/unexpected（但我建议至少做个轻量校验）
        if hasattr(model, "load_lora_state_dict"):
            model.load_lora_state_dict(lora_sd, adapter_name=lora_name, strict=False)
        else:
            filtered = {k: v for k, v in lora_sd.items() if ".adapters." in k}
            model.load_state_dict(filtered, strict=False)

        del lora_sd
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[load] lora done.", flush=True)
    else:
        if stage == 2:
            print("[load] stage2: no lora file found -> start with fresh LoRA init (ok).", flush=True)


def load_base_only(model, resume_root, stage, map_location="cpu"):
    base_path = _pick_first_existing([
        os.path.join(resume_root, "base", "model.safetensors"),
        os.path.join(resume_root, "pytorch_model.bin"),
        os.path.join(resume_root, "model.safetensors"),
    ])
    if base_path is None:
        raise FileNotFoundError(f"[load] base not found: {resume_root}")
    sd = load_file(base_path, device=map_location) if base_path.endswith(".safetensors") else torch.load(base_path, map_location=map_location, weights_only=True)
    sd = _strip_module_prefix(sd)
    normalized = {k.replace(".base.weight", ".weight").replace(".base.bias", ".bias"): v for k, v in sd.items()}
    if len(normalized) != len(sd):
        raise RuntimeError("Ambiguous duplicate base parameter names")
    sd = normalized
    expected = set(model.state_dict())
    missing, unexpected = expected - set(sd), set(sd) - expected
    vision_prefixes = ("query_embed", "qformer_", "vision_")
    allowed_missing = {k for k in missing if stage == 1 and k.startswith(vision_prefixes)}
    if unexpected or missing - allowed_missing:
        raise RuntimeError(f"Base key mismatch: missing={sorted(missing-allowed_missing)[:30]}, unexpected={sorted(unexpected)[:30]}")
    model.load_state_dict(sd, strict=not bool(allowed_missing))
    print(f"[load] complete base <- {base_path}; newly initialized vision tensors={len(allowed_missing)}", flush=True)
    del sd


def load_lora_only_after_attach(model, resume_root, lora_name, map_location="cpu"):
    lora_path = _pick_first_existing([os.path.join(resume_root, "lora", "lora.safetensors"), os.path.join(resume_root, "lora.safetensors")])
    if lora_path is None:
        print("[load] No saved LoRA; keeping new adapter initialization.", flush=True)
        return
    sd = _strip_module_prefix(load_file(lora_path, device=map_location))
    expected = {k for k in model.state_dict() if f".adapters.{lora_name}." in k}
    if set(sd) != expected:
        raise RuntimeError(f"LoRA key mismatch: missing={sorted(expected-set(sd))[:20]}, unexpected={sorted(set(sd)-expected)[:20]}")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected or any(k in expected for k in missing):
        raise RuntimeError("LoRA failed to load completely")
    model.activate_single_lora(lora_name)

def _rm_dir(p: str):
    if os.path.exists(p):
        shutil.rmtree(p)

MAGIC = b"VITREC03"
HDR_STRUCT = struct.Struct("<8sIIIIIIII")  # 40 bytes

FLAG_FEAT_BF16 = 1
FLAG_MASK_U8   = 2
FLAG_RATIO_F16 = 4   # 本实现不存 ratio（len_ratio=0, flags 不带这个）
FLAG_GPOS_U16  = 8
FLAG_GOFF_F16  = 16
GOFF_DIM = 4

def make_vision_key(dataset_name: str, idx_val: int, img_name: str) -> bytes:
    """
    跟你现有脚本一致：xxhash128 -> (hi, lo) -> pack 成 16 bytes
    这样你之前离线跑过的 LMDB 也能直接复用。
    """
    s = f"{dataset_name}\n{int(idx_val)}\n{img_name}"
    d = xxhash.xxh128(s.encode("utf-8")).intdigest()
    hi = (d >> 64) & ((1 << 64) - 1)
    lo = d & ((1 << 64) - 1)
    return struct.pack("<QQ", hi, lo)


def encode_record(
    feat_bf16: torch.Tensor,          # [Nv, D] torch.bfloat16 (CPU or GPU都行，会转CPU)
    attn_mask_u8: np.ndarray,         # [Nv] uint8
    global_pos_u16: np.ndarray,        # [Nv] uint16
    global_off_f16: np.ndarray,        # [Nv,4] float16

) -> bytes:
    if feat_bf16.dtype != torch.bfloat16:
        raise TypeError(f"feat_bf16 must be bfloat16, got {feat_bf16.dtype}")
    if feat_bf16.ndim != 2:
        raise ValueError(f"feat_bf16 must be [Nv,D], got {feat_bf16.shape}")

    Nv, D = int(feat_bf16.shape[0]), int(feat_bf16.shape[1])

    if attn_mask_u8.dtype != np.uint8 or attn_mask_u8.shape != (Nv,):
        raise ValueError(f"attn_mask_u8 must be uint8 [Nv], got {attn_mask_u8.dtype} {attn_mask_u8.shape}")
    if global_pos_u16.dtype != np.uint16 or global_pos_u16.shape != (Nv,):
        raise ValueError(f"global_pos_u16 must be uint16 [Nv], got {global_pos_u16.dtype} {global_pos_u16.shape}")
    if global_off_f16.dtype != np.float16 or global_off_f16.shape != (Nv, GOFF_DIM):
        raise ValueError(f"global_off_f16 must be float16 [Nv,{GOFF_DIM}], got {global_off_f16.dtype} {global_off_f16.shape}")

    feat_u16 = feat_bf16.contiguous().view(torch.uint16).cpu().numpy()  # [Nv,D] uint16 view
    feat_bytes = feat_u16.tobytes()

    mask_bytes = attn_mask_u8.tobytes()
    ratio_bytes = b""  # 不存
    gpos_bytes = global_pos_u16.tobytes()
    goff_bytes = global_off_f16.tobytes()

    flags = FLAG_FEAT_BF16 | FLAG_MASK_U8 | FLAG_GPOS_U16 | FLAG_GOFF_F16
    header = HDR_STRUCT.pack(
        MAGIC,
        Nv,
        D,
        flags,
        len(feat_bytes),
        len(mask_bytes),
        0,  # len_ratio = 0
        len(gpos_bytes),
        len(goff_bytes),
    )
    return header + feat_bytes + mask_bytes + ratio_bytes + gpos_bytes + goff_bytes


def decode_record(buf) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    mv = memoryview(buf)  # buf 可以是 bytes 或 memoryview
    magic, Nv, D, flags, lfeat, lmask, lratio, lgpos, lgoff = HDR_STRUCT.unpack_from(mv, 0)
    off = HDR_STRUCT.size

    feat_mv = mv[off:off+lfeat]; off += lfeat
    mask_mv = mv[off:off+lmask]; off += lmask
    off += lratio
    gpos_mv = mv[off:off+lgpos]; off += lgpos
    goff_mv = mv[off:off+lgoff]; off += lgoff

    feat_u16 = np.frombuffer(feat_mv, dtype=np.uint16).reshape(int(Nv), int(D))
    feat_bf16 = torch.from_numpy(feat_u16).view(torch.bfloat16)

    attn_mask_u8 = np.frombuffer(mask_mv, dtype=np.uint8).reshape(int(Nv))
    global_pos_u16 = np.frombuffer(gpos_mv, dtype=np.uint16).reshape(int(Nv))
    global_off_f16 = np.frombuffer(goff_mv, dtype=np.float16).reshape(int(Nv), GOFF_DIM)
    return feat_bf16, attn_mask_u8, global_pos_u16, global_off_f16




def _infer_hidden_size_from_config(vit_dir: str) -> int:
    cfg = AutoConfig.from_pretrained(vit_dir, local_files_only=True, trust_remote_code=True)
    if hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)
    if hasattr(cfg, "vision_config") and hasattr(cfg.vision_config, "hidden_size"):
        return int(cfg.vision_config.hidden_size)
    raise ValueError("cannot infer hidden_size from vit config")


# -------- VisionLMDB (read+write, missing -> build) --------
class VisionLMDB:
    def __init__(
        self,
        root_dir: str,
        vit_dir: str,
        image_dir: str,
        device: torch.device,
        map_size_gb: int = 200,
        auto_create_manifest: bool = True,
    ) -> None:
        self.root_dir = root_dir
        self.vit_dir = vit_dir
        self.image_dir = image_dir
        self.device = device

        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))

        self.lmdb_shards_dir = os.path.join(self.root_dir, "lmdb_shards")
        self.db_path = os.path.join(self.lmdb_shards_dir, f"vit_rank{self.rank}.lmdb")
        self.manifest_path = os.path.join(self.root_dir, "manifest.json")

        self.map_size = int(map_size_gb) * (1 << 30)

        self.tokens_per_image: int = 1024  # 先给默认值；第一次 build 后会校验
        self.feature_dim: int = _infer_hidden_size_from_config(self.vit_dir)

        self._env: Optional[lmdb.Environment] = None
        self._vip = VIPRuntime(self.vit_dir, device=self.device, amp_dtype=torch.float16)

        if auto_create_manifest:
            self._ensure_dirs_and_manifest()

    def _ensure_dirs_and_manifest(self) -> None:
        os.makedirs(self.lmdb_shards_dir, exist_ok=True)
        if not os.path.exists(self.manifest_path):
            man = {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "vit_dir": self.vit_dir,
                "world_size": self.world_size,
                "feature_shape": {
                    "tokens_per_image": int(self.tokens_per_image),
                    "feature_dim": int(self.feature_dim),
                },
                "lmdb_db_basename": "vit_rank{rank}.lmdb",
            }
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(man, f, ensure_ascii=False, indent=2)

    def _open(self) -> None:
        if self._env is not None:
            return
        os.makedirs(self.lmdb_shards_dir, exist_ok=True)
        in_train = int(os.environ.get("VIT_IN_TRAIN_LOOP", "0")) == 1
        readonly = in_train

        self._env = lmdb.open(
            self.db_path,
            map_size=self.map_size,
            subdir=True,
            readonly=readonly,
            lock=(not readonly),  # 只读时关锁，支持多 worker 并发读
            readahead=readonly,
            meminit=False,
            max_dbs=1,
        )
    def close(self) -> None:
        if self._env is None:
            return
        self._env.close()
        self._env = None

    def unload_vip(self) -> None:
        self._vip.unload()

    def rotate(self) -> None:
        """
        清空本 rank 的 LMDB（释放磁盘），仅用于 chunk 边界。
        训练循环中途调用会直接报错（防止 strict_read 读到一半库没了）。
        """
        chunk_steps = int(os.environ.get("VIT_CHUNK_STEPS", "0"))
        in_train = int(os.environ.get("VIT_IN_TRAIN_LOOP", "0"))
        if chunk_steps > 0 and in_train == 1:
            raise RuntimeError(
                "VisionLMDB.rotate() called INSIDE training loop while VIT_CHUNK_STEPS>0. "
                "Move rotate() to chunk boundary (outside trainer.train())."
            )

        # 1) 关闭 env
        self.close()
        self.unload_vip()
        # 2) 删除本 shard LMDB（你是单卡：vit_rank0.lmdb）
        if os.path.isdir(self.db_path):
            shutil.rmtree(self.db_path, ignore_errors=True)

        # 3) 可选：manifest 不删（保留 vit_dir / shape 等），你也可以按需删
        # if os.path.isfile(self.manifest_path):
        #     os.remove(self.manifest_path)

        # 4) 重建目录并 reopen
        os.makedirs(self.lmdb_shards_dir, exist_ok=True)
        self._ensure_dirs_and_manifest()
        self._open()

    def get(self, dataset_name: str, idx_val: int, img_name: str) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
        self._open()
        key = make_vision_key(dataset_name, idx_val, img_name)
        with self._env.begin(write=False) as txn:
            buf = txn.get(key)
        if buf is None:
            raise KeyError(f"LMDB missing key for {dataset_name}/{idx_val}/{img_name}")
        return decode_record(buf)
    def get_many(self, dataset_name: str, idx_val: int, img_names: list[str]):
        self._open()
        keys = [make_vision_key(dataset_name, idx_val, n) for n in img_names]
        with self._env.begin(write=False, buffers=True) as txn:
            bufs = [txn.get(k) for k in keys]
        # 任何一个缺失直接炸（训练期就是要严格）
        miss = [img_names[i] for i,b in enumerate(bufs) if b is None]
        if miss:
            raise KeyError(f"LMDB missing {len(miss)} keys for {dataset_name}/{idx_val}: {miss[:3]}")
        return [decode_record(b) for b in bufs]
    def get_or_build(
        self,
        ex: Dict[str, Any],
        dataset_name: str,
        idx_val: int,
        img_name: str,
    ) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
        self._open()
        key = make_vision_key(dataset_name, idx_val, img_name)

        with self._env.begin(write=False) as txn:
            buf = txn.get(key)
        if buf is not None:
            return decode_record(buf)

        # --- build now ---
        # 1) load image
        img_path = img_name
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.image_dir, img_name)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"image not found: {img_path}")

        with Image.open(img_path) as im0:
            im = ImageOps.exif_transpose(im0).convert("RGB")

        # 只对 thumb 做退化（如果你担心 eval 被影响，可再加环境变量开关）
        if img_name == _thumb_file_from_ex(ex):
            im = maybe_degrade_thumb(im)

        feat_bf16 = self._vip.encode_one(im)
        # 2) mask + mapping (strict)
        attn_mask_u8, _pending_ratio = compute_masks_for_image(ex, img_name)
        gpos_u16, goff_f16 = compute_global_pos_for_image(ex, img_name, attn_mask_u8)

        # 3) sanity check
        Nv = int(attn_mask_u8.shape[0])
        if feat_bf16.shape[0] != Nv:
            raise ValueError(f"VIT Nv mismatch: feat {feat_bf16.shape} vs mask Nv={Nv}")
        if feat_bf16.shape[1] != int(self.feature_dim):
            raise ValueError(f"VIT D mismatch: feat D={feat_bf16.shape[1]} vs expected {self.feature_dim}")

        # 4) write
        rec = encode_record(feat_bf16, attn_mask_u8, gpos_u16, goff_f16)
        with self._env.begin(write=True) as txn:
            txn.put(key, rec, overwrite=True)

        return feat_bf16, attn_mask_u8, gpos_u16, goff_f16
    def get_or_build_many(
        self,
        ex: Dict[str, Any],
        dataset_name: str,
        idx_val: int,
        img_names: list[str],
    ):
        self._open()
        keys = [make_vision_key(dataset_name, idx_val, n) for n in img_names]

        # 1) 先批量读
        with self._env.begin(write=False) as txn:
            bufs = [txn.get(k) for k in keys]

        miss_idx = [i for i, b in enumerate(bufs) if b is None]
        if not miss_idx:
            return [decode_record(b) for b in bufs]

        # 2) build missing（一次性 VIP batch）
        def _img_abs_path(img_name: str) -> str:
            if os.path.isabs(img_name):
                return img_name
            return os.path.join(self.image_dir, img_name)

        pil_imgs: list[Image.Image] = []
        miss_names: list[str] = []
        for i in miss_idx:
            img_name = img_names[i]
            img_path = _img_abs_path(img_name)
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"image not found: {img_path}")
            with Image.open(img_path) as im0:
                im = ImageOps.exif_transpose(im0).convert("RGB")
            pil_imgs.append(im)
            miss_names.append(img_name)

        self._vip.load()
        feats_b = self._vip.encode_batch(pil_imgs)  # [Bm,Nv,D] bf16 cpu

        # 3) 写 LMDB（一个 txn）
        new_bufs = {}
        with self._env.begin(write=True) as txn:
            for j, img_name in enumerate(miss_names):
                feat_bf16 = feats_b[j]  # [Nv,D]
                attn_mask_u8, _pending_ratio = compute_masks_for_image(ex, img_name)
                gpos_u16, goff_f16 = compute_global_pos_for_image(ex, img_name, attn_mask_u8)

                Nv = int(attn_mask_u8.shape[0])
                if feat_bf16.shape[0] != Nv:
                    raise ValueError(f"VIT Nv mismatch: feat {feat_bf16.shape} vs mask Nv={Nv}")
                if feat_bf16.shape[1] != int(self.feature_dim):
                    raise ValueError(f"VIT D mismatch: feat D={feat_bf16.shape[1]} vs expected {self.feature_dim}")

                rec = encode_record(feat_bf16, attn_mask_u8, gpos_u16, goff_f16)
                k = make_vision_key(dataset_name, idx_val, img_name)
                txn.put(k, rec, overwrite=True)
                new_bufs[k] = rec

        # 4) 拼回原顺序返回
        out = []
        for i, k in enumerate(keys):
            b = bufs[i]
            if b is None:
                b = new_bufs[k]
            out.append(decode_record(b))
        return out

class VisionCacheRotateCallback(TrainerCallback):
    def __init__(self, vit_db: VisionLMDB, rotate_every_steps: int) -> None:
        self.vit_db = vit_db
        self.rotate_every_steps = int(rotate_every_steps)
        self._pending = False

    def _maybe_rotate(self, state):
        if not self._pending:
            return
        step = int(state.global_step)
        self.vit_db.rotate()
        self._pending = False
        print(f"[vit-rotate] rotated AFTER eval/save at step={step}", flush=True)

    def on_step_end(self, args, state, control, **kwargs):
        if self.rotate_every_steps <= 0:
            return control
        step = int(state.global_step)
        if step > 0 and (step % self.rotate_every_steps) == 0:
            # 这里只标记，不在这里真删（避免和 eval/save 同步点撞车）
            self._pending = True
        return control

    def on_evaluate(self, args, state, control, **kwargs):
        self._maybe_rotate(state)
        return control

    def on_save(self, args, state, control, **kwargs):
        self._maybe_rotate(state)
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        # 万一这一步没触发 eval/save，也要在进入下一步训练前 rotate
        self._maybe_rotate(state)
        return control

# -------- meta helpers (严格：缺字段就报错) --------
def _jsonish(x: Any) -> Dict[str, Any]:
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, (bytes, bytearray)):
        x = x.decode("utf-8")
    if isinstance(x, str):
        return json.loads(x)
    raise TypeError(f"vision_meta must be dict/str/bytes, got {type(x)}")



def build_token_mask_from_content_box(
    content_box,          # [x0,y0,x1,y1] on canvas
    canvas_size: int,
    patch_size: int,
    grid_hw: Tuple[int, int],   # (h,w)
    use_unshuffle: bool,
    unshuffle_r: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回：
      - attn_mask_u8: [Nv] uint8  (ratio>0 -> 1 else 0)
      - pending_ratio_f16: [Nv] float16 in [0,1]
    """
    Hc = Wc = int(canvas_size)
    gy, gx = int(grid_hw[0]), int(grid_hw[1])  # grid_hw=(h,w)

    if use_unshuffle:
        th = int(gy // unshuffle_r)
        tw = int(gx // unshuffle_r)
        block = int(patch_size * unshuffle_r)
    else:
        th, tw = gy, gx
        block = int(patch_size)

    x0, y0, x1, y1 = [float(v) for v in content_box]
    x0 = max(0.0, min(float(Wc), x0))
    x1 = max(0.0, min(float(Wc), x1))
    y0 = max(0.0, min(float(Hc), y0))
    y1 = max(0.0, min(float(Hc), y1))

    xs0 = (np.arange(tw, dtype=np.float32) * block)[None, :]
    ys0 = (np.arange(th, dtype=np.float32) * block)[:, None]
    xs1 = xs0 + block
    ys1 = ys0 + block

    ix0 = np.maximum(xs0, x0)
    iy0 = np.maximum(ys0, y0)
    ix1 = np.minimum(xs1, x1)
    iy1 = np.minimum(ys1, y1)

    iw = np.maximum(0.0, ix1 - ix0)
    ih = np.maximum(0.0, iy1 - iy0)
    inter = iw * ih
    ratio = inter / float(block * block)

    attn = (ratio > 0).astype(np.uint8)     # [th,tw]
    pending = ratio.astype(np.float16)      # [th,tw]

    if use_unshuffle:
        attn = attn.reshape(th, 1, tw, 1).repeat(unshuffle_r, axis=1).repeat(unshuffle_r, axis=3)
        pending = pending.reshape(th, 1, tw, 1).repeat(unshuffle_r, axis=1).repeat(unshuffle_r, axis=3)

    attn = attn.reshape(-1)
    pending = pending.reshape(-1)

    return attn, pending
def _find_view_meta(ex: Dict[str, Any], image_file: str) -> Dict[str, Any]:
    vm = _jsonish(ex.get("vision_meta"))
    views = vm.get("views")
    if not isinstance(views, list):
        raise KeyError("vision_meta.views missing or not a list")

    for v in views:
        # 兼容：老字段 image_file、新字段 file
        fn = v.get("image_file")
        if fn is None:
            fn = v.get("file")
        if fn == image_file:
            return v

    raise KeyError(f"cannot find view meta for image_file={image_file}")

def _find_thumb_view(vm: Dict[str, Any]) -> Dict[str, Any]:
    views = vm.get("views")
    if not isinstance(views, list):
        raise KeyError("vision_meta.views missing or not a list")

    for v in views:
        kind = v.get("view_kind")
        if kind is None:
            kind = v.get("type")   # 兼容你现在的 schema
        if kind == "thumb":
            return v

    raise KeyError("cannot find thumb view in vision_meta.views")
def _thumb_file_from_ex(ex: Dict[str, Any]) -> str:
    vm = _jsonish(ex.get("vision_meta"))
    thumb = _find_thumb_view(vm)
    fn = thumb.get("image_file")
    if fn is None:
        fn = thumb.get("file")
    if not fn:
        raise KeyError("thumb view missing image_file/file")
    return str(fn)

def maybe_degrade_thumb(im: Image.Image) -> Image.Image:
    """
    互斥采样（总概率=THUMB_DOWNSAMPLE_P+THUMB_GAUSS_P）：
      - 10%：下采样->上采样 + 极轻高斯（更像低分辨率缩略图，布局保留）
      - 5% ：仅轻高斯
    """
    u = random.random()
    if u < THUMB_DOWNSAMPLE_P:
        w, h = im.size
        # 强度别太猛：2~3 足够“打掉小字”，布局仍稳定
        r = random.choice([2, 3])
        w2, h2 = max(1, w // r), max(1, h // r)
        im2 = im.resize((w2, h2), resample=Image.BILINEAR)
        im2 = im2.resize((w, h), resample=Image.BICUBIC)
        # 再加一点点极轻高斯，专打细线/小字锐度
        radius = random.uniform(0.3, 0.9)
        im2 = im2.filter(ImageFilter.GaussianBlur(radius=radius))
        return im2

    if u < (THUMB_DOWNSAMPLE_P + THUMB_GAUSS_P):
        radius = random.uniform(0.5, 1.2)
        return im.filter(ImageFilter.GaussianBlur(radius=radius))

    return im
def _vm_core(vm: Dict[str, Any]) -> Tuple[int, int, Tuple[int, int], bool, int]:
    """
    返回：patch_size, canvas_size, grid_hw(h,w), use_unshuffle, unshuffle_r
    兼容：
      - schema v0: vision_meta.patch_size / canvas_size / grid_hw
      - schema v1: vision_meta.vision_encoder.patch_size / image_size / grid_hw
    """
    # v0 (扁平)
    if "patch_size" in vm and "canvas_size" in vm and "grid_hw" in vm:
        patch_size = int(vm["patch_size"])
        canvas_size = int(vm["canvas_size"])
        grid_hw = tuple(vm["grid_hw"])
        use_unshuffle = bool(vm.get("use_unshuffle", False))
        unshuffle_r = int(vm.get("unshuffle_r", 1))
        return patch_size, canvas_size, (int(grid_hw[0]), int(grid_hw[1])), use_unshuffle, unshuffle_r

    # v1 (你现在贴的这种：vision_encoder)
    ve = vm.get("vision_encoder")
    if not isinstance(ve, dict):
        raise KeyError("vision_meta missing both flat keys and vision_encoder dict")

    patch_size = int(ve["patch_size"])
    canvas_size = int(ve.get("image_size", 448))
    grid_hw = tuple(ve["grid_hw"])
    use_unshuffle = bool(ve.get("use_unshuffle", vm.get("use_unshuffle", False)))
    unshuffle_r = int(ve.get("unshuffle_r", vm.get("unshuffle_r", 1)))
    return patch_size, canvas_size, (int(grid_hw[0]), int(grid_hw[1])), use_unshuffle, unshuffle_r


def compute_masks_for_image(ex: Dict[str, Any], image_file: str) -> Tuple[np.ndarray, np.ndarray]:
    vm = _jsonish(ex.get("vision_meta"))
    patch_size, canvas_size, grid_hw, use_unshuffle, unshuffle_r = _vm_core(vm)

    v = _find_view_meta(ex, image_file)
    pad = v.get("pad")
    if not isinstance(pad, dict) or "content_box" not in pad:
        raise KeyError(f"view.pad.content_box missing for image_file={image_file}")

    return build_token_mask_from_content_box(
        pad["content_box"],
        canvas_size=canvas_size,
        patch_size=patch_size,
        grid_hw=grid_hw,
        use_unshuffle=use_unshuffle,
        unshuffle_r=unshuffle_r,
    )


def compute_global_pos_for_image(
    ex: Dict[str, Any],
    image_file: str,
    attn_mask_u8: np.ndarray,   # [Nv]
) -> Tuple[np.ndarray, np.ndarray]:
    vm = _jsonish(ex.get("vision_meta"))
    patch_size, canvas_size, grid_hw, use_unshuffle, unshuffle_r = _vm_core(vm)

    gy, gx = int(grid_hw[0]), int(grid_hw[1])
    if use_unshuffle:
        th = int(gy // unshuffle_r)
        tw = int(gx // unshuffle_r)
        block = int(patch_size * unshuffle_r)
    else:
        th, tw = gy, gx
        block = int(patch_size)

    Nv = int(th * tw)
    if attn_mask_u8.shape != (Nv,):
        raise ValueError(f"attn_mask_u8 shape mismatch: got {attn_mask_u8.shape}, expect {(Nv,)}")

    v = _find_view_meta(ex, image_file)
    view_kind = v.get("view_kind")
    if view_kind is None:
        view_kind = v.get("type")  # 兼容你现在的 schema

    # thumb：恒等映射
    if view_kind == "thumb":
        global_pos = np.arange(Nv, dtype=np.uint16)

        # 由 token index -> (gy_idx, gx_idx)
        gx_idx = (global_pos % tw).astype(np.int32)
        gy_idx = (global_pos // tw).astype(np.int32)

        denom_x = max(tw - 1, 1)
        denom_y = max(th - 1, 1)
        x_norm = (gx_idx.astype(np.float32) / denom_x) * 2.0 - 1.0
        y_norm = (gy_idx.astype(np.float32) / denom_y) * 2.0 - 1.0

        zeros = np.zeros_like(x_norm, dtype=np.float32)
        global_off = np.stack([x_norm, y_norm, zeros, zeros], axis=-1).astype(np.float16)  # [Nv,4]

        # 跟你现在一样：padding token 清零
        valid = attn_mask_u8.astype(np.bool_)
        global_pos = np.where(valid, global_pos, np.zeros_like(global_pos))
        global_off = np.where(valid[:, None], global_off, np.zeros_like(global_off))
        return global_pos, global_off

    thumb = _find_thumb_view(vm)

    # ---- 如果是老 schema（有 tile_origin/thumb_origin/thumb_scale），继续沿用旧逻辑 ----
    if ("tile_origin" in v) and ("thumb_origin" in thumb) and ("thumb_scale" in thumb):
        tile_origin = np.array(v["tile_origin"], dtype=np.float32)
        thumb_origin = np.array(thumb["thumb_origin"], dtype=np.float32)
        thumb_scale = float(thumb["thumb_scale"])

        xs = (np.arange(tw, dtype=np.float32) + 0.5) * block
        ys = (np.arange(th, dtype=np.float32) + 0.5) * block
        xx, yy = np.meshgrid(xs, ys)
        centers_tile = np.stack([xx, yy], axis=-1).reshape(-1, 2)

        p_work = centers_tile + tile_origin[None, :]
        p_thumb = (p_work - thumb_origin[None, :]) * thumb_scale

    else:
        # ---- 新 schema：用 crop_box_work + pad.scale/offset 做 tile->work->thumb 映射 ----
        pad_t = v.get("pad")
        if not isinstance(pad_t, dict):
            raise KeyError(f"view.pad missing for image_file={image_file}")
        if "scale" not in pad_t or "offset" not in pad_t:
            raise KeyError(f"view.pad.scale/offset missing for image_file={image_file}")
        if "crop_box_work" not in v:
            raise KeyError(f"view.crop_box_work missing for image_file={image_file}")

        pad_th = thumb.get("pad")
        if not isinstance(pad_th, dict):
            raise KeyError("thumb.pad missing")
        if "scale" not in pad_th or "offset" not in pad_th:
            raise KeyError("thumb.pad.scale/offset missing")
        if "crop_box_work" not in thumb:
            raise KeyError("thumb.crop_box_work missing")

        tile_scale = float(pad_t["scale"])
        tile_off = np.array(pad_t["offset"], dtype=np.float32)
        tile_crop0 = np.array(v["crop_box_work"][:2], dtype=np.float32)

        thumb_scale = float(pad_th["scale"])
        thumb_off = np.array(pad_th["offset"], dtype=np.float32)
        thumb_crop0 = np.array(thumb["crop_box_work"][:2], dtype=np.float32)

        # token 中心点（tile canvas 坐标，448 上）
        xs = (np.arange(tw, dtype=np.float32) + 0.5) * block
        ys = (np.arange(th, dtype=np.float32) + 0.5) * block
        xx, yy = np.meshgrid(xs, ys)
        centers_tile = np.stack([xx, yy], axis=-1).reshape(-1, 2)  # [Nv,2]

        # tile canvas -> tile src (crop内) -> work
        p_src = (centers_tile - tile_off[None, :]) / tile_scale
        p_work = p_src + tile_crop0[None, :]

        # work -> thumb canvas
        p_thumb = (p_work - thumb_crop0[None, :]) * thumb_scale + thumb_off[None, :]

    # thumb token cell index
    gx_idx = np.floor(p_thumb[:, 0] / block).astype(np.int32)
    gy_idx = np.floor(p_thumb[:, 1] / block).astype(np.int32)

    gx_idx = np.clip(gx_idx, 0, tw - 1)
    gy_idx = np.clip(gy_idx, 0, th - 1)

    gpos = (gy_idx * tw + gx_idx).astype(np.uint16)

    # ==== 新增：绝对坐标归一化到 [-1,1] ====
    denom_x = max(tw - 1, 1)
    denom_y = max(th - 1, 1)
    x_norm = (gx_idx.astype(np.float32) / denom_x) * 2.0 - 1.0
    y_norm = (gy_idx.astype(np.float32) / denom_y) * 2.0 - 1.0

    # 偏移：cell 内 [-0.5,0.5]（中心为 0）
    cx = (gx_idx.astype(np.float32) + 0.5) * block
    cy = (gy_idx.astype(np.float32) + 0.5) * block
    dx = (p_thumb[:, 0] - cx) / block
    dy = (p_thumb[:, 1] - cy) / block

    # ==== 关键：global_off 变成 4 维 ====
    goff = np.stack([x_norm, y_norm, dx, dy], axis=-1).astype(np.float16)

    valid = attn_mask_u8.astype(np.bool_)
    gpos = np.where(valid, gpos, np.zeros_like(gpos))
    goff = np.where(valid[:, None], goff, np.zeros_like(goff))
    return gpos, goff

# -------- VIP runtime (lazy load / unload) --------
class VIPRuntime:
    def __init__(
        self,
        vit_dir: str,
        device: torch.device,
        amp_dtype: torch.dtype = torch.bfloat16,   # 你也可以改成 torch.bfloat16
    ) -> None:
        self.vit_dir = vit_dir
        self.device = device
        self.amp_dtype = amp_dtype
        self._model = None
        self._proc = None
        self._checked_once = False

    def load(self) -> None:
        if self._model is not None:
            return

        self._proc = AutoImageProcessor.from_pretrained(
            self.vit_dir, local_files_only=True, trust_remote_code=True
        )

        # 关键：加载时就指定 torch_dtype，避免权重默认 float32
        self._model = AutoModel.from_pretrained(
            self.vit_dir,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=self.amp_dtype,
            low_cpu_mem_usage=True,
        ).eval()

        # 关键：再强制整体 to 到同一 dtype（防止某些参数没跟着 torch_dtype 走）
        self._model = self._model.to(self.device)
        self._model = self._model.to(dtype=self.amp_dtype)

    def unload(self) -> None:
        if self._model is None:
            return
        del self._model
        del self._proc
        self._model = None
        self._proc = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def encode_one(self, pil_img: Image.Image) -> torch.Tensor:
        """
        返回：feat_bf16_cpu [Nv, D]，Nv=drop CLS 后的 token 数
        """
        self.load()
        inputs = self._proc(images=pil_img, return_tensors="pt")

        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.amp_dtype)

        # 只在第一次做一次 dtype 自检，避免刷屏
        if not self._checked_once:
            # 找一个参数看 dtype（conv bias/weight 都属于参数）
            p = next(self._model.parameters())
            if p.dtype != pixel_values.dtype:
                raise RuntimeError(
                    f"[VIPRuntime] dtype mismatch: pixel_values={pixel_values.dtype}, "
                    f"model_param={p.dtype}. You are still using a wrong VIPRuntime "
                    f"definition or model wasn't cast correctly."
                )
            self._checked_once = True

        out = self._model(pixel_values=pixel_values)

        feats = out.last_hidden_state      # [1, 1025, D]
        feats = feats[:, 1:, :]            # drop CLS -> [1, Nv, D]
        feats = feats.to(torch.bfloat16).cpu().contiguous()
        return feats[0]
    @torch.inference_mode()
    def encode_batch(self, pil_imgs: List[Image.Image]) -> torch.Tensor:
        """
        返回：feats_bf16_cpu [B, Nv, D]，Nv=drop CLS 后的 token 数
        """
        if not isinstance(pil_imgs, list) or len(pil_imgs) == 0:
            raise ValueError("encode_batch expects a non-empty list of PIL images")

        self.load()
        inputs = self._proc(images=pil_imgs, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.amp_dtype)

        if not self._checked_once:
            p = next(self._model.parameters())
            if p.dtype != pixel_values.dtype:
                raise RuntimeError(
                    f"[VIPRuntime] dtype mismatch: pixel_values={pixel_values.dtype}, model_param={p.dtype}"
                )
            self._checked_once = True

        out = self._model(pixel_values=pixel_values)
        feats = out.last_hidden_state      # [B, 1+Nv, D]
        feats = feats[:, 1:, :]            # drop CLS -> [B, Nv, D]
        feats = feats.to(torch.bfloat16).cpu().contiguous()
        return feats

vit_db: VisionLMDB | None = None
VIT_TOKENS_PER_IMAGE = 1024
VISION_FEATURE_DIM: int | None = None
def _fmt_hms(sec: float) -> str:
    sec = int(max(0, sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def enforce_single_gpu_only():
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    lr = int(os.environ.get("LOCAL_RANK", "0"))
    rk = int(os.environ.get("RANK", "0"))
    if ws != 1 or lr != 0 or rk != 0:
        raise RuntimeError(
            f"[FATAL] This script is single-GPU only. "
            f"Got WORLD_SIZE={ws}, LOCAL_RANK={lr}, RANK={rk}."
        )
    if dist.is_available() and dist.is_initialized():
        if dist.get_world_size() != 1:
            raise RuntimeError("[FATAL] DDP initialized but world_size != 1.")
def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed = time.time() - START_TIME
    print(f"[{ts}] (+{_fmt_hms(elapsed)}) {msg}", flush=True)


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def is_main_process() -> bool:
    return get_rank() == 0


# =====================
# RoPE state 清理（和 CPT 一致）
# =====================
ROPE_OWNER_PAT = re.compile(r"\.(rotary_emb|rotary|rope|rope_emb)\.")
ROPE_LEAF_ALLOW = {"inv_freq"}
ROPE_LEAF_DROP = {
    "cos_cached",
    "sin_cached",
    "cos_cached_long",
    "sin_cached_long",
    "max_seq_len_cached",
    "_seq_len_cached",
    "seq_len_cached",
    "cached_seq_len",
    "cached_max_seq_len",
    "base",
    "theta",
    "rope_base",
    "rope_theta",
}


def drop_rope_buffers_from_state(state: Dict[str, torch.Tensor]) -> int:
    to_drop = []
    for k in list(state.keys()):
        if ROPE_OWNER_PAT.search(k):
            leaf = k.rsplit(".", 1)[-1]
            if (leaf in ROPE_LEAF_DROP) or (leaf not in ROPE_LEAF_ALLOW):
                to_drop.append(k)
    for k in to_drop:
        state.pop(k, None)
    return len(to_drop)


# =====================
# 保存相关
# =====================
def _assert_no_top_numeric_keys(keys):
    bad = [k for k in keys if re.match(r"^\d+\.", k)]
    if bad:
        raise RuntimeError(
            f"State dict has top-level numeric keys (e.g. {bad[:5]}); "
            f"this usually means you're saving a submodule instead of the full model."
        )


def safe_save_full_model(model, out_dir: str, cfg_obj=None):
    os.makedirs(out_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    state = {
        k: (v.detach().cpu() if v.is_floating_point() else v.cpu())
        for k, v in model_to_save.state_dict().items()
    }
    _assert_no_top_numeric_keys(state.keys())
    save_file(state, os.path.join(out_dir, "model.safetensors"))

    if cfg_obj is not None:
        with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(getattr(cfg_obj, "__dict__", {}), f, ensure_ascii=False, indent=2)

def safe_save_base_and_lora(model, out_dir: str, cfg_obj=None):
    """
    在 out_dir 下生成：
      - base/model.safetensors + base/config.json
      - lora/lora.safetensors + lora/lora_config.json
      - meta.json  → 和 GRPO 同一风格的 LoRA 元信息
    """
    os.makedirs(out_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    full_state = model_to_save.state_dict()

    base_state = {}
    lora_state = {}

    for k, v in full_state.items():
        dest = lora_state if ".adapters." in k else base_state
        dest[k] = v.detach().cpu() if v.is_floating_point() else v.cpu()

    _assert_no_top_numeric_keys(base_state.keys())
    _assert_no_top_numeric_keys(lora_state.keys())

    base_dir = os.path.join(out_dir, "base")
    lora_dir = os.path.join(out_dir, "lora")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(lora_dir, exist_ok=True)

    # ---------- base ----------
    save_file(base_state, os.path.join(base_dir, "model.safetensors"))
    if cfg_obj is not None:
        with open(os.path.join(base_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(getattr(cfg_obj, "__dict__", {}), f, ensure_ascii=False, indent=2)

    # ---------- LoRA ----------
    if lora_state:
        # 1) 真正的 LoRA 权重
        save_file(lora_state, os.path.join(lora_dir, "lora.safetensors"))

        # 2) 当前脚本内部用的 LoRA 配置（你已经有了）
        lora_meta = {
            "adapter_name": LORA_NAME,
            "rank": LORA_RANK,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "target": LORA_TARGET,
        }
        with open(os.path.join(lora_dir, "lora_config.json"), "w", encoding="utf-8") as f:
            json.dump(lora_meta, f, ensure_ascii=False, indent=2)

        # 3) 和 GRPO 一样风格的 meta.json（放在 checkpoint 根目录）
        meta_path = os.path.join(out_dir, "meta.json")
        if not os.path.exists(meta_path):
            meta = {
                "run_name": RUN_NAME,
                "adapter_name": LORA_NAME,
                "description": (
                    "LoRA adapter for TinyLLM 0.5B, VLM SFT "
                    "on mixed image-text instructions."
                ),
                "lora_target": LORA_TARGET,
                "lora_rank": LORA_RANK,
                "lora_dropout": LORA_DROPOUT,
                "lora_alpha": LORA_ALPHA,
                "created_at": RUN_ID,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
class SafeTensorsCallback(TrainerCallback):
    """
    每次 Trainer 保存 checkpoint-xxxx 时，额外写 base+LoRA 拆开的权重。
    不依赖 self.trainer，直接用 kwargs["model"]。
    """
    def on_save(self, args, state, control, **kwargs):
        # ✅ 只在主进程跑（单卡等价于 True）
        if hasattr(state, "is_world_process_zero") and (not state.is_world_process_zero):
            return

        model = kwargs.get("model", None)
        if model is None:
            print("[safe-save][WARN] kwargs has no model; skip.")
            return

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            print(f"[safe-save][WARN] checkpoint dir not found: {ckpt_dir}")
            return

        model_to_save = model.module if hasattr(model, "module") else model
        try:
            safe_save_base_and_lora(
                model_to_save,
                ckpt_dir,
                cfg_obj=getattr(model_to_save, "cfg", None),
            )
            print(f"[safe-save] wrote base+LoRA under {ckpt_dir}")
        except Exception as e:
            print(f"[safe-save][WARN] {e}")

# =====================
# LoRA 准备（挂 adapter + 冻结 base）
# =====================

STAGE1_STRICT_QFORMER_ONLY = bool(int(os.environ.get("SFT_STAGE1_STRICT_QFORMER_ONLY", "1")))

def attach_lora_once_for_stage2(model, stage):
    if stage != 2:
        return
    existing = [k for k in model.state_dict() if f".adapters.{LORA_NAME}." in k]
    if not existing:
        attach_lora_if_needed(model)
    model.activate_single_lora(LORA_NAME)


def attach_lora_if_needed(model: TinyLLM):
    if not USE_LORA:
        if is_main_process():
            log("[LoRA] USE_LORA=0, skip attaching LoRA adapters.")
        return
    model.attach_lora_adapter(
        adapter_name=LORA_NAME,
        rank=LORA_RANK,
        dropout=LORA_DROPOUT,
        alpha=LORA_ALPHA,
        target="all",
    )
    model.activate_single_lora(LORA_NAME)

def set_trainable_by_patterns(model: TinyLLM, trainable_patterns: list[str], tag: str):
    total_params = 0
    trainable_params = 0
    for name, p in model.named_parameters():
        total_params += p.numel()
        if any(pat in name for pat in trainable_patterns):
            p.requires_grad = True
            trainable_params += p.numel()
        else:
            p.requires_grad = False
    if is_main_process():
        log(
            f"[TRAINABLE:{tag}] trainable={trainable_params/1e6:.2f}M / "
            f"total={total_params/1e6:.2f}M ({100.0*trainable_params/total_params:.2f}%)"
        )

def configure_train_stage(model: TinyLLM, stage: int):
    if stage == 1:
        patterns = [
            "query_embed", "qformer_blocks", "qformer_final_norm",
            "vision_projector", "vision_type_embed",
            "vision_view_embed", "vision_pos_embed",
            "vision_film_mlp",
            "vision_gate",      # 一把匹配 vision_gate_view/pos/film
            "vision_kv_sep",    # 如果你用 sep
        ]
        set_trainable_by_patterns(model, patterns, tag="stage1_qformer+vision_mod")
        return

    if stage == 2:
        # stage2：挂 LoRA，然后 Q-former + LoRA 一起训

        patterns = [
            ".adapters.",  # LoRA
            "vision_projector",  # bridge
            "vision_type_embed",

            # Q-Former
            "query_embed",
            "qformer_blocks",
            "qformer_final_norm",

            # Vision-side embeddings / conditioning
            "vision_view_embed",
            "vision_pos_embed",
            "vision_film_mlp",
            "vision_gate",  # 匹配 vision_gate_view / pos / film
            "vision_kv_sep",  # KV separator（如果启用）
        ]
        set_trainable_by_patterns(model, patterns, tag="stage2_qformer+lora")
        return

    raise ValueError(f"bad SFT_STAGE={stage}")
# =====================
# Tokenizer & Model
# =====================
def build_tokenizer_and_model():
    assert os.path.isdir(LOCAL_SNAPSHOT), f"Tokenizer dir not found: {LOCAL_SNAPSHOT}"
    assert os.path.isfile(
        os.path.join(LOCAL_SNAPSHOT, "tokenizer.json")
    ), "Missing tokenizer.json"

    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_SNAPSHOT,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer.get_vocab())

    cfg = Config(
        vocab_size=vocab_size,
        train_maxlength=8192,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        moe_use_detach=False,
        use_moe=False,
        dropout=0.0,
        learnable_temp=True,
        drop_path=0.0,
        residual_dropout=0.0,
        rope_type="yarn",
        use_ssm=False,
        ssm_layers=[],
        # Vision / Q-Former
        use_vision=True,
        vision_feature_dim=VISION_FEATURE_DIM,
        vision_dropout=0.0,
        vision_use_rmsnorm=True,
        qformer_num_layers=4,
        qformer_mlp_ratio=2.0,
        loss_reduction=SFT_LOSS_REDUCTION,
        sample_mean_alpha=SFT_SAMPLE_MEAN_ALPHA,
    )
    cfg.use_checkpoint = False
    cfg.checkpoint_use_reentrant = False

    model = TinyLLM(cfg)
    if SFT_STAGE == 2:
        attach_lora_if_needed(model)
        # 从 CPT checkpoint 加载
    ckpt_path = RESUME_ROOT
    if ckpt_path is None:
        raise FileNotFoundError(f"[RESUME] 未设置 RESUME_ROOT；当前值={RESUME_ROOT!r}")

    safe_path = os.path.join(ckpt_path, "model.safetensors")
    bin_path = os.path.join(ckpt_path, "pytorch_model.bin")

    if os.path.exists(safe_path):
        log(f"[RESUME] loading safetensors: {safe_path}")
        state = load_file(safe_path)
        src_file = safe_path
    elif os.path.exists(bin_path):
        log(f"[RESUME] loading torch bin: {bin_path}")
        state = torch.load(bin_path, map_location="cpu")
        src_file = bin_path
    else:
        items = os.listdir(ckpt_path)
        raise FileNotFoundError(
            f"[RESUME] 在 {ckpt_path} 未找到 'model.safetensors' 或 'pytorch_model.bin'。\n"
            f"dir = {items}"
        )

    dropped = drop_rope_buffers_from_state(state)
    ret = model.load_state_dict(state, strict=False)
    log(f"[RESUME] loaded from {src_file}, dropped {dropped} rope-keys")
    if ret.missing_keys:
        log(f"[RESUME] missing_keys (first 20) = {ret.missing_keys[:20]}")
    if ret.unexpected_keys:
        log(f"[RESUME] unexpected_keys (first 20) = {ret.unexpected_keys[:20]}")

    # 准备 LoRA + 冻结
    configure_train_stage(model, SFT_STAGE)

    return tokenizer, model


# =====================
# SFT Collator (右 pad + 视觉特征)
# =====================
class SFTCollatorRightPad:
    def __init__(self, tokenizer, ignore_index=IGNORE_INDEX, vit_db: VisionLMDB | None = None, vit_strict_read: bool = True):
        self.tok = tokenizer
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.ignore_index = ignore_index
        self.vit_db = vit_db
        self.vit_strict_read = bool(vit_strict_read)
    def _to_int(self, x):
        if torch.is_tensor(x):
            return int(x.item())
        return int(x)

    def _to_str(self, x):
        if isinstance(x, str):
            return x
        return "" if x is None else str(x)

    def _to_str_list(self, x):
        if isinstance(x, list):
            out = []
            for it in x:
                s = self._to_str(it)
                if not s:
                    raise RuntimeError("[COLLATOR][FATAL] empty image file in list")
                out.append(s)
            return out
        if isinstance(x, str) and x:
            # 兼容老数据（单图），但你说“只用新链路”，这里也可以直接 raise
            return [x]
        raise RuntimeError(f"[COLLATOR][FATAL] image_files must be list[str], got {type(x)}")

    def __call__(self, batch):
        seqs = []
        lengths = []

        tgt_masks = []

        vis_feats_list = []
        vis_mask_list  = []
        vis_gpos_list  = []
        vis_goff_list  = []
        any_vision = False

        vision_batch_indices = []
        for batch_index, ex in enumerate(batch):
            ids = ex["input_ids"]
            if not torch.is_tensor(ids):
                ids = torch.tensor(ids, dtype=torch.long)

            L = int(ex.get("length", ids.numel()))
            if L != ids.numel():
                raise RuntimeError(f"[COLLATOR][FATAL] length mismatch: length={L}, ids={ids.numel()}")
            seqs.append(ids)
            lengths.append(L)



            if "target_mask" not in ex:
                raise RuntimeError("[COLLATOR][FATAL] missing target_mask in example.")
            tm = ex["target_mask"]
            if not torch.is_tensor(tm):
                tm = torch.tensor(tm, dtype=torch.long)
            if tm.numel() != ids.numel():
                raise RuntimeError(f"[COLLATOR][FATAL] target_mask length mismatch: {tm.numel()} vs {ids.numel()}")
            tgt_masks.append(tm)

            # Text replay rows use an empty image_files list. They go through a
            # separate text-only forward pass and therefore receive no visual prefix.
            img_list = ex.get("image_files", None)
            if not isinstance(img_list, list):
                raise RuntimeError(f"[COLLATOR][FATAL] expect ex['image_files'] as list[str], got {type(img_list)}")
            if len(img_list) == 0:
                continue
            if len(img_list) != 5:
                raise RuntimeError(f"[COLLATOR][FATAL] expect 0 or 5 image files, got {len(img_list)}")
            if self.vit_db is None:
                raise RuntimeError("[COLLATOR][FATAL] vit_db is None but vision is required.")

            idx_val = self._to_int(ex.get("index", -1))
            if idx_val < 0:
                raise RuntimeError("[COLLATOR][FATAL] need valid ex['index']")
            dataset_name = self._to_str(ex.get("dataset", ""))
            if not dataset_name:
                raise RuntimeError("[COLLATOR][FATAL] ex['dataset'] is empty")
            thumb_fn = _thumb_file_from_ex(ex)
            tiles = [p for p in img_list if p != thumb_fn]
            if len(tiles) != 4:
                raise RuntimeError(f"[COLLATOR][FATAL] expect 4 tiles + 1 thumb, got tiles={len(tiles)}")

            # 永远保证 thumb 在最后
            # global_pos/global_off already encode each crop's real position.
            # Keep the canonical TL, TR, BL, BR order by default for exact
            # train/inference reproducibility; opt in only for ablations.
            if TILE_SHUFFLE_P > 0.0 and random.random() < TILE_SHUFFLE_P:
                random.shuffle(tiles)
            img_list = tiles + [thumb_fn]

            # 可选：硬断言，确保“默契”永远成立
            if img_list[-1] != thumb_fn:
                raise RuntimeError("[COLLATOR][FATAL] thumb is not the last view after reorder (bug)")
                # ✅ 真正从 LMDB 取 5 张图的 record
            if self.vit_strict_read:
                recs = self.vit_db.get_many(dataset_name, idx_val, img_list)
            else:
                recs = self.vit_db.get_or_build_many(ex, dataset_name, idx_val, img_list)

            feats_v = []
            mask_v  = []
            gpos_v  = []
            goff_v  = []

            for (feat_bf16, m_u8, gp_u16, go_f16) in recs:
                feats_v.append(feat_bf16)  # torch.bfloat16 [Nv,D] CPU
                mask_v.append(torch.from_numpy(m_u8).to(torch.bool))           # [Nv]
                gpos_v.append(torch.from_numpy(gp_u16).to(torch.int64))        # [Nv]
                goff_v.append(torch.from_numpy(go_f16).to(torch.float16))      # [Nv,4]

            # stack views: [5,Nv,*]
            vf = torch.stack(feats_v, dim=0)           # [5,Nv,D]
            vm = torch.stack(mask_v, dim=0)            # [5,Nv] bool
            vg = torch.stack(gpos_v, dim=0)            # [5,Nv] int64
            vo = torch.stack(goff_v, dim=0)            # [5,Nv,4] fp16

            vis_feats_list.append(vf)
            vis_mask_list.append(vm)
            vis_gpos_list.append(vg)
            vis_goff_list.append(vo)
            vision_batch_indices.append(batch_index)
            any_vision = True

        # ===== text pad (原逻辑) =====
        B = len(seqs)
        max_len = max(lengths) if lengths else 1
        input_ids = torch.full((B, max_len), self.pad_id, dtype=torch.long)
        tgt_pad = torch.zeros((B, max_len), dtype=torch.long)

        for i, (seq, L, tm) in enumerate(zip(seqs, lengths, tgt_masks)):
            Lc = min(L, seq.size(0), tm.size(0))
            input_ids[i, :Lc] = seq[:Lc]
            tgt_pad[i, :Lc] = tm[:Lc]

        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        for i, L in enumerate(lengths):
            attention_mask[i, :L] = 1

        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = self.ignore_index
        labels[attention_mask == 0] = self.ignore_index

        label_mask = torch.zeros_like(tgt_pad)
        label_mask[:, :-1] = tgt_pad[:, 1:]
        labels[label_mask == 0] = self.ignore_index

        lengths_t = torch.tensor(lengths, dtype=torch.long)


        # ===== vision batch stack (NEW, no padding needed if Nv fixed=1024) =====
        vision_feats = None
        vision_mask  = None
        global_pos   = None
        global_off   = None

        if any_vision:
            vision_feats = torch.stack(vis_feats_list, dim=0)  # [B,5,Nv,D]
            vision_mask  = torch.stack(vis_mask_list,  dim=0).to(torch.bool)      # [B,5,Nv]
            global_pos   = torch.stack(vis_gpos_list,  dim=0).to(torch.long)      # [B,5,Nv]
            global_off   = torch.stack(vis_goff_list,  dim=0).to(torch.float16)   # [B,5,Nv,4]


        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "length": lengths_t,

            "vision_feats": vision_feats,   # [B,5,1024,Dv]
            "vision_mask": vision_mask,     # [B,5,1024]
            "global_pos": global_pos,       # [B,5,1024]
            "global_off": global_off,       # [B,5,1024,4]
            "vision_batch_indices": torch.tensor(vision_batch_indices, dtype=torch.long),
        }



def build_train_and_eval_collators(*, tokenizer, vit_db, ignore_index: int):
    train_collator = SFTCollatorRightPad(
        tokenizer=tokenizer,
        ignore_index=ignore_index,
        vit_db=vit_db,
        vit_strict_read=True,
    )
    # ✅ eval 也 strict：eval 不允许现场 build
    eval_collator = SFTCollatorRightPad(
        tokenizer=tokenizer,
        ignore_index=ignore_index,
        vit_db=vit_db,
        vit_strict_read=True,
    )
    return train_collator, eval_collator

# =====================
# 动态 batch sampler
# =====================
class DynamicBatchSamplerDDP(Sampler):
    """
    按 token 数动态组 batch（单卡时 world_size=1）。
    cost_len = text_len + vision_tokens_per_example
    """

    def __init__(
        self,
        lengths,
        max_tokens_per_batch: int,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 42,
        drop_last: bool = True,
        vision_tokens_per_example: int = 0,
    ):
        self.text_lengths = list(map(int, lengths))
        v = int(vision_tokens_per_example)
        self.cost_lengths = [int(l) + v for l in self.text_lengths]
        self.N = len(self.cost_lengths)
        self.max_tokens = int(max_tokens_per_batch)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        if self.N == 0:
            return
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.N, generator=g).tolist()

        global_batches: List[List[int]] = []
        cur_batch: List[int] = []
        cur_max_len = 0

        for idx in indices:
            L = self.cost_lengths[idx]
            if L <= 0:
                continue
            new_max_len = max(cur_max_len, L)
            est_tokens = (len(cur_batch) + 1) * new_max_len

            if cur_batch and est_tokens > self.max_tokens:
                global_batches.append(cur_batch)
                cur_batch = [idx]
                cur_max_len = L
            else:
                cur_batch.append(idx)
                cur_max_len = new_max_len

        if cur_batch and (not self.drop_last or len(cur_batch) > 0):
            global_batches.append(cur_batch)

        for i, batch in enumerate(global_batches):
            if (i % self.world_size) == self.rank:
                yield batch

    def __len__(self):
        if self.N == 0:
            return 0

        # 用和 __iter__ 完全一致的随机序列与组 batch 规则，精确算 micro-batch 数
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.N, generator=g).tolist()

        count = 0
        cur_batch = []
        cur_max_len = 0

        for idx in indices:
            L = self.cost_lengths[idx]
            if L <= 0:
                continue
            new_max_len = max(cur_max_len, L)
            est_tokens = (len(cur_batch) + 1) * new_max_len

            if cur_batch and est_tokens > self.max_tokens:
                count += 1
                cur_batch = [idx]
                cur_max_len = L
            else:
                cur_batch.append(idx)
                cur_max_len = new_max_len

        if cur_batch:
            count += 1

        # 和 __iter__ 一致：按 i % world_size 分配给 rank
        # 单卡 world_size=1 时就是 count
        return (count + self.world_size - 1 - self.rank) // self.world_size
def build_dynamic_batches_once(
    lengths_list: List[int],
    max_tokens_per_batch: int,
    seed: int,
    vision_tokens_per_example: int,
    epoch_seed_offset: int = 0,
) -> List[List[int]]:
    """
    复刻 DynamicBatchSamplerDDP.__iter__，返回 global_batches（list[list[int]]）。
    注意：你现在 Trainer 并没有调用 sampler.set_epoch，所以 epoch 在实际训练里基本一直是 0；
    这里默认 epoch_seed_offset=0，保证和训练一致。
    """
    cost_lengths = [int(l) + int(vision_tokens_per_example) for l in lengths_list]
    N = len(cost_lengths)
    if N == 0:
        return []

    g = torch.Generator()
    g.manual_seed(int(seed) + int(epoch_seed_offset))
    indices = torch.randperm(N, generator=g).tolist()

    global_batches: List[List[int]] = []
    cur_batch: List[int] = []
    cur_max_len = 0

    for idx in indices:
        L = cost_lengths[idx]
        if L <= 0:
            continue
        new_max_len = max(cur_max_len, L)
        est_tokens = (len(cur_batch) + 1) * new_max_len

        if cur_batch and est_tokens > int(max_tokens_per_batch):
            global_batches.append(cur_batch)
            cur_batch = [idx]
            cur_max_len = L
        else:
            cur_batch.append(idx)
            cur_max_len = new_max_len

    if cur_batch:
        global_batches.append(cur_batch)

    return global_batches
@torch.inference_mode()
def precompute_vit_for_dataset(
    ds,
    vit_db: VisionLMDB,
    *,
    tag: str,
) -> None:
    import threading
    import queue

    n = len(ds)
    if n == 0:
        log(f"[PRECOMPUTE-{tag}] empty dataset -> skip")
        return

    t0 = time.time()
    built = 0
    hit = 0  # skip_get 时这里会一直是 0，但保留字段方便看 log

    vit_db._open()
    vit_db._vip.load()

    WRITE_QMAX = int(os.environ.get("VIP_LMDB_WRITE_QMAX", "128"))
    WRITE_TXN_BATCH = int(os.environ.get("VIP_LMDB_WRITE_TXN_BATCH", "128"))
    META_WORKERS = int(os.environ.get("VIP_META_WORKERS", "8"))

    write_q: "queue.Queue[tuple[bytes, bytes]]" = queue.Queue(maxsize=WRITE_QMAX)
    STOP = object()
    writer_err: dict[str, Any] = {"exc": None}

    def _writer_loop():
        try:
            buf: list[tuple[bytes, bytes]] = []
            while True:
                item = write_q.get()
                if item is STOP:
                    break
                buf.append(item)
                if len(buf) >= WRITE_TXN_BATCH:
                    with vit_db._env.begin(write=True) as txn:
                        for k, rec in buf:
                            txn.put(k, rec, overwrite=True)
                    buf.clear()

            if buf:
                with vit_db._env.begin(write=True) as txn:
                    for k, rec in buf:
                        txn.put(k, rec, overwrite=True)
                buf.clear()
        except Exception as e:
            writer_err["exc"] = e

    writer_th = threading.Thread(target=_writer_loop, daemon=True)
    writer_th.start()

    pending: List[Tuple[Dict[str, Any], str, int, str, bytes]] = []
    pending_keys: set[bytes] = set()

    seen_keys: set[bytes] = set()

    def _img_abs_path(img_name: str) -> str:
        return img_name if os.path.isabs(img_name) else os.path.join(vit_db.image_dir, img_name)

    meta_pool = ThreadPoolExecutor(max_workers=max(1, META_WORKERS))

    def _compute_meta(ex: Dict[str, Any], img_name: str):
        attn_mask_u8, _ = compute_masks_for_image(ex, img_name)
        gpos_u16, goff_f16 = compute_global_pos_for_image(ex, img_name, attn_mask_u8)
        return attn_mask_u8, gpos_u16, goff_f16

    def flush_pending() -> None:
        nonlocal built, pending, pending_keys

        if writer_err["exc"] is not None:
            raise writer_err["exc"]

        if not pending:
            return

        img_paths: list[str] = []
        meta_futs = []
        for (ex, dataset_name, idx_val, img_name, key) in pending:
            img_path = _img_abs_path(img_name)
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"image not found: {img_path}")
            img_paths.append(img_path)
            meta_futs.append(meta_pool.submit(_compute_meta, ex, img_name))

        def _load_one(path: str) -> Image.Image:
            with Image.open(path) as im0:
                im = ImageOps.exif_transpose(im0).convert("RGB")
            return im

        if VIP_IMG_LOAD_WORKERS > 1:
            with ThreadPoolExecutor(max_workers=VIP_IMG_LOAD_WORKERS) as pool:
                pil_imgs = list(pool.map(_load_one, img_paths))
        else:
            pil_imgs = [_load_one(p) for p in img_paths]

        feats_b = vit_db._vip.encode_batch(pil_imgs)  # [B, Nv, D] bf16 cpu
        del pil_imgs

        metas = [f.result() for f in meta_futs]

        for i, (ex, dataset_name, idx_val, img_name, key) in enumerate(pending):
            feat_bf16 = feats_b[i]
            attn_mask_u8, gpos_u16, goff_f16 = metas[i]

            Nv = int(attn_mask_u8.shape[0])
            if feat_bf16.shape[0] != Nv:
                raise ValueError(f"VIT Nv mismatch: feat {feat_bf16.shape} vs mask Nv={Nv}")
            if feat_bf16.shape[1] != int(vit_db.feature_dim):
                raise ValueError(
                    f"VIT D mismatch: feat D={feat_bf16.shape[1]} vs expected {vit_db.feature_dim}"
                )

            rec = encode_record(feat_bf16, attn_mask_u8, gpos_u16, goff_f16)
            write_q.put((key, rec))  # 这里会有背压，避免内存爆

        built += len(pending)
        pending.clear()
        pending_keys.clear()

        if writer_err["exc"] is not None:
            raise writer_err["exc"]

    for i in range(n):
        ex = ds[i]
        dataset_name = ex.get("dataset", "")
        if not dataset_name:
            raise RuntimeError("[PRECOMPUTE][FATAL] ex['dataset'] is empty")
        idx_val = int(ex["index"])

        img_files = ex.get("image_files", None)
        if img_files is None:
            img_files = ex.get("image_file", None)
        if isinstance(img_files, list) and len(img_files) == 0:
            continue
        if not isinstance(img_files, list) or len(img_files) != 5:
            raise RuntimeError(
                f"[PRECOMPUTE][FATAL] expect 0 or 5 image files, got {type(img_files)} len={len(img_files) if isinstance(img_files, list) else 'NA'}"
            )

        for img_name in img_files:
            key = make_vision_key(dataset_name, idx_val, img_name)

            if key in seen_keys:
                continue
            seen_keys.add(key)

            if not VIT_PRECOMPUTE_SKIP_GET:
                with vit_db._env.begin(write=False) as rtxn:
                    buf = rtxn.get(key)
                if buf is not None:
                    hit += 1
                    continue

            if key in pending_keys:
                continue

            pending.append((ex, dataset_name, idx_val, img_name, key))
            pending_keys.add(key)

            if len(pending) >= VIP_BATCH_IMAGES:
                flush_pending()

        if (i + 1) % max(1, VIT_PRECOMPUTE_VERBOSE_EVERY) == 0:
            log(f"[PRECOMPUTE-{tag}] examples {i+1}/{n}, built_imgs={built}, hit={hit}")

    flush_pending()

    write_q.put(STOP)
    writer_th.join()
    if writer_err["exc"] is not None:
        raise writer_err["exc"]

    meta_pool.shutdown(wait=True)

    log(
        f"[PRECOMPUTE-{tag}] done. examples={n}, built_imgs={built}, hit={hit}, time={_fmt_hms(time.time()-t0)}"
    )
@torch.inference_mode()
def precompute_vit_for_step_range(
    ds_train,
    vit_db: VisionLMDB,
    global_batches: List[List[int]],
    start_step: int,
    end_step: int,
) -> None:
    import threading
    import queue

    if end_step <= start_step:
        return
    n_steps = len(global_batches)
    if n_steps == 0:
        raise RuntimeError("[PRECOMPUTE] global_batches is empty")

    t0 = time.time()
    built = 0
    hit = 0

    vit_db._open()
    vit_db._vip.load()

    WRITE_QMAX = int(os.environ.get("VIP_LMDB_WRITE_QMAX", "32"))
    WRITE_TXN_BATCH = int(os.environ.get("VIP_LMDB_WRITE_TXN_BATCH", "32"))
    META_WORKERS = int(os.environ.get("VIP_META_WORKERS", "4"))

    write_q: "queue.Queue[tuple[bytes, bytes]]" = queue.Queue(maxsize=WRITE_QMAX)
    STOP = object()
    writer_err: dict[str, Any] = {"exc": None}

    def _writer_loop():
        try:
            buf: list[tuple[bytes, bytes]] = []
            while True:
                item = write_q.get()
                if item is STOP:
                    break
                buf.append(item)
                if len(buf) >= WRITE_TXN_BATCH:
                    with vit_db._env.begin(write=True) as txn:
                        for k, rec in buf:
                            txn.put(k, rec, overwrite=True)
                    buf.clear()

            if buf:
                with vit_db._env.begin(write=True) as txn:
                    for k, rec in buf:
                        txn.put(k, rec, overwrite=True)
                buf.clear()
        except Exception as e:
            writer_err["exc"] = e

    writer_th = threading.Thread(target=_writer_loop, daemon=True)
    writer_th.start()

    pending: List[Tuple[Dict[str, Any], str, int, str, bytes]] = []
    pending_keys: set[bytes] = set()

    seen_ex: set[int] = set()

    def _img_abs_path(img_name: str) -> str:
        return img_name if os.path.isabs(img_name) else os.path.join(vit_db.image_dir, img_name)

    meta_pool = ThreadPoolExecutor(max_workers=max(1, META_WORKERS))

    def _compute_meta(ex: Dict[str, Any], img_name: str):
        attn_mask_u8, _ = compute_masks_for_image(ex, img_name)
        gpos_u16, goff_f16 = compute_global_pos_for_image(ex, img_name, attn_mask_u8)
        return attn_mask_u8, gpos_u16, goff_f16

    def flush_pending() -> None:
        nonlocal built, pending, pending_keys

        if writer_err["exc"] is not None:
            raise writer_err["exc"]

        if not pending:
            return

        load_items: list[tuple[str, bool]] = []
        meta_futs = []

        for (ex, dataset_name, idx_val, img_name, key) in pending:
            img_path = _img_abs_path(img_name)
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"image not found: {img_path}")

            # 只对 thumb 做退化（train precompute），其它 4 张完全不动
            thumb_fn = _thumb_file_from_ex(ex)
            is_thumb = (img_name == thumb_fn)

            load_items.append((img_path, is_thumb))
            meta_futs.append(meta_pool.submit(_compute_meta, ex, img_name))

        def _load_one(item: tuple[str, bool]) -> Image.Image:
            path, is_thumb = item
            with Image.open(path) as im0:
                im = ImageOps.exif_transpose(im0).convert("RGB")

            if is_thumb:
                im = maybe_degrade_thumb(im)
            return im

        if VIP_IMG_LOAD_WORKERS > 1:
            with ThreadPoolExecutor(max_workers=VIP_IMG_LOAD_WORKERS) as pool:
                pil_imgs = list(pool.map(_load_one, load_items))
        else:
            pil_imgs = [_load_one(it) for it in load_items]


        feats_b = vit_db._vip.encode_batch(pil_imgs)  # [B, Nv, D] bf16 cpu
        del pil_imgs

        metas = [f.result() for f in meta_futs]

        for i, (ex, dataset_name, idx_val, img_name, key) in enumerate(pending):
            feat_bf16 = feats_b[i]
            attn_mask_u8, gpos_u16, goff_f16 = metas[i]

            Nv = int(attn_mask_u8.shape[0])
            if feat_bf16.shape[0] != Nv:
                raise ValueError(f"VIT Nv mismatch: feat {feat_bf16.shape} vs mask Nv={Nv}")
            if feat_bf16.shape[1] != int(vit_db.feature_dim):
                raise ValueError(
                    f"VIT D mismatch: feat D={feat_bf16.shape[1]} vs expected {vit_db.feature_dim}"
                )

            rec = encode_record(feat_bf16, attn_mask_u8, gpos_u16, goff_f16)
            write_q.put((key, rec))

        built += len(pending)
        pending.clear()
        pending_keys.clear()

        if writer_err["exc"] is not None:
            raise writer_err["exc"]

    for step in range(int(start_step), int(end_step)):
        step_in_epoch = step % n_steps
        batch_ids = global_batches[step_in_epoch]

        for ds_idx in batch_ids:
            ds_idx = int(ds_idx)
            if ds_idx in seen_ex:
                continue
            seen_ex.add(ds_idx)

            ex = ds_train[ds_idx]
            dataset_name = ex.get("dataset", "")
            if not dataset_name:
                raise RuntimeError("[PRECOMPUTE][FATAL] ex['dataset'] is empty -> LMDB key will be unstable")
            idx_val = int(ex["index"])

            img_files = ex.get("image_files", None)
            if img_files is None:
                img_files = ex.get("image_file", None)
            if isinstance(img_files, list) and len(img_files) == 0:
                continue
            if not isinstance(img_files, list) or len(img_files) != 5:
                raise RuntimeError(
                    f"[PRECOMPUTE][FATAL] expect 0 or 5 image files, got {type(img_files)} len={len(img_files) if isinstance(img_files, list) else 'NA'}"
                )

            for img_name in img_files:
                key = make_vision_key(dataset_name, idx_val, img_name)

                if not VIT_PRECOMPUTE_SKIP_GET:
                    with vit_db._env.begin(write=False) as rtxn:
                        buf = rtxn.get(key)
                    if buf is not None:
                        hit += 1
                        continue

                if key in pending_keys:
                    continue

                pending.append((ex, dataset_name, idx_val, img_name, key))
                pending_keys.add(key)

                if len(pending) >= VIP_BATCH_IMAGES:
                    flush_pending()

        if (step - start_step + 1) % max(1, VIT_PRECOMPUTE_VERBOSE_EVERY) == 0:
            log(f"[PRECOMPUTE] steps {step+1-start_step}/{end_step-start_step}, built={built}, hit={hit}")

    flush_pending()

    write_q.put(STOP)
    writer_th.join()
    if writer_err["exc"] is not None:
        raise writer_err["exc"]

    meta_pool.shutdown(wait=True)

    log(
        f"[PRECOMPUTE] done range [{start_step},{end_step}), built={built}, hit={hit}, time={_fmt_hms(time.time()-t0)}"
    )

# =====================
# 构建 SFT 数据集
# =====================
def build_sft_datasets():
    ds_short_all = load_from_disk(SFT_CACHE_SHORT)
    if ("image_files" not in ds_short_all.column_names) and ("image_file" in ds_short_all.column_names):
        ds_short_all = ds_short_all.rename_column("image_file", "image_files")

    # （可选）如果 pack 是 input_len 而不是 length
    if ("length" not in ds_short_all.column_names) and ("input_len" in ds_short_all.column_names):
        ds_short_all = ds_short_all.rename_column("input_len", "length")
    must_cols = ["input_ids", "target_mask"]
    missing = [c for c in must_cols if c not in ds_short_all.column_names]
    if missing:
        raise RuntimeError(
            f"[DATA][FATAL] dataset missing columns: {missing}. "
            f"columns={ds_short_all.column_names}"
        )

    def ensure_length_col(ds):
        if "length" in ds.column_names:
            return ds
        return ds.map(
            lambda ex: {"length": len(ex["input_ids"])},
            num_proc=4,
            desc="add length column",
        )

    ds_short_all = ensure_length_col(ds_short_all)

    n_total = len(ds_short_all)
    split_values = set(str(value or "").lower() for value in ds_short_all["split"]) if "split" in ds_short_all.column_names else set()
    if "eval" in split_values and "train" in split_values:
        ds_eval_short = ds_short_all.filter(lambda ex: str(ex["split"]).lower() == "eval")
        ds_short_train = ds_short_all.filter(lambda ex: str(ex["split"]).lower() == "train")
        ds_eval_short = ds_eval_short.shuffle(seed=SEED)
        ds_short_train = ds_short_train.shuffle(seed=SEED)
        if SFT_EXPLICIT_EVAL_MAX > 0 and len(ds_eval_short) > SFT_EXPLICIT_EVAL_MAX:
            ds_eval_short = ds_eval_short.select(range(SFT_EXPLICIT_EVAL_MAX))
            log(
                f"[DATA] capped explicit eval split to {len(ds_eval_short)} rows "
                f"for rolling vision-cache safety"
            )
        if len(ds_eval_short) == 0 or len(ds_short_train) == 0:
            raise RuntimeError("[DATA][FATAL] explicit split produced an empty train or eval dataset")
        log(f"[DATA] using explicit split column: train={len(ds_short_train)}, eval={len(ds_eval_short)}")
    else:
        ds_short_all = ds_short_all.shuffle(seed=SEED)
        n_eval_by_frac = int(n_total * SFT_EVAL_FRACTION)
        n_eval = max(SFT_EVAL_MIN, n_eval_by_frac)
        n_eval = min(n_eval, SFT_EVAL_MAX, n_total - 1)
        ds_eval_short = ds_short_all.select(range(n_eval))
        ds_short_train = ds_short_all.select(range(n_eval, n_total))

    vision_train_rows = len(ds_short_train)
    ds_eval_text = None
    text_train_rows = 0
    if SFT_TEXT_REPLAY_RATIO > 0.0:
        if not SFT_TEXT_REPLAY_CACHE:
            raise RuntimeError(
                "SFT_TEXT_REPLAY_RATIO is positive but SFT_TEXT_REPLAY_CACHE is empty"
            )
        ds_text_all = load_from_disk(SFT_TEXT_REPLAY_CACHE)
        if "length" not in ds_text_all.column_names and "input_len" in ds_text_all.column_names:
            ds_text_all = ds_text_all.rename_column("input_len", "length")
        replay_required = [
            "input_ids", "length", "target_mask", "index", "image_file",
            "dataset", "lang", "has_cot", "split", "vision_meta",
        ]
        replay_missing = [c for c in replay_required if c not in ds_text_all.column_names]
        if replay_missing:
            raise RuntimeError(
                f"[DATA][FATAL] text replay missing columns: {replay_missing}; "
                f"columns={ds_text_all.column_names}"
            )
        if "image_files" not in ds_text_all.column_names:
            ds_text_all = ds_text_all.rename_column("image_file", "image_files")
        bad_image_rows = sum(1 for files in ds_text_all["image_files"] if len(files) != 0)
        if bad_image_rows:
            raise RuntimeError(
                f"[DATA][FATAL] text replay contains {bad_image_rows} rows with images"
            )
        split_text = [str(value or "").lower() for value in ds_text_all["split"]]
        text_train_idx = [i for i, value in enumerate(split_text) if value == "train"]
        text_eval_idx = [i for i, value in enumerate(split_text) if value == "eval"]
        if not text_train_idx or not text_eval_idx:
            raise RuntimeError("[DATA][FATAL] text replay requires non-empty train/eval splits")
        ds_text_train = ds_text_all.select(text_train_idx).shuffle(seed=SEED + 101)
        ds_eval_text = ds_text_all.select(text_eval_idx).shuffle(seed=SEED + 103)
        target_text_rows = int(round(
            vision_train_rows * SFT_TEXT_REPLAY_RATIO / (1.0 - SFT_TEXT_REPLAY_RATIO)
        ))
        # Explicit replay exposures, not falsely reported as unique samples.
        fraction = float(os.environ.get("SFT_TEXT_REPLAY_ZH_FRACTION", "0.6666666667"))
        max_repeats = int(os.environ.get("SFT_TEXT_REPLAY_MAX_REPEATS", "32"))
        langs = list(ds_text_train["lang"])
        chosen = []
        for lang, count in (("zh", round(target_text_rows*fraction)),
                            ("en", target_text_rows-round(target_text_rows*fraction))):
            pool = [i for i, value in enumerate(langs) if value == lang]
            if not pool or count > len(pool)*max_repeats:
                raise RuntimeError(f"text replay {lang}: {count} exposures exceeds {len(pool)} unique rows x {max_repeats}")
            rng = random.Random(SEED + (211 if lang == "zh" else 223))
            cycle = []
            while len(cycle) < count:
                current = pool.copy(); rng.shuffle(current); cycle.extend(current)
            chosen.extend(cycle[:count])
            log(f"[REPLAY] {lang}: unique={len(pool)}, exposures={count}, mean_reuse={count/len(pool):.2f}")
        ds_text_train = ds_text_train.select(chosen).shuffle(seed=SEED+227)
        if SFT_TEXT_REPLAY_EVAL_MAX > 0 and len(ds_eval_text) > SFT_TEXT_REPLAY_EVAL_MAX:
            ds_eval_text = ds_eval_text.select(range(SFT_TEXT_REPLAY_EVAL_MAX))
        if ds_text_train.features != ds_short_train.features:
            raise RuntimeError(
                "[DATA][FATAL] text replay schema differs from VLM schema: "
                f"text={ds_text_train.features}, vision={ds_short_train.features}"
            )
        text_train_rows = len(ds_text_train)
        ds_short_train = concatenate_datasets([ds_short_train, ds_text_train]).shuffle(seed=SEED + 107)

    if is_main_process():
        log(
            f"[DATA] short_total={n_total}, eval_short={len(ds_eval_short)}, "
            f"vision_train={vision_train_rows}, text_replay_train={text_train_rows}, "
            f"combined_train={len(ds_short_train)}, text_replay_ratio="
            f"{(text_train_rows / max(1, len(ds_short_train))):.4f}"
        )

    needed_cols = (
        "input_ids",
        "length",
        "isMath",
        "target_mask",
        "vision_feats",
        "vision_mask",
    )
    needed_must = ["input_ids", "length", "target_mask", "index", "image_files", "dataset", "vision_meta"]

    for c in needed_must:
        if c not in ds_short_train.column_names:
            raise RuntimeError(f"[DATA][FATAL] train missing required col: {c}")
        if c not in ds_eval_short.column_names:
            raise RuntimeError(f"[DATA][FATAL] eval missing required col: {c}")

    torch_cols_train = ["input_ids", "length", "target_mask", "index"]

    ds_short_train = ds_short_train.with_format(
        "torch",
        columns=torch_cols_train,
        output_all_columns=True,  # ✅ 保留 image_file/dataset 等非tensor列
    )

    ds_eval_short = ds_eval_short.with_format(
        "torch",
        columns=torch_cols_train,  # eval 同样逻辑
        output_all_columns=True,
    )

    if ds_eval_text is not None:
        ds_eval_text = ds_eval_text.with_format(
            "torch",
            columns=torch_cols_train,
            output_all_columns=True,
        )

    return ds_short_train, ds_eval_short, ds_eval_text
class GradFreezeDebugCallback(TrainerCallback):
    """
    只打印你关心的模块（embedding / vision / qformer / lora adapters）
    并且最多打印少量条目，方便你把 log 粘给我一眼看出来问题。
    """
    def __init__(
        self,
        enabled: bool = True,
        max_show_trainables: int = 40,
        start_step: int = 0,
        max_show_bad: int = 30,
        name_filters: tuple[str, ...] = (
            "tok_embed",          # 文本 embedding
            "lm_bias",
            "vision_",            # 视觉侧所有
            "query_embed",        # Q-former queries
            "qformer_",           # Q-former blocks/norm
            ".adapters.",         # LoRA
        ),
    ):
        self.enabled = bool(enabled)
        self.max_show_trainables = int(max_show_trainables)
        self.max_show_bad = int(max_show_bad)
        self.name_filters = tuple(name_filters)
        self.start_step = int(start_step)
        self._printed_trainables = False
        self._printed_after_micro_backward = False
        self._printed_before_opt_step = False

    def _unwrap(self, model):
        return model.module if hasattr(model, "module") else model

    def _selected(self, name: str) -> bool:
        return any(s in name for s in self.name_filters)

    def _bucket(self, name: str) -> str:
        # 你关心的几个大类，做聚合统计用
        if name.startswith("tok_embed"):
            return "text_tok_embed"
        if "vision_view_embed" in name:
            return "vision_view_embed"
        if "vision_pos_embed" in name:
            return "vision_pos_embed"
        if "vision_type_embed" in name:
            return "vision_type_embed"
        if "vision_gate_" in name:
            return "vision_gates"
        if "vision_film_mlp" in name:
            return "vision_film_mlp"
        if "vision_projector" in name:
            return "vision_projector"
        if "vision_kv_sep" in name:
            return "vision_kv_sep"
        if "query_embed" in name:
            return "qformer_query_embed"
        if "qformer_blocks" in name or "qformer_final_norm" in name:
            return "qformer_blocks"
        if ".adapters." in name:
            return "lora_adapters"
        if "vision_" in name:
            return "vision_other"
        return "other"

    def _iter_selected_params(self, model):
        for n, p in model.named_parameters():
            if self._selected(n):
                yield n, p

    def _print_trainables_once(self, model):
        if self._printed_trainables or (not self.enabled):
            return
        self._printed_trainables = True

        model = self._unwrap(model)

        total_numel = 0
        train_numel = 0
        sel = []
        for n, p in self._iter_selected_params(model):
            total_numel += p.numel()
            if p.requires_grad:
                train_numel += p.numel()
            sel.append((n, p))

        print("\n[DBG][trainables] ===== selected requires_grad snapshot =====", flush=True)
        print(f"[DBG][trainables] selected_tensors={len(sel)} selected_params={total_numel/1e6:.2f}M "
              f"trainable_params={train_numel/1e6:.2f}M", flush=True)

        # 列举少量名字（T/F）
        shown = 0
        for n, p in sel:
            if shown >= self.max_show_trainables:
                break
            flag = "T" if p.requires_grad else "F"
            print(f"[DBG][trainables] [{flag}] {n} shape={tuple(p.shape)}", flush=True)
            shown += 1
        if len(sel) > shown:
            print(f"[DBG][trainables] ... truncated, total_selected={len(sel)}", flush=True)

    @torch.no_grad()
    def _summarize_grads(self, model, tag: str, state=None):
        if not self.enabled:
            return
        model = self._unwrap(model)

        # bucket -> stats
        stats = {}
        bad_none = []
        bad_zero = []

        def _ensure(b):
            if b not in stats:
                stats[b] = {
                    "tensors": 0,
                    "numel": 0,
                    "grad_none": 0,
                    "grad_zero": 0,
                    "grad_nonzero": 0,
                    "absmax_max": 0.0,
                }

        for n, p in self._iter_selected_params(model):
            if not p.requires_grad:
                continue

            b = self._bucket(n)
            _ensure(b)
            st = stats[b]
            st["tensors"] += 1
            st["numel"] += p.numel()

            g = p.grad
            if g is None:
                st["grad_none"] += 1
                if len(bad_none) < self.max_show_bad:
                    bad_none.append(n)
                continue

            # 用 absmax 判定是否为“全 0 梯度”
            # 这是一次性 debug：宁可准一点，少做花活
            absmax = float(g.detach().abs().max().item()) if g.numel() > 0 else 0.0
            st["absmax_max"] = max(st["absmax_max"], absmax)

            if absmax == 0.0:
                st["grad_zero"] += 1
                if len(bad_zero) < self.max_show_bad:
                    bad_zero.append(n)
            else:
                st["grad_nonzero"] += 1

        step = int(getattr(state, "global_step", -1)) if state is not None else -1
        print(f"\n[DBG][grads:{tag}] ===== grad snapshot (global_step={step}) =====", flush=True)

        if not stats:
            print("[DBG][grads] (no selected trainable params?)", flush=True)
            return

        # 稳定排序输出
        for b in sorted(stats.keys()):
            st = stats[b]
            print(
                f"[DBG][grads:{tag}] {b:18s} "
                f"tensors={st['tensors']:3d} "
                f"numel={st['numel']/1e6:7.2f}M "
                f"none={st['grad_none']:3d} "
                f"zero={st['grad_zero']:3d} "
                f"nonzero={st['grad_nonzero']:3d} "
                f"absmax_max={st['absmax_max']:.3e}",
                flush=True
            )

        if bad_none:
            print(f"[DBG][grads:{tag}] grad=None (first {len(bad_none)}):", flush=True)
            for n in bad_none:
                print(f"  - {n}", flush=True)

        if bad_zero:
            print(f"[DBG][grads:{tag}] grad==0 (first {len(bad_zero)}):", flush=True)
            for n in bad_zero:
                print(f"  - {n}", flush=True)

    # ---------- callbacks ----------
    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is not None:
            self._print_trainables_once(model)
        return control

    # 1) 尽量在“micro-step backward 完成后”抓一次（你想看的‘第一次回传’）
    def on_substep_end(self, args, state, control, **kwargs):
        step = int(getattr(state, "global_step", -1)) if state is not None else -1
        if step < self.start_step:
            return control
        if (not self.enabled) or self._printed_after_micro_backward:
            return control
        model = kwargs.get("model", None)
        if model is None:
            return control
        # 第一次 substep_end 就抓
        self._printed_after_micro_backward = True
        self._summarize_grads(model, tag="after_micro_backward", state=state)
        return control

    # 2) 再在“第一次 optimizer step 前”抓一次（累计完 grad_accum 后）
    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        step = int(getattr(state, "global_step", -1)) if state is not None else -1
        if step < self.start_step:
            return control
        if (not self.enabled) or self._printed_before_opt_step:
            return control
        model = kwargs.get("model", None)
        if model is None:
            return control
        self._printed_before_opt_step = True
        self._summarize_grads(model, tag="before_optimizer_step", state=state)
        return control

class NudgeZeroTrainablesOnceCallback(TrainerCallback):
    """
    只在训练真正从 global_step==0 开始时执行一次：
    - 点醒 gate_view / gate_pos（避免乘法 gate 把 embedding 梯度掐死）
    - 可选：对极少数“requires_grad=True 且全 0”的张量加一点点噪声
    """
    def __init__(
        self,
        eps_gate: float = 1e-3,
        eps_weight: float = 1e-6,
        tol: float = 0.0,
        enable_weights: bool = False,
    ):
        self.eps_gate = float(eps_gate)
        self.eps_weight = float(eps_weight)
        self.tol = float(tol)
        self.enable_weights = bool(enable_weights)
        self._done = False

        # 只动你“视觉侧”这些，坚决别碰 LoRA（很多 LoRA 的 B 本来就 0，是故意的）
        self._allow_substr = (
            "vision_gate_view",
            "vision_gate_pos",
            "vision_gate_film",  # ✅ 加上它
            "vision_kv_sep",
            "vision_film_mlp.2.weight",
            "vision_film_mlp.2.bias",
        )
        self._deny_substr = (".adapters.",)

    def on_train_begin(self, args, state, control, **kwargs):
        if self._done:
            return control

        # ✅ 关键：只在“真正从 0 step 开始训练”时执行
        if int(getattr(state, "global_step", 0)) != 0:
            self._done = True
            return control

        model = kwargs.get("model")
        if model is None:
            self._done = True
            return control
        model = model.module if hasattr(model, "module") else model

        changed = []

        with torch.no_grad():
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if any(d in name for d in self._deny_substr):
                    continue
                if not any(a in name for a in self._allow_substr):
                    continue
                if (not p.is_floating_point()):
                    continue

                absmax = float(p.detach().abs().max().item()) if p.numel() > 0 else 0.0
                if absmax > self.tol:
                    continue

                # 1) gate：直接加一个极小常数（更稳定、可复现）
                if "vision_gate_" in name:
                    p.add_(self.eps_gate)
                    changed.append((name, "gate+const"))
                    continue
                if name.endswith("vision_film_mlp.2.bias"):
                    p.add_(self.eps_weight)  # 用一个极小常数，不用噪声更稳定
                    changed.append((name, "film_last_bias+const"))
                    continue
                # 2) 其它“全 0 tensor”：默认不动，除非你显式开启 enable_weights
                if self.enable_weights:
                    noise = torch.randn_like(p) * self.eps_weight
                    p.add_(noise)
                    changed.append((name, "weight+noise"))

        if changed:
            print(f"[nudge-zero-once] changed={len(changed)} (show first 20): {changed[:20]}", flush=True)
        else:
            print("[nudge-zero-once] nothing to change.", flush=True)

        self._done = True
        return control

# =====================
# TinyLLMTrainer
# =====================
class TinyLLMTrainer(Trainer):
    """
    - 用 DynamicBatchSamplerDDP 做“按 token 数”动态 batch。
    - compute_loss 里把  vision_feats 等传给 forward。
    """
    def __init__(self, *args, eval_data_collator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._eval_data_collator = eval_data_collator if eval_data_collator is not None else self.data_collator
    def create_optimizer(self):
        # stage1：保持默认（不动）
        if SFT_STAGE != 2:
            return super().create_optimizer()

        # stage2：语言 LoRA、Q-former、视觉桥接分别使用独立学习率。
        if self.optimizer is not None:
            return self.optimizer

        model = self.model
        base_lr = float(self.args.learning_rate)
        qformer_lr = base_lr * float(STAGE2_QFORMER_LR_RATIO)
        bridge_lr = base_lr * float(STAGE2_BRIDGE_LR_RATIO)

        # HF 默认：对 LN / bias 不做 weight decay
        decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [n for n in decay_parameters if "bias" not in n]

        def is_qformer_param(name: str) -> bool:
            # 精准只压 Q-former：你 stage2 patterns 里就是这几个核心 key
            return (
                ("query_embed" in name)
                or ("qformer_blocks" in name)
                or ("qformer_final_norm" in name)
            )

        def is_bridge_param(name: str) -> bool:
            return any(
                key in name
                for key in (
                    "vision_projector",
                    "vision_type_embed",
                    "vision_view_embed",
                    "vision_pos_embed",
                    "vision_film_mlp",
                    "vision_gate",
                    "vision_kv_sep",
                )
            )

        def is_lora_param(name: str) -> bool:
            return ".adapters." in name

        base_decay, base_nodecay = [], []
        q_decay, q_nodecay = [], []
        bridge_decay, bridge_nodecay = [], []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            use_decay = (name in decay_parameters)
            if is_qformer_param(name):
                (q_decay if use_decay else q_nodecay).append(p)
            elif is_bridge_param(name):
                (bridge_decay if use_decay else bridge_nodecay).append(p)
            elif is_lora_param(name):
                (base_decay if use_decay else base_nodecay).append(p)
            else:
                raise RuntimeError(
                    f"Unexpected trainable stage2 parameter outside LoRA/Q-former/bridge groups: {name}"
                )

        param_groups = []
        if base_decay:
            param_groups.append({"params": base_decay, "weight_decay": self.args.weight_decay, "lr": base_lr})
        if base_nodecay:
            param_groups.append({"params": base_nodecay, "weight_decay": 0.0, "lr": base_lr})
        if q_decay:
            param_groups.append({"params": q_decay, "weight_decay": self.args.weight_decay, "lr": qformer_lr})
        if q_nodecay:
            param_groups.append({"params": q_nodecay, "weight_decay": 0.0, "lr": qformer_lr})
        if bridge_decay:
            param_groups.append({"params": bridge_decay, "weight_decay": self.args.weight_decay, "lr": bridge_lr})
        if bridge_nodecay:
            param_groups.append({"params": bridge_nodecay, "weight_decay": 0.0, "lr": bridge_lr})

        self.optimizer = AdamW(
            param_groups,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )

        if is_main_process():
            log(f"[OPT] stage2: lora_lr={base_lr:.3e}, qformer_lr={qformer_lr:.3e} "
                f"(ratio={STAGE2_QFORMER_LR_RATIO}), bridge_lr={bridge_lr:.3e} "
                f"(ratio={STAGE2_BRIDGE_LR_RATIO})")
            log(f"[OPT] params: base_decay={len(base_decay)}, base_nodecay={len(base_nodecay)}, "
                f"q_decay={len(q_decay)}, q_nodecay={len(q_nodecay)}, "
                f"bridge_decay={len(bridge_decay)}, bridge_nodecay={len(bridge_nodecay)}")

        return self.optimizer
    def get_train_dataloader(self):
        ds = self.train_dataset
        lengths = ds["length"]
        if torch.is_tensor(lengths):
            lengths_list = [int(x) for x in lengths.tolist()]
        else:
            lengths_list = [int(x) for x in lengths]

        sampler = DynamicBatchSamplerDDP(
            lengths=lengths_list,
            max_tokens_per_batch=MAX_TOKENS_PER_BATCH_SHORT,
            world_size=1,   # 单卡
            rank=0,
            seed=SEED,
            drop_last=True,
            vision_tokens_per_example=int( NUM_VISION_VIEWS * VIT_TOKENS_PER_IMAGE*0.2 +NUM_VISION_VIEWS - 1),
        )

        dl_kwargs = dict(
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=SFT_NUM_WORKERS,
            pin_memory=True,
            persistent_workers=False
        )
        if SFT_NUM_WORKERS > 0:
            dl_kwargs["prefetch_factor"] = SFT_PREFETCH
            dl_kwargs["multiprocessing_context"] = SFT_MP_CTX  # "forkserver" / "spawn"
        return DataLoader(ds, **dl_kwargs)

    def get_eval_dataloader(self, eval_dataset=None):
        ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        lengths = ds["length"]
        lengths_list = [int(x) for x in (lengths.tolist() if torch.is_tensor(lengths) else lengths)]

        sampler = DynamicBatchSamplerDDP(
            lengths=lengths_list,
            max_tokens_per_batch=MAX_TOKENS_PER_BATCH_SHORT,
            world_size=1,
            rank=0,
            seed=SEED,
            drop_last=False,
            vision_tokens_per_example=int(NUM_VISION_VIEWS * VIT_TOKENS_PER_IMAGE * 0.2 + NUM_VISION_VIEWS - 1),
        )

        num_workers = SFT_EVAL_NUM_WORKERS
        dl_kwargs = dict(
            batch_sampler=sampler,
            collate_fn=self._eval_data_collator,  # 顺便把 eval_collator 真用上
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False

        )
        if num_workers > 0:
            dl_kwargs["prefetch_factor"] = SFT_PREFETCH
            dl_kwargs["multiprocessing_context"] = SFT_MP_CTX
        return DataLoader(ds, **dl_kwargs)
    def compute_loss(self, model, inputs, return_outputs=False,**kwargs):
        """
        Run text-only replay without a fake visual prefix.

        A dynamic batch may contain both visual and text-only rows. Visual
        tensors are packed only for the visual rows, so split the language
        tensors into two forwards and combine the two scalar losses by row
        count. This keeps the configured replay ratio meaningful under the
        existing sample-mean loss while ensuring text replay updates only the
        language LoRA (Q-Former/bridge are absent from that computation graph).
        """
        labels = inputs.pop("labels")
        inputs.pop("length", None)  # 只给 sampler 用
        vision_batch_indices = inputs.pop("vision_batch_indices", None)
        vision_keys = ("vision_feats", "vision_mask", "global_pos", "global_off")
        vision_inputs = {key: inputs.pop(key, None) for key in vision_keys}

        batch_size = int(labels.shape[0])
        if vision_batch_indices is None:
            # Backward compatibility for old collators: presence of visual
            # features means every row is visual.
            n_visual = batch_size if vision_inputs["vision_feats"] is not None else 0
            vision_batch_indices = torch.arange(n_visual, device=labels.device)
        else:
            vision_batch_indices = vision_batch_indices.to(device=labels.device, dtype=torch.long)
        n_visual = int(vision_batch_indices.numel())

        def _loss_from(outputs):
            return outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if n_visual == 0:
            outputs = model(**inputs, labels=labels)
            loss = _loss_from(outputs)
            return (loss, outputs) if return_outputs else loss

        if n_visual == batch_size:
            outputs = model(**inputs, **vision_inputs, labels=labels)
            loss = _loss_from(outputs)
            return (loss, outputs) if return_outputs else loss

        if vision_inputs["vision_feats"] is None:
            raise RuntimeError("[LOSS][FATAL] mixed batch has visual rows but no vision_feats")
        if int(vision_inputs["vision_feats"].shape[0]) != n_visual:
            raise RuntimeError(
                "[LOSS][FATAL] packed visual tensor count does not match vision_batch_indices: "
                f"{vision_inputs['vision_feats'].shape[0]} vs {n_visual}"
            )

        all_indices = torch.arange(batch_size, device=labels.device)
        text_mask = torch.ones(batch_size, dtype=torch.bool, device=labels.device)
        text_mask[vision_batch_indices] = False
        text_batch_indices = all_indices[text_mask]

        # Model input tensors all have batch as dimension zero here.
        visual_language_inputs = {
            key: value.index_select(0, vision_batch_indices)
            for key, value in inputs.items()
        }
        text_language_inputs = {
            key: value.index_select(0, text_batch_indices)
            for key, value in inputs.items()
        }
        visual_labels = labels.index_select(0, vision_batch_indices)
        text_labels = labels.index_select(0, text_batch_indices)

        visual_outputs = model(
            **visual_language_inputs,
            **vision_inputs,
            labels=visual_labels,
        )
        text_outputs = model(**text_language_inputs, labels=text_labels)
        visual_loss = _loss_from(visual_outputs)
        text_loss = _loss_from(text_outputs)
        n_text = batch_size - n_visual
        loss = (visual_loss * n_visual + text_loss * n_text) / batch_size

        if return_outputs:
            # Evaluation datasets are modality-pure, but keep a useful result
            # shape if a mixed diagnostic dataset is passed explicitly.
            return loss, {
                "loss": loss,
                "visual_loss": visual_loss.detach(),
                "text_loss": text_loss.detach(),
            }
        return loss
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        optimizer = optimizer or self.optimizer

        # Respect either an explicit warmup_steps value or TrainingArguments'
        # warmup_ratio. The old code read warmup_steps directly, so a runner
        # using warmup_ratio with warmup_steps=0 silently got no warm-up.
        warmup_steps = int(self.args.get_warmup_steps(num_training_steps))
        min_ratio = float(os.environ.get("SFT_MIN_LR_RATIO", str(SFT_MIN_LR_RATIO)))
        if is_main_process():
            log(
                f"[SCHED] total_steps={num_training_steps}, warmup_steps={warmup_steps}, "
                f"min_lr_ratio={min_ratio:.3f}"
            )

        def lr_lambda(step: int) -> float:
            # warmup: 0 -> 1
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))

            # cosine decay: 1 -> min_ratio
            denom = max(1, num_training_steps - warmup_steps)
            progress = float(step - warmup_steps) / float(denom)
            progress = min(1.0, max(0.0, progress))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        self.lr_scheduler = LambdaLR(optimizer, lr_lambda)
        return self.lr_scheduler
class StopAtGlobalStepCallback(TrainerCallback):
    def __init__(self):
        self.target_step = None
        self.evaluate_on_stop = False

    def on_train_begin(self, args, state, control, **kwargs):
        # ✅ 每次 trainer.train() 重新开始时，清掉上次残留的 stop
        control.should_training_stop = False
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.target_step is None:
            return control

        step = int(state.global_step)
        if step >= int(self.target_step):
            # Chunk boundary must be resumable. Evaluation is forced only for
            # the final chunk; regular eval_steps boundaries remain unchanged.
            control.should_save = True
            control.should_evaluate = control.should_evaluate or self.evaluate_on_stop
            control.should_training_stop = True
        return control


class UnloadVIPAfterEvalSaveCallback(TrainerCallback):
    def __init__(self, vit_db: VisionLMDB) -> None:
        self.vit_db = vit_db

    def on_evaluate(self, args, state, control, **kwargs):
        # eval 结束后释放 VIP，避免训练阶段常驻
        self.vit_db.unload_vip()
        return control

    def on_save(self, args, state, control, **kwargs):
        # 有的 save 也会触发 eval 或占显存，顺手再清一次
        self.vit_db.unload_vip()
        return control

# =====================
# main
# =====================
def main():
    enforce_single_gpu_only()
    clear_on_start = int(os.environ.get("CLEAR_VIT_CACHE_ON_START", "0")) == 1
    if mp.current_process().name == "MainProcess" and clear_on_start:
        if os.path.isdir(VIT_LMDB_ROOT):
            shutil.rmtree(VIT_LMDB_ROOT, ignore_errors=True)
    os.makedirs(VIT_LMDB_ROOT, exist_ok=True)

    global vit_db, VIT_TOKENS_PER_IMAGE, VISION_FEATURE_DIM
    vit_db = VisionLMDB(
        root_dir=VIT_LMDB_ROOT,
        vit_dir=VIT_DIR,
        image_dir=IMAGE_DIR,
        device=torch.device("cuda", 0),
        map_size_gb=int(os.environ.get("VIT_LMDB_MAP_SIZE_GB", "1200")),
    )
    VIT_TOKENS_PER_IMAGE = vit_db.tokens_per_image
    VISION_FEATURE_DIM = vit_db.feature_dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)

    assert os.path.isdir(LOCAL_SNAPSHOT)
    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_SNAPSHOT,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ===== build model cfg =====
    vocab_size = len(tokenizer.get_vocab())
    cfg = Config(
        vocab_size=vocab_size,
        train_maxlength=8192,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        moe_use_detach=False,
        use_moe=False,
        dropout=0.0,
        learnable_temp=True,
        drop_path=0.0,
        residual_dropout=0.0,
        rope_type="yarn",
        use_ssm=False,
        ssm_layers=[],
        use_vision=True,
        vision_feature_dim=VISION_FEATURE_DIM,
        vision_dropout=0.0,
        vision_use_rmsnorm=True,
        qformer_num_layers=4,
        qformer_mlp_ratio=2.0,
        loss_reduction=SFT_LOSS_REDUCTION,
        sample_mean_alpha=SFT_SAMPLE_MEAN_ALPHA,
    )
    cfg.use_checkpoint = False
    cfg.checkpoint_use_reentrant = False

    model = TinyLLM(cfg)

    # ===== stage2 attach LoRA once (if stage2) =====


    # ===== load base + optional lora =====
    load_base_only(model, RESUME_ROOT, stage=SFT_STAGE, map_location="cpu")

    if SFT_STAGE == 2:
        attach_lora_once_for_stage2(model, SFT_STAGE)  # ✅ 先注射 adapter（改变 key）
        load_lora_only_after_attach(model, RESUME_ROOT, LORA_NAME, map_location="cpu")  # ✅ 再 load lora

    configure_train_stage(model, SFT_STAGE)


    model.to(device)

    ds_short, ds_eval_short, ds_eval_text = build_sft_datasets()
    if SFT_STAGE == 1:
        max_steps = SFT_STAGE1_STEPS
        save_steps = min(SAVE_INTERVAL_STEPS, SFT_STAGE1_STEPS)  # 确保 <=2000
        eval_steps = save_steps
    else:
        max_steps = -1
        save_steps = SAVE_INTERVAL_STEPS
        eval_steps = save_steps
    if MAX_STEPS_OVERRIDE > 0:
        max_steps = MAX_STEPS_OVERRIDE

    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=EPOCHS,
        learning_rate=LR_BASE,
        max_steps=max_steps,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=os.environ.get("SFT_LR_SCHEDULER", "cosine"),

        per_device_train_batch_size=1,  # 动态 batch，设 1 即可
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,

        max_grad_norm=MAX_GRAD_NORM,
        bf16=USE_BF16,
        tf32=True,

        logging_steps=LOG_SCALAR_EVERY,
        save_steps= save_steps ,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_total_limit=2,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Chunked training calls trainer.train() repeatedly. Loading the best
        # weights after every chunk would disturb continuation, so track and
        # preserve the best checkpoint without auto-loading it mid-run.
        load_best_model_at_end=False,

        dataloader_num_workers=0,
        report_to=["tensorboard"],
        logging_dir=LOG_DIR,
        remove_unused_columns=False,  # 保留 vision_feats 等字段
    )


    train_collator, eval_collator = build_train_and_eval_collators(
        tokenizer=tokenizer,
        vit_db=vit_db,

        ignore_index=IGNORE_INDEX,
    )

    trainer = TinyLLMTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_short,  # ✅ 不是 ds_train
        eval_dataset=ds_eval_short,
        data_collator=train_collator,
        eval_data_collator=eval_collator,
        callbacks=[
            NudgeZeroTrainablesOnceCallback(
                eps_gate=float(os.environ.get("SFT_NUDGE_GATE_EPS", "3e-2")),
                eps_weight=float(os.environ.get("SFT_NUDGE_WEIGHT_EPS", "1e-6")),
                tol=float(os.environ.get("SFT_NUDGE_TOL", "0.0")),
                enable_weights=int(os.environ.get("SFT_NUDGE_ENABLE_WEIGHTS", "0")) == 1,
            ),
            GradFreezeDebugCallback(
                enabled=int(os.environ.get("SFT_DEBUG_GRAD", "1")) == 1,  # 需要就=1，不要就=0
                max_show_trainables=40,
                max_show_bad=30,
                 start_step=10
            ),
            SafeTensorsCallback(),
            UnloadVIPAfterEvalSaveCallback(vit_db),
        ],
    )
    stop_cb = StopAtGlobalStepCallback()
    trainer.add_callback(stop_cb)

    if is_main_process():
        log(
            f"[SFT] start HF Trainer on 1 GPU, "
            f"train_samples={len(ds_short)}, eval_samples={len(ds_eval_short)}"
        )
        log(
            f"[LOSS] reduction={SFT_LOSS_REDUCTION}, "
            f"sample_mean_alpha={SFT_SAMPLE_MEAN_ALPHA:.2f}"
        )

    lengths = ds_short["length"]
    lengths_list = [int(x) for x in (lengths.tolist() if torch.is_tensor(lengths) else lengths)]

    vision_tokens_per_example = int(NUM_VISION_VIEWS * VIT_TOKENS_PER_IMAGE * 0.2 + NUM_VISION_VIEWS - 1)
    global_batches = build_dynamic_batches_once(
        lengths_list=lengths_list,
        max_tokens_per_batch=MAX_TOKENS_PER_BATCH_SHORT,
        seed=SEED,
        vision_tokens_per_example=vision_tokens_per_example,
        epoch_seed_offset=0,   # ✅ 和你当前训练一致（trainer 没 set_epoch）
    )
    num_micro_steps_per_epoch = len(global_batches)
    num_opt_steps_per_epoch = math.ceil(num_micro_steps_per_epoch / GRAD_ACCUM_STEPS)

    computed_total_steps = num_opt_steps_per_epoch * EPOCHS
    if training_args.max_steps is not None and int(training_args.max_steps) > 0:
        total_steps = int(training_args.max_steps)
    else:
        total_steps = int(computed_total_steps)
    log(f"[CHUNK] micro_steps/epoch={num_micro_steps_per_epoch}, opt_steps/epoch={num_opt_steps_per_epoch}, epochs={EPOCHS}, total_steps={total_steps}, chunk={VIT_CHUNK_STEPS}")

    # ===== 2) 自动 chunk-run：预计算 -> 卸载VIP -> 训练到 target_step -> (可选)清LMDB =====
    last_ckpt = None
    start_step = 0
    baseline_eval_done = False

    while start_step < total_steps:
        # End a cache chunk at an evaluation boundary when necessary.
        next_eval_step = (start_step // max(1, int(eval_steps)) + 1) * max(1, int(eval_steps))
        target_step = min(total_steps, start_step + VIT_CHUNK_STEPS, next_eval_step)
        log(f"[CHUNK] start_step={start_step} -> target_step={target_step}")

        # (A) 可选：每个 chunk 开始前清空 LMDB，保证磁盘不爆
        if VIT_CLEAR_BETWEEN_CHUNKS and not ((not clear_on_start) and start_step == 0):
            vit_db.rotate()

        micro_start = start_step * GRAD_ACCUM_STEPS
        micro_end = target_step * GRAD_ACCUM_STEPS
        # (B) 离线预计算：把接下来 chunk 用到的特征全写进 LMDB
        eval_period = max(1, int(eval_steps))
        needs_eval_cache = (
            start_step == 0
            or
            target_step >= total_steps
            or (start_step // eval_period) != (target_step // eval_period)
        )
        if needs_eval_cache:
            precompute_vit_for_dataset(
                ds_eval_short,
                vit_db,
                tag="EVAL",
            )
        else:
            log("[PRECOMPUTE-EVAL] skipped: this cache chunk has no eval boundary")
        precompute_vit_for_step_range(
            ds_train=ds_short,
            vit_db=vit_db,
            global_batches=global_batches,
            start_step=micro_start,
            end_step=micro_end,
        )

        # (C) 预计算完卸载 VIP，训练阶段不需要它
        vit_db.unload_vip()
        vit_db.close()
        if start_step == 0 and not baseline_eval_done:
            baseline_metrics = trainer.evaluate(metric_key_prefix="baseline")
            trainer.log_metrics("baseline", baseline_metrics)
            trainer.save_metrics("baseline", baseline_metrics)
            if ds_eval_text is not None:
                trainer._eval_dataloader = None
                baseline_text_metrics = trainer.evaluate(
                    eval_dataset=ds_eval_text,
                    metric_key_prefix="baseline_text",
                )
                trainer.log_metrics("baseline_text", baseline_text_metrics)
                trainer.save_metrics("baseline_text", baseline_text_metrics)
                trainer._eval_dataloader = None
                log(
                    "[BASELINE-TEXT] step=0 "
                    + json.dumps(baseline_text_metrics, ensure_ascii=False, sort_keys=True)
                )
            baseline_eval_done = True
            log(
                "[BASELINE] step=0 "
                + json.dumps(baseline_metrics, ensure_ascii=False, sort_keys=True)
            )
        # (D) 训练：只读 LMDB，缺就 KeyError 直接炸；并且让 Trainer 只跑到 target_step
        stop_cb.target_step = int(target_step)
        stop_cb.evaluate_on_stop = target_step >= total_steps

        train_kwargs = {}
        if last_ckpt is not None:
            train_kwargs["resume_from_checkpoint"] = last_ckpt
        os.environ["VIT_IN_TRAIN_LOOP"] = "1"
        trainer.train(**train_kwargs)
        trainer._train_dataloader = None
        trainer._eval_dataloader = None
        gc.collect()
        os.environ["VIT_IN_TRAIN_LOOP"] = "0"
        # (E) 更新位置
        start_step = int(trainer.state.global_step)

        # The Trainer's scheduled eval tracks visual validation loss. Run the
        # small text-only holdout at the same boundaries so language retention
        # is visible and a visually improved but linguistically regressed
        # checkpoint is not selected silently.
        reached_eval_boundary = (
            start_step >= total_steps
            or start_step % eval_period == 0
        )
        if ds_eval_text is not None and reached_eval_boundary:
            trainer._eval_dataloader = None
            text_metrics = trainer.evaluate(
                eval_dataset=ds_eval_text,
                metric_key_prefix="text_replay",
            )
            trainer.log_metrics("text_replay", text_metrics)
            trainer.save_metrics(f"text_replay_step_{start_step}", text_metrics)
            trainer._eval_dataloader = None
            log(
                f"[TEXT-EVAL] step={start_step} "
                + json.dumps(text_metrics, ensure_ascii=False, sort_keys=True)
            )

        # (F) 记录本 chunk 的 checkpoint 路径（save_steps=chunk 保证 checkpoint-{target_step} 存在；最后一段可能不足 chunk，手动兜底）
        cand = os.path.join(OUT_DIR, f"checkpoint-{start_step}")
        if os.path.isdir(cand):
            last_ckpt = cand
        else:
            log(f"[CHUNK] checkpoint-{start_step} not found, saving manual checkpoint to {cand}")
            # A chunk continuation needs optimizer, scheduler and RNG state,
            # not just weights. Use the same full save as scheduled checkpoints.
            trainer._save_checkpoint(trainer.model, trial=None)
            safe_save_base_and_lora(trainer.model, cand, cfg_obj=getattr(trainer.model, "cfg", None))
            last_ckpt = cand

        log(f"[CHUNK] done. global_step={start_step}, last_ckpt={last_ckpt}")

    best_manifest = {
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric,
        "final_checkpoint": last_ckpt,
        "selection_metric": "eval_loss",
        "greater_is_better": False,
    }
    with open(os.path.join(OUT_DIR, "best_checkpoint.json"), "w", encoding="utf-8") as f:
        json.dump(best_manifest, f, ensure_ascii=False, indent=2)
    log(f"[CHUNK] all done. {best_manifest}")


if __name__ == "__main__":
    main()
