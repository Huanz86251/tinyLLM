from project_paths import legacy_path, path as project_path
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, json, time, random, re, shutil
from datetime import datetime
from pathlib import Path
from itertools import chain
from safetensors.torch import load_file
import numpy as np
import torch

import torch.nn as nn
from contextlib import nullcontext
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler, default_collate
import torch.distributed as dist
from safetensors.torch import save_file
from datasets import load_from_disk, concatenate_datasets, load_from_disk as hf_load
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from typing import Dict
from model.config import Config
from model.model import TinyLLM
from train.pretrain.kd_reader import KDFetcher  # 保留导入但第三阶段不再使用KD
import time
from datetime import datetime
from train.pretrain.math_mask import build_mathish_masks, build_per_id_weight_from_masks,preview_mathish_tokens


# ============================================================
# 环境 / 性能选项
# ============================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
from torch.backends.cuda import sdp_kernel
sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True)

SAVE_TOTAL_LIMIT = int(os.environ.get("SAVE_TOTAL_LIMIT", "2"))
CKPT_PREFIX = "checkpoint-"
LOG_INTERVAL_LONG_STEPS = int(os.environ.get("LOG_INTERVAL_LONG_STEPS", "10"))
LOG_SCALAR_EVERY = int(os.environ.get("LOG_SCALAR_EVERY", "20"))
USE_BF16 = True
AUTOCAST_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
CHECKPOINT_BARRIER = int(os.environ.get("CHECKPOINT_BARRIER", "1"))
BARRIER_EVERY_N = int(os.environ.get("BARRIER_EVERY_N", "0"))
CTRL_LOG_EVERY_ADJUST = int(os.environ.get("CTRL_LOG_EVERY_ADJUST", "20"))
USE_CHECKPOINT = False
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
ATT_WARMUP_STEPS = int(os.environ.get("ATT_WARMUP_STEPS", "2000"))
# ============================================================
# ========== 基本训练超参(短模式/主模式) ==========
# ============================================================
MAX_LEN_SHORT = 2048
BATCH_SHORT = 3                        # per-device
GRAD_ACCUM_SHORT = 3                   # 第三阶段：前2个micro用main，最后1个micro用more
LR_BASE = 6e-5
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.2
EPOCHS = 1
LONG_LOSS_BOOST = 1.10
SEED=10
XLONG_LOSS_BOOST = 1.20
KD_DIR = None  # 第三阶段：不再使用KD

# ========== 长上下文子模式 (8K) ==========
MAX_LEN_LONG = 8192
BATCH_LONG = 1                         # per-device
GRAD_ACCUM_LONG = 2
LONG_LR_SCALE = 1.0

# ========== 额外超长上下文子模式 (16K) ==========
# 每出现 10 次 long(8K) step，则插入 1 次 xlong(16K) step
MAX_LEN_XLONG = 16384
BATCH_XLONG = 1
GRAD_ACCUM_XLONG = 1
XLONG_EVERY_LONG_STEPS = 5
XLONG_LR_SCALE = 1.0  # 与 long 保持一致；如需单独缩放可改此项

# ========== 采样调度 ==========
SHORT_STEPS_PER_CYCLE = 20
LONG_STEPS_PER_CYCLE  = 1

# ========== 评估 / 日志 / 保存 ==========
LOG_INTERVAL_STEPS = 200
SAVE_INTERVAL_STEPS = 6000
EVAL_INTERVAL_STEPS = 6000
R_TARGET = float(os.environ.get("R_TARGET", "0.7"))  #  65%
S_LANG_MIN, S_LANG_MAX = 0.20, 1.3                   # 语言步 loss 的缩放边界
CTRL_EMA = 0.9                                         # 观测的 EMA
CTRL_EVERY = 10                                      # 每多少个短周期调整一次
Kp = 0.5
S_LANG = 2.0 * (1.0 - R_TARGET) / max(1e-6, R_TARGET)
S_LANG = float(min(max(S_LANG, S_LANG_MIN), S_LANG_MAX))
RUN_ID   = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_NAME = f"tinyllm-mixctx-{RUN_ID}"
OUT_DIR  = str(project_path("runs/training/cpt3"))
RESUME_ROOT = legacy_path("/root/autodl-tmp/llm/tiny_05B_cpt2/checkpoint-396000")
LOG_DIR  = os.path.join(legacy_path("/root/autodl-tmp/llm/checkpoints"), "tb", RUN_NAME)
os.makedirs(LOG_DIR, exist_ok=True)

# ========== 数据路径 ==========
LOCAL_SNAPSHOT          = legacy_path("/root/autodl-tmp/llm/KDmodel/minicpm3_4b/snapshot")
# 第三阶段：明确两份短缓存路径（no-suffix 主集 + _more 锚定集）
PK_CACHE_SHORT_MAIN     = legacy_path("/root/autodl-tmp/llm/cache_pretrain_large/packed_len2048_reason")        # 无后缀：主集（指令/代码/推理等）
PK_CACHE_SHORT_MORE     = legacy_path("/root/autodl-tmp/llm/cache_pretrain_large/packed_len2048_language")   # 锚定用语言分布
PK_CACHE_LONG           = legacy_path("/root/autodl-tmp/llm/cache_pretrain_large/packed_len8k_last")
# 新增：16K 缓存路径（按需修改为你的实际位置）
PK_CACHE_XLONG          = legacy_path("/root/autodl-tmp/llm/cache_pretrain_large/packed_len16384")

FORCE_OFFLINE = True

if FORCE_OFFLINE:
    os.environ.update({
        "HF_HOME": legacy_path("/root/autodl-tmp/hf_home_strict_offline"),
        "HF_DATASETS_CACHE": legacy_path("/root/autodl-tmp/hf_datasets_cache_force"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_ENABLE_ONLINE_MODE": "0",
        "TOKENIZERS_PARALLELISM": "true",
    })
ema_g_reason, ema_g_lang = 0.0, 0.0
short_cycle_count = 0
# ============================================================
# 杂项小工具
# ============================================================
START_TIME = time.time()
MIN_LR_FLOOR = float(os.environ.get("MIN_LR_FLOOR", "2e-5"))
# 两种模式：
# "global": 地板不随层缩放（更“硬”的地板）
# "per_scale": 地板也随层的 lr_scale 成比例缩放（更温和）
MIN_LR_FLOOR_MODE = os.environ.get("MIN_LR_FLOOR_MODE", "per_scale")  # "global" | "per_scale"
def force_mamba2_torch_forward(model: torch.nn.Module):
    """
    Disable fused CUDA kernels in HF Mamba2 so we always go through torch_forward().
    (You had this already.)
    """
    patched = 0
    for m in model.modules():
        if hasattr(m, "cuda_kernels_forward") and hasattr(m, "torch_forward"):
            m.forward = m.torch_forward
            patched += 1
    print(f"[patch] force_mamba2_torch_forward: patched {patched} submodules to use torch_forward()")

def get_group_floor(lr_scale: float) -> float:
    if MIN_LR_FLOOR_MODE == "per_scale":
        return MIN_LR_FLOOR * lr_scale
    return MIN_LR_FLOOR
def log_param_stats(model, prefix="model"):
    """
    打印总参数量 / 可训练参数量，并按 embedding / blocks / lm_head 粗略拆分。
    """
    root = model.module if hasattr(model, "module") else model

    total = 0
    trainable = 0
    for p in root.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n

    # 粗分一下 emb / blocks / head
    emb = sum(p.numel() for n, p in root.named_parameters() if n.startswith("tok_embed"))
    head = sum(p.numel() for n, p in root.named_parameters()
               if n.startswith("lm_head") or n.startswith("lm_bias"))

    block_params = []
    for i, blk in enumerate(root.blocks):
        n = sum(p.numel() for p in blk.parameters())
        block_params.append(n)

    blocks_total = sum(block_params) if block_params else 0

    log(f"[PARAM] {prefix}: total={total:,} ({total/1e6:.3f}M), "
        f"trainable={trainable:,} ({trainable/1e6:.3f}M)")
    log(f"[PARAM]   tok_embed={emb/1e6:.3f}M, blocks={blocks_total/1e6:.3f}M, "
        f"lm_head+bias={head/1e6:.3f}M")

    if block_params:
        mean_blk = blocks_total / len(block_params)
        log(f"[PARAM]   per-block: mean={mean_blk/1e6:.3f}M, "
            f"min={min(block_params)/1e6:.3f}M, "
            f"max={max(block_params)/1e6:.3f}M")

def print_trainable_detail(model, top_k=200):
    print("===== TRAINABLE PARAMS (detail) =====")
    rows = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            rows.append((name, p.numel()))
    rows.sort(key=lambda x: -x[1])
    total = sum(n for _, n in rows)
    print(f"trainable params = {total:,}")
    for name, n in rows[:top_k]:
        print(f"{n:>10}  {name}")
def apply_min_lr_floor(lr_eff: float, lr_scale: float) -> float:
    floor = get_group_floor(lr_scale)
    return lr_eff if lr_eff >= floor else floor
def effective_scale(lr_scale: float, mode: str) -> float:
    """
    mode='reason'  → 原样返回（下窄上宽：0.5/0.6/1.0）
    mode='lang'    → 反转三段比例（下宽上窄：1.0/0.6/0.5）
    其他值保持不动（以防将来你扩更多档位）
    """
    if mode != "lang":
        return lr_scale
    # 仅对 0.5/0.6/1.0 这三档做映射
    if abs(lr_scale - 1.0) < 1e-8:
        return 0.5
    if abs(lr_scale - 0.6) < 1e-8:
        return 0.6
    if abs(lr_scale - 0.5) < 1e-8:
        return 1.0
    return lr_scale

def grad_total_norm(parameters, norm_type: float = 2.0) -> float:
    """Compute total grad norm over all params (ignores None grads)."""
    device = None
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        g = p.grad
        if device is None:
            device = g.device
        if norm_type == float('inf'):
            total = max(total, g.detach().abs().max().to(torch.float32))
        else:
            param_norm = g.detach().to(torch.float32).norm(norm_type)
            total += param_norm.item() ** norm_type
    if norm_type == float('inf'):
        return float(total)
    return float(total ** (1.0 / norm_type))

def to_float(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return float(x.detach().item())
    if isinstance(x, (int, float)):
        return float(x)
    return None

def _maybe_barrier_every_n(global_step: int):
    if BARRIER_EVERY_N > 0 and dist.is_available() and dist.is_initialized():
        if global_step % BARRIER_EVERY_N == 0:
            dist.barrier()

def _ddp_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def prune_old_checkpoints(base_dir: str, keep: int = SAVE_TOTAL_LIMIT, prefix: str = CKPT_PREFIX):
    if keep is None or keep <= 0 or not os.path.isdir(base_dir):
        return
    items = []
    for name in os.listdir(base_dir):
        if not name.startswith(prefix):
            continue
        step_str = name[len(prefix):]
        if not step_str.isdigit():
            continue
        items.append((int(step_str), os.path.join(base_dir, name)))
    items.sort(key=lambda x: x[0])
    for _, p in items[:-keep]:
        try:
            shutil.rmtree(p)
            log(f"[SAVE] pruned old checkpoint: {p}")
        except Exception as e:
            log(f"[SAVE] prune failed ({p}): {e}")

def _fmt_hms(sec: float) -> str:
    sec = int(max(0, sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed = time.time() - START_TIME
    print(f"[{ts}] (+{_fmt_hms(elapsed)}) {msg}", flush=True)
def save_tokenizer_files(tokenizer, out_dir: str):
    """
    把 tokenizer 相关文件 (tokenizer.json / tokenizer_config.json / special_tokens_map.json / merges.txt / vocab.json / added_tokens.json ...)
    直接保存到 out_dir 顶层，便于 AutoTokenizer.from_pretrained(out_dir) 直接加载。
    """
    os.makedirs(out_dir, exist_ok=True)
    tokenizer.save_pretrained(out_dir)
class ETAHelper:
    def __init__(self, total_steps: int, ema=0.9):
        self.total_steps = max(1, int(total_steps))
        self.ema = float(ema)
        self.step = 0
        self._ema_step_time = None
        self._ema_tokens_per_step = 0.0
        self._t0 = time.perf_counter()

    def tick(self, tokens_this_step: float = 0.0):
        now = time.perf_counter()
        dt = now - self._t0
        self._t0 = now
        if self._ema_step_time is None:
            self._ema_step_time = dt
        else:
            self._ema_step_time = self.ema * self._ema_step_time + (1 - self.ema) * dt
        self._ema_tokens_per_step = self.ema * self._ema_tokens_per_step + (1 - self.ema) * float(tokens_this_step)
        self.step += 1

    @property
    def steps_per_sec(self) -> float:
        return 1.0 / self._ema_step_time if self._ema_step_time and self._ema_step_time > 0 else 0.0

    @property
    def tokens_per_sec(self) -> float:
        if not self._ema_step_time or self._ema_step_time <= 0:
            return 0.0
        return self._ema_tokens_per_step / self._ema_step_time

    @property
    def eta_seconds(self) -> float:
        remain = max(0, self.total_steps - self.step)
        if not self._ema_step_time:
            return float("inf")
        return remain * self._ema_step_time

    def brief(self) -> str:
        pct = 100.0 * min(1.0, self.step / self.total_steps)
        eta = _fmt_hms(self.eta_seconds)
        sps = self.steps_per_sec
        tps = self.tokens_per_sec
        return (f"step {self.step}/{self.total_steps} ({pct:5.1f}%) | "
                f"ETA {eta} | {sps:5.2f} step/s | {tps/1e6:6.2f} M tok/s")
# ==== TensorBoard helpers ====
def tb_add(writer, tag, scalar, step):
    if writer is not None and scalar is not None:
        writer.add_scalar(tag, float(scalar), int(step))

def log_ctrl_to_tb(writer, step, s_lang, ema_reason, ema_lang, R_obs, R_target):
    if writer is None:
        return
    tb_add(writer, "ctrl/S_LANG", s_lang, step)
    tb_add(writer, "ctrl/ema_reason_gnorm", ema_reason, step)
    tb_add(writer, "ctrl/ema_lang_gnorm",   ema_lang,   step)
    tb_add(writer, "ctrl/R_obs_reason_share", R_obs, step)
    tb_add(writer, "ctrl/R_target", R_target, step)
    # 直接记录 post-clip 的语言强度；另起一个曲线记录 base 估计（在控制块里写入）
    if ema_lang > 0:
        tb_add(writer, "ctrl/lang_effective_postclip", ema_lang, step)
        tb_add(writer, "ctrl/reason_vs_lang_effective_ratio",
               (2.0 * ema_reason) / max(1e-6, ema_lang), step)

def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))

def get_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))

def is_main_process():
    return get_rank() == 0

def save_training_state(ckpt_dir, optimizer, global_optimizer_step, epoch):
    state = {
        "optimizer": optimizer.state_dict(),
        "global_optimizer_step": int(global_optimizer_step),
        "epoch": int(epoch),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    tmp = os.path.join(ckpt_dir, "training_state.pt.tmp")
    final = os.path.join(ckpt_dir, "training_state.pt")
    torch.save(state, tmp)
    os.replace(tmp, final)

def safe_save_full_model(model, out_dir, cfg_obj=None):
    os.makedirs(out_dir, exist_ok=True)
    state = {
        k: (v.detach().cpu() if v.is_floating_point() else v.cpu())
        for k, v in model.state_dict().items()
    }
    bad = [k for k in state.keys() if re.match(r"^\d+\.", k)]
    if bad:
        raise RuntimeError(
            f"State dict has top-level numeric keys (e.g., {bad[:5]}) - "
            f"you're likely saving a submodule instead of the full model."
        )
    save_file(state, os.path.join(out_dir, "model.safetensors"))
    if cfg_obj is not None:
        with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(getattr(cfg_obj, "__dict__", {}), f, ensure_ascii=False, indent=2)

ROPE_OWNER_PAT = re.compile(r"\.(rotary_emb|rotary|rope|rope_emb)\.")
ROPE_LEAF_ALLOW = {"inv_freq"}
ROPE_LEAF_DROP = {
    "cos_cached", "sin_cached", "cos_cached_long", "sin_cached_long",
    "max_seq_len_cached", "_seq_len_cached", "seq_len_cached",
    "cached_seq_len", "cached_max_seq_len",
    "base", "theta", "rope_base", "rope_theta",
}

def try_load_training_state(ckpt_dir, optimizer):
    st_file = os.path.join(ckpt_dir, "training_state.pt")
    if not os.path.isfile(st_file):
        return 0, 0
    st = torch.load(st_file, map_location="cpu")
    try:
        optimizer.load_state_dict(st["optimizer"])
    except Exception as e:
        print(f"[RESUME] optimizer state load failed: {e}")
    try:
        random.setstate(st["rng"]["python"])()
    except Exception:
        random.setstate(st["rng"]["python"])  # 兼容不同版本结构
    try:
        np.random.set_state(st["rng"]["numpy"])()
    except Exception:
        np.random.set_state(st["rng"]["numpy"])  # 兼容不同版本结构
    try:
        torch.set_rng_state(st["rng"]["torch"])()
    except Exception:
        torch.set_rng_state(st["rng"]["torch"])  # 兼容不同版本结构
    try:
        if torch.cuda.is_available() and st["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(st["rng"]["cuda"])()
    except Exception:
        if torch.cuda.is_available() and st["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(st["rng"]["cuda"])  # 兼容不同版本结构
    return int(st.get("global_optimizer_step", 0)), int(st.get("epoch", 0))

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

# ============================================================
# Collator（第三阶段：KD禁用，保持接口不变）
# ============================================================
class MixedCollator:
    def __init__(
        self,
        tokenizer,
        kd_dir=None,
        p_denoise=0.0,
        p_kd=0.0,
        ignore_index=-100,
        span_ratio=(0.15,0.30),
        mask_token=None,
        kd_topk_use=0,
        kd_pos_keep=0.0,
        shortlist_neg=128,
        exclude_special_in_neg=True,
        enable_shortlist=False
    ):
        self.tok = tokenizer
        self.p_denoise = p_denoise
        self.p_kd = p_kd
        self.ignore_index = ignore_index
        self.span_ratio = span_ratio
        self.mask_id = tokenizer.pad_token_id if mask_token is None else tokenizer.convert_tokens_to_ids(mask_token)

        self.kd_dir = None  # 第三阶段强制禁用KD
        self.kd_topk_use = 0
        self.kd_pos_keep = 0.0

        self.shortlist_neg = shortlist_neg
        self.exclude_special_in_neg = exclude_special_in_neg
        self.enable_shortlist = enable_shortlist

        self.vocab_size = len(tokenizer.get_vocab())
        self.kd = None

        probs = np.ones(self.vocab_size, dtype=np.float64) / float(self.vocab_size)
        self.unigram_probs = probs.copy()

        self.special_ids = set(
            i for i in [
                tokenizer.pad_token_id,
                tokenizer.eos_token_id,
                getattr(tokenizer, "unk_token_id", None),
                getattr(tokenizer, "bos_token_id", None),
            ] if i is not None
        )
        if self.exclude_special_in_neg and len(self.special_ids) > 0:
            for sid in self.special_ids:
                if 0 <= sid < self.unigram_probs.shape[0]:
                    self.unigram_probs[sid] = 0.0
            s = self.unigram_probs.sum()
            assert s > 0
            self.unigram_probs /= s

        self._torch_unigram_p = torch.from_numpy(self.unigram_probs).to(dtype=torch.float32)
        self.logq_vec = torch.log(self._torch_unigram_p + 1e-12)

    def _ar(self, input_ids):
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = self.ignore_index
        return input_ids, labels

    def __call__(self, batch):
        input_ids = default_collate([e["input_ids"] for e in batch]).long()

        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = self.ignore_index

        # attention_mask：当前位置是否为有效 token
        if self.tok.pad_token_id is not None:
            attn_mask = (input_ids != self.tok.pad_token_id)  # [B,T] bool
            # 仅当“下一位仍有效”时，当前位才需要学习
            next_valid = attn_mask[:, 1:]
            labels[:, :-1] = torch.where(
                next_valid,
                labels[:, :-1],
                torch.full_like(labels[:, :-1], self.ignore_index)
            )
        else:
            attn_mask = torch.ones_like(input_ids, dtype=torch.bool)

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}

# ============================================================
# 组装 tokenizer / 模型 / 数据
# ============================================================

def build_tokenizer_and_model():
    TOK_DIR = LOCAL_SNAPSHOT
    assert os.path.isdir(TOK_DIR), f"Tokenizer dir not found: {TOK_DIR}"
    assert os.path.isfile(os.path.join(TOK_DIR, "tokenizer.json")), "Missing tokenizer.json"

    tokenizer = AutoTokenizer.from_pretrained(
        TOK_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )
    masks = build_mathish_masks(tokenizer)

    per_id_w = build_per_id_weight_from_masks(masks, w_mathish=1.2, cap=3.0)
    if is_main_process():
        _ = preview_mathish_tokens(tokenizer, masks, k=100)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer.get_vocab())

    cfg = Config(
        vocab_size=vocab_size,
        train_maxlength=MAX_LEN_LONG,  # 保持 8192；依赖 YARN 以泛化到 16K
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
    )
    cfg.use_checkpoint = False
    cfg.checkpoint_use_reentrant = False

    model = TinyLLM(cfg)
    model.register_buffer("class_w_mathish", per_id_w, persistent=False)
    def _strip_mamba_token_embedding(ssm):
        if hasattr(ssm, "embeddings"):
            try:
                delattr(ssm, "embeddings")
            except Exception:
                ssm._parameters.pop("embeddings", None)
                ssm._modules.pop("embeddings", None)
        ssm.register_buffer("embeddings", torch.empty(0), persistent=False)
        ssm.get_input_embeddings = lambda: (_ for _ in ()).throw(RuntimeError("Mamba2 embeddings disabled"))
        ssm.set_input_embeddings = lambda _: (_ for _ in ()).throw(RuntimeError("Mamba2 embeddings disabled"))

    for blk in model.blocks:
        if hasattr(blk, "ssm"):
            _strip_mamba_token_embedding(blk.ssm)


    ckpt_path = RESUME_ROOT

    if ckpt_path is None:
        raise FileNotFoundError(f"[RESUME] 未设置 RESUME_ROOT；当前值={RESUME_ROOT!r}")

    safe_path = os.path.join(ckpt_path, "model.safetensors")
    bin_path = os.path.join(ckpt_path, "pytorch_model.bin")

    # 只要有 safetensors 就优先用它
    if os.path.exists(safe_path):
        print(f"[RESUME] loading safetensors: {safe_path}")
        state = load_file(safe_path)
    elif os.path.exists(bin_path):
        print(f"[RESUME] loading torch bin: {bin_path}")
        state = torch.load(bin_path, map_location="cpu")
    else:
        # 打印目录帮助定位路径是否写错 / 权限问题
        try:
            items = os.listdir(ckpt_path)
        except Exception as e:
            items = f"<listdir failed: {e}>"
        raise FileNotFoundError(
            f"[RESUME] 在 {ckpt_path} 未找到 'model.safetensors' 或 'pytorch_model.bin'。\n"
            f"dir = {items}"
        )

    dropped = drop_rope_buffers_from_state(state)
    ret = model.load_state_dict(state, strict=False)
    print(
        f"[RESUME] loaded (dropped {dropped} rope-keys) from {safe_path if 'safetensors' in locals() and os.path.exists(safe_path) else bin_path}")

    bad_missing = [k for k in ret.missing_keys if not ROPE_OWNER_PAT.search(k)]
    if bad_missing:
        print(f"[RESUME] ignore unexpected keys (likely old SSM weights, etc.): {ret.unexpected_keys[:8]}")
    force_mamba2_torch_forward(model)
    return tokenizer, model


def build_datasets(tokenizer):
    # 第三阶段：short 主集与 more 各自切 eval，互不泄题
    ds_main = load_from_disk(PK_CACHE_SHORT_MAIN)
    ds_more = load_from_disk(PK_CACHE_SHORT_MORE)

    EVAL_FRAC_MAIN = float(os.environ.get("EVAL_FRAC_MAIN", "0.0001"))
    EVAL_FRAC_MORE = float(os.environ.get("EVAL_FRAC_MORE", "0.0001"))

    split_main = ds_main.train_test_split(test_size=EVAL_FRAC_MAIN, seed=42)
    train_main = split_main["train"]
    eval_main  = split_main["test"]

    split_more = ds_more.train_test_split(test_size=EVAL_FRAC_MORE, seed=43)
    train_more = split_more["train"]
    eval_more  = split_more["test"]

    train_long = load_from_disk(PK_CACHE_LONG)

    # 新增：16K 数据集
    train_xlong = load_from_disk(PK_CACHE_XLONG)

    # 设置 torch 格式
    train_main = train_main.with_format("torch", columns=["input_ids"])
    train_more = train_more.with_format("torch", columns=["input_ids"])
    eval_main  = eval_main.with_format("torch", columns=["input_ids"])
    eval_more  = eval_more.with_format("torch", columns=["input_ids"])
    train_long = train_long.with_format("torch", columns=["input_ids"])
    train_xlong = train_xlong.with_format("torch", columns=["input_ids"])

    return train_main, train_more, train_long, train_xlong, eval_main, eval_more


def make_loader(dataset, batch_size, collator, shuffle=True, drop_last=True):
    sampler = DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=shuffle,
        drop_last=drop_last,
        seed=SEED,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=6,
        drop_last=drop_last,
    )
    return loader, sampler

# ============================================================
# 训练 step & eval step
# ============================================================

def forward_batch(model, batch, device, collator_obj,enhance_math=True,if_xlong=False):
    inputs = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device, non_blocking=True)
        else:
            inputs[k] = v
    if if_xlong:
        model_outputs = model(**inputs,enhance_math=enhance_math,force_checkpoint=True)
    else:
        model_outputs = model(**inputs,enhance_math=enhance_math)
    loss = model_outputs["loss"]
    metrics = {
        "ce_loss": to_float(model_outputs.get("ce_loss", None)),
        "kd_loss": to_float(model_outputs.get("kd_loss", None)),
        "aux_loss": to_float(model_outputs.get("aux_loss", None)),
    }
    return loss, metrics


def run_eval(model, eval_loader, device, collator_obj, max_batches=10):
    model.train(False)
    total_loss = 0.0
    total_count = 0
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=AUTOCAST_DTYPE):
        for i, batch in enumerate(eval_loader):
            if i >= max_batches:
                break
            loss, _ = forward_batch(model, batch, device, collator_obj,False)
            bs = batch["input_ids"].size(0)
            total_loss += float(loss.item()) * bs
            total_count += bs

    if dist.is_available() and dist.is_initialized():
        vec = torch.tensor([total_loss, total_count], dtype=torch.float64, device=device)
        dist.all_reduce(vec, op=dist.ReduceOp.SUM)
        total_loss, total_count = vec.tolist()
    model.train(True)
    if total_count == 0:
        return None
    return total_loss / total_count


# ============================================================
# 分层学习率 param groups（前1/3×0.5，中1/3×0.6，后1/3×1.0）
# ============================================================
_BLOCK_RE = re.compile(r"^(?:module\.)?blocks\.(\d+)\.")

def param_groups_layerwise(nn_module, weight_decay):
    root = nn_module.module if hasattr(nn_module, "module") else nn_module
    n_layers = len(root.blocks)
    b1 = n_layers // 3
    b2 = (2 * n_layers) // 3

    groups = {
        ("decay", 0.5, "blocks"): [],
        ("no_decay", 0.5, "blocks"): [],
        ("decay", 0.6, "blocks"): [],
        ("no_decay", 0.6, "blocks"): [],
        ("decay", 1.0, "blocks"): [],
        ("no_decay", 1.0, "blocks"): [],
        ("decay", 1.0, "nonblocks"): [],
        ("no_decay", 1.0, "nonblocks"): [],
    }

    for name, p in nn_module.named_parameters():
        if not p.requires_grad:
            continue
        m = _BLOCK_RE.match(name)
        if m:
            idx = int(m.group(1))
            lr_scale = 0.5 if idx < b1 else (0.6 if idx < b2 else 1.0)
            tag = "blocks"
        else:
            lr_scale = 1.0
            tag = "nonblocks"

        is_no_decay = (p.ndim == 1) or any(k in name for k in [
            "norm", "bias", "tok_embed", "gamma_att", "gamma_mlp",
            "tcc_gate", "router", "decider", "temp_param","embedding","embed","lm_head"
        ])
        key = ("no_decay" if is_no_decay else "decay", lr_scale, tag)
        groups.setdefault(key, []).append(p)

    param_groups = []
    for (kind, scale, tag), params in groups.items():
        if not params:
            continue
        param_groups.append({
            "params": params,
            "weight_decay": 0.0 if kind == "no_decay" else weight_decay,
            "lr_scale": scale,
            "group_tag": tag,            # <<< 新增
        })
    return param_groups
def collect_lr_snapshot(optimizer):
    """
    返回一个快照：{ (group_tag, lr_scale, decay/no_decay, idx): lr_value }
    方便既看 blocks/nonblocks，也看反转后实际 lr。
    """
    snap = []
    for gi, pg in enumerate(optimizer.param_groups):
        tag = pg.get("group_tag", "blocks")
        scale = pg.get("lr_scale", 1.0)
        kind = "no_decay" if float(pg.get("weight_decay", 0.0)) == 0.0 else "decay"
        lr = float(pg.get("lr", 0.0))
        snap.append(((tag, scale, kind, gi), lr))
    return snap

def log_lr_groups(writer, step, optimizer, prefix):
    if writer is None:
        return
    snap = collect_lr_snapshot(optimizer)
    # 逐组写 scalar，便于对比
    for (tag, scale, kind, gi), lr in snap:
        writer.add_scalar(f"lr/{prefix}/{tag}/{kind}/scale{scale:.2f}/group{gi}", lr, step)
    # 也记录一个全局 min/max/mean 辅助审计
    if snap:
        vals = [lr for _, lr in snap]
        writer.add_scalar(f"lr/{prefix}/_min", min(vals), step)
        writer.add_scalar(f"lr/{prefix}/_max", max(vals), step)
        writer.add_scalar(f"lr/{prefix}/_mean", sum(vals)/len(vals), step)

def rope_debug_dump(model, writer=None, step=0):
    root = model.module if hasattr(model, "module") else model
    # 1) 配置级别
    try:
        cfg = getattr(root, "cfg", None)
        if cfg is not None:
            max_pos  = getattr(cfg, "max_position_embeddings", None)
            trainlen = getattr(cfg, "train_maxlength", None)
            rope_type= getattr(cfg, "rope_type", None)
            rope_base= getattr(cfg, "RoPE_base", None) or getattr(cfg, "rope_theta", None)
            print(f"[rope] cfg: rope_type={rope_type} base/theta={rope_base} "
                  f"max_position_embeddings={max_pos} train_maxlength={trainlen}")
            if writer is not None:
                writer.add_scalar("rope/cfg/max_position_embeddings", max_pos or -1, step)
                writer.add_scalar("rope/cfg/train_maxlength", trainlen or -1, step)
                writer.add_text("rope/cfg",
                    json.dumps({"rope_type": rope_type, "base_or_theta": rope_base}, ensure_ascii=False), step)
    except Exception as e:
        print(f"[rope] cfg read fail: {e}")

    # 2) 模块级别（寻找 inv_freq + base/theta 的模块）
    found = 0
    for name, m in root.named_modules():
        inv = getattr(m, "inv_freq", None)
        base = getattr(m, "base", None) or getattr(m, "theta", None)
        if inv is not None and (base is not None):
            found += 1
            ishape = tuple(getattr(inv, "shape", [])) or "?"
            idtype = str(getattr(inv, "dtype", "?"))
            idevice= str(getattr(inv, "device","?"))
            try:
                base_val = float(base) if hasattr(base, "item") else base
            except Exception:
                base_val = str(base)
            print(f"[rope] {name}: base/theta={base_val} inv_freq.shape={ishape} dtype={idtype} device={idevice}")
            if writer is not None:
                writer.add_text("rope/module",
                    json.dumps({"module": name, "base_or_theta": base_val,
                                "inv_freq_shape": str(ishape), "dtype": idtype, "device": idevice},
                               ensure_ascii=False),
                    step)
    if found == 0:
        print("[rope] WARN: no module exposing (inv_freq, base/theta) found.")

# ============================================================
# 主训练循环
# ============================================================

def train_loop():
    first_lr_dump_done = False
    second_lr_dump_done=False
    if not dist.is_initialized() and "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    tokenizer, model = build_tokenizer_and_model()
    ema_lang_base = 0.0
    model.to(device)
    model.train()
    if is_main_process():
        log_param_stats(model, prefix="after_resume_before_freeze")
    if get_world_size() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            static_graph=False,
        )
    root = model.module if hasattr(model, "module") else model
    SSM_OLD_LAYERS = [4, 11, 19]   # 你原来开 SSM 的三层

    def freeze_all_but_new_att():
        # 1) 全局先全部冻结
        for p in root.parameters():
            p.requires_grad = False

        # 2) 只解冻这几层的 Attention 支路（包括前置 norm 和 gamma_att）
        for idx in SSM_OLD_LAYERS:
            blk = root.blocks[idx]

            # 2.1 attention 主体：Wq/Wk/Wv/Wo、QK-RMSNorm、RoPE 中的可训练部分等
            if hasattr(blk, "selfatt"):
                for p in blk.selfatt.parameters():
                    p.requires_grad = True

            # 2.2 attention 之前的 RMSNorm（名字按你的 Block 实现来，下面列了几个常见的）
            for attr in ["attRMS_norm", "qk_norm", "qk_rmsnorm"]:
                if hasattr(blk, attr):
                    for p in getattr(blk, attr).parameters():
                        p.requires_grad = True

            # 2.3 attention 的 gated 标量
            if hasattr(blk, "gamma_att"):
                getattr(blk, "gamma_att").requires_grad = True

            # **注意**：这里故意不动 blk.MLP / blk.MLPrms_norm / blk.gamma_mlp，
            # 它们保持 requires_grad=False → MLP 完全 frozen

        # 3) 为了让 readout 能微调一点点：放开 final_norm + lm_bias
        if hasattr(root, "final_norm"):
            for p in root.final_norm.parameters():
                p.requires_grad = True

        if getattr(root, "lm_bias", None) is not None:
            root.lm_bias.requires_grad = True

    def unfreeze_all():
        for p in root.parameters():
            p.requires_grad = True

    freeze_all_but_new_att()
    if is_main_process():
        log_param_stats(model, prefix="warmup_trainable")
        print_trainable_detail(model, top_k=300)
    warmup_done = False
    writer = SummaryWriter(log_dir=LOG_DIR, flush_secs=5, max_queue=20) if is_main_process() else None
    if is_main_process():
        log(f"[TB] SummaryWriter ready, writing heartbeat to {LOG_DIR}")
        writer.add_scalar("z/heartbeat", 1, 0)
        writer.add_text("run/meta", json.dumps({
            "run_name": RUN_NAME,
            "pid": os.getpid(),
            "rank": get_rank(),
            "world_size": get_world_size(),
            "short_steps_per_cycle": SHORT_STEPS_PER_CYCLE,
            "long_steps_per_cycle": LONG_STEPS_PER_CYCLE,
            "LOG_SCALAR_EVERY": LOG_SCALAR_EVERY,
        }, ensure_ascii=False, indent=2), 0)
        writer.flush()
        try:
            ev = list(Path(LOG_DIR).glob("events*"))
            log(f"[TB] events files: {[p.name for p in ev] or 'None yet (will appear after first step)'}")
        except Exception:
            pass
        # 再做 RoPE 自检（这时 writer 已经可用）
        rope_debug_dump(model, writer=writer, step=0)

    # ===== 数据集 & DataLoader =====
    train_main, train_more, train_long, train_xlong, eval_main, eval_more = build_datasets(tokenizer)

    collator_short_main = MixedCollator(tokenizer, kd_dir=None, p_kd=0.0, kd_topk_use=0, kd_pos_keep=0.0, enable_shortlist=False)
    collator_short_more = MixedCollator(tokenizer, kd_dir=None, p_kd=0.0, kd_topk_use=0, kd_pos_keep=0.0, enable_shortlist=False)
    collator_long       = MixedCollator(tokenizer, kd_dir=None, p_kd=0.0, kd_topk_use=0, kd_pos_keep=0.0, enable_shortlist=False)
    collator_xlong      = MixedCollator(tokenizer, kd_dir=None, p_kd=0.0, kd_topk_use=0, kd_pos_keep=0.0, enable_shortlist=False)

    loader_short_main, sampler_short_main = make_loader(train_main, batch_size=BATCH_SHORT, collator=collator_short_main, shuffle=True)
    loader_short_more, sampler_short_more = make_loader(train_more, batch_size=BATCH_SHORT, collator=collator_short_more, shuffle=True)
    loader_long, sampler_long             = make_loader(train_long, batch_size=BATCH_LONG, collator=collator_long, shuffle=True)
    loader_xlong, sampler_xlong           = make_loader(train_xlong, batch_size=BATCH_XLONG, collator=collator_xlong, shuffle=True)
    eval_loader_main, _ = make_loader(eval_main, batch_size=BATCH_SHORT, collator=collator_short_main,shuffle=False, drop_last=False)
    eval_loader_more, _ = make_loader(eval_more, batch_size=BATCH_SHORT, collator=collator_short_more,shuffle=False, drop_last=False)

    # ===== 优化器（分层学习率 param groups）=====
    base_groups = param_groups_layerwise(model, WEIGHT_DECAY)

    optimizer = torch.optim.AdamW(
        base_groups,
        lr=LR_BASE,
        betas=(0.9, 0.95),
        fused=True if torch.cuda.get_device_capability(local_rank)[0] >= 8 else False,
        weight_decay=WEIGHT_DECAY,
    )

    # ===== scheduler: warmup + cosine，epoch由主集(main)定义 =====
    warmup_ratio = 0.003

    # 关键：每个 optimizer step 消耗 (GRAD_ACCUM_SHORT-1) 个 main micro-batches
    iters_per_epoch = min(len(loader_short_main) // 2, len(loader_short_more))  # 每迭代消耗：main=2, more=1
    steps_short_per_epoch = 2 * iters_per_epoch  # 每迭代产生2个optimizer.step()
    steps_long_per_epoch = (iters_per_epoch // SHORT_STEPS_PER_CYCLE) * LONG_STEPS_PER_CYCLE
    steps_xlong_per_epoch =steps_long_per_epoch // XLONG_EVERY_LONG_STEPS  # 见第3条，如启用 xlong 再补上
    steps_total_per_epoch = steps_short_per_epoch + steps_long_per_epoch + steps_xlong_per_epoch

    total_steps = steps_total_per_epoch * EPOCHS
    warmup_steps = 4000
    min_lr = LR_BASE * 0.1
    boost_step=int(total_steps*0.3)
    def boost(mode, step):
        if step >= boost_step:
            return 1.0
        # 线性从 boost_max 衰减到 1.0
        remain = 1.0 - (step / float(boost_step))
        base = LONG_LOSS_BOOST if mode == "long" else XLONG_LOSS_BOOST
        return 1.0 + (base - 1.0) * remain

    def lr_schedule(global_step, base_lr, mode="short"):
        if mode == "short":
            target_base = base_lr
        else:
            # long/xlong 共用一套缩放
            scale = LONG_LR_SCALE if mode == "long" else XLONG_LR_SCALE
            target_base = base_lr * scale
        if global_step < warmup_steps:
            return target_base * float(global_step + 1) / float(warmup_steps)
        else:
            progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = 0.0 if progress < 0 else (1.0 if progress > 1.0 else progress)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr_val = min_lr + (target_base - min_lr) * cosine
            return lr_val

    long_step_count = 0
    eta = ETAHelper(total_steps=total_steps, ema=0.9)

    WORLD = get_world_size()
    TOKENS_PER_STEP_SHORT = WORLD * BATCH_SHORT * MAX_LEN_SHORT * GRAD_ACCUM_SHORT
    TOKENS_PER_STEP_LONG  = WORLD * BATCH_LONG  * MAX_LEN_LONG  * GRAD_ACCUM_LONG
    TOKENS_PER_STEP_XLONG = WORLD * BATCH_XLONG * MAX_LEN_XLONG * GRAD_ACCUM_XLONG

    if is_main_process():
        tok_reason_step = WORLD * BATCH_SHORT * MAX_LEN_SHORT * 2  # 推理步（2个micro）
        tok_lang_step = WORLD * BATCH_SHORT * MAX_LEN_SHORT * 1  # 语言步（1个micro）
        log(f"[INIT] total_steps≈{total_steps} | short_reason_step_tok≈{tok_reason_step}, "
            f"short_lang_step_tok≈{tok_lang_step}, long_step_tok≈{TOKENS_PER_STEP_LONG}, xlong_step_tok≈{TOKENS_PER_STEP_XLONG}")

    os.makedirs(OUT_DIR, exist_ok=True)
    resume_train_dir = get_last_checkpoint(OUT_DIR)

    if resume_train_dir is not None and USE_CHECKPOINT:
        start_step, start_epoch = try_load_training_state(resume_train_dir, optimizer)
    else:
        start_step, start_epoch = 0, 0
    global_optimizer_step = start_step

    # ========= 仅为修复 UnboundLocalError：改为局部变量 =========
    s_lang = float(S_LANG)
    ema_reason = float(ema_g_reason)
    ema_lang = float(ema_g_lang)
    short_cycles = int(short_cycle_count)
    ctrl_adjust_count = 0
    # =======================================================

    for epoch in range(EPOCHS):
        eta_epoch = ETAHelper(total_steps=steps_total_per_epoch, ema=0.9)

        sampler_short_main.set_epoch(epoch)
        sampler_short_more.set_epoch(epoch)
        sampler_long.set_epoch(epoch)
        sampler_xlong.set_epoch(epoch)

        short_main_iter = iter(loader_short_main)
        short_more_iter = iter(loader_short_more)
        long_iter       = iter(loader_long)
        xlong_iter      = iter(loader_xlong)

        done = False
        while not done:
            if (not warmup_done) and (global_optimizer_step >= ATT_WARMUP_STEPS):
                if is_main_process():
                    log(f"[WARMUP] ATT warmup finished at step {global_optimizer_step}, "
                        f"unfreezing all layers & rebuilding optimizer")
                unfreeze_all()
                base_groups = param_groups_layerwise(model, WEIGHT_DECAY)
                optimizer = torch.optim.AdamW(
                    base_groups,
                    lr=LR_BASE,
                    betas=(0.9, 0.95),
                    fused=True if torch.cuda.get_device_capability(local_rank)[0] >= 8 else False,
                    weight_decay=WEIGHT_DECAY,
                )
                warmup_done = True
            # ===== 短块：20个短step，每个step：2×main + 1×more（按 GRAD_ACCUM_SHORT=3）=====
            # ===== 短块：20个短step，每个step：2×main + 1×more（按 GRAD_ACCUM_SHORT=3）=====
            for _ in range(SHORT_STEPS_PER_CYCLE):
                # ===============================
                # (1) 推理步：2 个 micro，单独 step
                # ===============================
                optimizer.zero_grad(set_to_none=True)
                accumulated_loss = torch.zeros((), device=device)
                metrics_last = None

                for micro_idx in range(2):  # grad_accum = 2
                    try:
                        batch = next(short_main_iter)
                    except StopIteration:
                        done = True
                        break
                    # 推理批：开启 math 增强（语言批会关掉）
                    ctx = nullcontext() if (micro_idx == 1 or get_world_size() == 1) else model.no_sync()
                    with ctx, torch.autocast(device_type="cuda", dtype=AUTOCAST_DTYPE):
                        loss, metrics = forward_batch(model, batch, device, collator_short_main, enhance_math=True)
                        # 这里不额外缩放，保持“推理”为基准
                        (loss / 2.0).backward()  # 2 个 micro 做平均
                    accumulated_loss += loss.detach()
                    metrics_last = metrics

                if done:
                    break

                # 观测：推理步的 grad-norm（用于配比控制）
                g_reason = grad_total_norm(model.parameters(), norm_type=2.0)
                g_reason_eff = min(g_reason, MAX_GRAD_NORM)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

                # 短步内 first-step 使用短步 LR（和你原先一致）
                base_lr_short = lr_schedule(global_optimizer_step, LR_BASE, mode="short")
                floor_hits = 0
                min_applied = 1e9
                for pg in optimizer.param_groups:
                    s = pg.get("lr_scale", 1.0)
                    lr_eff = apply_min_lr_floor(base_lr_short * s, s)
                    pg["lr"] = lr_eff
                    min_applied = min(min_applied, lr_eff)
                    floor_hits += int(abs(lr_eff - get_group_floor(s)) <= 1e-12)
                if is_main_process() and not first_lr_dump_done:
                    log_lr_groups(writer, global_optimizer_step, optimizer, prefix="FIRST_apply_short_reason")
                    first_lr_dump_done = True

                optimizer.step()
                global_optimizer_step += 1
                _maybe_barrier_every_n(global_optimizer_step)
                eta.tick(tokens_this_step=WORLD * BATCH_SHORT * MAX_LEN_SHORT * 2)  # 2 个 micro
                eta_epoch.tick(tokens_this_step=WORLD * BATCH_SHORT * MAX_LEN_SHORT * 2)

                # 日志（可保留你原有的 writer 记录）
                if writer is not None and (global_optimizer_step % LOG_SCALAR_EVERY == 0):
                    ce_f = to_float(metrics_last.get("ce_loss")) if metrics_last else None
                    if ce_f is not None: writer.add_scalar("short_split/reason_ce", ce_f, global_optimizer_step)
                    writer.add_scalar("short_split/g_reason", g_reason, global_optimizer_step)
                    writer.add_scalar("train/base_lr_short", base_lr_short, global_optimizer_step)
                if LOG_INTERVAL_STEPS > 0 and is_main_process() and (global_optimizer_step % LOG_INTERVAL_STEPS) == 0:
                    ce_now = to_float(metrics_last.get("ce_loss")) if metrics_last else None
                    if ce_now is not None:
                        pct = 100.0 * min(1.0, global_optimizer_step / max(1, total_steps))
                        log(f"[STEP ] #{global_optimizer_step} short_reason ce={ce_now:.4f} | progress={pct:5.1f}% | ETA {_fmt_hms(eta.eta_seconds)}")
                # EMA 更新（推理）
                ema_reason = CTRL_EMA * ema_reason + (1.0 - CTRL_EMA) * float(g_reason_eff)
                if (global_optimizer_step % EVAL_INTERVAL_STEPS == 0):
                    eval_loss_main = run_eval(model, eval_loader_main, device, collator_short_main)
                    eval_loss_more = run_eval(model, eval_loader_more, device, collator_short_more)
                    if is_main_process():
                        log(f"[EVAL] step={global_optimizer_step} "
                            f"eval_reason={eval_loss_main:.4f} | eval_nature_language={eval_loss_more:.4f} | {eta.brief()}")
                        if writer is not None:
                            writer.add_scalar("eval/reason", eval_loss_main, global_optimizer_step)
                            writer.add_scalar("eval/nature_language", eval_loss_more, global_optimizer_step)
                if (global_optimizer_step % SAVE_INTERVAL_STEPS == 0) and is_main_process():
                    ckpt_dir = os.path.join(OUT_DIR, f"checkpoint-{global_optimizer_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    model_to_save = model.module if hasattr(model, "module") else model
                    safe_save_full_model(model_to_save, ckpt_dir, cfg_obj=getattr(model_to_save, "cfg", None))
                    save_tokenizer_files(tokenizer, ckpt_dir)
                    save_training_state(ckpt_dir, optimizer, global_optimizer_step, epoch)
                    log(f"[SAVE] wrote {ckpt_dir}/model.safetensors | {eta.brief()}")
                    prune_old_checkpoints(OUT_DIR, keep=SAVE_TOTAL_LIMIT)
                # ===============================
                # (2) 语言步：1 个 micro，单独 step（loss 乘 S_LANG）
                # ===============================
                optimizer.zero_grad(set_to_none=True)
                try:
                    batch = next(short_more_iter)
                except StopIteration:
                    short_more_iter = iter(loader_short_more)
                    batch = next(short_more_iter)

                with torch.autocast(device_type="cuda", dtype=AUTOCAST_DTYPE):
                    # 语言批：关闭 math 增强，避免把语言也朝“数字权重”拉偏
                    loss_lang, metrics_lang = forward_batch(model, batch, device, collator_short_more,
                                                            enhance_math=False)
                    (loss_lang * s_lang).backward()

                g_lang = grad_total_norm(model.parameters(), norm_type=2.0)
                g_lang_base_inst = g_lang / max(1e-6, s_lang)
                ema_lang_base = CTRL_EMA * ema_lang_base + (1.0 - CTRL_EMA) * float(g_lang_base_inst)
                g_lang_eff = min(g_lang, MAX_GRAD_NORM)

                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

                # 语言步同样用短步 LR；也可以选用稍小的步长（不建议和 S_LANG 叠加缩放，避免“双降”）
                base_lr_short_lang = base_lr_short
                for pg in optimizer.param_groups:
                    s = pg.get("lr_scale", 1.0)
                    tag = pg.get("group_tag", "blocks")
                    s_eff = effective_scale(s, mode="lang") if tag == "blocks" else s
                    lr_eff = apply_min_lr_floor(base_lr_short_lang * s_eff, s_eff)
                    pg["lr"] = lr_eff
                if is_main_process() and not second_lr_dump_done:
                    log_lr_groups(writer, global_optimizer_step, optimizer, prefix="short_lang_after_reverse")
                    second_lr_dump_done = True

                optimizer.step()
                global_optimizer_step += 1
                _maybe_barrier_every_n(global_optimizer_step)
                eta.tick(tokens_this_step=WORLD * BATCH_SHORT * MAX_LEN_SHORT * 1)  # 1 个 micro
                eta_epoch.tick(tokens_this_step=WORLD * BATCH_SHORT * MAX_LEN_SHORT * 1)

                if writer is not None and (global_optimizer_step % LOG_SCALAR_EVERY == 0):
                    ce_l = to_float(metrics_lang.get("ce_loss")) if metrics_lang else None
                    if ce_l is not None: writer.add_scalar("short_split/lang_ce", ce_l, global_optimizer_step)
                    writer.add_scalar("short_split/g_lang", g_lang, global_optimizer_step)
                    writer.add_scalar("short_split/S_LANG", s_lang, global_optimizer_step)
                if LOG_INTERVAL_STEPS > 0 and is_main_process() and (global_optimizer_step % LOG_INTERVAL_STEPS) == 0:
                    ce_now = to_float(metrics_lang.get("ce_loss")) if metrics_lang else None
                    if ce_now is not None:
                        pct = 100.0 * min(1.0, global_optimizer_step / max(1, total_steps))
                        log(f"[STEP ] #{global_optimizer_step} short_lang  ce={ce_now:.4f} | progress={pct:5.1f}% | ETA {_fmt_hms(eta.eta_seconds)}")
                if (global_optimizer_step % EVAL_INTERVAL_STEPS == 0):
                    eval_loss_main = run_eval(model, eval_loader_main, device, collator_short_main)
                    eval_loss_more = run_eval(model, eval_loader_more, device, collator_short_more)
                    if is_main_process():
                        log(f"[EVAL] step={global_optimizer_step} "
                            f"eval_reason={eval_loss_main:.4f} | eval_nature_language={eval_loss_more:.4f} | {eta.brief()}")
                        if writer is not None:
                            writer.add_scalar("eval/reason", eval_loss_main, global_optimizer_step)
                            writer.add_scalar("eval/nature_language", eval_loss_more, global_optimizer_step)
                if (global_optimizer_step % SAVE_INTERVAL_STEPS == 0) and is_main_process():
                    ckpt_dir = os.path.join(OUT_DIR, f"checkpoint-{global_optimizer_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    model_to_save = model.module if hasattr(model, "module") else model
                    safe_save_full_model(model_to_save, ckpt_dir, cfg_obj=getattr(model_to_save, "cfg", None))
                    save_tokenizer_files(tokenizer, ckpt_dir)
                    save_training_state(ckpt_dir, optimizer, global_optimizer_step, epoch)
                    log(f"[SAVE] wrote {ckpt_dir}/model.safetensors | {eta.brief()}")
                    prune_old_checkpoints(OUT_DIR, keep=SAVE_TOTAL_LIMIT)
                # EMA 更新（语言）
                ema_lang = CTRL_EMA * ema_lang + (1.0 - CTRL_EMA) * float(g_lang_eff)

                # ===============================
                # (3) 极轻量配比控制：每若干短周期调整一次 S_LANG
                #     目标：在一个“短周期（2推理+1语言）”里，推理份额 ≈ R_TARGET
                #     份额估计： R_obs ≈ (2 * ema_reason) / (2 * ema_reason + s_lang * ema_lang)
                # ===============================
                short_cycles += 1
                if short_cycles % CTRL_EVERY == 0 and ema_reason > 0 and ema_lang > 0:
                    # 1) 观测份额：用 post-clip 有效强度；推理按 tokens 乘 2
                    R_obs = (2.0 * ema_reason) / max(1e-6, (2.0 * ema_reason + ema_lang))

                    # 2) 比例控制：Kp 建议 0.5，更稳（乘性更新，避免一步跳太大）
                    err = R_TARGET - R_obs
                    s_cmd = s_lang * (1.0 - 0.5 * err)

                    # 3) anti-saturation 守门：把“未缩放”的语言范数压到 ~0.85×max_norm
                    tau = 0.85
                    s_guard = tau * MAX_GRAD_NORM / max(1e-6, ema_lang_base)

                    # 4) 先 obey 守门，再限幅
                    s_next = min(s_cmd, s_guard)
                    s_next = float(min(max(s_next, S_LANG_MIN), S_LANG_MAX))
                    s_lang = s_next

                    if is_main_process():
                        if CTRL_LOG_EVERY_ADJUST > 0:
                            ctrl_adjust_count += 1
                            if (ctrl_adjust_count % CTRL_LOG_EVERY_ADJUST) == 0:
                                log(f"[CTRL] R_obs={R_obs:.3f} -> target={R_TARGET:.3f}, "
                                    f"S_LANG -> {s_lang:.4f} (guard={s_guard:.4f})")
                        log_ctrl_to_tb(writer, global_optimizer_step, s_lang, ema_reason, ema_lang, R_obs, R_TARGET)
                        if writer is not None:
                            writer.add_scalar("ctrl/s_guard", s_guard, global_optimizer_step)
                            writer.add_scalar("ctrl/ema_lang_base", ema_lang_base, global_optimizer_step)


            # ===== 长块：1个长step，long 用尽后循环 =====
            accumulated_loss = torch.zeros((), device=device)
            optimizer.zero_grad(set_to_none=True)
            for micro_idx in range(GRAD_ACCUM_LONG):
                try:
                    batch = next(long_iter)
                except StopIteration:
                    long_iter = iter(loader_long)
                    batch = next(long_iter)

                with torch.autocast(device_type="cuda", dtype=AUTOCAST_DTYPE):
                    loss, metrics = forward_batch(model, batch, device, collator_long,False)
                    loss = loss * boost("long", global_optimizer_step)
                    loss = loss / GRAD_ACCUM_LONG

                loss.backward()
                accumulated_loss = accumulated_loss + loss.detach()
            preclip_gn = grad_total_norm(model.parameters(), norm_type=2.0)
            if writer is not None and (global_optimizer_step % LOG_SCALAR_EVERY == 0):
                writer.add_scalar("grad/total_norm_preclip_long", preclip_gn, global_optimizer_step)
            # grad clip → 再设置分组学习率（长步 base_lr*LONG_LR_SCALE）
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            base_lr_long = lr_schedule(global_optimizer_step, LR_BASE, mode="long")
            floor_hits = 0
            min_applied = 1e9
            for pg in optimizer.param_groups:
                lr_scale = pg.get("lr_scale", 1.0)
                lr_eff = apply_min_lr_floor(base_lr_long * lr_scale, lr_scale)
                pg["lr"] = lr_eff
                min_applied = min(min_applied, lr_eff)
                floor_hits += int(abs(lr_eff - get_group_floor(lr_scale)) <= 1e-12)
            optimizer.step()
            global_optimizer_step += 1
            _maybe_barrier_every_n(global_optimizer_step)

            eta.tick(tokens_this_step=TOKENS_PER_STEP_LONG)
            eta_epoch.tick(tokens_this_step=TOKENS_PER_STEP_LONG)

            if writer is not None:
                ce_f  = to_float(metrics.get("ce_loss"))
                kd_f  = to_float(metrics.get("kd_loss"))
                aux_f = to_float(metrics.get("aux_loss"))
                if ce_f is not None:  writer.add_scalar("long/loss/ce", ce_f, global_optimizer_step)
                if kd_f is not None:  writer.add_scalar("long/loss/kd", kd_f, global_optimizer_step)
                if aux_f is not None: writer.add_scalar("long/loss/aux", aux_f, global_optimizer_step)
                writer.add_scalar("train/base_lr_long", base_lr_long, global_optimizer_step)
                writer.add_scalar("speed/steps_per_sec", eta.steps_per_sec, global_optimizer_step)
                writer.add_scalar("speed/tokens_per_sec", eta.tokens_per_sec, global_optimizer_step)
                writer.add_scalar("eta/seconds_remaining_total", eta.eta_seconds, global_optimizer_step)
                writer.add_scalar("eta/seconds_remaining_epoch", eta_epoch.eta_seconds, global_optimizer_step)
                writer.add_scalar("progress/step_total", global_optimizer_step, global_optimizer_step)
                writer.add_scalar("train/lr_min_applied_long", min_applied, global_optimizer_step)
                writer.add_scalar("train/lr_floor_hits_long", floor_hits, global_optimizer_step)

            long_step_count += 1
            if is_main_process() and (long_step_count % LOG_INTERVAL_LONG_STEPS == 0):
                log(f"[LONG ] step={global_optimizer_step} loss={accumulated_loss:.4f} "
                    f"ce={metrics.get('ce_loss')} kd={metrics.get('kd_loss')} "
                    f"lr(base)={base_lr_long:.6e} (long ctx) | {eta.brief()}")

            if (global_optimizer_step % EVAL_INTERVAL_STEPS == 0):
                eval_loss_main = run_eval(model, eval_loader_main, device, collator_short_main)
                eval_loss_more = run_eval(model, eval_loader_more, device, collator_short_more)
                if is_main_process():
                    log(f"[EVAL] step={global_optimizer_step} "
                        f"eval_reason={eval_loss_main:.4f} | eval_nature_language={eval_loss_more:.4f} | {eta.brief()}")
                    if writer is not None:
                        writer.add_scalar("eval/reason", eval_loss_main, global_optimizer_step)
                        writer.add_scalar("eval/nature_language", eval_loss_more, global_optimizer_step)

            if (global_optimizer_step % SAVE_INTERVAL_STEPS == 0) and is_main_process():
                ckpt_dir = os.path.join(OUT_DIR, f"checkpoint-{global_optimizer_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                model_to_save = model.module if hasattr(model, "module") else model
                safe_save_full_model(model_to_save, ckpt_dir, cfg_obj=getattr(model_to_save, "cfg", None))
                save_tokenizer_files(tokenizer, ckpt_dir)
                save_training_state(ckpt_dir, optimizer, global_optimizer_step, epoch)
                log(f"[SAVE] wrote {ckpt_dir}/model.safetensors | {eta.brief()}")
                prune_old_checkpoints(OUT_DIR, keep=SAVE_TOTAL_LIMIT)

            # ====== 每 10 个 long step → 插入 1 次 xlong(16K) step ======
            if XLONG_EVERY_LONG_STEPS > 0 and (long_step_count % XLONG_EVERY_LONG_STEPS) == 0:
                accumulated_loss_x = torch.zeros((), device=device)
                optimizer.zero_grad(set_to_none=True)
                for micro_idx in range(GRAD_ACCUM_XLONG):
                    try:
                        batch = next(xlong_iter)
                    except StopIteration:
                        xlong_iter = iter(loader_xlong)
                        batch = next(xlong_iter)
                    with torch.autocast(device_type="cuda", dtype=AUTOCAST_DTYPE):
                        loss, metrics = forward_batch(model, batch, device, collator_xlong,False,if_xlong=True)
                        loss = loss * boost("xlong", global_optimizer_step)
                        loss = loss / GRAD_ACCUM_XLONG
                    loss.backward()
                    accumulated_loss_x = accumulated_loss_x + loss.detach()
                preclip_gn_x = grad_total_norm(model.parameters(), norm_type=2.0)
                if writer is not None and (global_optimizer_step % LOG_SCALAR_EVERY == 0):
                    writer.add_scalar("grad/total_norm_preclip_xlong", preclip_gn_x, global_optimizer_step)

                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

                base_lr_xlong = lr_schedule(global_optimizer_step, LR_BASE, mode="long")  # 复用 long 缩放
                floor_hits_x = 0
                min_applied_x = 1e9
                for pg in optimizer.param_groups:
                    lr_scale = pg.get("lr_scale", 1.0)
                    lr_eff = apply_min_lr_floor(base_lr_xlong * lr_scale, lr_scale)
                    pg["lr"] = lr_eff
                    min_applied_x = min(min_applied_x, lr_eff)
                    floor_hits_x += int(abs(lr_eff - get_group_floor(lr_scale)) <= 1e-12)

                optimizer.step()
                global_optimizer_step += 1
                _maybe_barrier_every_n(global_optimizer_step)

                eta.tick(tokens_this_step=TOKENS_PER_STEP_XLONG)
                eta_epoch.tick(tokens_this_step=TOKENS_PER_STEP_XLONG)

                if writer is not None:
                    ce_f  = to_float(metrics.get("ce_loss"))
                    kd_f  = to_float(metrics.get("kd_loss"))
                    aux_f = to_float(metrics.get("aux_loss"))
                    if ce_f is not None:  writer.add_scalar("xlong/loss/ce", ce_f, global_optimizer_step)
                    if kd_f is not None:  writer.add_scalar("xlong/loss/kd", kd_f, global_optimizer_step)
                    if aux_f is not None: writer.add_scalar("xlong/loss/aux", aux_f, global_optimizer_step)
                    writer.add_scalar("train/base_lr_xlong", base_lr_xlong, global_optimizer_step)
                    writer.add_scalar("speed/steps_per_sec", eta.steps_per_sec, global_optimizer_step)
                    writer.add_scalar("speed/tokens_per_sec", eta.tokens_per_sec, global_optimizer_step)
                    writer.add_scalar("eta/seconds_remaining_total", eta.eta_seconds, global_optimizer_step)
                    writer.add_scalar("eta/seconds_remaining_epoch", eta_epoch.eta_seconds, global_optimizer_step)
                    writer.add_scalar("progress/step_total", global_optimizer_step, global_optimizer_step)
                    writer.add_scalar("train/lr_min_applied_xlong", min_applied_x, global_optimizer_step)
                    writer.add_scalar("train/lr_floor_hits_xlong", floor_hits_x, global_optimizer_step)

                if is_main_process():
                    log(f"[XLONG] step={global_optimizer_step} loss={accumulated_loss_x:.4f} "
                        f"ce={metrics.get('ce_loss')} kd={metrics.get('kd_loss')} "
                        f"lr(base)={base_lr_xlong:.6e} (16k ctx) | {eta.brief()}")

                if (global_optimizer_step % EVAL_INTERVAL_STEPS == 0):
                    eval_loss_main = run_eval(model, eval_loader_main, device, collator_short_main)
                    eval_loss_more = run_eval(model, eval_loader_more, device, collator_short_more)
                    if is_main_process():
                        log(f"[EVAL] step={global_optimizer_step} "
                            f"eval_reason={eval_loss_main:.4f} | eval_nature_language={eval_loss_more:.4f} | {eta.brief()}")
                        if writer is not None:
                            writer.add_scalar("eval/reason", eval_loss_main, global_optimizer_step)
                            writer.add_scalar("eval/nature_language", eval_loss_more, global_optimizer_step)

                if (global_optimizer_step % SAVE_INTERVAL_STEPS == 0) and is_main_process():
                    ckpt_dir = os.path.join(OUT_DIR, f"checkpoint-{global_optimizer_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    model_to_save = model.module if hasattr(model, "module") else model
                    safe_save_full_model(model_to_save, ckpt_dir, cfg_obj=getattr(model_to_save, "cfg", None))
                    save_tokenizer_files(tokenizer, ckpt_dir)
                    save_training_state(ckpt_dir, optimizer, global_optimizer_step, epoch)
                    log(f"[SAVE] wrote {ckpt_dir}/model.safetensors | {eta.brief()}")
                    prune_old_checkpoints(OUT_DIR, keep=SAVE_TOTAL_LIMIT)

        # end while epoch

    # final save
    _ddp_barrier()
    if is_main_process():
        final_dir = os.path.join(OUT_DIR, f"final-{global_optimizer_step}")
        os.makedirs(final_dir, exist_ok=True)
        model_to_save = model.module if hasattr(model, "module") else model
        safe_save_full_model(model_to_save, final_dir, cfg_obj=getattr(model_to_save, "cfg", None))
        save_tokenizer_files(tokenizer, final_dir)
        log(f"[FINAL SAVE] {final_dir} | {eta.brief()}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    train_loop()
