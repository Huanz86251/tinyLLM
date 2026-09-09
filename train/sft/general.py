from project_paths import legacy_path, path as project_path
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import math
import json
import time
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter
from safetensors.torch import load_file
from datasets import load_from_disk
from transformers import AutoTokenizer

from model.config import Config
from model.model import TinyLLM
from train.pretrain.math_mask import (
    build_mathish_masks,
    build_per_id_weight_from_masks,
    preview_mathish_tokens,
)

# =====================
# 环境 / 全局配置
# =====================

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
GRAD_ACCUM_STEPS = int(os.environ.get("SFT_GRAD_ACCUM_STEPS", "2"))
try:
    from torch.backends.cuda import sdp_kernel
    sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True)
except Exception:
    pass

IGNORE_INDEX = -100

# ---- 这些路径需要你按自己机器改一下 ----
RESUME_ROOT = legacy_path("/root/autodl-tmp/llm/tiny_05B_cpt3/checkpoint-648000")  # 从哪个 CPT ckpt 接着 SFT
OUT_DIR     = str(project_path("runs/training/text_sft"))                     # SFT 输出目录

# 两个 SFT 数据集：short (<=2048) / long (<=16k) —— 你自己准备好的 HF Dataset 路径
SFT_CACHE_SHORT = legacy_path("/root/autodl-tmp/llm/cache_sft/short_varlen")   # 改成真实路径
SFT_CACHE_LONG  = legacy_path("/root/autodl-tmp/llm/cache_sft/long_varlen")    # 改成真实路径
SFT_EVAL_FRACTION = float(os.environ.get("SFT_EVAL_FRACTION", "0.0001"))  # 默认 0.01%
SFT_EVAL_MIN = int(os.environ.get("SFT_EVAL_MIN", "2000"))    # 至少多少条 eval 样本
SFT_EVAL_MAX = int(os.environ.get("SFT_EVAL_MAX", "6000"))   # 最多多少条 eval 样本
# tokenizer 快照（通常跟 CPT 同一个目录，或单独 copy 出来的）
LOCAL_SNAPSHOT  = legacy_path("/root/autodl-tmp/llm/KDmodel/minicpm3_4b/snapshot")

LOG_DIR = os.path.join(legacy_path("/root/autodl-tmp/llm/checkpoints"), "tb",
                       f"tinyllm-sft-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# =====================
# SFT 超参
# =====================
SEED = 10
USE_BF16 = True
AUTOCAST_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16

LR_BASE = 1.5e-5
WARMUP_STEPS = 500
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.01
EPOCHS = 1

# 每个 rank 上的 token budget（大概值，跑一趟看显存再调）
LR_MIN_FACTOR = float(os.environ.get("SFT_LR_MIN_FACTOR", "0.2"))
MAX_TOKENS_PER_BATCH_SHORT = int(os.environ.get("SFT_MAX_TOKENS_PER_BATCH_SHORT", "6144"))
MAX_TOKENS_PER_BATCH_LONG  = int(os.environ.get("SFT_MAX_TOKENS_PER_BATCH_LONG",  "16384"))

# 调度：多少个 short step 插一次 long step
LONG_EVERY_K_SHORT = int(os.environ.get("SFT_LONG_EVERY_K_SHORT", "50"))

# 日志&保存
LOG_SCALAR_EVERY = 20
SAVE_INTERVAL_STEPS = 2500
START_TIME = time.time()


# =====================
# 小工具
# =====================
def prune_old_checkpoints(base_dir: str, max_keep: int = 2):
    """
    只清理形如 checkpoint-xxxx 的目录，最多保留最近 max_keep 个。
    epoch1-stepX 这类按需自己删。
    """
    if max_keep <= 0:
        return

    if not os.path.isdir(base_dir):
        return

    subdirs = []
    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if os.path.isdir(full) and name.startswith("checkpoint-"):
            try:
                step = int(name.split("-")[-1])
            except ValueError:
                continue
            subdirs.append((step, full))

    if len(subdirs) <= max_keep:
        return

    # 按 step 排序，保留最后 max_keep 个
    subdirs.sort(key=lambda x: x[0])
    to_delete = subdirs[:-max_keep]

    for step, path in to_delete:
        try:
            shutil.rmtree(path)
            log(f"[PRUNE] remove old checkpoint step={step} path={path}")
        except Exception as e:
            log(f"[PRUNE] failed to remove {path}: {e}")
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


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    return get_rank() == 0


def to_float(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return float(x.detach().item())
    if isinstance(x, (int, float)):
        return float(x)
    return None


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


# 🔹 新增：ETA 打印工具
def log_eta(global_step: int, total_steps_est: int):
    """
    简单 ETA 估计：
      - total_steps_est：预估的总 step 数（short+long）
      - global_step：当前已经走到第几个 step
    只在主进程调用即可。
    """
    if total_steps_est <= 0:
        return
    done = max(1, min(global_step, total_steps_est))
    frac = done / total_steps_est
    elapsed = time.time() - START_TIME
    if frac > 0:
        total_est = elapsed / frac
        remaining = max(0.0, total_est - elapsed)
    else:
        remaining = 0.0
    log(f"[ETA] progress={done}/{total_steps_est} ({frac*100:.2f}%) "
        f"elapsed={_fmt_hms(elapsed)} remaining≈{_fmt_hms(remaining)}")


# =====================
# RoPE state 清理（复用你预训练里的逻辑）
# =====================
ROPE_OWNER_PAT = re.compile(r"\.(rotary_emb|rotary|rope|rope_emb)\.")
ROPE_LEAF_ALLOW = {"inv_freq"}
ROPE_LEAF_DROP = {
    "cos_cached", "sin_cached", "cos_cached_long", "sin_cached_long",
    "max_seq_len_cached", "_seq_len_cached", "seq_len_cached",
    "cached_seq_len", "cached_max_seq_len",
    "base", "theta", "rope_base", "rope_theta",
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
def save_tokenizer_files(tokenizer, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    tokenizer.save_pretrained(out_dir)


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
    from safetensors.torch import save_file
    save_file(state, os.path.join(out_dir, "model.safetensors"))
    if cfg_obj is not None:
        with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(getattr(cfg_obj, "__dict__", {}), f, ensure_ascii=False, indent=2)


# =====================
# 分层学习率 groups
# =====================
_BLOCK_RE = re.compile(r"^(?:module\.)?blocks\.(\d+)\.")


def param_groups_layerwise(nn_module, weight_decay: float):
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
            lr_scale = 1.0
            tag = "blocks"
        else:
            lr_scale = 1.0
            tag = "nonblocks"

        is_no_decay = (p.ndim == 1) or any(k in name for k in [
            "norm", "bias", "tok_embed", "gamma_att", "gamma_mlp",
            "tcc_gate", "router", "decider", "temp_param", "embedding", "embed", "lm_head"
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
            "group_tag": tag,
        })
    return param_groups


# =====================
# Tokenizer & Model
# =====================
def build_tokenizer_and_model():
    assert os.path.isdir(LOCAL_SNAPSHOT), f"Tokenizer dir not found: {LOCAL_SNAPSHOT}"
    assert os.path.isfile(os.path.join(LOCAL_SNAPSHOT, "tokenizer.json")), "Missing tokenizer.json"

    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_SNAPSHOT,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer.get_vocab())

    # mathish 权重（沿用你 pretrain 的做法）
    masks = build_mathish_masks(tokenizer)
    per_id_w = build_per_id_weight_from_masks(masks, w_mathish=1.2, cap=3.0)
    if is_main_process():
        _ = preview_mathish_tokens(tokenizer, masks, k=100)

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
    )
    cfg.use_checkpoint = False
    cfg.checkpoint_use_reentrant = False

    model = TinyLLM(cfg)
    model.register_buffer("class_w_mathish", per_id_w, persistent=False)

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

    return tokenizer, model


# =====================
# SFT Collator (右 pad)
# =====================
class SFTCollatorRightPad:
    """
    - 右侧 padding
    - attention_mask 只根据真实长度来造（和 pad_id 无关）
    - 构造 autoregressive labels:
        labels[t] = input_ids[t+1]
      然后用两层 mask:
        1) 长度 / padding：t >= length → IGNORE_INDEX
        2) target_mask: 只在“下一个 token 属于 assistant”时监督
    """

    def __init__(self, tokenizer, ignore_index: int = IGNORE_INDEX):
        self.tok = tokenizer
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.ignore_index = ignore_index

    def __call__(self, batch):
        seqs: List[torch.Tensor] = []
        lengths: List[int] = []
        is_math_flags: List[float] = []
        tgt_masks: List[torch.Tensor] = []

        for ex in batch:
            ids = ex["input_ids"]
            if not torch.is_tensor(ids):
                ids = torch.tensor(ids, dtype=torch.long)

            # 长度：优先用 length 字段，否则 fallback 为 len(ids)
            L = int(ex.get("length", ids.numel()))
            seqs.append(ids)
            lengths.append(L)

            is_math = 1.0 if ex.get("isMath", False) else 0.0
            is_math_flags.append(is_math)

            # target_mask：如果不存在，就退化成全 1（全监督）
            if "target_mask" in ex:
                tm = ex["target_mask"]
                if not torch.is_tensor(tm):
                    tm = torch.tensor(tm, dtype=torch.long)
            else:
                tm = torch.ones_like(ids, dtype=torch.long)
            tgt_masks.append(tm)

        B = len(seqs)
        max_len = max(lengths) if lengths else 1

        # ========= 右 pad input_ids / target_mask =========
        input_ids = torch.full((B, max_len), self.pad_id, dtype=torch.long)
        tgt_pad   = torch.zeros((B, max_len), dtype=torch.long)

        for i, (seq, L, tm) in enumerate(zip(seqs, lengths, tgt_masks)):
            L_clamp = min(L, seq.size(0), tm.size(0))
            input_ids[i, :L_clamp] = seq[:L_clamp]
            tgt_pad[i, :L_clamp]   = tm[:L_clamp]

        # ========= attention_mask：只根据 length 来造 =========
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        for i, L in enumerate(lengths):
            attention_mask[i, :L] = 1

        # ========= labels：标准自回归 + 双重屏蔽 =========
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = self.ignore_index  # 最后一位没有下一个 token

        # 1) 长度 / padding 屏蔽
        labels[attention_mask == 0] = self.ignore_index

        # 2) target_mask → label_mask（向右平移一格）
        #   label_mask[t] = 1  表示  labels[t] = input_ids[t+1] 这个目标属于 assistant 段
        label_mask = torch.zeros_like(tgt_pad)
        label_mask[:, :-1] = tgt_pad[:, 1:]
        labels[label_mask == 0] = self.ignore_index

        lengths_t = torch.tensor(lengths, dtype=torch.long)
        is_math   = torch.tensor(is_math_flags, dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "length": lengths_t,
            "is_math": is_math,
        }


# =====================
# 动态 batch sampler (DDP)
# =====================
class DynamicBatchSamplerDDP(Sampler):
    """
    按 token 数动态组 batch，并做 DDP 切分。

    - lengths: List[int]，每个样本的长度（不含 pad）
    - max_tokens_per_batch: 每个 rank 上一个 step 近似的 token 上限
    """

    def __init__(
        self,
        lengths,
        max_tokens_per_batch: int,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 42,
        drop_last: bool = True,
    ):
        self.lengths = list(map(int, lengths))
        self.N = len(self.lengths)
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
        # 1) 打乱所有样本索引（所有 rank 相同顺序）
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.N, generator=g).tolist()

        # 2) greedy 组 batch
        global_batches: List[List[int]] = []
        cur_batch: List[int] = []
        cur_max_len = 0

        for idx in indices:
            L = self.lengths[idx]
            if L <= 0:
                continue
            new_max_len = max(cur_max_len, L)
            est_tokens = (len(cur_batch) + 1) * new_max_len

            if cur_batch and est_tokens > self.max_tokens:
                # 当前 batch 满了
                global_batches.append(cur_batch)
                cur_batch = [idx]
                cur_max_len = L
            else:
                cur_batch.append(idx)
                cur_max_len = new_max_len

        if cur_batch and (not self.drop_last or len(cur_batch) > 0):
            global_batches.append(cur_batch)

        # 3) DDP: 每个 rank 拿一部分 batch
        for i, batch in enumerate(global_batches):
            if (i % self.world_size) == self.rank:
                yield batch

    def __len__(self):
        # 粗略估计一下（用于进度条，不要求精确）
        if self.N == 0:
            return 0
        median_len = int(np.median(self.lengths))
        approx_bs = max(1, self.max_tokens // max(1, median_len))
        return max(1, (self.N // self.world_size) // approx_bs)



# =====================
def build_sft_datasets():
    ds_short_all = load_from_disk(SFT_CACHE_SHORT)
    ds_long = load_from_disk(SFT_CACHE_LONG)

    def ensure_length_col(ds):
        if "length" in ds.column_names:
            return ds
        return ds.map(
            lambda ex: {"length": len(ex["input_ids"])},
            num_proc=4,
            desc="add length column",
        )

    ds_short_all = ensure_length_col(ds_short_all)
    ds_long = ensure_length_col(ds_long)

    # 先 shuffle 一下 short，再切一块出来做 eval，剩下当 train
    n_total = len(ds_short_all)
    ds_short_all = ds_short_all.shuffle(seed=SEED)

    # 计算 eval 数量：按比例 + [SFT_EVAL_MIN, SFT_EVAL_MAX] 限制
    n_eval_by_frac = int(n_total * SFT_EVAL_FRACTION)
    n_eval = max(SFT_EVAL_MIN, n_eval_by_frac)
    n_eval = min(n_eval, SFT_EVAL_MAX, n_total - 1)  # 至少留 1 条给 train

    ds_eval_short = ds_short_all.select(range(n_eval))
    ds_short_train = ds_short_all.select(range(n_eval, n_total))

    if is_main_process():
        log(
            f"[DATA] short_total={n_total}, eval_short={len(ds_eval_short)}, "
            f"train_short={len(ds_short_train)}, long={len(ds_long)}"
        )

    # ✅ 把 target_mask 也一起拿出来
    cols_short = [c for c in ds_short_train.column_names
                  if c in ("input_ids", "length", "isMath", "target_mask")]
    cols_eval = [c for c in ds_eval_short.column_names
                 if c in ("input_ids", "length", "isMath", "target_mask")]
    cols_long = [c for c in ds_long.column_names
                 if c in ("input_ids", "length", "isMath", "target_mask")]

    ds_short_train = ds_short_train.with_format("torch", columns=cols_short)
    ds_eval_short = ds_eval_short.with_format("torch", columns=cols_eval)
    ds_long = ds_long.with_format("torch", columns=cols_long)

    # 返回：训练用 short、long + eval short
    return ds_short_train, ds_long, ds_eval_short


# =====================
# SFT eval（直接从 SFT_CACHE_SHORT 抽样；math / non-math 分开）
# =====================

def _make_eval_loader(dataset, tokenizer, max_tokens_per_batch: int, seed: int = 42):
    """
    eval 也用 DynamicBatchSamplerDDP，区别是 drop_last=False。
    """
    lengths = dataset["length"]
    if torch.is_tensor(lengths):
        lengths_list = [int(x) for x in lengths.tolist()]
    else:
        lengths_list = [int(x) for x in lengths]

    sampler = DynamicBatchSamplerDDP(
        lengths=lengths_list,
        max_tokens_per_batch=max_tokens_per_batch,
        world_size=get_world_size(),
        rank=get_rank(),
        seed=seed,
        drop_last=False,   # eval 不丢最后一个 batch
    )

    collator = SFTCollatorRightPad(tokenizer)

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,
    )
    return loader

def eval_sft_short_math_lang(model, tokenizer, ds_eval_short):
    """
    只在事先切好的 ds_eval_short 上评估；
    再按 isMath 拆成 math / non-math 两部分。

    返回: (math_ce, lang_ce)
    """
    device = next(model.parameters()).device

    # ---- 1. 按 isMath 拆成 math / non-math 子集 ----
    ds_math = ds_eval_short.filter(lambda ex: bool(ex.get("isMath", False)), num_proc=4)
    ds_lang = ds_eval_short.filter(lambda ex: not bool(ex.get("isMath", False)), num_proc=4)

    if is_main_process():
        log(f"[EVAL] eval_total={len(ds_eval_short)}, math={len(ds_math)}, lang={len(ds_lang)}")

    def _safe_loader(ds, name: str):
        if len(ds) == 0:
            if is_main_process():
                log(f"[EVAL-{name}] dataset empty, skip")
            return None
        return _make_eval_loader(ds, tokenizer, max_tokens_per_batch=MAX_TOKENS_PER_BATCH_SHORT)

    loader_math = _safe_loader(ds_math, "MATH")
    loader_lang = _safe_loader(ds_lang, "LANG")

    def _run_one_split(loader, name: str):
        if loader is None:
            return float("nan")

        model.eval()
        loss_times_tokens = torch.zeros(1, device=device, dtype=torch.float32)
        token_count = torch.zeros(1, device=device, dtype=torch.float32)

        with torch.no_grad():
            for batch in loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device, non_blocking=True)

                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    enhance_math=False,   # ✅ eval 不加数学权重，公平比较
                )
                ce = out["ce_loss"].to(torch.float32)
                valid_tokens = (batch["labels"] != IGNORE_INDEX).sum().to(torch.float32)

                loss_times_tokens += ce * valid_tokens
                token_count += valid_tokens

        # DDP：所有卡聚合
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss_times_tokens, op=dist.ReduceOp.SUM)
            dist.all_reduce(token_count, op=dist.ReduceOp.SUM)

        avg_ce = (loss_times_tokens / token_count.clamp_min(1.0)).item()
        if is_main_process():
            log(f"[EVAL-{name}] ce={avg_ce:.4f}, tokens={int(token_count.item())}")
        return avg_ce

    math_ce = _run_one_split(loader_math, "MATH")
    lang_ce = _run_one_split(loader_lang, "LANG")

    model.train()
    return math_ce, lang_ce


def make_sft_loader(dataset, tokenizer, max_tokens_per_batch: int, seed: int = 42, drop_last: bool = True):
    lengths = dataset["length"]  # 这是 torch.Tensor / list[Int]

    # 把 lengths 转为 python list[int]
    if torch.is_tensor(lengths):
        lengths_list = [int(x) for x in lengths.tolist()]
    else:
        lengths_list = [int(x) for x in lengths]

    sampler = DynamicBatchSamplerDDP(
        lengths=lengths_list,
        max_tokens_per_batch=max_tokens_per_batch,
        world_size=get_world_size(),
        rank=get_rank(),
        seed=seed,
        drop_last=drop_last,
    )

    collator = SFTCollatorRightPad(tokenizer)

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    return loader, sampler


# =====================
# 训练 & LR schedule
# =====================
def sft_lr_schedule(global_step: int, total_steps_est: int) -> float:
    """
    warmup + 余弦退火：
      - 0 ~ WARMUP_STEPS: 线性从 0 -> LR_BASE
      - 之后: 从 LR_BASE 余弦退火到 LR_BASE * LR_MIN_FACTOR
    total_steps_est 是预估的总 update 数（不是 micro step）
    """
    # ------ warmup ------
    if global_step < WARMUP_STEPS:
        return LR_BASE * float(global_step + 1) / float(max(1, WARMUP_STEPS))

    # ------ 退火阶段 ------
    # 防止 total_steps_est 比 warmup 小/相等导致除零
    if total_steps_est <= WARMUP_STEPS + 1:
        return LR_BASE * LR_MIN_FACTOR

    # 退火从 step = WARMUP_STEPS 开始
    t = max(0, min(global_step - WARMUP_STEPS, total_steps_est - WARMUP_STEPS))
    T = max(1, total_steps_est - WARMUP_STEPS)

    # 余弦系数：从 1 缓慢降到 ~0
    cos_factor = 0.5 * (1.0 + math.cos(math.pi * t / T))

    lr_max = LR_BASE
    lr_min = LR_BASE * LR_MIN_FACTOR
    lr = lr_min + (lr_max - lr_min) * cos_factor
    return lr


def train_sft():
    # ---- DDP 初始化 ----
    if not dist.is_initialized() and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # ---- 构建 tokenizer & model ----
    tokenizer, model = build_tokenizer_and_model()
    model.to(device)
    model.train()

    if get_world_size() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            static_graph=False,
        )

    # ---- 数据集 & DataLoader ----
    ds_short, ds_long, ds_eval_short = build_sft_datasets()
    loader_short, sampler_short = make_sft_loader(
        ds_short, tokenizer, max_tokens_per_batch=MAX_TOKENS_PER_BATCH_SHORT, seed=SEED
    )
    loader_long, sampler_long = make_sft_loader(
        ds_long, tokenizer, max_tokens_per_batch=MAX_TOKENS_PER_BATCH_LONG, seed=SEED
    )

    # 🔹 估算总 step 数（用来打 ETA）
    total_short_micro = len(loader_short) * EPOCHS
    if LONG_EVERY_K_SHORT > 0:
        total_long_micro = total_short_micro // LONG_EVERY_K_SHORT
    else:
        total_long_micro = 0
    total_micro = total_short_micro + total_long_micro
    total_steps_est = max(1, math.ceil(total_micro / GRAD_ACCUM_STEPS))

    if is_main_process():
        log(
            f"[SFT] est total updates ≈ {total_steps_est} "
            f"(micro_short≈{total_short_micro}, micro_long≈{total_long_micro}, "
            f"grad_accum={GRAD_ACCUM_STEPS})"
        )
    # ---- optimizer ----
    base_groups = param_groups_layerwise(model, WEIGHT_DECAY)
    fused_ok = torch.cuda.get_device_capability(local_rank)[0] >= 8
    optimizer = torch.optim.AdamW(
        base_groups,
        lr=LR_BASE,
        betas=(0.9, 0.95),
        fused=fused_ok,
        weight_decay=WEIGHT_DECAY,
    )

    writer = SummaryWriter(LOG_DIR) if is_main_process() else None
    if is_main_process():
        log(f"[TB] SummaryWriter -> {LOG_DIR}")

    global_step = 0
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(EPOCHS):
        sampler_short.set_epoch(epoch)
        sampler_long.set_epoch(epoch)

        short_iter = iter(loader_short)
        long_iter = iter(loader_long)

        while True:
            # ========= 1) short step（支持梯度累积） =========
            try:
                batch = next(short_iter)
            except StopIteration:
                # 当前 epoch 的 short 用完
                break

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=AUTOCAST_DTYPE):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    enhance_math=True,   # SFT 也保留 mathish 加权
                    math_sample_mask=batch["is_math"],
                )
                # 🔹 梯度累积分摊 loss
                loss = out["loss"] / GRAD_ACCUM_STEPS

            loss.backward()
            micro_step += 1
            do_update = (micro_step % GRAD_ACCUM_STEPS == 0)

            if do_update:
                gnorm = grad_total_norm(model.parameters(), norm_type=2.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

                lr_now = sft_lr_schedule(global_step, total_steps_est)
                for pg in optimizer.param_groups:
                    s = pg.get("lr_scale", 1.0)
                    pg["lr"] = lr_now * s

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if writer is not None and (global_step % LOG_SCALAR_EVERY == 0):
                    ce = to_float(out["ce_loss"]) if "ce_loss" in out else to_float(loss) * GRAD_ACCUM_STEPS
                    writer.add_scalar("sft/short_ce", ce, global_step)
                    writer.add_scalar("sft/grad_norm", gnorm, global_step)
                    writer.add_scalar("sft/lr", lr_now, global_step)

                if is_main_process() and (global_step % (LOG_SCALAR_EVERY * 5) == 0):
                    ce = to_float(out["ce_loss"]) if "ce_loss" in out else to_float(loss) * GRAD_ACCUM_STEPS
                    log(f"[SFT] step={global_step} epoch={epoch} short_ce={ce:.4f} lr={lr_now:.2e} gnorm={gnorm:.2f}")
                    log_eta(global_step, total_steps_est)

                if (global_step % SAVE_INTERVAL_STEPS == 0):
                    # ⭐⭐ 所有 rank 一起进 eval（里面有 all_reduce）
                    try:
                        math_ce, lang_ce = eval_sft_short_math_lang(
                            model.module if hasattr(model, "module") else model,
                            tokenizer,
                            ds_eval_short,
                        )

                        if is_main_process() and writer is not None:
                            writer.add_scalar("eval_short/math_ce", math_ce, global_step)
                            writer.add_scalar("eval_short/lang_ce", lang_ce, global_step)
                    except Exception as e:
                        if is_main_process():
                            log(f"[EVAL] failed: {e}")

                    # ⭐⭐ 只在主进程保存和清理 checkpoint
                    if is_main_process():
                        ckpt_dir = os.path.join(OUT_DIR, f"checkpoint-{global_step}")
                        os.makedirs(ckpt_dir, exist_ok=True)
                        model_to_save = model.module if hasattr(model, "module") else model
                        safe_save_full_model(model_to_save, ckpt_dir, cfg_obj=getattr(model_to_save, "cfg", None))
                        save_tokenizer_files(tokenizer, ckpt_dir)
                        log(f"[SAVE] wrote {ckpt_dir}")
                        prune_old_checkpoints(OUT_DIR, max_keep=2)
            # ========= 2) 每 K 个 short step 插一个 long step =========
            if LONG_EVERY_K_SHORT > 0 and (global_step % LONG_EVERY_K_SHORT == 0):
                try:
                    batch_long = next(long_iter)
                except StopIteration:
                    long_iter = iter(loader_long)
                    batch_long = next(long_iter)

                for k, v in batch_long.items():
                    if isinstance(v, torch.Tensor):
                        batch_long[k] = v.to(device, non_blocking=True)

                with torch.autocast(device_type="cuda", dtype=AUTOCAST_DTYPE):
                    out_long = model(
                        input_ids=batch_long["input_ids"],
                        attention_mask=batch_long["attention_mask"],
                        labels=batch_long["labels"],
                        enhance_math=False,
                        force_checkpoint=True,
                    )
                    # 🔹 同样分摊 loss
                    loss_long = out_long["loss"] / GRAD_ACCUM_STEPS

                loss_long.backward()
                micro_step += 1
                do_update_long = (micro_step % GRAD_ACCUM_STEPS == 0)

                if do_update_long:
                    gnorm_long = grad_total_norm(model.parameters(), norm_type=2.0)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

                    lr_now = sft_lr_schedule(global_step, total_steps_est)
                    for pg in optimizer.param_groups:
                        s = pg.get("lr_scale", 1.0)
                        pg["lr"] = lr_now * s

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if writer is not None and (global_step % LOG_SCALAR_EVERY == 0):
                        ce_l = to_float(out_long["ce_loss"]) if "ce_loss" in out_long else to_float(loss_long) * GRAD_ACCUM_STEPS
                        writer.add_scalar("sft/long_ce", ce_l, global_step)
                        writer.add_scalar("sft/grad_norm_long", gnorm_long, global_step)

                    if is_main_process():
                        ce_l = to_float(out_long["ce_loss"]) if "ce_loss" in out_long else to_float(loss_long) * GRAD_ACCUM_STEPS
                        log(f"[SFT-LONG] step={global_step} epoch={epoch} long_ce={ce_l:.4f} lr={lr_now:.2e} gnorm={gnorm_long:.2f}")
                        log_eta(global_step, total_steps_est)



        # 每个 epoch 结束后额外存一份
        if is_main_process():
            ckpt_dir = os.path.join(OUT_DIR, f"epoch{epoch+1}-step{global_step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model_to_save = model.module if hasattr(model, "module") else model
            safe_save_full_model(model_to_save, ckpt_dir, cfg_obj=getattr(model_to_save, "cfg", None))
            save_tokenizer_files(tokenizer, ckpt_dir)
            log(f"[EPOCH SAVE] {ckpt_dir} completed epoch={epoch+1}")

    if writer is not None:
        writer.close()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    train_sft()
