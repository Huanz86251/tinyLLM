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
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler, default_collate
import torch.distributed as dist
from safetensors.torch import save_file
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from typing import Dict
from model.config import Config
from model.model import TinyLLM
from train.pretrain.kd_reader import KDFetcher
import time
from datetime import datetime
# ============================================================
# 环境 / 性能选项
# ============================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
from torch.backends.cuda import sdp_kernel
sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
SAVE_TOTAL_LIMIT = int(os.environ.get("SAVE_TOTAL_LIMIT", "2"))
CKPT_PREFIX = "checkpoint-"
LOG_INTERVAL_LONG_STEPS = int(os.environ.get("LOG_INTERVAL_LONG_STEPS", "10"))
LOG_SCALAR_EVERY = int(os.environ.get("LOG_SCALAR_EVERY", "20"))
USE_BF16 = True
AUTOCAST_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
CHECKPOINT_BARRIER = int(os.environ.get("CHECKPOINT_BARRIER", "1"))
BARRIER_EVERY_N = int(os.environ.get("BARRIER_EVERY_N", "0"))
USE_CHECKPOINT=False
# 我们会按 world_size 确定全局 batch/token吞吐
# 需要 DDP: WORLD_SIZE / RANK / LOCAL_RANK 由 torchrun/accelerate 启动时给
# 例子: torchrun --nproc_per_node=4 train_mix.py
# ============================================================

# ========== 基本训练超参(短模式/主模式) ==========
MAX_LEN_SHORT = 2048
BATCH_SHORT = 3                        # per-device
GRAD_ACCUM_SHORT = 3
LR_BASE = 1.8e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.2
EPOCHS = 3                             # 逻辑epoch(我们用短池大小来界定走几轮)

KD_DIR = legacy_path("/root/autodl-tmp/llm/data/kd_lmdb_minicpm3_4b")

# ========== 长上下文子模式 ==========
MAX_LEN_LONG = 8192
BATCH_LONG = 1                         # per-device
GRAD_ACCUM_LONG = 2
LONG_LR_SCALE = 1.0                  # 长步时 LR = LR_BASE * 1

# ========== 采样调度 ==========
SHORT_STEPS_PER_CYCLE = 20             # 20个短step
LONG_STEPS_PER_CYCLE  = 1              # 然后1个长step
# 最终长step占比 ≈ 1 / (20+1) ~= 4.8%

# ========== 评估 / 日志 / 保存 ==========
LOG_INTERVAL_STEPS = 200               # 打印/日志间隔(按optimizer step计)
SAVE_INTERVAL_STEPS = 4000             # checkpoint间隔(按optimizer step计)
EVAL_INTERVAL_STEPS = 4000             # eval间隔(按optimizer step计)

RUN_ID   = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_NAME = f"tinyllm-mixctx-{RUN_ID}"
OUT_DIR  = str(project_path("runs/training/cpt_mixed"))   # 新输出目录(不要和旧目录互相覆盖)
LOG_DIR  = os.path.join(legacy_path("/root/autodl-tmp/llm/checkpoints"), "tb", RUN_NAME)
os.makedirs(LOG_DIR, exist_ok=True)

# ========== 数据路径 ==========
LOCAL_SNAPSHOT      = legacy_path("/root/autodl-tmp/llm/KDmodel/minicpm3_4b/snapshot")
PK_CACHE_SHORT_KD   = legacy_path("/root/autodl-tmp/llm/cache_pretrain_large/packed_len2048")
PK_CACHE_SHORT_PLAIN= legacy_path("/root/autodl-tmp/llm/cache_pretrain_large/packed_len2048_more")
PK_CACHE_LONG       = legacy_path("/root/autodl-tmp/llm/cache_pretrain_large/packed_len8192")

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

# ============================================================
# 杂项小工具: 主进程判断 / 安全保存
# ============================================================


START_TIME = time.time()
def to_float(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return float(x.detach().item())
    if isinstance(x, (int, float)):     # ← 新增：直接返回数字
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
        if not step_str.isdigit():   # 跳过 final-* 等
            continue
        items.append((int(step_str), os.path.join(base_dir, name)))
    items.sort(key=lambda x: x[0])   # 按 step 升序
    for _, p in items[:-keep]:       # 删掉较旧的
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
    # 主进程打印 + 时间戳 + 累计用时
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed = time.time() - START_TIME
    print(f"[{ts}] (+{_fmt_hms(elapsed)}) {msg}", flush=True)

class ETAHelper:
    """
    以“optimizer step”为单位做 EMA 估计，给出剩余步数的 ETA；
    同时做 steps/s 和 tokens/s 的 EMA。
    """
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

        # 更新 EMA
        if self._ema_step_time is None:
            self._ema_step_time = dt
        else:
            self._ema_step_time = self.ema * self._ema_step_time + (1 - self.ema) * dt

        self._ema_tokens_per_step = self.ema * self._ema_tokens_per_step + (1 - self.ema) * float(tokens_this_step)

        # 全局 step +1
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

    # 防止 "0.xxx" 键名问题
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

ROPE_OWNER_PAT = re.compile(r"\.(rotary_emb|rotary|rope|rope_emb)\.")  # 覆盖常见命名
ROPE_LEAF_ALLOW = {
    # 真正会被 init 的“结构性”字段（如果是参数/缓冲）
    "inv_freq",
}
ROPE_LEAF_DROP = {
    # 一切缓存/长度标记等，统统丢弃
    "cos_cached", "sin_cached", "cos_cached_long", "sin_cached_long",
    "max_seq_len_cached", "_seq_len_cached", "seq_len_cached",
    "cached_seq_len", "cached_max_seq_len",
    # 有些实现会把 base/theta 也当 buffer 放在 rope 子模块里
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
    # 恢复 RNG
    try:
        random.setstate(st["rng"]["python"])
        np.random.set_state(st["rng"]["numpy"])
        torch.set_rng_state(st["rng"]["torch"])
        if torch.cuda.is_available() and st["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(st["rng"]["cuda"])
    except Exception as e:
        print(f"[RESUME] RNG restore failed: {e}")
    return int(st.get("global_optimizer_step", 0)), int(st.get("epoch", 0))

def drop_rope_buffers_from_state(state: Dict[str, torch.Tensor]) -> int:
    """删除 state_dict 中所有 RoPE 子模块下的缓存/缓冲字段。返回删除的 key 数量。"""
    to_drop = []
    for k in state.keys():
        if ROPE_OWNER_PAT.search(k):
            leaf = k.rsplit(".", 1)[-1]
            if (leaf in ROPE_LEAF_DROP) or (leaf not in ROPE_LEAF_ALLOW):
                # 默认保守：除了 allow 的，其余都丢（防止奇怪的实现把缓存名换了）
                to_drop.append(k)
    for k in to_drop:
        state.pop(k, None)
    return len(to_drop)
# ============================================================
# Collator: 短模式 vs 长模式
# ============================================================
class MixedCollator:
    """
    你的原版 collator，稍微整理了一点注释。
    短模式: kd_dir != None, p_kd ~1.0
    长模式: kd_dir=None, p_kd=0.0, batch_size更小
    """
    def __init__(
        self,
        tokenizer,
        kd_dir=None,
        p_denoise=0.0,
        p_kd=1.0,
        ignore_index=-100,
        span_ratio=(0.15,0.30),
        mask_token=None,
        kd_topk_use=16,
        kd_pos_keep=1.0,
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

        self.kd_dir = kd_dir
        self.kd_topk_use = kd_topk_use
        self.kd_pos_keep = kd_pos_keep

        self.shortlist_neg = shortlist_neg
        self.exclude_special_in_neg = exclude_special_in_neg
        self.enable_shortlist = enable_shortlist

        self.vocab_size = len(tokenizer.get_vocab())
        self.kd = None  # lazy init

        # 简易 unigram 抽样分布 (你给的是均匀 over 20k，作为负样本)
        probs = np.ones(20000, dtype=np.float64) / 20000
        self.unigram_probs = probs.astype(np.float64).copy()

        # 把特殊token概率清零再renorm
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
        # next-token prediction, 最后一位 label = ignore
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = self.ignore_index
        return input_ids, labels

    def __call__(self, batch):
        # kd fetcher lazy init
        if self.kd is None and self.kd_dir is not None:
            self.kd = KDFetcher(self.kd_dir)

        input_ids = default_collate([e["input_ids"] for e in batch]).long()
        ids, lbs = self._ar(input_ids)
        B, T = ids.size()

        out = {"input_ids": ids, "labels": lbs}

        # ========== KD 支持 (仅短模式生效) ==========
        use_kd_any = False
        kd_idx_list, kd_val_list, kd_mask_list = [], [], []
        K = self.kd_topk_use

        if (self.kd is not None) and (self.p_kd > 0.0) and (self.kd_topk_use > 0):
            for b in range(B):
                if random.random() < self.p_kd:
                    fetched = self.kd.get(ids[b].tolist())
                    if fetched is not None:
                        top_idx_np, top_val_np = fetched
                        # 截断 topk
                        if top_idx_np.shape[1] > K:
                            top_idx_np = top_idx_np[:, :K]
                            top_val_np = top_val_np[:, :K]
                        keep = (np.random.rand(T) < self.kd_pos_keep) if (self.kd_pos_keep < 1.0) else np.ones(T, bool)

                        kd_idx_list.append(torch.tensor(top_idx_np, dtype=torch.long))
                        kd_val_list.append(torch.tensor(top_val_np, dtype=torch.float32))
                        kd_mask_list.append(torch.from_numpy(keep).bool())
                        use_kd_any = True
                        continue
                # no kd for this sample
                kd_idx_list.append(None)
                kd_val_list.append(None)
                kd_mask_list.append(None)

        if use_kd_any:
            for i in range(B):
                if kd_idx_list[i] is None:
                    kd_idx_list[i] = torch.zeros((T, K), dtype=torch.long)
                    kd_val_list[i] = torch.zeros((T, K), dtype=torch.float32)
                    kd_mask_list[i] = torch.zeros((T,), dtype=torch.bool)

            kd_idx_btK = torch.stack(kd_idx_list, dim=0)   # [B,T,K]
            kd_val_btK = torch.stack(kd_val_list, dim=0)   # [B,T,K]
            kd_mask_bt = torch.stack(kd_mask_list, dim=0)  # [B,T]

            out["kd_idx"]  = kd_idx_btK
            out["kd_val"]  = kd_val_btK
            out["kd_mask"] = kd_mask_bt

        # 负样本 shortlist 逻辑（你原本保留的那套）；长模式我们其实不太依赖它，
        # 但保留没坏处
        gold = lbs.clone()
        mask_ign = (gold == -100)
        gold[mask_ign] = ids[mask_ign]
        gold = gold.long()
        gold_col = torch.zeros_like(gold)

        # 如果我们有 kd_idx_btK，否则空
        if use_kd_any:
            kd_width = self.kd_topk_use
            kd_idx_btK_final = kd_idx_btK
        else:
            kd_width = 0
            kd_idx_btK_final = torch.empty((B, T, 0), dtype=torch.long)

        # 负采样
        K_neg = 0
        if K_neg > 0:
            # 注意：self._torch_unigram_p 在CPU，这里multinomial必须在同device的话我们后面可以.to(device)再用。
            neg_flat = torch.multinomial(
                self._torch_unigram_p,
                num_samples=B * T * K_neg,
                replacement=True
            )
            neg = neg_flat.view(B, T, K_neg).long()
        else:
            neg = torch.empty((B, T, 0), dtype=torch.long)

        short_all = torch.cat([gold.unsqueeze(-1), kd_idx_btK_final, neg], dim=-1)
        Kp = short_all.size(-1)

        short_logq = torch.zeros((B, T, Kp), dtype=torch.float32)
        neg_start = 1 + kd_width
        if K_neg > 0 and neg_start < Kp:
            # 只给负样本列赋 log q
            # 注意：logq_vec在CPU，后面我们会在to(device)后再用. 这里先放CPU版
            # 我们在训练step里会把 short_logq.to(device) 再算loss
            idx_slice = short_all[..., neg_start:]
            # 这里不能直接索引 logq_vec[idx_slice] 因为device不同；延后到train_step
            # 我们只先存 short_all / gold_col，后面再处理
        out["short_idx"]   = short_all
        out["gold_col"]    = gold_col
        out["short_logq"]  = short_logq  # 会在train_step里搬到GPU并填充
        return out


# ============================================================
# 组装 tokenizer / 模型 / 数据
# ============================================================
def build_tokenizer_and_model():
    # tokenizer
    TOK_DIR = LOCAL_SNAPSHOT
    assert os.path.isdir(TOK_DIR), f"Tokenizer dir not found: {TOK_DIR}"
    assert os.path.isfile(os.path.join(TOK_DIR, "tokenizer.json")), "Missing tokenizer.json"

    tokenizer = AutoTokenizer.from_pretrained(
        TOK_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer.get_vocab())

    # 构建 config
    cfg = Config(
        vocab_size=vocab_size,
        train_maxlength=MAX_LEN_LONG,
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
    )

    # gradient checkpoint开关
    cfg.use_checkpoint = False
    cfg.checkpoint_use_reentrant = True

    model = TinyLLM(cfg)

    # 重要：Mamba块里原本的 embeddings 我们移除（你的原脚本里做了）
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

    # 加载已有checkpoint继续训
    # 这里我们假设你已经在 /root/autodl-tmp/llm/tiny_05B 里有之前的checkpoint
    RESUME_ROOT = legacy_path("/root/autodl-tmp/llm/tiny_05B")
    ckpt_path = get_last_checkpoint(RESUME_ROOT)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"[RESUME] 没有在 {RESUME_ROOT} 找到 checkpoint-* 目录，无法续训。"
        )
    # HF-style checkpoint: model weights在 pytorch_model.bin 或类似；你之前用 Trainer,
    # 加了我们safe_save_full_model()同步写了 model.safetensors
    # 我们优先尝试从 model.safetensors 里load
    safetensor_path = os.path.join(ckpt_path, "model.safetensors")
    if os.path.isfile(safetensor_path):
        state = load_file(safetensor_path)
        dropped = drop_rope_buffers_from_state(state)
        ret = model.load_state_dict(state, strict=False)
        print(f"[RESUME] loaded {safetensor_path} (dropped {dropped} rope-keys)")
        bad_missing = [k for k in ret.missing_keys if not ROPE_OWNER_PAT.search(k)]
        if bad_missing:
            raise RuntimeError(f"Unexpected missing keys (non-ROPE): {bad_missing[:8]}")
    else:
        # 回退到 Trainer 默认的pytorch_model.bin
        bin_path = os.path.join(ckpt_path, "pytorch_model.bin")
        state = torch.load(bin_path, map_location="cpu")
        dropped = drop_rope_buffers_from_state(state)
        ret = model.load_state_dict(state, strict=False)
        print(f"[RESUME] loaded {bin_path} (dropped {dropped} rope-keys)")
        bad_missing = [k for k in ret.missing_keys if not ROPE_OWNER_PAT.search(k)]
        if bad_missing:
            raise RuntimeError(f"Unexpected missing keys (non-ROPE): {bad_missing[:8]}")

    return tokenizer, model


def build_datasets(tokenizer):
    # 加载2048 KD + 2048 plain, concat
    ds_kd    = load_from_disk(PK_CACHE_SHORT_KD)
    ds_plain = load_from_disk(PK_CACHE_SHORT_PLAIN)
    ds_short_all = concatenate_datasets([ds_kd, ds_plain])

    # train/eval split (和你原来一样极小eval frac)
    eval_frac = 0.0001
    split = ds_short_all.train_test_split(
        test_size=eval_frac,
        seed=42
    )
    train_short = split["train"]
    eval_short  = split["test"]

    # long
    train_long = load_from_disk(PK_CACHE_LONG)

    eval_long = None

    # 设置成 torch 格式
    train_short = train_short.with_format("torch", columns=["input_ids"])
    eval_short  = eval_short.with_format("torch", columns=["input_ids"])
    train_long  = train_long.with_format("torch", columns=["input_ids"])

    return train_short, train_long, eval_short, eval_long


def make_loader(dataset, batch_size, collator, shuffle=True):
    """
    我们用 DistributedSampler 确保多卡DDP每张卡拿不同样本。
    drop_last=True 保证形状一致。
    """
    sampler = DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=shuffle,
        drop_last=True,
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
        drop_last=True,
    )
    return loader, sampler


# ============================================================
# 训练 step & eval step
# ============================================================
def forward_batch(model, batch, device, kd_collator_obj):
    """
    把collator产物搬到device上做前向，并补上shortlist的logq张量device对齐。
    返回 (loss, logging_info_dict)
    """
    # 把 batch 所有tensor都搬到device
    inputs = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device, non_blocking=True)
        else:
            inputs[k] = v
    if not getattr(kd_collator_obj, "enable_shortlist", False):
        inputs.pop("short_idx",  None)
        inputs.pop("short_logq", None)
        inputs.pop("gold_col",  None)

    model_outputs = model(**inputs)
    loss = model_outputs["loss"]

    # for logging
    ce_v = model_outputs.get("ce_loss", None)
    kd_v = model_outputs.get("kd_loss", None)
    aux_v= model_outputs.get("aux_loss", None)



    metrics = {
        "ce_loss": to_float(ce_v),
        "kd_loss": to_float(kd_v),
        "aux_loss": to_float(aux_v),
    }


    return loss, metrics


def run_eval(model, eval_loader, device, kd_collator_obj, max_batches=10):
    """
    简易eval: 只跑几批short eval看看loss趋势, 不做梯度。
    """
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=AUTOCAST_DTYPE):
        for i, batch in enumerate(eval_loader):
            if i >= max_batches:
                break
            loss, _ = forward_batch(model, batch, device, kd_collator_obj)
            bs = batch["input_ids"].size(0)
            total_loss += float(loss.item()) * bs
            total_count += bs
    model.train()
    if total_count == 0:
        return None
    return total_loss / total_count


# ============================================================
# 主训练循环
# ============================================================
def train_loop():
    # 准备分布式
    if not dist.is_initialized() and "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        # 假定torchrun已export了以下env: RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # tokenizer, model
    tokenizer, model = build_tokenizer_and_model()

    # DDP wrap
    model.to(device)
    model.train()
    if get_world_size() > 1:
        # 我们用 find_unused_parameters=False（和你原始Trainer一致）
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            static_graph=False,
        )
    if is_main_process():
        root = model.module if hasattr(model, "module") else model
        att0 = root.blocks[0].selfatt
        rb = getattr(att0, "rope_backend", None)
        if rb is not None:
            inv = getattr(rb, "inv_freq", None)
            base = getattr(rb, "base", getattr(rb, "theta", None))
            rope_cls = rb.__class__.__name__
            log(
                f"[rope] backend={rope_cls} type={getattr(root.cfg, 'rope_type', None)} "
                f"theta={getattr(root.cfg, 'rope_theta', None) or base} "
                f"inv_freq.shape={tuple(inv.shape) if inv is not None else None} "
                f"dtype={getattr(inv, 'dtype', None)} device={getattr(inv, 'device', None)}"
            )
        else:
            log("[rope] rope_backend has no persistent buffers (expected).")
    writer = SummaryWriter(log_dir=LOG_DIR, flush_secs=5, max_queue=20) if is_main_process() else None
    if writer is not None:
        log(f"[TB] SummaryWriter ready, writing heartbeat to {LOG_DIR}")
        # 心跳与元信息（立即触发创建 events 文件）
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
        # 可选：确认 events 文件是否出现
        try:
            from pathlib import Path
            ev = list(Path(LOG_DIR).glob("events*"))
            log(f"[TB] events files: {[p.name for p in ev] or 'None yet (will appear after first step)'}")
        except Exception:
            pass
    # 数据集和dataloader
    train_short, train_long, eval_short, _ = build_datasets(tokenizer)

    # collator 短模式 (KD开)
    collator_short = MixedCollator(
        tokenizer,
        kd_dir=KD_DIR,
        p_denoise=0.0,
        p_kd=1.0,
        kd_topk_use=16,
        kd_pos_keep=1.0,
        ignore_index=-100,
        enable_shortlist=False,
    )

    # collator 长模式 (KD关)
    collator_long = MixedCollator(
        tokenizer,
        kd_dir=None,          # 不读取KD
        p_denoise=0.0,
        p_kd=0.0,
        kd_topk_use=0,
        kd_pos_keep=0.0,
        ignore_index=-100,
        enable_shortlist=False,
    )

    loader_short, sampler_short = make_loader(
        train_short,
        batch_size=BATCH_SHORT,
        collator=collator_short,
        shuffle=True
    )
    loader_long, sampler_long   = make_loader(
        train_long,
        batch_size=BATCH_LONG,
        collator=collator_long,
        shuffle=True
    )
    eval_loader_short, _ = make_loader(
        eval_short,
        batch_size=BATCH_SHORT,
        collator=collator_short,
        shuffle=False
    )



    # optimizer
    def param_groups_no_decay(nn_module, weight_decay):
        decay, no_decay = [], []
        for n, p in nn_module.named_parameters():
            if not p.requires_grad:
                continue
            if (p.ndim == 1) or any(k in n for k in [
                "norm", "bias", "tok_embed", "gamma_att", "gamma_mlp",
                "tcc_gate", "router", "decider", "temp_param"
            ]):
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    optimizer = torch.optim.AdamW(
        param_groups_no_decay(model, WEIGHT_DECAY),
        lr=LR_BASE,
        betas=(0.9, 0.95),
        fused=True if torch.cuda.get_device_capability(local_rank)[0] >= 8 else False
    )

    # scheduler: cosine with min_lr = LR_BASE * 0.1 跟你之前的思路类似
    # 为简单起见，我们手写个warmup+cosine
    warmup_ratio = 0.01
    # 假设一个epoch相当于 len(loader_short)/GRAD_ACCUM_SHORT 个optimizer step
    steps_short_per_epoch = len(loader_short) // GRAD_ACCUM_SHORT
    steps_long_per_epoch = (steps_short_per_epoch // SHORT_STEPS_PER_CYCLE) * LONG_STEPS_PER_CYCLE
    steps_total_per_epoch = steps_short_per_epoch + steps_long_per_epoch
    total_steps = steps_total_per_epoch * EPOCHS
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    min_lr = LR_BASE * 0.1

    def lr_schedule(global_step, base_lr, mode="short"):
        # 我们把长步LR缩小到 base_lr * LONG_LR_SCALE
        target_base = base_lr if mode == "short" else (base_lr * LONG_LR_SCALE)

        if global_step < warmup_steps:
            # linear warmup
            return target_base * float(global_step + 1) / float(warmup_steps)
        else:
            # cosine decay to min_lr
            progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = 0.0 if progress < 0 else (1.0 if progress > 1.0 else progress)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr_val = min_lr + (target_base - min_lr) * cosine
            return lr_val

    global_optimizer_step = 0  # 我们的“真正 optimizer.step()”计数器
    long_step_count = 0
    eta = ETAHelper(total_steps=total_steps, ema=0.9)
    WORLD = get_world_size()
    TOKENS_PER_STEP_SHORT = WORLD * BATCH_SHORT * MAX_LEN_SHORT * GRAD_ACCUM_SHORT
    TOKENS_PER_STEP_LONG = WORLD * BATCH_LONG * MAX_LEN_LONG * GRAD_ACCUM_LONG
    eta_epoch = ETAHelper(total_steps=steps_total_per_epoch, ema=0.9)
    if is_main_process():
        log(f"[INIT] total_steps={total_steps} | tokens/step≈ short={TOKENS_PER_STEP_SHORT}, long={TOKENS_PER_STEP_LONG}")
    # 主循环
    resume_train_dir = get_last_checkpoint(OUT_DIR)
    if resume_train_dir is not None and USE_CHECKPOINT:
        start_step, start_epoch = try_load_training_state(resume_train_dir, optimizer)
    else:
        start_step, start_epoch = 0, 0

    global_optimizer_step = start_step
    for epoch in range(EPOCHS):
        eta_epoch = ETAHelper(total_steps=steps_total_per_epoch, ema=0.9)
        # sampler要在每个epoch set_epoch() 保证shuffle一致性
        sampler_short.set_epoch(epoch)
        sampler_long.set_epoch(epoch)
        short_iter = iter(loader_short)
        long_iter = iter(loader_long)
        # 每个epoch我们就无脑循环 loader_short 为主驱动，
        # 遇到StopIteration则break，这意味着可能没吃完long，这没关系
        done = False
        while not done:
            # 先跑一组短step块
            for _ in range(SHORT_STEPS_PER_CYCLE):
                optimizer.zero_grad(set_to_none=True)
                accumulated_loss = torch.zeros((), device=device)
                micro_count = 0
                # 一个optimizer step = GRAD_ACCUM_SHORT 个micro-step

                for micro_idx in range(GRAD_ACCUM_SHORT):
                    try:
                        batch = next(short_iter)
                    except StopIteration:
                        done = True
                        break


                    with torch.autocast(device_type="cuda", dtype=AUTOCAST_DTYPE):
                        loss, metrics = forward_batch(model, batch, device, collator_short)
                        loss = loss / GRAD_ACCUM_SHORT
                    loss.backward()
                    accumulated_loss = accumulated_loss + loss.detach()
                    micro_count += 1
                if micro_count == 0:
                    break
                if micro_count < GRAD_ACCUM_SHORT:
                    scale = GRAD_ACCUM_SHORT / micro_count
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(scale)
                # lr schedule (短模式)
                lr_val = lr_schedule(global_optimizer_step, LR_BASE, mode="short")
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_val

                # grad clip
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

                optimizer.step()
                global_optimizer_step += 1
                _maybe_barrier_every_n(global_optimizer_step)
                eta.tick(tokens_this_step=TOKENS_PER_STEP_SHORT)
                eta_epoch.tick(tokens_this_step=TOKENS_PER_STEP_SHORT)
                if writer is not None and (global_optimizer_step % LOG_SCALAR_EVERY == 0):

                    ce_f = to_float(metrics.get("ce_loss"))
                    kd_f = to_float(metrics.get("kd_loss"))
                    aux_f = to_float(metrics.get("aux_loss"))
                    # 训练标量
                    if ce_f is not None:  writer.add_scalar("short/loss/ce", ce_f, global_optimizer_step)
                    if kd_f is not None:  writer.add_scalar("short/loss/kd", kd_f, global_optimizer_step)
                    if aux_f is not None: writer.add_scalar("short/loss/aux", aux_f, global_optimizer_step)

                    # kd_ratio
                    kd_ratio = (kd_f / (ce_f + kd_f)) if (
                                ce_f is not None and kd_f is not None and (ce_f + kd_f) > 0) else None
                    if kd_ratio is not None:
                        writer.add_scalar("short/loss/kd_ratio", kd_ratio, global_optimizer_step)
                    writer.add_scalar("train/lr", lr_val, global_optimizer_step)
                    writer.add_scalar("speed/steps_per_sec", eta.steps_per_sec, global_optimizer_step)
                    writer.add_scalar("speed/tokens_per_sec", eta.tokens_per_sec, global_optimizer_step)
                    writer.add_scalar("eta/seconds_remaining_total", eta.eta_seconds, global_optimizer_step)
                    writer.add_scalar("eta/seconds_remaining_epoch", eta_epoch.eta_seconds, global_optimizer_step)
                    writer.add_scalar("progress/step_total", global_optimizer_step, global_optimizer_step)
                # logging / eval / save
                if is_main_process() and (global_optimizer_step % LOG_INTERVAL_STEPS == 0):
                    if writer is not None:
                        writer.flush()

                    log(f"[SHORT] step={global_optimizer_step} "f"loss={accumulated_loss:.4f} "f"ce={metrics.get('ce_loss')} kd={metrics.get('kd_loss')}  "f"lr={lr_val:.6e} | {eta.brief()}")
                if (global_optimizer_step % EVAL_INTERVAL_STEPS == 0) and is_main_process():
                    eval_loss = run_eval(model, eval_loader_short, device, collator_short)
                    if eval_loss is not None:
                        log(f"[EVAL] step={global_optimizer_step} eval_loss={eval_loss:.4f} | {eta.brief()}")
                        if writer is not None:
                            writer.add_scalar("eval/loss", eval_loss, global_optimizer_step)
                if (global_optimizer_step % SAVE_INTERVAL_STEPS == 0) and is_main_process():
                    ckpt_dir = os.path.join(OUT_DIR, f"checkpoint-{global_optimizer_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    # unwrap model if DDP
                    model_to_save = model.module if hasattr(model, "module") else model
                    safe_save_full_model(model_to_save, ckpt_dir, cfg_obj=getattr(model_to_save, "cfg", None))

                    save_training_state(ckpt_dir, optimizer, global_optimizer_step, epoch)
                    log(f"[SAVE] wrote {ckpt_dir}/model.safetensors | {eta.brief()}")
                    prune_old_checkpoints(OUT_DIR, keep=SAVE_TOTAL_LIMIT)

                if done:
                    break

            if done:
                break

            # 跑一组长step块 (1个长step)
            # 一个optimizer step = GRAD_ACCUM_LONG 个micro-step
            accumulated_loss = torch.zeros((), device=device)
            optimizer.zero_grad(set_to_none=True)
            for micro_idx in range(GRAD_ACCUM_LONG):
                try:
                    batch = next(long_iter)
                except StopIteration:
                    # 长loader用完了就重置一下iterator，继续循环（因为长池比短池小很多，
                    # 我们允许重复采样，这和我们5%占比+轻LR的策略契合）
                    long_iter = iter(loader_long)
                    batch = next(long_iter)



                with torch.autocast(device_type="cuda", dtype=AUTOCAST_DTYPE):
                    loss, metrics = forward_batch(model, batch, device, collator_long)
                    loss = loss / GRAD_ACCUM_LONG
                loss.backward()
                accumulated_loss = accumulated_loss + loss.detach()

            # lr schedule (长模式, 会自动乘 LONG_LR_SCALE)
            lr_val = lr_schedule(global_optimizer_step, LR_BASE, mode="long")
            for pg in optimizer.param_groups:
                pg["lr"] = lr_val

            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            global_optimizer_step += 1
            _maybe_barrier_every_n(global_optimizer_step)
            eta.tick(tokens_this_step=TOKENS_PER_STEP_LONG)
            eta_epoch.tick(tokens_this_step=TOKENS_PER_STEP_LONG)
            if writer is not None:
                acc_float = float(accumulated_loss.item())

                # metrics 转 float（若非 None）

                ce_f = to_float(metrics.get("ce_loss"))
                kd_f = to_float(metrics.get("kd_loss"))
                aux_f = to_float(metrics.get("aux_loss"))

                if ce_f is not None:  writer.add_scalar("long/loss/ce", ce_f, global_optimizer_step)
                if kd_f is not None:  writer.add_scalar("long/loss/kd", kd_f, global_optimizer_step)
                if aux_f is not None: writer.add_scalar("long/loss/aux", aux_f, global_optimizer_step)

                # kd_ratio
                if ce_f is not None and kd_f is not None and (ce_f + kd_f) > 0:
                    writer.add_scalar("long/loss/kd_ratio", kd_f / (ce_f + kd_f), global_optimizer_step)

                writer.add_scalar("train/lr", lr_val, global_optimizer_step)
                writer.add_scalar("speed/steps_per_sec", eta.steps_per_sec, global_optimizer_step)
                writer.add_scalar("speed/tokens_per_sec", eta.tokens_per_sec, global_optimizer_step)
                writer.add_scalar("eta/seconds_remaining_total", eta.eta_seconds, global_optimizer_step)
                writer.add_scalar("eta/seconds_remaining_epoch", eta_epoch.eta_seconds, global_optimizer_step)
                writer.add_scalar("progress/step_total", global_optimizer_step, global_optimizer_step)
            long_step_count += 1
            if is_main_process() and (long_step_count % LOG_INTERVAL_LONG_STEPS == 0):
                log(f"[LONG ] step={global_optimizer_step} "f"loss={accumulated_loss:.4f} "f"ce={metrics.get('ce_loss')} kd={metrics.get('kd_loss')} "f"lr={lr_val:.6e} (long ctx) | {eta.brief()}")
            if (global_optimizer_step % EVAL_INTERVAL_STEPS == 0) and is_main_process():
                eval_loss = run_eval(model, eval_loader_short, device, collator_short)
                if eval_loss is not None:
                    log(f"[EVAL] step={global_optimizer_step} eval_loss={eval_loss:.4f} | {eta.brief()}")
                    if writer is not None:
                        writer.add_scalar("eval/loss", eval_loss, global_optimizer_step)
            if (global_optimizer_step % SAVE_INTERVAL_STEPS == 0) and is_main_process():
                ckpt_dir = os.path.join(OUT_DIR, f"checkpoint-{global_optimizer_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                model_to_save = model.module if hasattr(model, "module") else model
                safe_save_full_model(model_to_save, ckpt_dir, cfg_obj=getattr(model_to_save, "cfg", None))
                save_training_state(ckpt_dir, optimizer, global_optimizer_step, epoch)
                log(f"[SAVE] wrote {ckpt_dir}/model.safetensors | {eta.brief()}")
                prune_old_checkpoints(OUT_DIR, keep=SAVE_TOTAL_LIMIT)

        # end while epoch
    # end for epoch

    # final save
    _ddp_barrier()
    if is_main_process():
        final_dir = os.path.join(OUT_DIR, f"final-{global_optimizer_step}")
        os.makedirs(final_dir, exist_ok=True)
        model_to_save = model.module if hasattr(model, "module") else model
        safe_save_full_model(model_to_save, final_dir, cfg_obj=getattr(model_to_save, "cfg", None))
        log(f"[FINAL SAVE] {final_dir} | {eta.brief()}")

    # cleanup dist
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    train_loop()
