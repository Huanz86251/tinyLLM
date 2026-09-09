from project_paths import legacy_path, path as project_path
import os, math, json
import torch
from datasets import load_dataset
from train.pretrain.kd_reader import KDFetcher
from transformers import Trainer, TrainingArguments
from torch.utils.data import default_collate
from model.config import Config
from itertools import chain
from datasets import load_from_disk
from model.model import TinyLLM
from datasets import concatenate_datasets
import torch.nn.functional as F
from collections import Counter
import torch
import numpy as np
import os, glob
import re
from pathlib import Path
from datasets import interleave_datasets
from safetensors.torch import save_file
from datetime import datetime
from transformers.trainer_utils import get_last_checkpoint
import random
import math
import contextlib
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import TrainerCallback, TrainerState, TrainerControl
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets import interleave_datasets
import time, shutil
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.backends.cuda import sdp_kernel
sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)

os.environ["HF_DATASETS_CACHE"] = os.path.abspath("./hf_datasets_cache_force")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
USE_BF16 = True
AUTOCAST_DTYPE=torch.bfloat16 if USE_BF16 else torch.float16
use_checkpoint=False
checkpoint_use_reentrant=True
probs = np.ones(20000, dtype=np.float64) / 20000
gradient_accumulation_steps=3
# ========== 基本配置 ==========

MAX_LEN = 2048

BATCH = 3
LR = 2.5e-4
EPOCHS = 3

IGNORE_INDEX = -100
eval_frac = 0.0001 #分割eval比例
REPAK=False
SEED=42
#torch.manual_seed(SEED)
CACHE_DIR = "cache_pretrain_large"
TOK_CACHE = os.path.join(CACHE_DIR, f"tokenized")
PK_CACHE  = os.path.join(CACHE_DIR, f"packed_len{MAX_LEN}")
EXTRA_PK_CACHE = os.path.join(CACHE_DIR, f"packed_len{MAX_LEN}_more")
PK_SENTINEL = Path(PK_CACHE) / "dataset_info.json"
KD_DIR = legacy_path("/root/autodl-tmp/llm/data/kd_lmdb_minicpm3_4b")
from transformers import AutoTokenizer
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# ==== HARD OFFLINE & LOCAL PATHS (DO NOT TOUCH INTERNET) ====
import os
FORCE_OFFLINE = True

# 你的本地快照与tokenizer所在目录（就是你刚刚冻结出来的）
LOCAL_SNAPSHOT = legacy_path("/root/autodl-tmp/llm/KDmodel/minicpm3_4b/snapshot")
LOCAL_PK_CACHE = legacy_path("/root/autodl-tmp/llm/cache_pretrain_large/packed_len2048")

# 允许用环境变量覆盖；不设环境变量时使用上面默认值
TOK_DIR  = os.environ.get("TOK_DIR",  LOCAL_SNAPSHOT)
SNAP_DIR = os.environ.get("SNAP_DIR", LOCAL_SNAPSHOT)
PK_CACHE = os.environ.get("PK_CACHE", LOCAL_PK_CACHE)

if FORCE_OFFLINE:
    os.environ.update({
        "HF_HOME": legacy_path("/root/autodl-tmp/hf_home_strict_offline"),
        "HF_DATASETS_CACHE": legacy_path("/root/autodl-tmp/hf_datasets_cache_force"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        # 关键：禁用“在线模式”回退；找不到文件就报错
        "HF_HUB_ENABLE_ONLINE_MODE": "0",
        # Tokenizers 并行
        "TOKENIZERS_PARALLELISM": "true",
    })

# 硬校验：本地必须有这些文件，否则立刻报错
assert os.path.isdir(TOK_DIR), f"Tokenizer dir not found: {TOK_DIR}"
assert os.path.isfile(os.path.join(TOK_DIR, "tokenizer.json")), f"Missing tokenizer.json under {TOK_DIR}"
assert os.path.isfile(os.path.join(SNAP_DIR, "config.json")), f"Missing config.json under {SNAP_DIR}"


tokenizer = AutoTokenizer.from_pretrained(
    TOK_DIR,
    trust_remote_code=True,
    local_files_only=True,   # <== 只允许本地
)

VOCLEN=len(tokenizer.get_vocab())
tokenizer.padding_side = "right"
assert tokenizer.eos_token_id is not None, "需要 eos_token用于样本连接与分块"

print(f"[TOKENIZER] vocab_size = {VOCLEN}")
print(f"[TOKENIZER] eos_token/id = {tokenizer.eos_token}/{tokenizer.eos_token_id}")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"[TOKENIZER] pad_token/id = {tokenizer.pad_token}/{tokenizer.pad_token_id}")

print(f"[TOKENIZER] pad_token/id = {tokenizer.pad_token}/{tokenizer.pad_token_id}")


# ===== 安全保存函数：整模型 state_dict + cfg =====

def is_main_process():
    # LOCAL_RANK 在单卡/非DDP时一般不存在
    lr = int(os.environ.get("LOCAL_RANK", "0"))
    return (lr == 0)

def wait_for_file(path, timeout=36000, interval=5):
    # 轮询等待 rank0 生成的数据
    start = time.time()
    while not os.path.exists(path):
        if time.time() - start > timeout:
            raise TimeoutError(f"Timeout waiting for {path}")
        time.sleep(interval)

def get_world_size():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    # 兼容从环境读取（Trainer 初始化前）
    return int(os.environ.get("WORLD_SIZE", "1"))

def _assert_no_top_numeric_keys(state_dict_keys):
    # 防止保存到子模块导致键名变成 "0.xxx" 这种顶级数字键
    bad = [k for k in state_dict_keys if re.match(r"^\d+\.", k)]
    if bad:
        raise RuntimeError(
            f"State dict has top-level numeric keys (e.g., {bad[:5]}). "
            f"You're likely saving a submodule instead of the full model."
        )

def safe_save_full_model(model, out_dir, cfg_obj=None):
    os.makedirs(out_dir, exist_ok=True)
    state = {k: (v.detach().cpu() if v.is_floating_point() else v.cpu())
             for k, v in model.state_dict().items()}
    _assert_no_top_numeric_keys(state.keys())  # 早发现坏键名
    save_file(state, os.path.join(out_dir, "model.safetensors"))
    # 额外保存 config（基于你的 Config 对象）
    if cfg_obj is not None:
        with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(getattr(cfg_obj, "__dict__", {}), f, ensure_ascii=False, indent=2)

# ===== 回调：在每次保存时，再落一份 safetensors + config =====
class SafeTensorsCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        # 只在主进程执行
        if hasattr(self, "trainer") and not self.trainer.is_world_process_zero:
            return
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(ckpt_dir):
            try:
                safe_save_full_model(self.trainer.model, ckpt_dir, cfg_obj=getattr(self.trainer.model, "cfg", None))
                print(f"[safe-save] wrote {os.path.join(ckpt_dir, 'model.safetensors')}")
            except Exception as e:
                print(f"[safe-save][WARN] {e}")

def tokenize_fn(batch):

    texts = [t for t in batch["TXT"]]
    out = tokenizer(texts, add_special_tokens=False,truncation=False,padding=False,return_attention_mask=False)
    eos = tokenizer.eos_token_id
    input_ids = [ids + [eos] for ids in out["input_ids"]]
    return {"input_ids": input_ids}
def group_texts(examples):

    concatenated = list(chain.from_iterable(examples["input_ids"])) # 展平
    total_len = (len(concatenated) // MAX_LEN) * MAX_LEN
    if total_len == 0:
        return {"input_ids": []}
    blocks = [concatenated[i:i + MAX_LEN] for i in range(0, total_len, MAX_LEN)]
    return {"input_ids": blocks}



ds_kd    = load_from_disk(PK_CACHE)
ds_plain = load_from_disk(EXTRA_PK_CACHE)




mixed = concatenate_datasets([ds_kd, ds_plain])
print(f"[MIX] lengths(KD,Plain,Total) = {len(ds_kd)}, {len(ds_plain)}, {len(mixed)}")

# 3) 划分 train/eval；分层保证 eval 里两类都有
split = mixed.train_test_split(
    test_size=eval_frac,
    seed=SEED
)

train_ds, eval_ds = split["train"], split["test"]



num_blocks_all  = len(mixed)
num_blocks_tr   = len(train_ds)
num_blocks_eval = len(eval_ds)

total_tokens_all  = num_blocks_all  * MAX_LEN
total_tokens_tr   = num_blocks_tr   * MAX_LEN
total_tokens_eval = num_blocks_eval * MAX_LEN

# 预估“参与 loss 的 token”（每块最后一位是 -100）
loss_tokens_tr = num_blocks_tr * (MAX_LEN - 1)

print(f"[TOKENS] blocks(all/train/eval) = {num_blocks_all}/{num_blocks_tr}/{num_blocks_eval}")
print(f"[TOKENS] total(all/train/eval)  = {total_tokens_all:,}/{total_tokens_tr:,}/{total_tokens_eval:,}")
print(f"[TOKENS] train loss-effective    = {loss_tokens_tr:,} (≈ train_blocks * (MAX_LEN-1))")
# to torch
train_ds = train_ds.with_format("torch", columns=["input_ids"])
eval_ds  = eval_ds.with_format("torch", columns=["input_ids"])

train_ds=train_ds.with_format("torch",columns=["input_ids"])
eval_ds=eval_ds.with_format("torch",columns=["input_ids"])





class MixedCollator:
    def __init__(self, tokenizer, kd_dir=None,
                 p_denoise=0.2, p_kd=0.2,
                 ignore_index=-100, span_ratio=(0.15,0.30),
                 mask_token=None, kd_topk_use=32, kd_pos_keep=1.0,shortlist_neg=128,
                 exclude_special_in_neg=True,enable_shortlist: bool = True):
        self.tok = tokenizer
        self.p_denoise = p_denoise
        self.p_kd = p_kd
        self.ignore_index = ignore_index
        self.span_ratio = span_ratio
        self.mask_id = tokenizer.pad_token_id if mask_token is None else tokenizer.convert_tokens_to_ids(mask_token)
        self.kd_dir=kd_dir
        self.kd_topk_use = kd_topk_use
        self.kd_pos_keep = kd_pos_keep
        self.shortlist_neg = shortlist_neg
        self.vocab_size = len(tokenizer.get_vocab())
        self.exclude_special_in_neg = exclude_special_in_neg
        self.special_ids = set(
            [i for i in [
                tokenizer.pad_token_id,
                tokenizer.eos_token_id,
                getattr(tokenizer, "unk_token_id", None),
                getattr(tokenizer, "bos_token_id", None),
            ] if i is not None]
        )
        self.kd = None
        self.unigram_probs = probs
        self.unigram_probs = probs.astype(np.float64).copy()
        if self.exclude_special_in_neg and len(self.special_ids) > 0:
            for sid in self.special_ids:
                if 0 <= sid < self.unigram_probs.shape[0]:
                    self.unigram_probs[sid] = 0.0
            s = self.unigram_probs.sum()
            assert s > 0
            self.unigram_probs /= s
        self._torch_unigram_p=torch.from_numpy(self.unigram_probs).to(dtype=torch.float32)
        self.logq_vec = torch.log(self._torch_unigram_p + 1e-12)
        self.enable_shortlist = enable_shortlist

    def _ar(self, input_ids):
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = self.ignore_index
        return input_ids, labels

    def _fim_one(self, ids_1d: torch.Tensor):
        # 形如： <|fim_prefix|> prefix <|fim_suffix|> suffix <|fim_middle|> middle
        # 只对最后的 middle 监督；标准因果掩码下可以“读到”prefix+suffix（它们在左边）
        T = ids_1d.size(0)
        min_r, max_r = self.span_ratio
        r = random.uniform(min_r, max_r)
        m = max(1, int(T * r))
        s = random.randint(0, max(0, T - m))
        e = s + m

        prefix, middle, suffix = ids_1d[:s], ids_1d[s:e], ids_1d[e:]

        p_tok = torch.tensor([self.tok.convert_tokens_to_ids("<|fim_prefix|>")], dtype=ids_1d.dtype,
                             device=ids_1d.device)
        s_tok = torch.tensor([self.tok.convert_tokens_to_ids("<|fim_suffix|>")], dtype=ids_1d.dtype,
                             device=ids_1d.device)
        m_tok = torch.tensor([self.tok.convert_tokens_to_ids("<|fim_middle|>")], dtype=ids_1d.dtype,
                             device=ids_1d.device)

        x = torch.cat([p_tok, prefix, s_tok, suffix, m_tok, middle], dim=0)

        # 长度对齐：优先从 prefix/suffix 截断；不足则右 pad（这些位 labels=IGNORE）
        if x.size(0) > T:
            overflow = x.size(0) - T
            cut_left = min(overflow // 2, prefix.size(0))
            cut_right = min(overflow - cut_left, suffix.size(0))
            prefix = prefix[: prefix.size(0) - cut_left] if cut_left > 0 else prefix
            suffix = suffix[: suffix.size(0) - cut_right] if cut_right > 0 else suffix
            x = torch.cat([p_tok, prefix, s_tok, suffix, m_tok, middle], dim=0)
            if x.size(0) > T:
                extra = x.size(0) - T
                if prefix.size(0) >= suffix.size(0):
                    prefix = prefix[: max(0, prefix.size(0) - extra)]
                else:
                    suffix = suffix[: max(0, suffix.size(0) - extra)]
                x = torch.cat([p_tok, prefix, s_tok, suffix, m_tok, middle], dim=0)
        if x.size(0) < T:
            pad = torch.full((T - x.size(0),), self.tok.pad_token_id, dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=0)

        labels = torch.full((T,), self.ignore_index, dtype=ids_1d.dtype, device=ids_1d.device)
        mid_len = min(middle.size(0), T)
        mid_start = T - mid_len  # 末尾部分是 middle
        labels[mid_start: mid_start + mid_len] = x[mid_start: mid_start + mid_len]
        return x, labels

    def _fim(self, input_ids):
        B, _ = input_ids.size()
        xs, ys = zip(*(self._fim_one(input_ids[b]) for b in range(B)))
        return torch.stack(xs, 0), torch.stack(ys, 0)

    def _span_denoise(self, input_ids):
        B, T = input_ids.size()
        labels = torch.full_like(input_ids, self.ignore_index)
        out_ids = input_ids.clone()
        min_r, max_r = self.span_ratio
        for b in range(B):
            L = T
            r = random.uniform(min_r, max_r)
            m = max(1, int(L * r))
            start = random.randint(0, max(0, L - m))
            end = start + m
            labels[b, start:end] = out_ids[b, start:end]
            out_ids[b, start:end] = self.mask_id
        return out_ids, labels

    def __call__(self, batch):
        if self.kd is None and self.kd_dir is not None:
            from train.pretrain.kd_reader import KDFetcher
            self.kd = KDFetcher(self.kd_dir)
        input_ids = default_collate([e["input_ids"] for e in batch]).long()

        if random.random() < self.p_denoise:
            ids, lbs = self._fim(input_ids)
            return {"input_ids": ids, "labels": lbs}

        ids, lbs = self._ar(input_ids)

        # ========= 可选 KD =========
        B, T = ids.size()
        K = self.kd_topk_use
        kd_idx_list, kd_val_list, kd_mask_list = [], [], []
        use_kd_any = False
        if not self.enable_shortlist:
            B, T = ids.size()
            out = {"input_ids": ids, "labels": lbs}

            use_kd_any = False
            if (self.kd is not None) and (self.p_kd > 0.0) and (self.kd_topk_use > 0):
                K = self.kd_topk_use
                kd_idx_list, kd_val_list, kd_mask_list = [], [], []
                for b in range(B):
                    if random.random() < self.p_kd:
                        fetched = self.kd.get(ids[b].tolist())
                        if fetched is not None:
                            top_idx_np, top_val_np = fetched
                            if top_idx_np.shape[1] > K:
                                top_idx_np = top_idx_np[:, :K]
                                top_val_np = top_val_np[:, :K]
                            keep = (np.random.rand(T) < self.kd_pos_keep) if (self.kd_pos_keep < 1.0) else np.ones(T,
                                                                                                                   bool)
                            kd_idx_list.append(torch.tensor(top_idx_np, dtype=torch.long))
                            kd_val_list.append(torch.tensor(top_val_np, dtype=torch.float32))
                            kd_mask_list.append(torch.from_numpy(keep).bool())
                            use_kd_any = True
                            continue
                    kd_idx_list.append(None);
                    kd_val_list.append(None);
                    kd_mask_list.append(None)

                if use_kd_any:
                    for i in range(B):
                        if kd_idx_list[i] is None:
                            kd_idx_list[i] = torch.zeros((T, K), dtype=torch.long)
                            kd_val_list[i] = torch.zeros((T, K), dtype=torch.float32)
                            kd_mask_list[i] = torch.zeros((T,), dtype=torch.bool)
                    out["kd_idx"] = torch.stack(kd_idx_list, dim=0)  # [B,T,K]
                    out["kd_val"] = torch.stack(kd_val_list, dim=0)  # [B,T,K]
                    out["kd_mask"] = torch.stack(kd_mask_list, dim=0)  # [B,T]
            return out

        if (self.kd is not None) and (self.p_kd > 0.0):
            for b in range(B):
                if random.random() < self.p_kd:
                    fetched = self.kd.get(ids[b].tolist())
                    if fetched is not None:
                        top_idx_np, top_val_np = fetched
                        if top_idx_np.shape[1] > K:
                            top_idx_np = top_idx_np[:, :K]
                            top_val_np = top_val_np[:, :K]
                        keep = (np.random.rand(T) < self.kd_pos_keep) if (self.kd_pos_keep < 1.0) else np.ones(T, bool)
                        kd_idx_list.append(torch.tensor(top_idx_np, dtype=torch.long))
                        kd_val_list.append(torch.tensor(top_val_np, dtype=torch.float32))
                        kd_mask_list.append(torch.from_numpy(keep).bool())
                        use_kd_any = True
                        continue

                kd_idx_list.append(None)
                kd_val_list.append(None)
                kd_mask_list.append(None)

        batch_out = {"input_ids": ids, "labels": lbs}

        # —— 统一构造 shortlist（即使没有 KD）——
        gold = lbs.clone()
        mask_ign = (gold == self.ignore_index)
        gold[mask_ign] = ids[mask_ign]
        gold = gold.long()  # [B,T]
        gold_col = torch.zeros_like(gold)  # [B,T]

        if use_kd_any:
            K = self.kd_topk_use
            for i in range(B):
                if kd_idx_list[i] is None:
                    kd_idx_list[i] = torch.zeros((T, K), dtype=torch.long)
                    kd_val_list[i] = torch.zeros((T, K), dtype=torch.float32)
                    kd_mask_list[i] = torch.zeros((T,), dtype=torch.bool)
            kd_idx_btK = torch.stack(kd_idx_list, dim=0)  # [B,T,K_kd]
            kd_val_btK = torch.stack(kd_val_list, dim=0)  # [B,T,K_kd]
            kd_mask_bt = torch.stack(kd_mask_list, dim=0)  # [B,T]
            batch_out["kd_idx"] = kd_idx_btK
            batch_out["kd_val"] = kd_val_btK
            batch_out["kd_mask"] = kd_mask_bt
        else:
            # 没有 KD 时给 0 宽度占位，便于拼接
            kd_idx_btK = torch.empty((B, T, 0), dtype=torch.long)
            # 不必把 kd_val/kd_mask 放进 batch_out


        K_neg = self.shortlist_neg
        if K_neg > 0:
            p = self._torch_unigram_p
            neg_flat = torch.multinomial(p, num_samples=B * T * K_neg, replacement=True)
            neg = neg_flat.view(B, T, K_neg)  # [B,T,K_neg]
        else:
            neg = torch.empty((B, T, 0), dtype=torch.long)

        # 最终 shortlist: [gold | KD | neg]
        short_all = torch.cat([gold.unsqueeze(-1), kd_idx_btK, neg], dim=-1)  # [B,T,1+K_kd+K_neg]
        B, T, Kp = short_all.shape
        short_logq = torch.zeros((B, T, Kp), dtype=torch.float32)
        kd_width = self.kd_topk_use if use_kd_any else 0
        neg_start = 1 + kd_width
        if self.shortlist_neg > 0 and neg_start < Kp:
    # 只给负样本列写入 log q；gold/KD 列保持 0
            short_logq[..., neg_start:] = self.logq_vec[short_all[..., neg_start:]]
        batch_out["short_idx"] = short_all
        batch_out["gold_col"] = gold_col
        batch_out["short_logq"] = short_logq
        return batch_out


collator = MixedCollator(
    tokenizer,
    kd_dir=KD_DIR,
    p_denoise=0.0,     # 30% span-denoise
    p_kd=1,          # 20% AR 样本带 KD
    kd_topk_use=16,    # 用 16/32
    kd_pos_keep=1,   # 只在 50% token 上做 KD
    ignore_index=IGNORE_INDEX,
    enable_shortlist=False
)


cfg = Config(
    vocab_size=VOCLEN,
    train_maxlength=MAX_LEN,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    moe_use_detach=False,
    use_moe = False,
    dropout=0.0,
    learnable_temp = True,
    drop_path = 0.0,
    residual_dropout = 0.00,
    rope_type="yarn",
)

cfg.use_checkpoint = use_checkpoint
cfg.checkpoint_use_reentrant = checkpoint_use_reentrant
model = TinyLLM(cfg)
class LoggingTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"]                 # 这个保留梯度

        # 只在 log 时把分量损失转为 Python float；先 detach 再 item
        metrics = {}
        prefix = "train/" if model.training else "eval/"
        for k in ("ce_loss", "kd_loss", "aux_loss"):
            v = outputs.get(k, None)
            if v is None:
                continue
            if isinstance(v, torch.Tensor):

                if v.dim() > 0:
                    v = v.detach().mean()
                else:
                    v = v.detach()
                v = v.item()   # 从 GPU 读出 4B 标量，同步一次（只在 logging_steps 发生）
            else:
                v = float(v)
            metrics[prefix + k] = v
        gs = self.state.global_step
        if metrics and (gs % self.args.logging_steps == 0 or gs == 0) and self.is_world_process_zero():
            ki = inputs.get("kd_idx", None)
            km = inputs.get("kd_mask", None)
            lbs = inputs.get("labels", None)

            if ki is not None and km is not None and lbs is not None and ki.numel() > 0:
                with torch.no_grad():
                    m = km.bool() & (lbs != IGNORE_INDEX)  # 有 KD 且不是 ignore 的位置
                    cover = (m.float().mean().item()) if m.numel() > 0 else 0.0

                    if m.any():
                        lab = lbs[m]  # [M]
                        hit_any = (ki[m] == lab.unsqueeze(-1)).any(-1)  # [M]
                        hit_rate = hit_any.float().mean().item()
                    else:
                        hit_rate = 0.0

                    metrics[prefix + "kd_cover"] = cover  # e.g. 0.80 -> 80% 位置有 KD
                    metrics[prefix + "kd_hit@K"] = hit_rate  # label 是否落在 teacher 的 top-K

            # KD 在 (CE+KD) 中的占比，直观看是不是接近你想要的 ~0.6
            ce_v = outputs.get("ce_loss")
            kd_v = outputs.get("kd_loss")
            if isinstance(ce_v, torch.Tensor): ce_v = ce_v.detach().float().item()
            if isinstance(kd_v, torch.Tensor): kd_v = kd_v.detach().float().item()
            if isinstance(ce_v, (int, float)) and isinstance(kd_v, (int, float)) and (ce_v + kd_v) > 0:
                metrics[prefix + "kd_ratio"] = kd_v / (kd_v + ce_v)
            self.log(metrics)

        return (loss, outputs) if return_outputs else loss
def _strip_mamba_token_embedding(ssm):
    # 1) 物理移除 Parameter（如果还在的话）
    if hasattr(ssm, "embeddings"):
        try:
            delattr(ssm, "embeddings")
        except Exception:
            ssm._parameters.pop("embeddings", None)
            ssm._modules.pop("embeddings", None)
    # 2) 放一个非持久化占位，防误访问
    ssm.register_buffer("embeddings", torch.empty(0), persistent=False)
    # 3) 禁用 HF 习惯接口（有人想取/设，直接报错）
    ssm.get_input_embeddings = lambda: (_ for _ in ()).throw(RuntimeError("Mamba2 embeddings disabled"))
    ssm.set_input_embeddings = lambda _: (_ for _ in ()).throw(RuntimeError("Mamba2 embeddings disabled"))


for blk in model.blocks:
    if hasattr(blk, "ssm"):
        _strip_mamba_token_embedding(blk.ssm)
assert not any("ssm.embeddings" in n for n, _ in model.named_parameters()), "embeddings still present!"




global_batch = BATCH *gradient_accumulation_steps * get_world_size()
steps_per_epoch = math.ceil(len(train_ds) / global_batch)
save_eval_interval = max(1000, max(1, steps_per_epoch // 20))
RUN_ID   = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_NAME = f"tinyllm-pretrain-{RUN_ID}"
LOG_DIR  = os.path.join(legacy_path("/root/autodl-tmp/llm/checkpoints"), "tb", RUN_NAME)
args = TrainingArguments(
    output_dir=str(project_path("runs/training/pretrain")),
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH if eval_ds else None,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    warmup_ratio=0.01,
    weight_decay=0.01,
    logging_steps=200,
    save_steps=save_eval_interval,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr": LR * 0.1},
    bf16=True,
    ddp_bucket_cap_mb=64,

    tf32=True,  # 允许 TF32
    gradient_checkpointing=False,
    max_grad_norm=1.2,
    dataloader_num_workers=6,
    dataloader_persistent_workers=True,
    dataloader_pin_memory=True,
    dataloader_prefetch_factor=6,
    ddp_backend="nccl",
    ddp_find_unused_parameters=False,
    torch_compile=False,
    eval_strategy="steps" if eval_ds else "no",
    eval_steps=save_eval_interval if eval_ds else None,
    report_to="tensorboard",
    run_name=RUN_NAME,
    logging_dir=LOG_DIR,
    save_total_limit=2,


    dataloader_drop_last=True
)

def param_groups_no_decay(model):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or any(
                k in n for k in ["norm", "bias", "tok_embed", "gamma_att", "gamma_mlp","tcc_gate","router","decider","temp_param"]):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


optimizer = torch.optim.AdamW(param_groups_no_decay(model), lr=args.learning_rate, betas=(0.9, 0.95),fused=True if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else False)


trainer = LoggingTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
    processing_class=tokenizer,
    optimizers=(optimizer, None),

)
cb = SafeTensorsCallback()
cb.trainer = trainer
trainer.add_callback(cb)
if trainer.is_world_process_zero():
    tokenizer.save_pretrained(args.output_dir)

def probe_fast_once(run_once_fn, *args, **kwargs):
    import torch
    # 只包一小步，代价很小；use_cuda=True 能把 GPU kernel 事件也记录下来
    with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
        out = run_once_fn(*args, **kwargs)

    names = [e.key.lower() for e in prof.key_averages()]

    def seen(*subs):
        for n in names:
            for s in subs:
                if s in n:
                    return True
        return False

    used_scan   = seen("selective_scan")
    used_update = seen("selective_state_update", "causal_conv1d_update")
    used_causal = seen("causal_conv1d")

    print(f"[FASTPROBE] selective_scan kernels seen: {used_scan}")
    print(f"[FASTPROBE] selective_state_update kernels seen: {used_update}")
    print(f"[FASTPROBE] causal_conv1d kernels seen: {used_causal}")
    return out

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    RESUME_TRAINING = True
    RESUME_ROOT = legacy_path("/root/autodl-tmp/llm/tiny_05B")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank       = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    print(f"[DDP] rank={rank} local_rank={local_rank} world_size={world_size}")
    ckpt_path = None
    if RESUME_TRAINING:
        ckpt_path = get_last_checkpoint(RESUME_ROOT)

        if ckpt_path is None:
            raise FileNotFoundError(
                f"[RESUME] 没有在 {RESUME_ROOT} 下面找到 checkpoint-* 目录，无法续训。"
            )

        print(f"[RESUME] rank={rank} 将从 {ckpt_path} 继续训练")
    else:
        print("[RESUME] RESUME_TRAINING=False，将从随机初始化开始训练（不加载checkpoint）")

    with sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False):
        if RESUME_TRAINING:
            trainer.train(resume_from_checkpoint=ckpt_path)
        else:
            trainer.train()
