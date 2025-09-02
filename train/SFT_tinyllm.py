import os, math, json
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from torch.utils.data import default_collate
from model.config import Config
from itertools import chain
from datasets import load_from_disk
from model.model import TinyLLM
from transformers.trainer_utils import get_last_checkpoint
from safetensors.torch import load_file as safe_load
import regex as re
from typing import List, Dict, Tuple
import numpy as np
from transformers import TrainerCallback

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

USE_BF16 = True
AUTOCAST_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
use_checkpoint = False
checkpoint_use_reentrant = True

# ========== 基本配置 ==========
MAX_LEN = 512
SEED = 42
BATCH = 8
LR = 1e-4
EPOCHS = 20
SAVE_STEPS = 2000
IGNORE_INDEX = -100
eval_frac = 0.0001  # 分割eval比例
REPAK = True
torch.manual_seed(SEED)
CACHE_DIR = "cache_SFTshortest"
TOK_CACHE = os.path.join(CACHE_DIR, f"tokenized")
PK_CACHE = os.path.join(CACHE_DIR, f"packed_len{MAX_LEN}")

from model.model import Top1Router, MoEMLP



class Top2AlphaWarmup(TrainerCallback):
    def __init__(self, ratio=0.03, target=1.0):
        self.ratio  = ratio
        self.target = target

    def on_step_end(self, args, state, control, **kwargs):
        if not state.max_steps:
            return control
        prog = state.global_step / state.max_steps
        alpha = self.target * max(0.0, min(1.0, prog / self.ratio))
        model = kwargs["model"]
        for m in model.modules():
            if isinstance(m, MoEMLP):
                m.top2_alpha.fill_(alpha)
        return control
def sanity_check_batch(input_ids, labels, attention_mask, tokenizer, ignore_index=-100):
    import torch
    uniq = torch.unique(labels)
    print("[CHECK] unique(labels) size:", uniq.numel(),
          "has -100:", (uniq == ignore_index).any().item(),
          "min/max:", labels.min().item(), labels.max().item())
    if attention_mask is not None:
        pad_pos = (attention_mask == 0)
        bad_pad = ((labels != ignore_index) & pad_pos).any().item()
        print("[CHECK] padding positions all -100:", not bad_pad)
    unk = getattr(tokenizer, "unk_token_id", None)
    if unk is not None:
        has_unk = (labels == unk).any().item()
        print("[CHECK] labels contain raw <unk> id:", bool(has_unk))
    total = labels.numel()
    n_ignore = (labels == ignore_index).sum().item()
    print(f"[CHECK] ignore_ratio = {n_ignore/total:.3f}")
    shift_in = input_ids[:, 1:]
    shift_lb = labels[:, :-1]
    mask = (shift_lb != ignore_index)
    agree = (shift_in[mask] == shift_lb[mask]).float().mean().item() if mask.any() else float('nan')
    print(f"[CHECK] next-token alignment agreement ≈ {agree:.3f}")
    eos = tokenizer.eos_token_id
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is not None and eos is not None and pad == eos and attention_mask is not None:
        wrong = ((labels == eos) & (attention_mask == 0)).any().item()
        print("[CHECK] pad==eos case safe (pad labels=-100):", not wrong)


def attach_moe_probe(model):
    def _wrap_router_forward(router: Top1Router):
        if getattr(router, "_moe_wrapped", False):
            return
        orig_forward = router.forward

        def wrapped(x: torch.Tensor):
            out = orig_forward(x)
            try:
                e, prob_sel, prob = out
            except Exception:
                return out
            router._last_e = e.detach()
            router._last_prob = prob.detach()
            router._last_prob_sel = prob_sel.detach()
            router._last_N = e.numel()
            router._last_E = prob.size(-1)
            return out

        router.forward = wrapped
        router._moe_wrapped = True

    for m in model.modules():
        if isinstance(m, Top1Router):
            _wrap_router_forward(m)

    def _moe_hook(module: MoEMLP, inputs, output):
        x = inputs[0]
        B, T, H = x.shape
        N = B * T
        router: Top1Router = module.router
        if not hasattr(router, "_last_e"):
            module.last_stats = {"note": "router no data"}
            return
        e = router._last_e
        prob_sel = router._last_prob_sel
        E = router._last_E
        counts = torch.bincount(e.view(-1), minlength=E).to(torch.int64)
        cap = math.ceil(module.cap_factor * N / E)
        kept = torch.clamp(counts, max=cap)
        fallback = (counts - kept).clamp_min(0)
        fallback_total = int(fallback.sum().item())
        fallback_frac = float(fallback_total / max(N, 1))
        stats = {
            "B": int(B), "T": int(T), "N": int(N), "E": int(E),
            "cap": int(cap),
            "per_expert_count": counts.cpu().tolist(),
            "kept_per_expert": kept.cpu().tolist(),
            "fallback_per_expert": fallback.cpu().tolist(),
            "fallback_total": fallback_total,
            "fallback_frac": fallback_frac,
            "top1_prob_mean": float(prob_sel.mean().item()),
            "top1_prob_p95": float(torch.quantile(prob_sel, 0.95).item()),
            "top1_prob_p99": float(torch.quantile(prob_sel, 0.99).item()),
        }
        module.last_stats = stats

    for m in model.modules():
        if isinstance(m, MoEMLP):
            if hasattr(m, "_moe_hook_handle"):
                try:
                    m._moe_hook_handle.remove()
                except Exception:
                    pass
            m._moe_hook_handle = m.register_forward_hook(_moe_hook)


def detach_moe_probe(model):
    for m in model.modules():
        if isinstance(m, MoEMLP) and hasattr(m, "_moe_hook_handle"):
            try:
                m._moe_hook_handle.remove()
            except Exception:
                pass
            if hasattr(m, "last_stats"):
                delattr(m, "last_stats")
    for m in model.modules():
        if isinstance(m, Top1Router) and getattr(m, "_moe_wrapped", False):
            pass


def print_moe_summary(model, title="MoE probe"):
    print(f"\n===== {title} =====")
    layer_id = 0
    any_layer = False
    for m in model.modules():
        if isinstance(m, MoEMLP):
            any_layer = True
            layer_id += 1
            st: Dict = getattr(m, "last_stats", None)
            if not st:
                print(f"[moe@layer{layer_id}] no stats")
                continue
            print(f"[moe@layer{layer_id}] N={st['N']} E={st['E']} cap={st['cap']} "
                  f"fallback={st['fallback_total']} ({st['fallback_frac']*100:.2f}%) "
                  f"top1: mean={st['top1_prob_mean']:.3f} p95={st['top1_prob_p95']:.3f} p99={st['top1_prob_p99']:.3f}")
            print(f"  per_expert_count={st['per_expert_count']}")
            if st['fallback_total'] > 0:
                print(f"  fallback_per_expert={st['fallback_per_expert']}")
    if not any_layer:
        print("[warn] model has no MoEMLP layers (use_moe=False)")


def quick_probe_once(model, tokenizer, max_len=256):
    model.eval()
    with torch.no_grad():
        texts = [
            "你是谁？简单回答一句话。",
            "介绍一下MoE路由的工作原理。",
            "给我一个PyTorch中注册forward_hook的例子。",
        ]
        batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        _ = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    print_moe_summary(model, title="single forward on tiny batch")


def to_messages(ex: Dict) -> List[Dict]:
    if "messages" in ex and isinstance(ex["messages"], list):
        return ex["messages"]
    raise ValueError("样本缺少 messages ")


class DataCollatorSFT:
    def __init__(self, pad_id: int, ignore_index: int):
        self.pad_id = pad_id
        self.ignore_index = ignore_index

    def __call__(self, features: List[Dict]):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            ids = f["input_ids"]; lbs = f["labels"]
            pad = max_len - len(ids)
            input_ids.append(torch.tensor(ids + [self.pad_id] * pad, dtype=torch.long))
            labels.append(torch.tensor(lbs + [self.ignore_index] * pad, dtype=torch.long))
            attn.append(torch.tensor([1] * len(ids) + [0] * pad, dtype=torch.long))
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attn),
        }


def reset_gamma_to_target(model, target_eff=0.25, gmin=0.02, gmax=0.5):
    assert gmin < target_eff < gmax, "target_eff 必须在 (gmin, gmax) 内"
    ratio = (target_eff - gmin) / (gmax - gmin)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("gamma_att") or name.endswith("gamma_mlp"):
                bad = ~torch.isfinite(p)
                if bad.any():
                    p[bad] = 0.0
                raw_target = torch.logit(torch.tensor(ratio, dtype=p.dtype, device=p.device))
                p.fill_(raw_target)


class UnfreezeGammaCallback(TrainerCallback):
    def __init__(self, unfreeze_step=10000, gamma_lr_mult=0.1):
        self.unfreeze_step = unfreeze_step
        self.gamma_lr_mult = gamma_lr_mult


    def on_step_end(self, args, state, control, **kwargs):
        sched = kwargs.get("lr_scheduler", None)
        if  state.global_step < self.unfreeze_step:
            return control

        optimizer = kwargs["optimizer"]
        base_lrs = sched.get_last_lr() if sched is not None else [g["lr"] for g in optimizer.param_groups]
        ref_lr = max(base_lrs)
        for group in optimizer.param_groups:

            if group.get("name", "") == "gamma":
                group["lr"] = ref_lr* self.gamma_lr_mult

        return control



def main():

    global tokenizer, ASSIST_START_ID, SYSTEM_START_ID, USER_START_ID, IM_END_ID, NEWLINE_IDS, raw_ds

    #tokenizer 与 special ids
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("../model")
    assert tokenizer.eos_token_id is not None, "需要 eos_token用于样本连接与分块"
    assert tokenizer.pad_token_id is not None, "需要 pad_token_id；没有的话可设为 eos/endoftext"
    tokenizer.padding_side = "right"
    ASSIST_START_ID = tokenizer.convert_tokens_to_ids("<|im_start|>assistant")
    SYSTEM_START_ID = tokenizer.convert_tokens_to_ids("<|im_start|>system")
    USER_START_ID = tokenizer.convert_tokens_to_ids("<|im_start|>user")
    IM_END_ID = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert ASSIST_START_ID is not None and IM_END_ID is not None and SYSTEM_START_ID is not None and USER_START_ID is not None, \
        "词表中必须包含 <|im_start|>assistant 和 <|im_end|> 这两个特殊 token"
    NEWLINE_IDS = tokenizer.encode("\n", add_special_tokens=False)

    #原始数据加载
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "SFTshort.jsonl")
    data_path = os.path.abspath(data_path)
    raw_ds = load_dataset("json", data_files=data_path, split="train")


    def build_sft_ids_and_labels(ex: Dict) -> Dict[str, List[int]]:
        ids: List[int] = tokenizer.apply_chat_template(
            to_messages(ex),
            tokenize=True,
            add_generation_prompt=False,
            truncation=False,
        )
        if len(ids) > MAX_LEN:
            return {"input_ids": [], "labels": []}
        if tokenizer.unk_token_id in ids:
            return {"input_ids": [], "labels": []}
        labels = [IGNORE_INDEX] * len(ids)
        i, L = 0, len(ids)
        while i < L:
            if ids[i] == ASSIST_START_ID:
                j = i + 1
                if NEWLINE_IDS and ids[j:j + len(NEWLINE_IDS)] == NEWLINE_IDS:
                    j += len(NEWLINE_IDS)
                k = j
                if k - 1 >= 0:
                    labels[k - 1] = ids[k]
                while k < L and ids[k] != IM_END_ID:
                    labels[k] = ids[k + 1]
                    k += 1
                i = k
            i += 1
        return {"input_ids": ids, "labels": labels}


    if os.path.exists(PK_CACHE) and not REPAK:
        tokenized = load_from_disk(PK_CACHE)
    else:
        tokenized = raw_ds.map(
            build_sft_ids_and_labels,
            remove_columns=raw_ds.column_names,
            num_proc=1,
            desc="Build SFT samples via chat_template",
        )
        tokenized = tokenized.filter(lambda ex: len(ex["input_ids"]) > 0)
        tokenized.save_to_disk(PK_CACHE)

    split = tokenized.train_test_split(test_size=eval_frac, seed=SEED)
    train_ds, eval_ds = split["train"], split["test"]

    train_ds = train_ds.with_format(type=None)
    eval_ds = eval_ds.with_format(type=None)

    collator = DataCollatorSFT(pad_id=tokenizer.pad_token_id, ignore_index=IGNORE_INDEX)

    cfg = Config(
        vocab_size=tokenizer.vocab_size,
        train_maxlength=MAX_LEN,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        unk_token_id=tokenizer.unk_token_id
    )
    cfg.use_checkpoint = use_checkpoint
    cfg.checkpoint_use_reentrant = checkpoint_use_reentrant
    model = TinyLLM(cfg)

    last_ckpt = get_last_checkpoint("runs/tinyllm_SFTshort1")
    if last_ckpt is not None:
        sf_path = os.path.join(last_ckpt, "model.safetensors")
        state = safe_load(sf_path, device="cpu")
        _ = model.load_state_dict(state, strict=False)

    # reset_gamma_to_target(model, target_eff=0.4, gmin=0.2, gmax=0.7)#gamma 过于极端的时候暴力重设
    #
    save_eval_interval = max(1, len(train_ds) // (BATCH * 10))
    # raw_att, raw_mlp = [], []
    # eff_att, eff_mlp = [], []
    #
    # for i, blk in enumerate(model.blocks):#检查gamma值
    #     if hasattr(blk, "gamma_att_raw"):
    #         gatt_raw = float(blk.gamma_att_raw.item())
    #         gmlp_raw = float(blk.gamma_mlp_raw.item())
    #         gatt_eff = float(blk.gamma_min + (0.7 - 0.2) * torch.sigmoid(blk.gamma_att_raw))
    #         gmlp_eff = float(blk.gamma_min + (0.7 - 0.2) * torch.sigmoid(blk.gamma_mlp_raw))
    #     else:
    #         gatt_raw = float(blk.gamma_att.item())
    #         gmlp_raw = float(blk.gamma_mlp.item())
    #         gatt_eff = float(blk._bounded(blk.gamma_att))
    #         gmlp_eff = float(blk._bounded(blk.gamma_mlp))
    #     raw_att.append(gatt_raw); raw_mlp.append(gmlp_raw)
    #     eff_att.append(gatt_eff); eff_mlp.append(gmlp_eff)
    #
    # def show(name, arr):
    #     arr = np.array(arr, dtype=np.float64)
    #     print(f"{name}: min={arr.min():.6g}  p5={np.percentile(arr,5):.6g}  "
    #           f"median={np.median(arr):.6g}  p95={np.percentile(arr,95):.6g}  max={arr.max():.6g}")
    #
    # print("== RAW (pre-sigmoid, just for debugging) ==")
    # show("gamma_att_raw", raw_att)
    # show("gamma_mlp_raw", raw_mlp)
    #
    # print("\n== EFFECTIVE (after bounded mapping, truly used in forward) ==")
    # show("gamma_att_eff", eff_att)
    # show("gamma_mlp_eff", eff_mlp)
    #
    # attach_moe_probe(model)
    # quick_probe_once(model, tokenizer)#检查moe的概率，判断路由是否极端

    args = TrainingArguments(
        output_dir="runs/tinyllm_SFTshort2",
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH if eval_ds else None,
        gradient_accumulation_steps=1,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=save_eval_interval,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": LR * 0.1},
        bf16=True,
        tf32=True,
        gradient_checkpointing=False,
        max_grad_norm=1.0,
        dataloader_num_workers=0,
        dataloader_persistent_workers=False,
        dataloader_pin_memory=True,
        evaluation_strategy="steps" if eval_ds else "no",
        eval_steps=save_eval_interval if eval_ds else None,
        report_to="tensorboard",
        run_name="tinyllm-pretrain",
        logging_dir="runs/tensorboard",
        save_total_limit=2,
    )

    def param_groups_no_decay(model, base_lr, weight_decay):
        decay, no_decay, gamma = [], [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            is_gamma = any(t in n for t in ["gamma_att_raw", "gamma_mlp_raw", "gamma_att", "gamma_mlp"])
            if is_gamma:
                gamma.append(p)
                continue
            if p.ndim == 1 or any(k in n for k in ["norm", "bias", "tok_embed", "router", "decider"]):
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "weight_decay": weight_decay, "lr": base_lr,"name": "delay"},
            {"params": no_decay, "weight_decay": 0.0, "lr": base_lr * 0.3, "name": "no_delay"},
            {"params": gamma, "weight_decay": 0.0, "lr": base_lr * 0.2, "name": "gamma"},
        ]

    optimizer = torch.optim.AdamW(param_groups_no_decay(model, LR, 0.01), betas=(0.9, 0.95))
    steps = math.ceil(len(train_ds) / (BATCH * args.gradient_accumulation_steps)) * EPOCHS
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        optimizers=(optimizer, None),
        # callbacks=[
        #     # UnfreezeGammaCallback(unfreeze_step=40000, gamma_lr_mult=0.1),#暴力重置gamma后退火学习
        #     # Top2AlphaWarmup(ratio=0.03, target=1.0),#从top1MOE改为top2，退火学习逐渐恢复学习率
        # ],
    )
    trainer.train()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
