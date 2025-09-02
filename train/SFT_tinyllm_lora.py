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
import torch.nn as nn
from transformers import TrainerCallback
from model.model import LoraLinear
from collections import OrderedDict

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
EPOCHS = 1
SAVE_STEPS = 2000
IGNORE_INDEX = -100
eval_frac = 0.0001  # 分割eval比例
REPAK = False
torch.manual_seed(SEED)
CACHE_DIR = "cache_SFTshortest"
TOK_CACHE = os.path.join(CACHE_DIR, f"tokenized")
PK_CACHE = os.path.join(CACHE_DIR, f"packed_len{MAX_LEN}")
def savelora(model,adapter_name):
    state=OrderedDict()
    for module_name, module in model.named_modules():
        for attr, child in module._modules.items():
            if isinstance(child, LoraLinear) and adapter_name in child.adapters:
                key = f"{module_name}.{attr}" if module_name else attr
                state[key] = child.adapters[adapter_name].state_dict()
    return state


def iter_linear(module:nn.Module):
    for module_name,module in module.named_modules():
        for child_name,child in list(module.__dict__.get("_modules",{}).items()):
            if not isinstance(child, nn.Linear):
                continue
            name=f"{module_name}.{child_name}" if module_name else child_name
            yield name,module,child_name,child

def inject_lora(model:nn.Module,target_patterns=("w_q","w_k","w_v","w_o","upsamp","downsamp","swiGate"),mode="exclusive",cap_norm=None, global_scale=1.0):

    pat = re.compile("|".join([f"(?:{p})" for p in target_patterns]))
    for name,module,child_name,child in iter_linear(model):
        if not pat.search(child_name):
            continue
        if isinstance(child,LoraLinear):
            continue
        lora=LoraLinear(child,mode,cap_norm,global_scale)
        setattr(module,child_name,lora)
    return

def lora_register_activate_all(model: nn.Module,adapter_name="sft_lora",rank=8, alpha=16.0, dropout=0.05,weight=1.0,exclusive=True):

    count = 0
    for m in model.modules():
        if isinstance(m, LoraLinear):
            if adapter_name not in m.adapters:
                m.register_adapter(adapter_name, rank=rank, dropout=dropout, alpha=alpha, state_dict=None, weight=weight)
            m.activate(adapter_name, exclusive=exclusive)
            count += 1
    return count
def lora_train(model:nn.Module):
    trainble=[]
    for p in model.parameters():
        p.requires_grad_(False)
    from model.model import LoraLinear
    for m in model.modules():
        if isinstance(m,LoraLinear):
            for name,adp in m.adapters.items():
                for p in adp.parameters():
                    p.requires_grad_(True)
                    trainble.append(p)
    return trainble




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

    #预处理缓存
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


    save_eval_interval = max(1, len(train_ds) // (BATCH * 10))

    args = TrainingArguments(
        output_dir="runs/tinyllm_SFTshort3",
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


    inject_lora(model)
    lora_register_activate_all(model, adapter_name="sft_lora", rank=8, alpha=16.0, dropout=0.05)
    lora_params = lora_train(model)
    optimizer = torch.optim.AdamW(lora_params, betas=(0.9, 0.95),lr=2e-4,weight_decay=0.0)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        optimizers=(optimizer, None),

    )
    trainer.train()
    out_dir = args.output_dir
    os.makedirs(os.path.join(out_dir, "adapters"), exist_ok=True)
    torch.save(savelora(model, "sft_lora"),
               os.path.join(out_dir, "adapters", "sft_lora.pt"))

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
