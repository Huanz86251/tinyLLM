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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
USE_BF16 = True
AUTOCAST_DTYPE=torch.bfloat16 if USE_BF16 else torch.float16
use_checkpoint=False
checkpoint_use_reentrant=True

# ========== 基本配置 ==========

MAX_LEN = 2048
SEED = 42
BATCH = 2
LR = 3e-4
EPOCHS = 1
SAVE_STEPS = 2000
IGNORE_INDEX = -100
eval_frac = 0.0001 #分割eval比例
REPAK=False
torch.manual_seed(SEED)
CACHE_DIR = "cache_pretain_more"
TOK_CACHE = os.path.join(CACHE_DIR, f"tokenized")
PK_CACHE  = os.path.join(CACHE_DIR, f"packed_len{MAX_LEN}")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("../model")
assert tokenizer.eos_token_id is not None, "需要 eos_token用于样本连接与分块"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "wudao_filtered_more.jsonl")
DATA_PATH = os.path.abspath(DATA_PATH)
raw_ds = load_dataset("json", data_files=DATA_PATH, split="train")


def tokenize_fn(batch):
    texts = [t for t in batch["text"]]
    out = tokenizer(texts, add_special_tokens=False,truncation=False,padding=False,return_attention_mask=False)
    eos = tokenizer.eos_token_id
    input_ids = [ids + [eos] for ids in out["input_ids"]]
    return {"input_ids": input_ids}
def group_texts(examples):
    concatenated = list(chain.from_iterable(examples["input_ids"])) # 展平
    total_len = (len(concatenated) // MAX_LEN) * MAX_LEN
    if total_len == 0:
        return {"input_ids": []}  # 这批太短，跳过
    blocks = [concatenated[i:i + MAX_LEN] for i in range(0, total_len, MAX_LEN)]
    return {"input_ids": blocks}


if os.path.exists(PK_CACHE) and not REPAK:
    packed = load_from_disk(PK_CACHE)
else:
    tokenized = raw_ds.map(tokenize_fn, batched=True, remove_columns=raw_ds.column_names,num_proc=1,batch_size=10000,desc="Tokenizing")
    packed = tokenized.map(group_texts,batched=True,num_proc=1,batch_size=20000,desc="Packing")
    packed.save_to_disk(PK_CACHE)

split = packed.train_test_split(test_size=eval_frac, seed=SEED)
train_ds, eval_ds = split["train"], split["test"]
train_ds=train_ds.with_format("torch",columns=["input_ids"])
eval_ds=eval_ds.with_format("torch",columns=["input_ids"])

class CausalShiftCollator:
    def __init__(self, ignore_index=IGNORE_INDEX):
        self.ignore_index = ignore_index
    def __call__(self, batch):
        input_ids = default_collate([e["input_ids"] for e in batch]).long()  # [B, T]
        labels = input_ids.clone()
        labels[:,:-1]=input_ids[:,1:]
        labels[:,-1]=self.ignore_index
        return {"input_ids": input_ids, "labels": labels}


collator = CausalShiftCollator()


cfg = Config(
    vocab_size=tokenizer.vocab_size,
    train_maxlength=MAX_LEN,
)

cfg.use_checkpoint = use_checkpoint
cfg.checkpoint_use_reentrant = checkpoint_use_reentrant
model = TinyLLM(cfg)
save_eval_interval = len(train_ds) // (BATCH * 10)

args = TrainingArguments(
    output_dir="runs/tinyllm_2",
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH if eval_ds else None,
    gradient_accumulation_steps=1,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    warmup_ratio=0.01,
    weight_decay=0.01,
    logging_steps=50,
    save_steps=save_eval_interval,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr": LR * 0.1},
    bf16=True,
    tf32=True,  # 允许 TF32
    gradient_checkpointing=False,
    max_grad_norm=1.0,
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    evaluation_strategy="steps" if eval_ds else "no",
    eval_steps=save_eval_interval if eval_ds else None,
    report_to="tensorboard",
    run_name="tinyllm-pretrain",
    logging_dir="runs/tensorboard",
    save_total_limit=2,
)


#no-decay设置
def param_groups_no_decay(model):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or any(
                k in n for k in ["norm", "bias", "tok_embed", "gamma_att", "gamma_mlp","router","decider"]):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


optimizer = torch.optim.AdamW(param_groups_no_decay(model), lr=args.learning_rate, betas=(0.9, 0.95))
# old_ckpt = get_last_checkpoint("runs/tinyllm")#恢复训练
# if old_ckpt is None:
#     raise ValueError("找不到旧 checkpoint")
# sf_path = os.path.join(old_ckpt, "model.safetensors")
# state = safe_load(sf_path, device="cpu")
# missing, unexpected = model.load_state_dict(state, strict=False)
# print("[LOAD] missing:", missing[:10], "unexpected:", unexpected[:10])
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
    tokenizer=tokenizer,
    optimizers=(optimizer, None),
)

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    last_ckpt = get_last_checkpoint("runs/tinyllm")
    if last_ckpt:
        print(f"[RESUME] Found checkpoint: {last_ckpt} -> resume training")
        trainer.train(resume_from_checkpoint=last_ckpt)
    else:
        print("[RESUME] No checkpoint found -> start fresh")
        trainer.train()
