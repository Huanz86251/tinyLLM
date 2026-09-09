from project_paths import legacy_path, path as project_path
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from local_datasets import load_local_dataset
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer
from contextlib import contextmanager
from model.config import Config
from model.model import TinyLLM

# =========================
# 环境 & 路径（对齐你 CPT 脚本）
# =========================

# 显存分段（和 CPT 一致）
if os.name == "nt":
    # Windows CUDA does not support expandable_segments.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.8"
else:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
LORA_TARGET = "attn_mlp_skip_first4"
# 是否强制 HF 离线（默认 0 = 不强制；你要和 CPT 一样关网，可以 export FORCE_OFFLINE=1）
FORCE_OFFLINE = bool(int(os.environ.get("FORCE_OFFLINE", "0")))
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

# TensorBoard 主目录对齐：
# /root/autodl-tmp/llm/checkpoints/tb/<run-name>
RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_NAME = f"tinyllm-grpo-gsm8k-{RUN_ID}"
TB_ROOT = str(project_path("runs/tensorboard"))
DEFAULT_TB_DIR = os.path.join(TB_ROOT, RUN_NAME)

# LoRA ckpt 保存根目录（每个 run 建一个子目录）
GRPO_SAVE_ROOT_DEFAULT = str(project_path("runs/training/grpo_gsm8k"))

# =========================
# Global config
# =========================
LORA_ADAPTER_NAME = "math"
LORA_RANK = 64
LORA_DROPOUT = 0.0
LORA_ALPHA = 32.0

DTYPE = torch.bfloat16
DTYPE_EVAL = torch.float32
IGNORE_INDEX = -100
GRPO_DEBUG_PRINT_SAMPLES = bool(int(os.environ.get("GRPO_DEBUG_PRINT_SAMPLES", "0")))
GSM8K_EVAL_BATCH_SIZE_QUESTIONS = int(os.environ.get("GSM8K_EVAL_BATCH_SIZE", "8"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRPO_MAX_NEW_TOKENS_TRAIN = int(os.environ.get("GRPO_MAX_NEW_TOKENS_TRAIN", "600"))
GSM8K_SPLIT = os.environ.get("GSM8K_SPLIT", "train")   # "train" 或 "test"
GSM8K_MAX_EXAMPLES = int(os.environ.get("GSM8K_MAX_EXAMPLES", "0"))  # 0 = 用完整 split
GSM8K_BATCH_SIZE_QUESTIONS = int(os.environ.get("GSM8K_BATCH_SIZE", "2"))  # 每个 step 多少道题
GRPO_NUM_GENERATIONS = int(os.environ.get("GRPO_NUM_GENERATIONS", "10"))    # 每题采样几条
GRPO_EVAL_EVERY_STEPS = int(os.environ.get("GRPO_EVAL_EVERY_STEPS", "100"))
GRPO_EVAL_NUM_QUESTIONS = int(os.environ.get("GRPO_EVAL_NUM_QUESTIONS", "1319"))
GSM8K_EVAL_SPLIT = os.environ.get("GSM8K_EVAL_SPLIT", "test")

SAVE_STEPS = int(os.environ.get("GRPO_SAVE_STEPS", "100"))

# 梯度累积步数（默认 2，相当于 global batch = GSM8K_BATCH_SIZE_QUESTIONS * GRPO_GRAD_ACCUM_STEPS）
GRPO_GRAD_ACCUM_STEPS = int(os.environ.get("GRPO_GRAD_ACCUM_STEPS", "4"))

# ===== CUDA / TF32 / SDP（按你 CPT 脚本风格）=====
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
try:
    from torch.backends.cuda import sdp_kernel
    # PyTorch 2.1+ 新接口
    sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True)
except Exception:
    pass

# 为了 HF & manual 采样可复现
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ---------- Prompt & system COT ----------

SYSTEM_PROMPT_FOR_COT = (
    "Think step by step inside <|thought_start|> and <|thought_end|>. Then provide final answer."
)
USE_SYSTEM_PROMPT_FOR_COT = True

# =========================================================
# TinyLLM loading helpers
# =========================================================

def build_gsm8k_prompt(question: str) -> str:
    q = question.strip()
    return (
        f"{q}\n"
        f"Put your final answer in LaTeX boxed form like $\\boxed{{answer}}$. "
    )


def _clean_state_dict_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    tmp = {}
    for k, v in state.items():
        if k.startswith("module."):
            tmp[k[len("module."):]] = v
        else:
            tmp[k] = v

    out = {}
    for k, v in tmp.items():
        if re.match(r"^\d+\.", k):
            out[f"blocks.{k}"] = v
        else:
            out[k] = v
    return out


def load_cfg_from_json(cfg_path: str, tokenizer) -> Config:
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    raw.setdefault("vocab_size", len(tokenizer.get_vocab()))
    raw.setdefault("bos_token_id", tokenizer.bos_token_id)
    raw.setdefault("eos_token_id", tokenizer.eos_token_id)
    raw.setdefault("pad_token_id", tokenizer.pad_token_id)
    raw.setdefault("train_maxlength", raw.get("train_maxlength", 2048))
    raw.setdefault("ignore_index", IGNORE_INDEX)

    cfg = Config(**{
        k: v for k, v in raw.items()
        if k in Config.__init__.__code__.co_varnames
    })
    for k, v in raw.items():
        if not hasattr(cfg, k):
            setattr(cfg, k, v)
    if not hasattr(cfg, "ignore_index"):
        cfg.ignore_index = IGNORE_INDEX
    return cfg


def build_fallback_cfg(tokenizer) -> Config:
    cfg = Config(
        vocab_size=len(tokenizer.get_vocab()),
        train_maxlength=2048,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        moe_use_detach=False,
        use_moe=False,
        dropout=0.0,
        learnable_temp=True,
        drop_path=0.0,
        residual_dropout=0.00,
        rope_type="yarn",
    )
    cfg.ignore_index = IGNORE_INDEX
    cfg.use_checkpoint = False
    cfg.checkpoint_use_reentrant = False
    return cfg


def load_tinyllm_from_ckpt(
    ckpt_dir: str,
    tokenizer,
    strict: bool = False
) -> TinyLLM:
    ckpt_dir = Path(ckpt_dir)
    safepath = ckpt_dir / "model.safetensors"
    binpath  = ckpt_dir / "pytorch_model.bin"
    cfgpath  = ckpt_dir / "config.json"

    if safepath.is_file():
        raw_state = load_file(str(safepath), device="cpu")
        picked_path = safepath
    elif binpath.is_file():
        raw_state = torch.load(str(binpath), map_location="cpu")
        picked_path = binpath
    else:
        raise FileNotFoundError(
            f"No model.safetensors or pytorch_model.bin in {ckpt_dir}"
        )

    state = _clean_state_dict_keys(raw_state)

    if cfgpath.is_file():
        cfg = load_cfg_from_json(str(cfgpath), tokenizer)
    else:
        cfg = build_fallback_cfg(tokenizer)

    cfg.ignore_index = IGNORE_INDEX

    model = TinyLLM(cfg)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    print(f"[load-ok] {picked_path}")
    print("  missing[:10]   =", missing[:10],   f"(total {len(missing)})")
    print("  unexpected[:10]=", unexpected[:10],f"(total {len(unexpected)})")

    model.to(DEVICE, dtype=DTYPE)
    model.eval()
    return model

# =========================================================
# ChatML prompt 构造
# =========================================================

def _build_stop_sets(tokenizer, stop_on_punct: bool = False):
    eos_id = tokenizer.eos_token_id
    sent_end_ids = set()
    if stop_on_punct:
        for ch in ["。", "！", "？", ".", "!", "?"]:
            ids = tokenizer.encode(ch, add_special_tokens=False)
            if ids:
                sent_end_ids.add(int(ids[0]))
    return eos_id, sent_end_ids


class GSM8KDataset(Dataset):
    """
    HF gsm8k 封装；max_examples=0 表示全量。
    """
    def __init__(
        self,
        split: str = GSM8K_SPLIT,
        max_examples: int = GSM8K_MAX_EXAMPLES,
        seed: int = 42,
    ):
        ds = load_local_dataset("gsm8k", split)
        ds = ds.shuffle(seed=seed)
        if max_examples > 0:
            max_examples = min(max_examples, len(ds))
            ds = ds.select(range(max_examples))
        self._ds = ds

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        ex = self._ds[idx]
        return {
            "question": ex["question"],
            "answer": ex["answer"],
        }


def gsm8k_collate_fn(batch: List[Dict[str, str]]) -> Dict[str, List[str]]:
    questions = [item["question"] for item in batch]
    answers = [item["answer"] for item in batch]
    prompts = [build_gsm8k_prompt(q) for q in questions]
    return {
        "questions": questions,
        "answers": answers,
        "prompts": prompts,
    }


def make_gsm8k_dataloader(
    split: str = GSM8K_SPLIT,
    batch_size: int = GSM8K_BATCH_SIZE_QUESTIONS,
    max_examples: int = GSM8K_MAX_EXAMPLES,
    seed: int = 42,
) -> DataLoader:
    dataset = GSM8KDataset(split=split, max_examples=max_examples, seed=seed)
    print(f"[GSM8K] split={split}, total examples={len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=gsm8k_collate_fn,
        drop_last=False,
    )
    return loader

# ========= 全局格式相关配置 =========

BOX_RE = re.compile(
    r"\\boxed\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}",
    flags=re.MULTILINE
)

THOUGHT_START = "<|thought_start|>"
THOUGHT_END   = "<|thought_end|>"

def parse_gsm8k_prediction(
    pred_str: str,
    tokenizer,
) -> Dict:
    if pred_str is None:
        pred_str = ""
    raw = pred_str.strip()

    # ===== 1) 思维链内容 =====
    start_idx = raw.find(THOUGHT_START)
    end_idx   = raw.find(THOUGHT_END)

    has_cot_tags = (start_idx != -1) and (end_idx != -1) and (end_idx > start_idx)
    cot_text = None
    cot_token_len = 0

    if has_cot_tags:
        inner = raw[start_idx + len(THOUGHT_START): end_idx]
        cot_text = inner.strip()
        if cot_text:
            cot_token_ids = tokenizer.encode(
                cot_text,
                add_special_tokens=False,
            )
            cot_token_len = len(cot_token_ids)

    # ===== 2) boxed 里的答案 =====
    boxed_match = BOX_RE.search(raw)
    has_box = boxed_match is not None
    boxed_val: Optional[float] = None
    if has_box:
        try:
            boxed_val = float(boxed_match.group(1))
        except Exception:
            boxed_val = None

    # ===== 3) fallback：最后一个数字 =====
    fallback_val: Optional[float] = None
    text_no_commas = raw.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", text_no_commas)
    if nums:
        try:
            fallback_val = float(nums[-1])
        except Exception:
            fallback_val = None

    # ===== 4) 选最终答案 =====
    if boxed_val is not None:
        final_answer = boxed_val
        from_box = True
        has_box_and_valid = True
    else:
        final_answer = fallback_val
        from_box = False
        has_box_and_valid = False

    return {
        "raw": raw,
        "has_box": bool(has_box),
        "boxed_value": boxed_val,
        "fallback_value": fallback_val,
        "final_answer": final_answer,
        "from_box": from_box,
        "has_box_and_valid": has_box_and_valid,
        "has_cot_tags": bool(has_cot_tags),
        "cot_text": cot_text,
        "cot_token_len": int(cot_token_len),
    }

def render_prompt_with_tokenizer(tokenizer, user_prompt: str, system_prompt: Optional[str]):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# =========================================================
# Sampling helpers（手写 increment decode）
# 采样阶段彻底 no_grad，不建计算图
# =========================================================

def _apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: List[List[int]],
    penalty: float,
):
    if penalty is None or penalty == 1.0:
        return logits

    B, V = logits.shape
    for b in range(B):
        hist = generated_ids[b]
        if not hist:
            continue
        for tid in set(hist):
            if tid < 0 or tid >= V:
                continue
            val = logits[b, tid]
            if val > 0:
                logits[b, tid] = val / penalty
            else:
                logits[b, tid] = val * penalty
    return logits


def _calc_banned_tokens_for_ngram(
    gen_tokens: List[int],
    no_repeat_ngram_size: int,
) -> List[int]:
    n = no_repeat_ngram_size
    if n <= 0 or len(gen_tokens) < n:
        return []

    ngram_dict: Dict[Tuple[int, ...], set] = {}
    for i in range(len(gen_tokens) - n + 1):
        ngram = gen_tokens[i: i + n]
        prefix = tuple(ngram[:-1])
        next_tok = ngram[-1]
        if prefix not in ngram_dict:
            ngram_dict[prefix] = set()
        ngram_dict[prefix].add(next_tok)

    prefix = tuple(gen_tokens[-(n - 1):])
    banned = ngram_dict.get(prefix, set())
    return list(banned)


def _apply_no_repeat_ngram(
    logits: torch.Tensor,
    generated_ids: List[List[int]],
    no_repeat_ngram_size: int,
):
    if no_repeat_ngram_size is None or no_repeat_ngram_size <= 0:
        return logits

    B, V = logits.shape
    for b in range(B):
        gen = generated_ids[b]
        banned = _calc_banned_tokens_for_ngram(gen, no_repeat_ngram_size)
        if not banned:
            continue
        for tid in banned:
            if 0 <= tid < V:
                logits[b, tid] = float("-inf")
    return logits


def sample_for_grpo_manual(
    model: TinyLLM,
    tokenizer,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    num_generations: int = 8,
    max_new_tokens: int = 700,
    temperature: float = 0.4,
    top_p: float = 0.9,
    stop_on_punct: bool = False,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 0,
    greedy: bool = False,
):
    """
    注意：这里已经改成纯推理：
    - 用 no_grad 包裹
    - 不再返回带梯度的 logprob_sums
    """
    was_training = model.training
    model.eval()

    chatml = render_prompt_with_tokenizer(tokenizer, user_prompt, system_prompt)

    ctx_ids_1 = tokenizer.encode(
        chatml,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(DEVICE)  # [1, T0]

    B = num_generations
    ctx_ids = ctx_ids_1.expand(B, -1).contiguous()
    attn_ctx = torch.ones_like(ctx_ids, dtype=torch.long, device=DEVICE)

    eos_id, sent_end_ids = _build_stop_sets(tokenizer, stop_on_punct=stop_on_punct)
    pad_id = tokenizer.pad_token_id
    unk_id = getattr(getattr(model, "cfg", None), "unk_token_id", None)
    if isinstance(unk_id, torch.Tensor):
        unk_id = int(unk_id.item())

    try:
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    except Exception:
        im_end_id = None

    generated_ids: List[List[int]] = [[] for _ in range(B)]
    finished: List[bool] = [False] * B

    def _top_p_filtering_row(row: torch.Tensor, top_p_val: float) -> torch.Tensor:
        if top_p_val >= 1.0 or top_p_val <= 0.0:
            return row
        sorted_logits, sorted_indices = torch.sort(row, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p_val
        sorted_indices_to_remove[..., 0] = False
        sorted_logits[sorted_indices_to_remove] = float("-inf")
        return row.scatter(0, sorted_indices, sorted_logits)

    with torch.no_grad():
        # prefill
        prefill = model(
            input_ids=ctx_ids,
            attention_mask=attn_ctx,
            labels=None,
            use_cache=True,
            past_states=None,
        )
        past_states = prefill["past_states"]
        logits = prefill["logits"][:, -1, :]

        for _ in range(max_new_tokens):
            if all(finished):
                break

            logits_step = logits.clone()

            # 屏蔽 pad / unk
            if pad_id is not None and pad_id != eos_id and 0 <= int(pad_id) < logits_step.size(-1):
                logits_step[:, int(pad_id)] = float("-inf")
            if unk_id is not None and 0 <= int(unk_id) < logits_step.size(-1):
                logits_step[:, int(unk_id)] = float("-inf")

            # finished 的样本强制采 eos
            if eos_id is not None and any(finished):
                mask = torch.tensor(finished, dtype=torch.bool, device=DEVICE)
                logits_step[mask] = float("-inf")
                logits_step[mask, int(eos_id)] = 0.0

            logits_step = _apply_no_repeat_ngram(
                logits_step,
                generated_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            logits_step = _apply_repetition_penalty(
                logits_step,
                generated_ids,
                penalty=repetition_penalty,
            )

            if greedy:
                # 纯 greedy：直接取 argmax，忽略 temperature / top_p
                next_ids = torch.argmax(logits_step, dim=-1, keepdim=True)
            else:
                # 采样模式（和之前一样）
                if temperature is not None and temperature > 0.0 and temperature != 1.0:
                    logits_step = logits_step / temperature

                if top_p is not None and 0.0 < top_p < 1.0:
                    new_rows = []
                    for b in range(B):
                        row = logits_step[b]
                        row = _top_p_filtering_row(row, top_p)
                        new_rows.append(row.unsqueeze(0))
                    logits_step = torch.cat(new_rows, dim=0)

                logprobs_step = F.log_softmax(logits_step.float(), dim=-1)
                probs_step = logprobs_step.exp()
                next_ids = torch.multinomial(probs_step, num_samples=1)

            for b in range(B):
                if finished[b]:
                    continue

                nid = int(next_ids[b, 0].item())
                generated_ids[b].append(nid)

                stop = False
                partial_text = tokenizer.decode(
                    generated_ids[b],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                if BOX_RE.search(partial_text):
                    stop = True
                if eos_id is not None and nid == int(eos_id):
                    stop = True
                if (not stop) and (im_end_id is not None) and nid == int(im_end_id):
                    stop = True
                if stop_on_punct and (not stop):
                    if nid in sent_end_ids and len(generated_ids[b]) >= 30:
                        stop = True

                if stop:
                    finished[b] = True

            attn_one = torch.ones_like(next_ids, dtype=torch.long, device=DEVICE)
            step_out = model(
                input_ids=next_ids.to(DEVICE),
                attention_mask=attn_one,
                labels=None,
                use_cache=True,
                past_states=past_states,
            )
            past_states = step_out["past_states"]
            logits = step_out["logits"][:, -1, :]

    # 还原训练/评估模式
    if was_training:
        model.train()

    texts: List[str] = []
    for b in range(B):
        text = tokenizer.decode(
            generated_ids[b],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        texts.append(text.strip())

    return {
        "texts": texts,
        "token_ids": generated_ids,
    }

def sample_for_grpo_prompt_batch(
    model: TinyLLM,
    tokenizer,
    user_prompts: List[str],
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 700,
    temperature: float = 0.4,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 0,
):
    """Sample one trajectory for each different prompt in a real GPU batch."""
    if not user_prompts:
        return {"texts": [], "token_ids": []}
    was_training = model.training
    model.eval()
    encoded = [torch.tensor(tokenizer.encode(
        render_prompt_with_tokenizer(tokenizer, prompt, system_prompt),
        add_special_tokens=False), dtype=torch.long) for prompt in user_prompts]
    batch_size = len(encoded)
    pad_id = tokenizer.pad_token_id
    eos_id, _ = _build_stop_sets(tokenizer, stop_on_punct=False)
    if pad_id is None:
        pad_id = eos_id
    max_context = max(item.numel() for item in encoded)
    ctx_ids = torch.full((batch_size, max_context), int(pad_id),
                         dtype=torch.long, device=DEVICE)
    attention = torch.zeros_like(ctx_ids)
    for row, ids in enumerate(encoded):
        length = ids.numel()
        ctx_ids[row, -length:] = ids.to(DEVICE)
        attention[row, -length:] = 1
    unk_id = getattr(getattr(model, "cfg", None), "unk_token_id", None)
    if isinstance(unk_id, torch.Tensor):
        unk_id = int(unk_id.item())
    try:
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    except Exception:
        im_end_id = None
    generated_ids: List[List[int]] = [[] for _ in range(batch_size)]
    finished = [False] * batch_size
    with torch.no_grad():
        prefill = model(input_ids=ctx_ids, attention_mask=attention, labels=None,
                        use_cache=True, past_states=None)
        past_states = prefill["past_states"]
        logits = prefill["logits"][:, -1, :]
        for _ in range(max_new_tokens):
            if all(finished):
                break
            step_logits = logits.clone()
            vocab = step_logits.size(-1)
            if pad_id is not None and pad_id != eos_id and 0 <= int(pad_id) < vocab:
                step_logits[:, int(pad_id)] = float("-inf")
            if unk_id is not None and 0 <= int(unk_id) < vocab:
                step_logits[:, int(unk_id)] = float("-inf")
            if eos_id is not None and any(finished):
                finished_mask = torch.tensor(finished, dtype=torch.bool, device=DEVICE)
                step_logits[finished_mask] = float("-inf")
                step_logits[finished_mask, int(eos_id)] = 0.0
            step_logits = _apply_no_repeat_ngram(
                step_logits, generated_ids, no_repeat_ngram_size=no_repeat_ngram_size)
            step_logits = _apply_repetition_penalty(
                step_logits, generated_ids, penalty=repetition_penalty)
            if temperature is not None and temperature > 0.0 and temperature != 1.0:
                step_logits = step_logits / temperature
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    step_logits, descending=True, dim=-1)
                cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumulative > top_p
                remove[:, 0] = False
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                filtered = torch.full_like(step_logits, float("-inf"))
                step_logits = filtered.scatter(1, sorted_indices, sorted_logits)
            next_ids = torch.multinomial(
                F.softmax(step_logits.float(), dim=-1), num_samples=1)
            for row in range(batch_size):
                if finished[row]:
                    continue
                token_id = int(next_ids[row, 0].item())
                generated_ids[row].append(token_id)
                partial = tokenizer.decode(
                    generated_ids[row], skip_special_tokens=True,
                    clean_up_tokenization_spaces=False)
                stop = bool(BOX_RE.search(partial))
                if eos_id is not None and token_id == int(eos_id):
                    stop = True
                if not stop and im_end_id is not None and token_id == int(im_end_id):
                    stop = True
                if stop:
                    finished[row] = True
            one_attention = torch.ones_like(next_ids, dtype=torch.long, device=DEVICE)
            step_output = model(
                input_ids=next_ids, attention_mask=one_attention, labels=None,
                use_cache=True, past_states=past_states)
            past_states = step_output["past_states"]
            logits = step_output["logits"][:, -1, :]
    if was_training:
        model.train()
    texts = [tokenizer.decode(
        ids, skip_special_tokens=True,
        clean_up_tokenization_spaces=False).strip() for ids in generated_ids]
    return {"texts": texts, "token_ids": generated_ids}


def sample_for_gsm8k_eval_batch(
    model: TinyLLM,
    tokenizer,
    user_prompts: List[str],
    system_prompt: Optional[str],
    max_new_tokens: int = 700,
):
    """
    Eval 专用（修正版）：
    - 手动左 padding ChatML prompt，保证 logits[:, -1, :] 是最后一个真实 token
    - 多题 batch greedy 解码
    - 仍然支持 BOX / eos / <|im_end|> 早停
    """
    was_training = model.training
    model.eval()

    # 1) 先逐条构造 ChatML，再各自 encode（无 padding）
    chatml_list: List[str] = [
        render_prompt_with_tokenizer(tokenizer, up, system_prompt)
        for up in user_prompts
    ]

    all_ids: List[torch.Tensor] = []
    lengths: List[int] = []
    for chatml in chatml_list:
        ids = tokenizer.encode(
            chatml,
            add_special_tokens=False,
        )
        t = torch.tensor(ids, dtype=torch.long)
        all_ids.append(t)
        lengths.append(t.size(0))

    B = len(all_ids)
    pad_id = tokenizer.pad_token_id
    device = DEVICE

    # 2) 手动左 padding：
    #    pad 在左边，真实 token 靠右，这样 index = -1 一定是每个样本的最后一个真实 token
    max_len = max(x.size(0) for x in all_ids)
    ctx_ids = torch.full(
        (B, max_len),
        fill_value=pad_id,
        dtype=torch.long,
        device=device,
    )
    attn_ctx = torch.zeros(
        (B, max_len),
        dtype=torch.long,
        device=device,
    )

    for i, ids in enumerate(all_ids):
        L = ids.size(0)
        # 👇 左 pad：真实 token 右对齐
        ctx_ids[i, max_len - L:] = ids.to(device)
        attn_ctx[i, max_len - L:] = 1

    # 一些特殊 token
    eos_id, sent_end_ids = _build_stop_sets(tokenizer, stop_on_punct=False)
    unk_id = getattr(getattr(model, "cfg", None), "unk_token_id", None)
    if isinstance(unk_id, torch.Tensor):
        unk_id = int(unk_id.item())
    try:
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    except Exception:
        im_end_id = None

    generated_ids: List[List[int]] = [[] for _ in range(B)]
    finished: List[bool] = [False] * B

    with torch.no_grad():
        # ===== prefill =====
        prefill = model(
            input_ids=ctx_ids,
            attention_mask=attn_ctx,
            labels=None,
            use_cache=True,
            past_states=None,
        )
        past_states = prefill["past_states"]
        # 现在 logits[:, -1, :] 对所有样本都是 “最后一个真实 token” 的分布
        logits = prefill["logits"][:, -1, :]   # [B, V]

        # ===== 增量 decode =====
        for _ in range(max_new_tokens):
            if all(finished):
                break

            logits_step = logits.clone()

            # 屏蔽 pad / unk
            V = logits_step.size(-1)
            if pad_id is not None and pad_id != eos_id and 0 <= int(pad_id) < V:
                logits_step[:, int(pad_id)] = float("-inf")
            if unk_id is not None and 0 <= int(unk_id) < V:
                logits_step[:, int(unk_id)] = float("-inf")

            # finished 的样本强制采 eos
            if eos_id is not None and any(finished):
                mask = torch.tensor(finished, dtype=torch.bool, device=device)
                logits_step[mask] = float("-inf")
                logits_step[mask, int(eos_id)] = 0.0

            # eval：纯 greedy
            next_ids = torch.argmax(logits_step, dim=-1, keepdim=True)  # [B,1]

            # 逐样本检查早停
            for b in range(B):
                if finished[b]:
                    continue

                nid = int(next_ids[b, 0].item())
                generated_ids[b].append(nid)

                stop = False
                partial_text = tokenizer.decode(
                    generated_ids[b],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                if BOX_RE.search(partial_text):
                    stop = True
                if eos_id is not None and nid == int(eos_id):
                    stop = True
                if (not stop) and (im_end_id is not None) and nid == int(im_end_id):
                    stop = True

                if stop:
                    finished[b] = True

            # 推进 KV cache
            attn_one = torch.ones_like(next_ids, dtype=torch.long, device=device)
            step_out = model(
                input_ids=next_ids.to(device),
                attention_mask=attn_one,
                labels=None,
                use_cache=True,
                past_states=past_states,
            )
            past_states = step_out["past_states"]
            logits = step_out["logits"][:, -1, :]

    if was_training:
        model.train()

    texts: List[str] = []
    for b in range(B):
        text = tokenizer.decode(
            generated_ids[b],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        texts.append(text.strip())

    return {
        "texts": texts,
        "token_ids": generated_ids,
    }

def compute_logprob_sums_for_model(
    model: TinyLLM,
    tokenizer,
    user_prompt: str,
    system_prompt: Optional[str],
    sampled_token_ids: List[List[int]],
    use_cache: bool = False,
    force_checkpoint:bool=False
) -> torch.Tensor:
    """
    第二遍前向：带梯度的 logprob 计算（policy），
    / 或者在 no_grad 里算 ref_model 的 logprob（reference）。
    """
    chatml = render_prompt_with_tokenizer(tokenizer, user_prompt, system_prompt)
    ctx_ids = tokenizer.encode(
        chatml,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(DEVICE)
    ctx = ctx_ids[0].tolist()
    ctx_len = len(ctx)

    B = len(sampled_token_ids)

    full_sequences: List[torch.Tensor] = []
    lengths: List[int] = []
    for ids in sampled_token_ids:
        seq = torch.tensor(
            ctx + ids,
            dtype=torch.long,
            device=DEVICE,
        )
        full_sequences.append(seq)
        lengths.append(seq.size(0))

    max_len = max(lengths)
    pad_id = tokenizer.pad_token_id

    batch_ids = torch.full(
        (B, max_len),
        fill_value=pad_id,
        dtype=torch.long,
        device=DEVICE,
    )
    attention_mask = torch.zeros(
        (B, max_len),
        dtype=torch.long,
        device=DEVICE,
    )
    for i, seq in enumerate(full_sequences):
        L = seq.size(0)
        batch_ids[i, :L] = seq
        attention_mask[i, :L] = 1
    T = batch_ids.size(1)
    CKPT_MIN_TOK = 400
    use_force_ckpt = force_checkpoint and (T >= CKPT_MIN_TOK)
    out = model(
        input_ids=batch_ids,
        attention_mask=attention_mask,
        labels=None,
        use_cache=use_cache,
        force_checkpoint=use_force_ckpt
    )
    logits = out["logits"]

    logprobs_all = F.log_softmax(logits[:, :-1, :].float(), dim=-1)

    next_tokens = batch_ids[:, 1:]

    chosen_logprobs = logprobs_all.gather(
        2, next_tokens.unsqueeze(-1)
    ).squeeze(-1)

    logprob_sums = []
    for i, gen_ids in enumerate(sampled_token_ids):
        Lg = len(gen_ids)
        start = ctx_len - 1
        end = start + Lg
        lp_seq = chosen_logprobs[i, start:end]
        logprob_sums.append(lp_seq.sum())

    logprob_sums = torch.stack(logprob_sums, dim=0)
    return logprob_sums

# =========================================================
# GRPO & reward 设计
# =========================================================

def extract_gsm8k_gt(answer_str: str) -> Optional[float]:
    s = answer_str.strip()
    if "####" in s:
        s = s.split("####")[-1].strip()
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        nums = re.findall(r"-?\d+\.?\d*", s)
        if not nums:
            return None
        try:
            return float(nums[-1])
        except Exception:
            return None

def _cot_length_score(L: int, target_len: int = 256, min_len: int = 50, hard_max: int = 400) -> float:
    if L is None or L <= min_len:
        return 0.0
    if L > hard_max:
        return -1.0
    if L >= target_len:
        return 1.0
    return (L - min_len) / float(target_len - min_len)

def compute_rewards_for_group(
    gt_answer_str: str,
    candidate_texts: List[str],
    tokenizer,
) -> Tuple[torch.Tensor, List[Dict]]:
    gt_val = extract_gsm8k_gt(gt_answer_str)
    rewards: List[float] = []
    parsed_list: List[Dict] = []

    for text in candidate_texts:
        info = parse_gsm8k_prediction(text, tokenizer)

        pred_val = info["final_answer"]
        is_correct = (
                (gt_val is not None)
                and (pred_val is not None)
                and math.isfinite(pred_val)
                and abs(pred_val - gt_val) < 1e-6
        )
        in_box = info["has_box_and_valid"]
        cot_len = info["cot_token_len"]
        format_score = 1.0 if info["has_cot_tags"] else 0.0
        len_score = _cot_length_score(cot_len)
        # Do not reinforce a correct answer embedded in a degenerate loop.
        loop_found = bool(re.search(r"(?P<unit>.{16,160}?)(?P=unit){2,}", text, re.S))
        info.update(is_correct=bool(is_correct), loop_found=loop_found,
                    format_score=format_score, length_score=len_score)
        if loop_found:
            r = -0.5
        elif is_correct and in_box:
            base = 1.0
            r = base + 0.3 * format_score + 0.2 * len_score
        elif is_correct and (not in_box):
            r = 0.3
        else:
            r = 0.0

            # 如果你想更硬一点，可以直接:
            # r = 0.0

        rewards.append(float(r))
        parsed_list.append(info)

    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
    return rewards_tensor, parsed_list


def grpo_loss_for_group(
    logprob_sums: torch.Tensor,
    rewards: torch.Tensor,
    logprob_sums_ref: Optional[torch.Tensor] = None,
    kl_coef: float = 0.0,
    eps: float = 1e-5,
):
    assert logprob_sums.dim() == 1
    assert rewards.dim() == 1
    assert logprob_sums.size(0) == rewards.size(0)
    if logprob_sums_ref is not None:
        assert logprob_sums_ref.shape == logprob_sums.shape

    if (logprob_sums_ref is not None) and (kl_coef > 0.0):
        kl_est = logprob_sums - logprob_sums_ref
        shaped_rewards = rewards - kl_coef * kl_est
    else:
        shaped_rewards = rewards

    r_mean = shaped_rewards.mean()
    r_std = shaped_rewards.std()
    if torch.isnan(r_std) or r_std < eps:
        adv = shaped_rewards - r_mean
    else:
        adv = (shaped_rewards - r_mean) / (r_std + eps)

    loss = -(adv * logprob_sums).mean()
    return loss, adv

def grpo_step_for_batch(
    policy_model: TinyLLM,
    ref_model: TinyLLM,
    tokenizer,
    batch: Dict[str, List[str]],
    system_prompt: Optional[str],
    num_generations: int = GRPO_NUM_GENERATIONS,
    max_new_tokens: int = 700,
    temperature: float = 0.4,
    top_p: float = 0.9,
    kl_coef: float = 0.05,
):
    """
    单个 batch（若干题）的 GRPO step：
    - 先用 policy_model 采样（无梯度）
    - 再分别用 policy_model（有梯度）和 ref_model（无梯度）算 logprob_sums
    """
    policy_model.train()
    ref_model.eval()

    questions = batch["questions"]
    answers   = batch["answers"]
    prompts   = batch["prompts"]

    per_question_losses = []
    debug_infos = []

    for q_idx, (user_prompt, gt_answer_str) in enumerate(zip(prompts, answers)):
        # 1) 采样：无梯度
        samples = sample_for_grpo_manual(
            model=policy_model,
            tokenizer=tokenizer,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            num_generations=num_generations,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_on_punct=False,
            repetition_penalty=1.1,
            no_repeat_ngram_size=0,
        )
        texts = samples["texts"]
        token_ids = samples["token_ids"]

        with torch.no_grad():
            logprob_sums_ref = compute_logprob_sums_for_model(
                model=ref_model,
                tokenizer=tokenizer,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                sampled_token_ids=token_ids,
                use_cache=False,
            )

            rewards, parsed_list = compute_rewards_for_group(
                gt_answer_str=gt_answer_str,
                candidate_texts=texts,
                tokenizer=tokenizer,
            )


        logprob_sums_policy = compute_logprob_sums_for_model(
            model=policy_model,
            tokenizer=tokenizer,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            sampled_token_ids=token_ids,
            use_cache=False,
            force_checkpoint=True,
        )

        # 3) ref：无梯度 logprob_sums_ref & rewards

        # 4) GRPO loss
        loss_q, adv_q = grpo_loss_for_group(
            logprob_sums=logprob_sums_policy,
            rewards=rewards,
            logprob_sums_ref=logprob_sums_ref,
            kl_coef=kl_coef,
        )
        per_question_losses.append(loss_q)

        debug_infos.append({
            "question_idx": q_idx,
            "user_prompt": user_prompt,
            "gt": gt_answer_str,
            "texts": texts,
            "rewards": rewards.detach().cpu().tolist(),
            "adv": adv_q.detach().cpu().tolist(),
            "logprob_sums_policy": logprob_sums_policy.detach().cpu().tolist(),
            "logprob_sums_ref": logprob_sums_ref.detach().cpu().tolist(),
            "parsed": parsed_list,
            "loss": float(loss_q.item()),
        })

    batch_loss = torch.stack(per_question_losses).mean()
    return batch_loss, debug_infos

# =========================================================
# Eval：GSM8K acc + reward
# =========================================================

def evaluate_on_gsm8k(
    policy_model: TinyLLM,
    tokenizer,
    eval_dataset: GSM8KDataset,
    system_prompt: Optional[str],
    max_new_tokens: int = 256,
    debug: bool = False,
    debug_max: int = 20,      # 最多打印多少道题
    debug_only_wrong: bool = False,  # 只打印错题
) -> Tuple[float, float]:
    policy_model.eval()

    correct = 0
    total = 0
    reward_list: List[float] = []

    for idx in range(len(eval_dataset)):
        ex = eval_dataset[idx]
        question = ex["question"]
        gt_answer = ex["answer"]

        user_prompt = build_gsm8k_prompt(question)

        with torch.no_grad():
            samples = sample_for_grpo_manual(
                model=policy_model,
                tokenizer=tokenizer,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                num_generations=1,
                max_new_tokens=max_new_tokens,
                # 下面这些参数在 greedy 模式下会被忽略，但写清楚也没问题
                temperature=None,
                top_p=None,
                stop_on_punct=False,
                repetition_penalty=1.0,
                no_repeat_ngram_size=0,
                greedy=True,   # <<< 关键：评估用纯 greedy
            )

        pred_text = samples["texts"][0]
        info = parse_gsm8k_prediction(pred_text, tokenizer)

        gt_val = extract_gsm8k_gt(gt_answer)
        pred_val = info["final_answer"]

        is_correct = (
            (gt_val is not None)
            and (pred_val is not None)
            and math.isfinite(pred_val)
            and abs(pred_val - gt_val) < 1e-6
        )
        if is_correct:
            correct += 1
        total += 1

        rewards, _ = compute_rewards_for_group(
            gt_answer_str=gt_answer,
            candidate_texts=[pred_text],
            tokenizer=tokenizer,
        )
        reward_list.append(float(rewards[0].item()))

        # ========= 调试打印区 =========
        if debug and idx < debug_max:
            # 如果只想看错题，且当前是对的，就跳过
            if debug_only_wrong and is_correct:
                pass
            else:
                print("=" * 80)
                print(f"idx = {idx}")
                print("Q:", question)
                print("GT raw:", gt_answer)
                print("GT val:", gt_val)
                print("PRED raw:", info['raw'][:500])  # 避免太长，截断一下
                print("PRED val:", pred_val, "  is_correct:", is_correct)
                print(
                    "has_box:", info["has_box"],
                    "from_box:", info["from_box"],
                    "cot_tokens:", info["cot_token_len"],
                )
                print("reward:", rewards[0].item())
                print("=" * 80)

        # 也可以顺便每隔 100 题打印一下当前总体 acc
        if debug and (idx + 1) % 100 == 0:
            print(f"[debug] processed {idx+1}/{len(eval_dataset)} "
                  f"current_acc={correct/total:.3f}")

    acc = correct / total if total > 0 else 0.0
    avg_reward = sum(reward_list) / len(reward_list) if reward_list else 0.0

    policy_model.train()
    return acc, avg_reward
def evaluate_on_gsm8k_batched(
    policy_model: TinyLLM,
    tokenizer,
    eval_dataset: GSM8KDataset,
    system_prompt: Optional[str],
    max_new_tokens: int = 256,
    batch_size: int = 8,
    debug: bool = False,
    debug_max: int = 20,
    debug_only_wrong: bool = False,
) -> Tuple[float, float]:
    """
    Batch 版 GSM8K 评估：fp32 + greedy
    """

    policy_model.eval()

    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=gsm8k_collate_fn,
        drop_last=False,
    )

    correct = 0
    total = 0
    reward_list: List[float] = []

    seen_examples = 0

    for batch_idx, batch in enumerate(loader):
        questions = batch["questions"]
        answers   = batch["answers"]
        prompts   = batch["prompts"]

        out = sample_for_gsm8k_eval_batch(
            model=policy_model,
            tokenizer=tokenizer,
            user_prompts=prompts,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
        )
        pred_texts = out["texts"]

        for q, gt_answer, pred_text in zip(questions, answers, pred_texts):
            info = parse_gsm8k_prediction(pred_text, tokenizer)
            gt_val   = extract_gsm8k_gt(gt_answer)
            pred_val = info["final_answer"]

            is_correct = (
                (gt_val is not None)
                and (pred_val is not None)
                and math.isfinite(pred_val)
                and abs(pred_val - gt_val) < 1e-6
            )
            if is_correct:
                correct += 1
            total += 1

            rewards, _ = compute_rewards_for_group(
                gt_answer_str=gt_answer,
                candidate_texts=[pred_text],
                tokenizer=tokenizer,
            )
            reward_float = float(rewards[0].item())
            reward_list.append(reward_float)

            if debug and seen_examples < debug_max:
                if (not debug_only_wrong) or (debug_only_wrong and not is_correct):
                    print("=" * 80)
                    print(f"[batched-eval] example #{seen_examples}  "
                          f"(global_idx={total-1})")
                    print("Q:", q.strip())
                    print("\nGT raw:")
                    print(gt_answer.strip())
                    print("GT val:", gt_val)
                    print("\nPRED raw (truncated to 500 chars):")
                    print(info["raw"][:500])
                    print("PRED val:", pred_val, "  is_correct:", is_correct)
                    print(
                        "has_box:", info["has_box"],
                        "from_box:", info["from_box"],
                        "cot_tokens:", info["cot_token_len"],
                    )
                    print("reward:", reward_float)
                    print("=" * 80)
                    seen_examples += 1

        if debug and (total % 100 == 0):
            print(f"[batched-eval] processed {total}/{len(eval_dataset)} "
                  f"current_acc={correct/total:.3f}")

    acc = correct / total if total > 0 else 0.0
    avg_reward = sum(reward_list) / len(reward_list) if reward_list else 0.0

    return acc, avg_reward


# =========================================================
# 训练主循环（对齐 AutoDL/TensorBoard 路径）
# 带梯度累积（GRPO_GRAD_ACCUM_STEPS，默认 2）
# =========================================================

def train_grpo_on_gsm8k(
    policy_model: TinyLLM,
    ref_model: TinyLLM,
    tokenizer,
    num_epochs: int = 1,
    lr: float = 4e-6,
    kl_coef: float = 0.05,
):
    loader = make_gsm8k_dataloader(
        split=GSM8K_SPLIT,
        batch_size=GSM8K_BATCH_SIZE_QUESTIONS,
        max_examples=GSM8K_MAX_EXAMPLES,
        seed=42,
    )

    tb_dir = os.environ.get("GRPO_TB_DIR", DEFAULT_TB_DIR)
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir, flush_secs=5, max_queue=20)
    debug_print_done = False
    print(f"[TB] logging to: {tb_dir}")

    # 固定 eval 集：例如 test split 前 N 道
    eval_dataset = GSM8KDataset(
        split=GSM8K_EVAL_SPLIT,
        max_examples=GRPO_EVAL_NUM_QUESTIONS,
        seed=1234,
    )
    system_prompt = SYSTEM_PROMPT_FOR_COT if USE_SYSTEM_PROMPT_FOR_COT else None
    print("[eval @ step 0] running baseline evaluation...")
    eval_acc0, eval_avg_reward0 = evaluate_on_gsm8k_batched(
        policy_model=policy_model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        max_new_tokens=1200,
        batch_size=GSM8K_EVAL_BATCH_SIZE_QUESTIONS,
        debug=True,  # 开启打印
        debug_max=30,  # 只看前 30 题
        debug_only_wrong=False,  # 也可以改成 True 只看错题
    )
    print(
        f"[eval @ step 0] gsm8k_acc={eval_acc0:.3f}  "
        f"avg_reward={eval_avg_reward0:.3f}"
    )
    writer.add_scalar("eval/gsm8k_acc", eval_acc0, 0)
    writer.add_scalar("eval/avg_reward", eval_avg_reward0, 0)
    optimizer = torch.optim.AdamW(
        (p for p in policy_model.parameters() if p.requires_grad),
        lr=lr,
    )
    num_batches_per_epoch = len(loader)
    # 每个 step = GRPO_GRAD_ACCUM_STEPS 个 micro step
    max_train_steps = math.ceil(
        num_batches_per_epoch * num_epochs / GRPO_GRAD_ACCUM_STEPS
    )

    # 可以自己改比例，这里给一个 10% warmup
    warmup_frac = 0.01
    warmup_steps = max(1, int(warmup_frac * max_train_steps))

    def lr_lambda(current_step: int):
        """
        current_step 从 0 开始：
        - [0, warmup_steps): 线性升到 1.0
        - [warmup_steps, max_train_steps): 线性降到 0.2
        """
        step = float(current_step)
        if step < warmup_steps:
            return (step + 1.0) / float(warmup_steps)

        # decay 阶段：从 1.0 线性到 0.2
        progress = (step - warmup_steps) / max(1.0, max_train_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 1.0 - 0.8 * progress  # 1.0 → 0.2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step_idx = 0          # 优化器 update 次数
    micro_step = 0        # 累积用的 micro step 计数
    accum_loss = 0.0
    accum_rewards_for_step: List[float] = []

    optimizer.zero_grad()

    for epoch in range(num_epochs):
        for batch in loader:
            micro_step += 1

            loss, debug_infos = grpo_step_for_batch(
                policy_model=policy_model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                batch=batch,
                system_prompt=system_prompt,
                num_generations=GRPO_NUM_GENERATIONS,
                max_new_tokens=700,
                temperature=0.4,
                top_p=0.9,
                kl_coef=kl_coef,
            )
            if GRPO_DEBUG_PRINT_SAMPLES:
                print("\n" + "=" * 80)
                print("[DEBUG] GRPO samples for the first training batch")
                print("=" * 80)
                for info in debug_infos:
                    q = info["user_prompt"]
                    gt = info["gt"]
                    texts = info["texts"]  # List[str], 长度 = num_generations
                    rewards = info["rewards"]  # List[float]
                    advs = info["adv"]  # List[float]
                    parsed_list = info["parsed"]  # List[Dict]（parse_gsm8k_prediction 的结果）

                    print("-" * 80)
                    print("Question:")
                    print(q.strip())
                    print("\nGT answer raw:")
                    print(gt.strip())

                    for i, text in enumerate(texts):
                        parsed = parsed_list[i]
                        final_val = parsed.get("final_answer", None)
                        has_box = parsed.get("has_box", False)
                        from_box = parsed.get("from_box", False)

                        print("\n" + "-" * 40)
                        print(f"Candidate #{i}  reward={rewards[i]:.4f}  adv={advs[i]:.4f}")
                        print(f"  final_answer={final_val}  has_box={has_box}  from_box={from_box}")
                        print("-" * 40)
                        print(text.strip()[:800])  # 防止太长，截断到前 800 字符
                        print()

                print("=" * 80)
                print("[DEBUG] End of first-batch samples")
                print("=" * 80 + "\n")

                debug_print_done = True
            # 梯度累积：先除以累积步数
            (loss / GRPO_GRAD_ACCUM_STEPS).backward()

            # 累积当前 loss/reward，方便做 step 级别的 logging
            accum_loss += float(loss.item())
            for info in debug_infos:
                accum_rewards_for_step.extend(info["rewards"])

            # 到了一个完整的累计步数，才真正更新一次参数 & 记录一次 TB
            if micro_step % GRPO_GRAD_ACCUM_STEPS == 0:
                step_idx += 1

                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                avg_loss = accum_loss / GRPO_GRAD_ACCUM_STEPS
                if accum_rewards_for_step:
                    avg_reward = float(
                        torch.tensor(accum_rewards_for_step, dtype=torch.float32).mean().item()
                    )
                else:
                    avg_reward = 0.0

                writer.add_scalar("train/loss", avg_loss, step_idx)
                writer.add_scalar("train/avg_reward", avg_reward, step_idx)

                if step_idx % 10 == 0:
                    print(f"[step {step_idx}] loss={avg_loss:.4f}  avg_reward={avg_reward:.3f}")

                # eval
                if step_idx % GRPO_EVAL_EVERY_STEPS == 0:
                    eval_acc, eval_avg_reward = evaluate_on_gsm8k_batched(
                        policy_model=policy_model,
                        tokenizer=tokenizer,
                        eval_dataset=eval_dataset,
                        system_prompt=system_prompt,
                        max_new_tokens=1200,
                        batch_size=GSM8K_EVAL_BATCH_SIZE_QUESTIONS,
                        debug=False,  # 训练中就先关掉打印，避免太吵
                    )
                    print(
                        f"[eval @ step {step_idx}] "
                        f"gsm8k_acc={eval_acc:.3f}  avg_reward={eval_avg_reward:.3f}"
                    )
                    writer.add_scalar("eval/gsm8k_acc", eval_acc, step_idx)
                    writer.add_scalar("eval/avg_reward", eval_avg_reward, step_idx)

                # LoRA checkpoint 保存：
                # /root/autodl-tmp/llm/checkpoints/grpo_gsm8k/<run-name>/lora_math_stepxxxxxx.pt
                if step_idx % SAVE_STEPS == 0:
                    save_root = os.environ.get("GRPO_SAVE_ROOT", GRPO_SAVE_ROOT_DEFAULT)
                    save_dir = os.path.join(save_root, RUN_NAME)
                    os.makedirs(save_dir, exist_ok=True)
                    lora_ckpt_path = os.path.join(
                        save_dir,
                        f"lora_{LORA_ADAPTER_NAME}_step{step_idx:06d}.pt"
                    )
                    lora_state = policy_model.get_lora_state_dict(adapter_name=LORA_ADAPTER_NAME)
                    torch.save(lora_state, lora_ckpt_path)
                    print(f"[checkpoint] LoRA saved to {lora_ckpt_path}")
                    meta_path = os.path.join(save_dir, "meta.json")
                    if not os.path.exists(meta_path):
                        meta = {
                            "run_name": RUN_NAME,
                            "adapter_name": LORA_ADAPTER_NAME,
                            # 你想写啥都行，这里给个简单英文描述
                            "description": (
                                "LoRA adapter for TinyLLM 0.5B, "
                                "GRPO on GSM8K (train split, eval on test split)."
                            ),
                            "lora_target": LORA_TARGET,        # 比如 "attn_mlp_skip_first4"
                            "lora_rank": LORA_RANK,            # 64
                            "lora_dropout": LORA_DROPOUT,      # 0.05
                            "lora_alpha": LORA_ALPHA,          # 16.0
                            # 额外留个时间戳，方便你以后翻日志
                            "created_at": RUN_ID,
                        }
                        with open(meta_path, "w", encoding="utf-8") as f:
                            json.dump(meta, f, indent=2, ensure_ascii=False)
                        print(f"[checkpoint] meta saved to {meta_path}")
                # 清空累积的统计量
                accum_loss = 0.0
                accum_rewards_for_step = []

    writer.close()
    print("GRPO training finished.")

# =========================================================
# main
# =========================================================

def main():
    # ===== 这里改成你的 CPT ckpt 目录 =====
    # 推荐用环境变量覆盖：export TINYLLM_CKPT=/root/autodl-tmp/llm/tiny_05B_cpt3/checkpoint-396000
    CKPT_DIR = os.environ.get(
        "TINYLLM_CKPT",
        legacy_path("/root/autodl-tmp/llm/tiny_05B_sft/checkpoint-50000"),
    )

    print(f"[CKPT] using student checkpoint: {CKPT_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(
        CKPT_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("[tokenizer] vocab_size =", len(tokenizer.get_vocab()))
    print("[tokenizer] eos/pad =", tokenizer.eos_token_id, tokenizer.pad_token_id)

    # policy / ref 用同一个底座
    policy_model = load_tinyllm_from_ckpt(CKPT_DIR, tokenizer, strict=False)
    ref_model = load_tinyllm_from_ckpt(CKPT_DIR, tokenizer, strict=False)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # ===== LoRA 接上，“math” adapter，只训 LoRA =====
    policy_model.attach_lora_adapter(
        adapter_name=LORA_ADAPTER_NAME,
        rank=LORA_RANK,
        dropout=LORA_DROPOUT,
        alpha=LORA_ALPHA,
        target=LORA_TARGET,
    )
    policy_model.activate_single_lora(LORA_ADAPTER_NAME)

    for name, param in policy_model.named_parameters():
        if f".adapters.{LORA_ADAPTER_NAME}." in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    total_params = sum(p.numel() for p in policy_model.parameters())
    trainable_params = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
    print(f"[LoRA] total params = {total_params/1e6:.2f}M, "
          f"trainable (LoRA) = {trainable_params/1e6:.2f}M")

    # ===== 开烧 GRPO =====
    train_grpo_on_gsm8k(
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        num_epochs=1,
        lr=2e-5,          # LoRA 学习率可以比全参数大
        kl_coef=0.02,
    )


if __name__ == "__main__":
    main()
