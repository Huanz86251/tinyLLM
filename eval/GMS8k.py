from project_paths import legacy_path
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import math
import json
from pathlib import Path
from typing import List, Dict, Optional

import torch
from datasets import load_dataset
from safetensors.torch import load_file
from transformers import AutoTokenizer

from model.config import Config
from model.model import TinyLLM

# =========================
# Global config
# =========================

DTYPE = torch.float32          # 你也可以改成 torch.bfloat16 看显存情况
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IGNORE_INDEX = -100

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    from torch.backends.cuda import sdp_kernel
    sdp_kernel.enable_flash_sdp(False)
    sdp_kernel.enable_mem_efficient_sdp(False)
    sdp_kernel.enable_math_sdp(True)
except Exception:
    pass

# ---------- GSM8K & prompt 设置 ----------

GSM8K_SPLIT = os.environ.get("GSM8K_SPLIT", "test")  # "train" or "test"

# 默认随机抽 200 题；你可以用 GSM8K_MAX_EXAMPLES 环境变量改，例如 150
MAX_EXAMPLES = int(os.environ.get("GSM8K_MAX_EXAMPLES", "0"))

# ✅ 系统 prompt 完全保持你原来的（不改一个字）
# SYSTEM_PROMPT_FOR_COT = (
#     "Think step by step inside <|thought_start|> and <|thought_end|>. Then provide final answer."
# )
SYSTEM_PROMPT_FOR_COT=None

def build_gsm8k_prompt(question: str) -> str:
    """
    把 GSM8K 的 question 包成一个纯 user prompt。
    不改 system，只在 user 末尾追加 boxed 指令。
    """
    q = question.strip()
    return (
        f"{q}\n"
        f"Put your final answer in LaTeX boxed form like $\\boxed{{answer}}$. "

    )


# =========================================================
# TinyLLM loading helpers（复用你之前的逻辑）
# =========================================================

def _clean_state_dict_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Strip 'module.' prefix and move '0.xxx' -> 'blocks.0.xxx' etc.
    """
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
    cfg.checkpoint_use_reentrant = True
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
# TinyLLM incremental generation (ChatML)
# =========================================================

def build_chatml_prompt_for_tiny(
    user_prompt: str,
    system_prompt: Optional[str] = None,
) -> str:
    """
    和你 SFT 一致的 ChatML：
    [system?] + user + assistant 起头（不含结尾 <|im_end|>）
    """
    parts: List[str] = []

    if system_prompt:
        parts.append("<|im_start|>system\n")
        parts.append(system_prompt.strip())
        parts.append("<|im_end|>\n")

    parts.append("<|im_start|>user\n")
    parts.append(user_prompt.strip())
    parts.append("<|im_end|>\n")

    parts.append("<|im_start|>assistant\n")
    # 不加 <|im_end|>，让模型自己生成

    return "".join(parts)


# boxed 正则：支持 \\boxed{91} / \\boxed{ 91.5 } / \\boxed{-3} 等
BOX_RE = re.compile(
    r"\\boxed\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}",
    flags=re.MULTILINE
)


@torch.no_grad()
def generate_chat_incremental_tiny(
    model: TinyLLM,
    tokenizer,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 512,
) -> str:
    """
    TinyLLM：ChatML prompt + KV cache 增量解码（greedy）。
    关键：不靠 system，也做 boxed early-stop（检测到 \\boxed{...} 就停）
    """
    model.eval()

    chatml = build_chatml_prompt_for_tiny(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
    )

    ctx_ids = tokenizer.encode(
        chatml,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(DEVICE)          # [1, T0]
    attn_ctx = torch.ones_like(ctx_ids, dtype=torch.long, device=DEVICE)

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    unk_id = getattr(getattr(model, "cfg", None), "unk_token_id", None)
    if isinstance(unk_id, torch.Tensor):
        unk_id = int(unk_id.item())

    try:
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    except Exception:
        im_end_id = None

    # prefill
    prefill = model(
        input_ids=ctx_ids,
        attention_mask=attn_ctx,
        labels=None,
        use_cache=True,
        past_states=None,
    )
    past_states = prefill["past_states"]
    logits = prefill["logits"][:, -1, :]  # [1, V]

    generated_ids: List[int] = []

    for _ in range(max_new_tokens):
        # 避免吐 pad / unk
        if pad_id is not None and pad_id != eos_id and 0 <= int(pad_id) < logits.size(-1):
            logits[:, int(pad_id)] = float("-inf")
        if unk_id is not None and 0 <= int(unk_id) < logits.size(-1):
            logits[:, int(unk_id)] = float("-inf")

        next_id = torch.argmax(logits, dim=-1, keepdim=True)  # greedy
        nid = int(next_id.item())
        generated_ids.append(nid)

        # early stop: 一旦出现 boxed，就停（防止后面继续乱写污染）
        partial_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if BOX_RE.search(partial_text):
            break

        # 原有停止条件：eos / <|im_end|>
        if eos_id is not None and nid == int(eos_id):
            break
        if im_end_id is not None and nid == int(im_end_id):
            break

        # 继续增量解码
        attn_one = torch.ones_like(next_id, dtype=torch.long, device=DEVICE)
        step_out = model(
            input_ids=next_id.to(DEVICE),
            attention_mask=attn_one,
            labels=None,
            use_cache=True,
            past_states=past_states,
        )
        past_states = step_out["past_states"]
        logits = step_out["logits"][:, -1, :]

    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return text.strip()


# =========================================================
# 答案解析 & 评测
# =========================================================

def extract_gsm8k_gt(answer_str: str) -> Optional[float]:
    """
    GSM8K 的标注一般是:
      '...推理...\\n\\n#### 21'
    优先从 #### 后面抽；然后转成 float。
    """
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


def extract_model_answer(pred_str: str) -> Optional[float]:
    """
    模型输出里面抽最终答案：
      1) 优先从 \\boxed{number} 里抽
      2) 兜底：最后一个数字
    """
    if not pred_str:
        return None

    m = BOX_RE.search(pred_str)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass

    text = pred_str.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", text)
    if not nums:
        return None
    try:
        return float(nums[-1])
    except Exception:
        return None


def evaluate_on_gsm8k(
    tiny_model: TinyLLM,
    tokenizer,
    split: str = "test",
    max_examples: int = 0,
    max_new_tokens: int = 512,
):
    print(f"[DATA] loading gsm8k (split={split}) ...")
    ds = load_dataset("gsm8k", "main", split=split)

    total_all = len(ds)
    print(f"[DATA] original total examples = {total_all}")

    ds = ds.shuffle(seed=42)
    if max_examples > 0:
        max_examples = min(max_examples, len(ds))
        ds = ds.select(range(max_examples))
    total = len(ds)
    print(f"[DATA] after shuffle & select: total examples = {total}")

    correct = 0
    examples_log = []

    for idx, ex in enumerate(ds):
        q = ex["question"]
        a = ex["answer"]
        prompt = build_gsm8k_prompt(q)
        gt_val = extract_gsm8k_gt(a)

        print(f"\n===== [#{idx}] =====")
        print("Question:")
        print(q)
        print("GT answer string:")
        print(a)
        print("GT parsed:", gt_val)

        tiny_out = generate_chat_incremental_tiny(
            model=tiny_model,
            tokenizer=tokenizer,
            user_prompt=prompt,
            system_prompt=SYSTEM_PROMPT_FOR_COT,
            max_new_tokens=max_new_tokens,
        )
        tiny_ans = extract_model_answer(tiny_out)

        tiny_ok = (
            gt_val is not None and tiny_ans is not None and math.isfinite(tiny_ans)
            and abs(tiny_ans - gt_val) < 1e-6
        )
        if tiny_ok:
            correct += 1

        print("\n--- TinyLLM ---")
        print(tiny_out)
        print(f"[TinyLLM parsed] {tiny_ans}  ->  {'CORRECT' if tiny_ok else 'WRONG'}")

        if idx < 50:
            examples_log.append({
                "idx": idx,
                "question": q,
                "gt_val": gt_val,
                "gt_raw": a,
                "tiny_out": tiny_out,
                "tiny_ans": tiny_ans,
                "tiny_ok": tiny_ok,
            })

    acc = correct / total if total > 0 else 0.0

    print("\n================ SUMMARY ================")
    print(f"Total examples (after shuffle/select): {total}")
    print(f"TinyLLM correct: {correct} / {total}  (acc = {acc:.4f})")

    out_path = f"gsm8k_eval_examples_{split}_tiny.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(examples_log, f, ensure_ascii=False, indent=2)
    print(f"[LOG] wrote first {len(examples_log)} examples to {out_path}")


# =========================================================
# main
# =========================================================

def main():
    CKPT_DIR = legacy_path(r"E:\learn\data\checkpoint-50000")

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

    tiny_model = load_tinyllm_from_ckpt(CKPT_DIR, tokenizer, strict=False)
    print("[TinyLLM] dtype:", next(tiny_model.parameters()).dtype)
    print("[TinyLLM] train_maxlength:", getattr(tiny_model.cfg, "train_maxlength", None))

    evaluate_on_gsm8k(
        tiny_model=tiny_model,
        tokenizer=tokenizer,
        split=GSM8K_SPLIT,
        max_examples=MAX_EXAMPLES,
        max_new_tokens=1200,
    )


if __name__ == "__main__":
    main()
