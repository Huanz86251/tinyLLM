from project_paths import legacy_path
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from safetensors.torch import load_file
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from model.config import Config
from model.model import TinyLLM

# =========================
# 基本配置（与训练脚本保持一致）
# =========================

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
IGNORE_INDEX = -100

SYSTEM_PROMPT_FOR_COT = (
    "Think step by step inside <|thought_start|> and <|thought_end|>. "
    "Then provide final answer."
)
USE_SYSTEM_PROMPT_FOR_COT = True

# ARC 数据相关（沿用你原来的命名）
GSM8K_SPLIT = "train"  # 这里只是类名保留，实际加载的是 ARC
GSM8K_EVAL_BATCH_SIZE_QUESTIONS = 8

# ChatML / COT 标记
THOUGHT_START = "<|thought_start|>"
THOUGHT_END = "<|thought_end|>"

# 匹配 boxed / 选项字母
BOX_RE = re.compile(
    r"\\boxed\s*\{\s*([^{}]+?)\s*\}",
    flags=re.MULTILINE,
)
CHOICE_STANDALONE_RE = re.compile(
    r"(?:^|[\s\u3000])([ABCDE])(?=[\s\.\,\!\?\:\;\)\]\u3000]|$)",
    flags=re.IGNORECASE,
)
ANSWER_LETTER_RE = re.compile(
    r"(?:answer|答案|选项)\s*[:：]?\s*([ABCDE])",
    flags=re.IGNORECASE,
)


# =========================================================
# TinyLLM 加载（原样拷贝）
# =========================================================

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
    strict: bool = False,
) -> TinyLLM:
    ckpt_dir = Path(ckpt_dir)
    safepath = ckpt_dir / "model.safetensors"
    binpath = ckpt_dir / "pytorch_model.bin"
    cfgpath = ckpt_dir / "config.json"

    if safepath.is_file():
        raw_state = load_file(str(safepath), device="cpu")   # ✅ 用 safetensors 加载
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
    print("  missing[:10]   =", missing[:10], f"(total {len(missing)})")
    print("  unexpected[:10]=", unexpected[:10], f"(total {len(unexpected)})")

    model.to(DEVICE, dtype=DTYPE)
    model.eval()
    return model


# =========================================================
# ChatML prompt & decode
# =========================================================

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


def _build_stop_sets(tokenizer, stop_on_punct: bool = False):
    eos_id = tokenizer.eos_token_id
    sent_end_ids = set()
    if stop_on_punct:
        for ch in ["。", "！", "？", ".", "!", "?"]:
            ids = tokenizer.encode(ch, add_special_tokens=False)
            if ids:
                sent_end_ids.add(int(ids[0]))
    return eos_id, sent_end_ids


def sample_for_arc_eval_batch(
    model: TinyLLM,
    tokenizer,
    user_prompts: List[str],
    system_prompt: Optional[str],
    max_new_tokens: int = 256,
):
    """
    多题 batch，纯 greedy 解码，用于 ARC 评测。
    逻辑基本等同于你原来的 sample_for_gsm8k_eval_batch。
    """
    was_training = model.training
    model.eval()

    chatml_list: List[str] = [
        render_prompt_with_tokenizer(tokenizer, up, system_prompt)
        for up in user_prompts
    ]

    all_ids: List[torch.Tensor] = []
    for chatml in chatml_list:
        ids = tokenizer.encode(
            chatml,
            add_special_tokens=False,
        )
        t = torch.tensor(ids, dtype=torch.long)
        all_ids.append(t)

    B = len(all_ids)
    pad_id = tokenizer.pad_token_id
    device = DEVICE

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
        ctx_ids[i, max_len - L:] = ids.to(device)
        attn_ctx[i, max_len - L:] = 1

    eos_id, _ = _build_stop_sets(tokenizer, stop_on_punct=False)
    try:
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    except Exception:
        im_end_id = None

    unk_id = getattr(getattr(model, "cfg", None), "unk_token_id", None)
    if isinstance(unk_id, torch.Tensor):
        unk_id = int(unk_id.item())

    generated_ids: List[List[int]] = [[] for _ in range(B)]
    finished: List[bool] = [False] * B

    with torch.no_grad():
        out = model(
            input_ids=ctx_ids,
            attention_mask=attn_ctx,
            labels=None,
            use_cache=True,
            past_states=None,
        )
        past_states = out["past_states"]
        logits = out["logits"][:, -1, :]  # [B, V]

        for _ in range(max_new_tokens):
            if all(finished):
                break

            logits_step = logits.clone()
            V = logits_step.size(-1)

            # 屏蔽 pad / unk
            if pad_id is not None and pad_id != eos_id and 0 <= int(pad_id) < V:
                logits_step[:, int(pad_id)] = float("-inf")
            if unk_id is not None and 0 <= int(unk_id) < V:
                logits_step[:, int(unk_id)] = float("-inf")

            # 已结束的样本强制 EOS
            if eos_id is not None and any(finished):
                mask = torch.tensor(finished, dtype=torch.bool, device=device)
                logits_step[mask] = float("-inf")
                logits_step[mask, int(eos_id)] = 0.0

            next_ids = torch.argmax(logits_step, dim=-1, keepdim=True)

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

            attn_one = torch.ones_like(next_ids, dtype=torch.long, device=device)
            out_step = model(
                input_ids=next_ids.to(device),
                attention_mask=attn_one,
                labels=None,
                use_cache=True,
                past_states=past_states,
            )
            past_states = out_step["past_states"]
            logits = out_step["logits"][:, -1, :]

    if was_training:
        model.train()

    texts: List[str] = []
    for b in range(B):
        text = tokenizer.decode(
            generated_ids[b],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        texts.append(text.strip())

    return {
        "texts": texts,
        "token_ids": generated_ids,
    }


# =========================================================
# ARC 数据集封装（沿用之前 GSM8KDataset 逻辑）
# =========================================================

class GSM8KDataset(Dataset):
    """
    实际上是 ARC 数据集封装：
    - configs 里传 ["ARC-Easy"] 或 ["ARC-Challenge"]
    - max_examples=0 表示全量
    """

    def __init__(
        self,
        split: str = "train",
        max_examples: int = 0,
        seed: int = 42,
        configs: Optional[List[str]] = None,
    ):
        if configs is None:
            configs = ["ARC-Easy"]

        ds_list = []
        for cfg_name in configs:
            ds_cfg = load_dataset("ai2_arc", cfg_name, split=split)
            ds_list.append(ds_cfg)

        if len(ds_list) == 1:
            ds = ds_list[0]
        else:
            from datasets import concatenate_datasets
            ds = concatenate_datasets(ds_list)

        ds = ds.shuffle(seed=seed)

        if max_examples > 0:
            max_examples = min(max_examples, len(ds))
            ds = ds.select(range(max_examples))

        self._ds = ds

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        ex = self._ds[idx]
        stem: str = ex["question"]
        choice_texts = ex["choices"]["text"]
        choice_labels = ex["choices"]["label"]

        pairs = sorted(zip(choice_labels, choice_texts), key=lambda x: x[0])
        opts_lines = [f"{label}. {text}" for label, text in pairs]
        q_full = stem.strip() + "\n\n" + "\n".join(opts_lines)

        ans = ex["answerKey"].strip()

        return {
            "question": q_full,
            "answer": ans,
        }


def build_gsm8k_prompt(question: str) -> str:
    q = question.strip()
    return (
        f"{q}\n"
        f"Put your final answer in LaTeX boxed form like $\\boxed{{answer}}$."
    )


def gsm8k_collate_fn(batch: List[Dict[str, str]]) -> Dict[str, List[str]]:
    questions = [item["question"] for item in batch]
    answers = [item["answer"] for item in batch]
    prompts = [build_gsm8k_prompt(q) for q in questions]
    return {
        "questions": questions,
        "answers": answers,
        "prompts": prompts,
    }


# =========================================================
# 预测解析 & 正确率计算
# =========================================================

def extract_gsm8k_gt(answer_str: str) -> Optional[str]:
    if answer_str is None:
        return None
    s = answer_str.strip().upper()
    if not s:
        return None
    m = re.search(r"[ABCDE]", s)
    return m.group(0) if m else None


def _is_truly_standalone_choice(raw: str, match_end: int) -> bool:
    tail = raw[match_end:]
    i = 0
    ALLOWED = " \t\r\n.,!?;:)]}）】。、，；：！？"
    while i < len(tail) and tail[i] in ALLOWED:
        i += 1
    return i == len(tail)


def parse_gsm8k_prediction(pred_str: str, tokenizer) -> Dict:
    if pred_str is None:
        pred_str = ""
    raw = pred_str.strip()

    # 1) COT 部分
    start_idx = raw.find(THOUGHT_START)
    end_idx = raw.find(THOUGHT_END)
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

    # 2) box 里找选项
    boxed_match = BOX_RE.search(raw)
    has_box = boxed_match is not None
    boxed_choice: Optional[str] = None
    if has_box:
        content = boxed_match.group(1).strip()
        m = re.fullmatch(r"[ABCDEabcde]", content)
        if m:
            boxed_choice = m.group(0).upper()

    # 3) 其他地方找选项字母
    letters: List[str] = []
    for m in ANSWER_LETTER_RE.finditer(raw):
        letters.append(m.group(1).upper())
    for m in CHOICE_STANDALONE_RE.finditer(raw):
        if _is_truly_standalone_choice(raw, m.end()):
            letters.append(m.group(1).upper())

    fallback_choice: Optional[str] = None
    if letters:
        fallback_choice = letters[-1]

    has_valid_box_choice = boxed_choice in ["A", "B", "C", "D"]
    if has_valid_box_choice:
        final_choice = boxed_choice
    else:
        final_choice = fallback_choice if fallback_choice in ["A", "B", "C", "D"] else None

    from_box = (final_choice is not None) and (final_choice == boxed_choice)

    return {
        "raw": raw,
        "has_box": bool(has_box),
        "boxed_value": None,
        "fallback_value": None,
        "final_answer": final_choice,
        "from_box": from_box,
        "has_box_and_valid": bool(has_valid_box_choice),
        "has_cot_tags": bool(has_cot_tags),
        "cot_text": cot_text,
        "cot_token_len": int(cot_token_len),
    }


def evaluate_on_arc_batched(
    policy_model: TinyLLM,
    tokenizer,
    eval_dataset: GSM8KDataset,
    system_prompt: Optional[str],
    max_new_tokens: int = 256,
    batch_size: int = 8,
    debug: bool = False,
    debug_max: int = 20,
    debug_only_wrong: bool = False,
) -> float:
    """
    只算 ARC-Easy 准确率，不算 reward。
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
    seen_examples = 0

    for batch_idx, batch in enumerate(loader):
        questions = batch["questions"]
        answers = batch["answers"]
        prompts = batch["prompts"]

        out = sample_for_arc_eval_batch(
            model=policy_model,
            tokenizer=tokenizer,
            user_prompts=prompts,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
        )
        pred_texts = out["texts"]

        for q, gt_answer, pred_text in zip(questions, answers, pred_texts):
            info = parse_gsm8k_prediction(pred_text, tokenizer)
            gt_val = extract_gsm8k_gt(gt_answer)
            pred_val = info["final_answer"]

            is_correct = (
                (gt_val is not None)
                and (pred_val is not None)
                and (pred_val == gt_val)
            )

            if is_correct:
                correct += 1
            total += 1

            if debug and seen_examples < debug_max:
                if (not debug_only_wrong) or (debug_only_wrong and not is_correct):
                    print("=" * 80)
                    print(f"[eval-debug] example #{seen_examples} (global_idx={total-1})")
                    print("Q:", q.strip())
                    print("\nGT raw:", gt_answer.strip(), "  GT val:", gt_val)
                    print("\nPRED raw (truncated 500):")
                    print(info["raw"][:500])
                    print("PRED val:", pred_val, "  is_correct:", is_correct)
                    print("has_box:", info["has_box"],
                          "from_box:", info["from_box"],
                          "cot_tokens:", info["cot_token_len"])
                    print("=" * 80)
                    seen_examples += 1

        if debug and (total % 100 == 0):
            print(f"[eval-debug] processed {total}/{len(eval_dataset)} "
                  f"current_acc={correct/total:.3f}")

    acc = correct / total if total > 0 else 0.0
    return acc


# =========================================================
# LoRA meta & ckpt 读取
# =========================================================

def load_lora_meta(run_dir: Path):
    meta_path = run_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found in {run_dir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    adapter_name = meta.get("adapter_name", "arc")
    lora_target = meta["lora_target"]
    lora_rank = int(meta["lora_rank"])
    lora_dropout = float(meta["lora_dropout"])
    lora_alpha = float(meta["lora_alpha"])

    print("[meta] loaded:")
    print("  adapter_name:", adapter_name)
    print("  lora_target :", lora_target)
    print("  lora_rank   :", lora_rank)
    print("  lora_dropout:", lora_dropout)
    print("  lora_alpha  :", lora_alpha)

    return adapter_name, lora_target, lora_rank, lora_dropout, lora_alpha


def find_lora_ckpts(run_dir: Path, adapter_name: str):
    patterns = sorted(run_dir.glob(f"lora_{adapter_name}_step*.pt"))
    ckpts: List[Tuple[int, Path]] = []
    for p in patterns:
        m = re.search(r"step(\d+)\.pt$", p.name)
        if not m:
            continue
        step = int(m.group(1))
        ckpts.append((step, p))
    ckpts.sort()
    return ckpts


# =========================================================
# main：baseline + 各步 LoRA 评测
# =========================================================
def main():
    # ===== 学生模型 ckpt 路径（与训练脚本保持一致）=====
    CKPT_DIR = os.environ.get(
        "TINYLLM_CKPT",
        legacy_path("/root/autodl-tmp/llm/tiny_05B_sft/checkpoint-50000"),
    )
    print(f"[CKPT] using student checkpoint: {CKPT_DIR}")

    # ===== GRPO run 目录（里面有 meta.json 和 lora_*.pt）=====
    run_dir_env = legacy_path("/root/autodl-tmp/llm/grpo_arc/tinyllm-grpo-arc-20251212-090356")
    run_dir = Path(os.environ.get("GRPO_RUN_DIR", run_dir_env))
    print(f"[RUN] using GRPO run dir: {run_dir}")

    # 用来收集所有结果（名称, acc）
    results: List[Tuple[str, float]] = []

    # ===== tokenizer =====
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

    # ===== system prompt =====
    system_prompt = SYSTEM_PROMPT_FOR_COT if USE_SYSTEM_PROMPT_FOR_COT else None

    # ===== eval 数据：全量 ARC-Easy test split =====
    eval_dataset = GSM8KDataset(
        split="test",
        max_examples=0,       # 0 = full split
        seed=1234,
        configs=["ARC-Easy"],
    )
    print(f"[ARC] eval split=test, configs=['ARC-Easy'], total examples={len(eval_dataset)}")

    # ===== baseline：不挂 LoRA =====
    print("\n" + "=" * 80)
    print("[EVAL] Baseline (no LoRA)")
    print("=" * 80)
    base_model = load_tinyllm_from_ckpt(CKPT_DIR, tokenizer, strict=False)
    acc_base = evaluate_on_arc_batched(
        policy_model=base_model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        max_new_tokens=1200,
        batch_size=GSM8K_EVAL_BATCH_SIZE_QUESTIONS,
        debug=False,
    )
    print(f"[EVAL] baseline ARC-Easy(test) acc = {acc_base:.4f}")
    results.append(("baseline", acc_base))      # <<< 记录 baseline

    # ===== LoRA meta & ckpt 列表 =====
    adapter_name, lora_target, lora_rank, lora_dropout, lora_alpha = load_lora_meta(run_dir)
    all_ckpts = find_lora_ckpts(run_dir, adapter_name)

    if not all_ckpts:
        print("[WARN] no LoRA ckpts found, only baseline was evaluated.")
        # 只保存 baseline 结果
        out_path = run_dir / "arc_eval_results.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            for name, acc in results:
                f.write(f"{name}\t{acc:.6f}\n")
        print(f"[EVAL] results saved to {out_path}")
        return

    print("\n[CKPTS] found LoRA ckpts:")
    for step, p in all_ckpts:
        print(f"  step={step:6d}  path={p}")

    # 如果你只想测特定步（比如 400, 500），可以在这里筛一下：
    target_steps = {400, 500,600}
    all_ckpts = [(s, p) for (s, p) in all_ckpts if s in target_steps]

    # ===== 逐个 LoRA ckpt 评测 =====
    for step, ckpt_path in all_ckpts:
        print("\n" + "=" * 80)
        print(f"[EVAL] LoRA step {step}")
        print("=" * 80)

        model = load_tinyllm_from_ckpt(CKPT_DIR, tokenizer, strict=False)
        model.attach_lora_adapter(
            adapter_name=adapter_name,
            rank=lora_rank,
            dropout=lora_dropout,
            alpha=lora_alpha,
            target=lora_target,
        )
        model.activate_single_lora(adapter_name)

        state = torch.load(ckpt_path, map_location="cpu")
        model.load_lora_state_dict(state, adapter_name=adapter_name)

        acc = evaluate_on_arc_batched(
            policy_model=model,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            system_prompt=system_prompt,
            max_new_tokens=1200,
            batch_size=GSM8K_EVAL_BATCH_SIZE_QUESTIONS,
            debug=False,
        )
        print(f"[EVAL] LoRA step {step:6d} ARC-Easy(test) acc = {acc:.4f}")
        results.append((f"lora_step_{step}", acc))   # <<< 记录每个 step

    # ===== 写出到 txt =====
    out_path = run_dir / "arc_eval_results.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for name, acc in results:
            f.write(f"{name}\t{acc:.6f}\n")

    print(f"\n[EVAL] all results saved to {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
