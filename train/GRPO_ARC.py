from project_paths import legacy_path, path as project_path
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from local_datasets import load_local_dataset
from datasets import load_dataset, concatenate_datasets
import random
import math
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional

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
# 当一题 16 条里没有任何“正例 reward”，就认为全错：禁用 KL
POS_REWARD_THR = float(os.environ.get("GRPO_POS_REWARD_THR", "0.05"))
REPLAY_MIN_BUF = 32
# 当 rewards 没区分度（max-min 很小）时，这题的 adv≈0，直接跳过可省算力
REWARD_SPAN_EPS = float(os.environ.get("GRPO_REWARD_SPAN_EPS", "1e-6"))
REPLAY_MAX_TEMP = 0.50          # ✅ 只用 temp<=0.5 的候选进 replay 做自蒸馏
REPLAY_CE_SCALE = 0.2        # ✅ replay 这一下 CE 的强度（“加一点点”）
# 是否在“无区分度/全错”时直接跳过 policy/ref logprob（省算力）
SKIP_ZERO_SIGNAL_QUESTION = bool(int(os.environ.get("GRPO_SKIP_ZERO_SIGNAL_QUESTION", "1")))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
LORA_TARGET = "attn_mlp_skip_first4"
MAX_EXTRA_REWARD = 1.0
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
GOOD_COT_MIN_TOK = 60
WEAK_COT_MIN_TOK = 60
# TensorBoard 主目录对齐：
# /root/autodl-tmp/llm/checkpoints/tb/<run-name>
# ===== GRPO-LEAD hyperparams =====
# 长度依赖奖励：alpha 越大，对「比平均更长」的正确样本惩罚越狠
LEN_ALPHA = 0.05
LEN_EPS = 1e-4
LEN_MIN_FACTOR = float(os.environ.get("GRPO_LEAD_LEN_MIN_FACTOR", "0.6"))
LEN_MAX_FACTOR = float(os.environ.get("GRPO_LEAD_LEN_MAX_FACTOR", "1.4"))

# 错误样本统一负奖励（论文里是显式 penalty，这里默认 -1）
# NEG_REWARD = -0.55
NEG_REWARD = -1.0
# 难度感知 advantage 重加权的 logistic 函数超参（简化版）
# w(ρ) = B + A / (1 + exp(k * (ρ - ρ0)))
# ρ 越小（题越难），w 越接近 B + A，梯度放大；ρ 接近 1（题很容易）时，w ~ B
DIFF_A = float(os.environ.get("GRPO_LEAD_DIFF_A", "2.0"))
DIFF_B = float(os.environ.get("GRPO_LEAD_DIFF_B", "1.0"))
DIFF_K = float(os.environ.get("GRPO_LEAD_DIFF_K", "10.0"))
DIFF_RHO0 = float(os.environ.get("GRPO_LEAD_DIFF_RHO0", "0.75"))

RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_NAME = f"tinyllm-grpo-arc-{RUN_ID}"
TB_ROOT = str(project_path("runs/tensorboard"))
DEFAULT_TB_DIR = os.path.join(TB_ROOT, RUN_NAME)

# LoRA ckpt 保存根目录（每个 run 建一个子目录）
GRPO_SAVE_ROOT_DEFAULT = str(project_path("runs/training/grpo_arc"))

REPLAY_BUFFER_MAX_SIZE = 256        # 最多存多少道“好题”
REPLAY_EVERY_N_BATCHES = 12           # 每看到多少个 DataLoader batch，插一次 replay
REPLAY_MIN_LEN = 180             # 判定“短 COT”的长度下界（token 数）
REPLAY_MAX_LEN = 350             # 判定“短 COT”的长度上界
REPLAY_MIN_REWARD = 1.1            # 判定“高奖励”的下限（根据你现在 reward 量纲）
# =========================
# Global config
# =========================
LORA_ADAPTER_NAME = "arc"
LORA_RANK = 64
LORA_DROPOUT = 0.0
LORA_ALPHA  = 32.0

DTYPE = torch.bfloat16
DTYPE_EVAL = torch.float32
IGNORE_INDEX = -100
GRPO_DEBUG_PRINT_SAMPLES = False
GSM8K_EVAL_BATCH_SIZE_QUESTIONS = 24
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRPO_MAX_NEW_TOKENS_TRAIN = 350
GSM8K_SPLIT ="train" # "train" 或 "test"
GSM8K_MAX_EXAMPLES = 0 # 0 = 用完整 split
GSM8K_BATCH_SIZE_QUESTIONS =3 # 每个 step 多少道题
GRPO_NUM_GENERATIONS = 16  # 每题采样几条
GRPO_EVAL_EVERY_STEPS =100
GRPO_EVAL_NUM_QUESTIONS = 0
GSM8K_EVAL_SPLIT = "test"
ARC_USE_EASY_FOR_TRAIN =1
SAVE_STEPS = 100

# 梯度累积步数（默认 2，相当于 global batch = GSM8K_BATCH_SIZE_QUESTIONS * GRPO_GRAD_ACCUM_STEPS）
GRPO_GRAD_ACCUM_STEPS = 2

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
def maybe_add_to_replay_buffer(
    replay_buffer: List[Dict[str, str]],
    debug_infos: List[Dict],
    tokenizer,
):
    for info in debug_infos:
        user_prompt = info["user_prompt"]
        gt_answer_str = info["gt"]

        texts: List[str] = info["texts"]
        rewards: List[float] = info["rewards"]
        parsed_list: List[Dict] = info["parsed"]
        temps_list: List[float] = info.get("temps", [1.0] * len(texts))  # ✅ 没有就当全是高温

        if not texts:
            continue

        gt_val = extract_gsm8k_gt(gt_answer_str)
        if gt_val is None:
            continue

        # ✅ 候选池：必须 temp<=阈值 + reward>=下限 + 正确 + 有COT + 合法box + 长度合格
        candidates = []
        for i in range(len(texts)):
            t = float(temps_list[i]) if temps_list[i] is not None else 0.0  # greedy 视作 0.0
            if t > REPLAY_MAX_TEMP:
                continue
            if rewards[i] < REPLAY_MIN_REWARD:
                continue

            p = parsed_list[i]
            pred_val = p.get("final_answer", None)
            is_correct = (pred_val is not None) and (pred_val == gt_val)
            has_cot = bool(p.get("has_cot", False))
            in_box  = bool(p.get("in_box", False))
            if not (is_correct and has_cot and in_box):
                continue

            token_ids = tokenizer.encode(texts[i], add_special_tokens=False)
            L_total = len(token_ids)
            if not (REPLAY_MIN_LEN <= L_total <= REPLAY_MAX_LEN):
                continue

            # 排序 key：温度越低越优先；同温度下 reward 越高越优先
            candidates.append((t, -float(rewards[i]), i))

        if not candidates:
            continue

        candidates.sort()
        best_t, _, best_idx = candidates[0]
        best_text = texts[best_idx]
        best_reward = float(rewards[best_idx])

        # ✅ entry 现在要存 “prompt + 你选出来的低温正确答案”（用于 SFT/自蒸馏）
        entry = {
            "user_prompt": user_prompt,
            "gt": gt_answer_str,
            "completion": best_text,      # ✅ 关键：自证流用这个
            "temp": float(best_t),
            "reward": float(best_reward),
        }

        # ✅ 去重策略：同一个 prompt 只保留“更低温/同温更高reward”的版本
        replaced = False
        for j in range(len(replay_buffer)):
            if replay_buffer[j].get("user_prompt") == user_prompt:
                old_t = float(replay_buffer[j].get("temp", 1e9))
                old_r = float(replay_buffer[j].get("reward", -1e9))
                better = (best_t < old_t) or ((best_t == old_t) and (best_reward > old_r))
                if better:
                    replay_buffer[j] = entry
                replaced = True
                break

        if not replaced:
            replay_buffer.append(entry)
            if len(replay_buffer) > REPLAY_BUFFER_MAX_SIZE:
                replay_buffer.pop(0)


def build_batch_from_replay_buffer(
    replay_buffer: List[Dict[str, str]],
    batch_size: int,
) -> Dict[str, List[str]]:
    """
    从 replay buffer 里随机抽一小批题，组装成和 DataLoader 一样结构的 batch：
    {questions, answers, prompts}
    这里 questions 用 user_prompt 代替，反正 GRPO 逻辑只用到 prompts/answers。
    """
    k = min(batch_size, len(replay_buffer))
    chosen = random.sample(replay_buffer, k=k)

    prompts = [item["user_prompt"] for item in chosen]
    answers = [item["gt"] for item in chosen]
    questions = prompts  # 这里只是占位，grpo_step_for_batch 不会用到 questions

    return {
        "questions": questions,
        "answers": answers,
        "prompts": prompts,
    }
def build_sft_batch_from_replay_buffer(
    replay_buffer: List[Dict[str, str]],
    batch_size: int,
) -> Dict[str, List[str]]:
    k = min(batch_size, len(replay_buffer))
    chosen = random.sample(replay_buffer, k=k)

    prompts = [item["user_prompt"] for item in chosen]
    completions = [item["completion"] for item in chosen]

    return {
        "prompts": prompts,
        "completions": completions,
    }
def sft_ce_loss_on_completions(
    model: TinyLLM,
    tokenizer,
    user_prompts: List[str],
    completions: List[str],
    system_prompt: Optional[str],
) -> Tuple[torch.Tensor, int]:
    assert len(user_prompts) == len(completions)
    # 过滤空 completion
    pairs = [(p, c) for p, c in zip(user_prompts, completions) if c and c.strip()]
    if not pairs:
        # 造一个带 grad 的 0，避免 backward 报错
        loss0 = None
        for p in model.parameters():
            if p.requires_grad:
                loss0 = p.sum() * 0.0
                break
        if loss0 is None:
            loss0 = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        return loss0, 0

    user_prompts, completions = zip(*pairs)
    user_prompts = list(user_prompts)
    completions = list(completions)

    # 1) 编码 prompt / completion
    prompt_ids_list = []
    comp_ids_list = []
    for up, comp in zip(user_prompts, completions):
        chatml = render_prompt_with_tokenizer(tokenizer, up, system_prompt)
        p_ids = tokenizer.encode(chatml, add_special_tokens=False)
        c_ids = tokenizer.encode(comp, add_special_tokens=False)
        prompt_ids_list.append(p_ids)
        comp_ids_list.append(c_ids)

    # 2) 组装 seq，并 pad
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    seq_ids_list = [p + c for p, c in zip(prompt_ids_list, comp_ids_list)]
    max_len = max(len(x) for x in seq_ids_list)

    batch_ids = torch.full((len(seq_ids_list), max_len), pad_id, dtype=torch.long, device=DEVICE)
    attn = torch.zeros((len(seq_ids_list), max_len), dtype=torch.long, device=DEVICE)

    prompt_lens = []
    comp_lens = []
    for i, (p_ids, c_ids, seq) in enumerate(zip(prompt_ids_list, comp_ids_list, seq_ids_list)):
        L = len(seq)
        batch_ids[i, :L] = torch.tensor(seq, dtype=torch.long, device=DEVICE)
        attn[i, :L] = 1
        prompt_lens.append(len(p_ids))
        comp_lens.append(len(c_ids))

    # 3) forward
    out = model(
        input_ids=batch_ids,
        attention_mask=attn,
        labels=None,
        use_cache=False,
        force_checkpoint=True,
    )
    logits = out["logits"]  # [B,T,V]

    # 4) NLL，只对 completion tokens 计 loss
    logits_next = logits[:, :-1, :].float()
    targets = batch_ids[:, 1:]  # [B,T-1]
    logprobs = torch.log_softmax(logits_next, dim=-1)
    nll = -logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B,T-1]

    mask = torch.zeros_like(targets, dtype=torch.bool, device=DEVICE)
    for i, (pl, cl) in enumerate(zip(prompt_lens, comp_lens)):
        # 预测第一个 completion token 的位置是 pl-1
        start = max(pl - 1, 0)
        end = min(pl + cl - 1, targets.size(1))
        if end > start:
            mask[i, start:end] = True

    # 也要把 padding 删掉
    mask = mask & (attn[:, 1:] == 1)

    denom = int(mask.sum().item())
    loss = (nll * mask.float()).sum() / (mask.float().sum().clamp_min(1.0))
    return loss, denom

def build_gsm8k_prompt(question: str) -> str:
    q = question.strip()
    return (
        f"{q}\n"
        f"Put your final answer in LaTeX boxed form like $\\boxed{{answer}}$."
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
    ARC 封装；max_examples=0 表示全量。
    支持同时加载 ARC-Easy / ARC-Challenge，并混合后 shuffle。
    """

    def __init__(
        self,
        split: str = GSM8K_SPLIT,
        max_examples: int = GSM8K_MAX_EXAMPLES,
        seed: int = 42,
        configs: Optional[List[str]] = None,  # 新增：要用哪些 config
    ):
        # 默认只用 ARC-Challenge（和之前行为相同）
        if configs is None:
            configs = ["ARC-Challenge"]

        ds_list = []
        for cfg_name in configs:
            ds_cfg = load_local_dataset("arc", split, config=cfg_name)
            ds_list.append(ds_cfg)

        if len(ds_list) == 1:
            ds = ds_list[0]
        else:
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
        stem: str = ex["question"]          # 题干
        choice_texts = ex["choices"]["text"]
        choice_labels = ex["choices"]["label"]   # 一般是 ["A","B","C","D"]

        # 保险起见按 label 排一下
        pairs = sorted(zip(choice_labels, choice_texts), key=lambda x: x[0])

        # 拼成一个完整的多选题文本
        opts_lines = [f"{label}. {text}" for label, text in pairs]
        q_full = stem.strip() + "\n\n" + "\n".join(opts_lines)

        # ARC 的 GT 就是一个字母，比如 "A"
        ans = ex["answerKey"].strip()

        return {
            "question": q_full,
            "answer": ans,
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
    # 训练：根据环境变量决定是否混合 ARC-Easy

    if split == "train":
        configs = ["ARC-Easy"]   # 或者根据 ARC_USE_EASY_FOR_TRAIN 来选
    else:
        configs = ["ARC-Easy"]   # 这里你可以换成 ["ARC-Challenge"] 或混合

    dataset = GSM8KDataset(
        split=split,
        max_examples=max_examples,
        seed=seed,
        configs=configs,
    )
    print(f"[ARC] split={split}, configs={configs}, total examples={len(dataset)}")

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
    r"\\boxed\s*\{\s*([^{}]+?)\s*\}",
    flags=re.MULTILINE,
)

# 匹配“独立的 A/B/C/D”（前面是空格或全角空格，后面是空格/标点/结束）
CHOICE_STANDALONE_RE = re.compile(
    r"(?:^|[\s\u3000])([ABCDE])(?=[\s\.\,\!\?\:\;\)\]\u3000]|$)",
    flags=re.IGNORECASE,
)
ANSWER_LETTER_RE = re.compile(
    r"(?:answer|答案|选项)\s*[:：]?\s*([ABCD])",
    flags=re.IGNORECASE,
)
THOUGHT_START = "<|thought_start|>"
THOUGHT_END   = "<|thought_end|>"
def _is_truly_standalone_choice(raw: str, match_end: int) -> bool:
    """
    只允许形如 '... D.' / '... D )' / '... D。' 等，后面只有空白/标点，直到字符串结束。
    一旦在后面再看到字母/数字/汉字，就视为不是“单独答案字母”。
    """
    tail = raw[match_end:]  # 从字母后面开始看
    i = 0
    # 可以根据需要扩一下允许的标点集合
    ALLOWED = " \t\r\n.,!?;:)]}）】。、，；：！？"

    while i < len(tail) and tail[i] in ALLOWED:
        i += 1

    # 如果跳过空白/标点后已经到结尾 => 说明这个字母真的是最后一个“独立”内容
    return i == len(tail)
def parse_gsm8k_prediction(
    pred_str: str,
    tokenizer,
) -> Dict:
    if pred_str is None:
        pred_str = ""
    raw = pred_str.strip()

    # ===== 1) 思维链内容（沿用你原来的逻辑） =====
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

    # ===== 2) 优先从 \boxed{...} 里找一个选项字母 =====
    boxed_match = BOX_RE.search(raw)
    has_box = boxed_match is not None
    boxed_choice: Optional[str] = None
    if has_box:
        content = boxed_match.group(1).strip()
        m = re.fullmatch(r"[ABCDabcd]", content)

        if m:
            boxed_choice = m.group(0).upper()

    # ===== 3) 其他地方找“独立 A/B/C/D” =====
    letters: List[str] = []

    # 3.1 Answer: C / 答案：C / 选项 C
    for m in ANSWER_LETTER_RE.finditer(raw):
        letters.append(m.group(1).upper())

    # 3.2 前后都是空格/标点的独立字母
    for m in CHOICE_STANDALONE_RE.finditer(raw):
        if _is_truly_standalone_choice(raw, m.end()):
            letters.append(m.group(1).upper())

    fallback_choice: Optional[str] = None
    if letters:
        # 取“最后一次出现”的那个字母
        fallback_choice = letters[-1]

    # ===== 4) 决定最终答案（字母） =====
    has_valid_box_choice = boxed_choice in ["A", "B", "C", "D"]

    # 2) 全局有没有合法字母（包括 fallback）
    has_any_valid_choice = (
        has_valid_box_choice
        or (fallback_choice in ["A", "B", "C", "D"])
    )

    # 3) final_answer：优先用 box 里的字母，其次 fallback
    if has_valid_box_choice:
        final_choice = boxed_choice
    else:
        final_choice = fallback_choice if fallback_choice in ["A", "B", "C", "D"] else None

    from_box = (final_choice is not None) and (final_choice == boxed_choice)

    return {
        "raw": raw,
        "has_box": bool(has_box),
        "boxed_value": None,       # 占位，兼容旧字段
        "fallback_value": None,    # 占位，兼容旧字段
        "final_answer": final_choice,   # 'A' / 'B' / 'C' / 'D' or None
        "from_box": from_box,
        # ✅ 现在语义：只有“box 里有合法 A-D”才是 True
        "has_box_and_valid": bool(has_valid_box_choice),
        # （可选）你也可以顺便加一个字段：
        # "has_any_valid_choice": bool(has_any_valid_choice),
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
    no_repeat_ngram_size: int = 4,
    greedy: bool = False,
    temperatures: Optional[List[Optional[float]]] = None,
    top_ps: Optional[List[Optional[float]]] = None,
    greedy_mask: Optional[List[bool]] = None,
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
    use_mixed = (temperatures is not None) or (top_ps is not None) or (greedy_mask is not None)
    if temperatures is None:
        temperatures = [temperature] * B
    if top_ps is None:
        top_ps = [top_p] * B
    if greedy_mask is None:
        greedy_mask = [greedy] * B
    assert len(temperatures) == B and len(top_ps) == B and len(greedy_mask) == B
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

            if use_mixed or greedy:
                # 纯 greedy：直接取 argmax，忽略 temperature / top_p
                next_ids = torch.empty((B, 1), dtype=torch.long, device=DEVICE)

                for b in range(B):
                    if finished[b]:
                        next_ids[b, 0] = int(eos_id) if eos_id is not None else 0
                        continue

                    row = logits_step[b]

                    if greedy_mask[b]:
                        next_ids[b, 0] = torch.argmax(row, dim=-1).item()
                        continue

                    t = temperatures[b]
                    if t is not None and t > 0.0 and t != 1.0:
                        row = row / t

                    p = top_ps[b]
                    if p is not None and 0.0 < p < 1.0:
                        row = _top_p_filtering_row(row, p)

                    probs = F.softmax(row.float(), dim=-1)
                    next_ids[b, 0] = torch.multinomial(probs, num_samples=1).item()

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
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        texts.append(text.strip())

    # ===== 训练期采样的 Debug 打印（按概率抽样） =====
    # 只在非 greedy（也就是训练 GRPO 时）打印，避免评估刷屏

    debug_prob=0.1
    if (not greedy) and debug_prob > 0.0 and random.random() < debug_prob:
        print("\n" + "=" * 80)
        print("[GRPO sample DEBUG] one training sample batch")
        print("=" * 80)
        print("User prompt:")
        # 避免太长，截断一下
        print(user_prompt.strip()[:800])
        if system_prompt:
            print("\nSystem prompt:")
            print(system_prompt.strip()[:400])

        for i, txt in enumerate(texts):
            print("-" * 80)
            print(f"[candidate #{i}] ")
            print(txt.strip())
        print("=" * 80 + "\n")

    return {
        "texts": texts,
        "token_ids": generated_ids,
    }
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
            skip_special_tokens=False,
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

def extract_gsm8k_gt(answer_str: str) -> Optional[str]:
    """
    ARC 的 GT 就是一个选项字母，比如 'A' / 'B' / 'C' / 'D'。
    这里容忍一点额外文字，只要里面能找到 A-D 之一就行。
    """
    if answer_str is None:
        return None
    s = answer_str.strip().upper()
    if not s:
        return None
    m = re.search(r"[ABCD]", s)

    return m.group(0) if m else None

def cot_length_bonus(
    L_cot: Optional[int],
    min_good: int = 60,
    max_good: int = 200,
    mega_bad: int = 380,
) -> float:
    """
    绝对 COT 长度奖励（只看 <|thought_start|>...<|thought_end|> 里的 token 数）：
    - L_cot <= 0             → -0.5 （视为没 COT / 空壳）
    - 0 < L_cot <= min_good  → -0.5 线性涨到 0
    - min_good < L_cot <= max_good → 0 线性涨到 +0.5
    - max_good < L_cot < mega_bad  → 固定 +0.5（不再继续涨）
    - L_cot >= mega_bad      → -0.5（极端超长，当成不合格 COT）
    """
    if L_cot is None or L_cot <= 0:
        return -0.5

    # 0 ~ min_good：-0.5 → 0
    if L_cot <= min_good:
        return -0.5 + 0.5 * (L_cot / float(min_good))

    # min_good ~ max_good：0 → +0.5
    if L_cot <= max_good:
        return 0.5 * ((L_cot - min_good) / float(max_good - min_good))
    if L_cot <= mega_bad:
        # t: 0 at max_good, 1 at mega_bad
        denom = float(max(1, mega_bad - max_good))
        t = (L_cot - max_good) / denom
        return 0.5 - 1.0 * t   # 0.5 -> -0.5

    return -0.5
def _total_gen_length_score(
    raw_text: str,
    tokenizer,
    min_good: int = 40,
    max_good: int = 120,
    hard_max: int = 200,
) -> Tuple[float, int]:
    """
    返回 (len_score, L_total)：
    - L_total: 总生成 token 数
    - len_score：长度奖励，越超长越负
    """
    if not raw_text:
        return 0.0, 0

    token_ids = tokenizer.encode(
        raw_text,
        add_special_tokens=False,
    )
    L = len(token_ids)

    # 这里只保留很粗的三挡：正常 / 偏长 / 超长
    if L <= min_good:
        # 太短，略微负一点
        return -0.2, L
    if L <= max_good:
        # 理想区间：给一点点正激励，但不要太大（避免盖过正确性）
        return 0.5, L
    if L <= hard_max:
        # 超过 max_good 但没爆 hard_max：轻微负
        return -0.3, L

    # 严重超长：直接大负分
    return -1.0, L
def compute_rewards_for_group(
    gt_answer_str: str,
    candidate_texts: List[str],
    tokenizer,
) -> Tuple[torch.Tensor, List[Dict]]:
    """
    简化版 reward（无组内长度缩放、无难度重权）：
    - 超长 hit_ceiling：NEG_REWARD
    - 正确 + box(valid A-D)： 1.0 + (cot_length_bonus) + fmt_bonus
    - 正确但没 box：0.2
    - 其它：NEG_REWARD
    """
    gt_val = extract_gsm8k_gt(gt_answer_str)

    rewards: List[float] = []
    parsed_list: List[Dict] = []

    for text in candidate_texts:
        info = parse_gsm8k_prediction(text, tokenizer)

        pred_val = info["final_answer"]
        in_box   = info["has_box_and_valid"]

        cot_len      = info["cot_token_len"]
        has_cot_tags = info["has_cot_tags"]

        good_cot = has_cot_tags and (cot_len >= GOOD_COT_MIN_TOK)         # >=100
        weak_cot = has_cot_tags and (cot_len >= WEAK_COT_MIN_TOK)         # >=60

        # 兼容你 replay 的字段语义
        info["good_cot"] = bool(good_cot)
        info["weak_cot"] = bool(weak_cot)
        info["has_cot"] = bool(good_cot)          # has_cot == 合格 COT
        info["has_any_cot"] = bool(weak_cot)

        is_correct = (
            (gt_val is not None)
            and (pred_val is not None)
            and (pred_val == gt_val)
        )
        is_correct_and_box = bool(is_correct and in_box)

        token_ids = tokenizer.encode(text or "", add_special_tokens=False)
        L_total = len(token_ids)
        hit_ceiling = (L_total >= GRPO_MAX_NEW_TOKENS_TRAIN - 10)

        info["is_correct"] = bool(is_correct)
        info["is_correct_and_box"] = bool(is_correct_and_box)
        info["total_tokens"] = int(L_total)
        info["hit_ceiling"] = bool(hit_ceiling)
        info["in_box"] = bool(in_box)

        # ===== reward =====
        if hit_ceiling:
            r = NEG_REWARD


        elif is_correct_and_box:

            base = 1.0

            if has_cot_tags:

                len_bonus = cot_length_bonus(

                    L_cot=cot_len,

                    min_good=WEAK_COT_MIN_TOK,  # 60

                    max_good=250,

                    mega_bad=380,

                )

                len_bonus *= 1.2  # 你说的 1.2/1.3 放这里

            else:

                len_bonus = 0.0  # ✅ 关键：不写 COT 也别被扣分

            fmt_bonus = 0.0  # ✅ 直接删

            extra = len_bonus + fmt_bonus

            extra = max(-0.5, min(extra, MAX_EXTRA_REWARD))

            r = base + extra


        elif is_correct:
            r = 0.2
        else:
            r = NEG_REWARD

        rewards.append(float(r))
        parsed_list.append(info)

    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
    return rewards_tensor, parsed_list


def _difficulty_weight(rho: float) -> float:
    """
    w(ρ) = DIFF_B + DIFF_A / (1 + exp(DIFF_K * (ρ - DIFF_RHO0)))
    ρ 越小（题越难），w 越大
    """
    rho = float(rho)
    return DIFF_B + DIFF_A / (1.0 + math.exp(DIFF_K * (rho - DIFF_RHO0)))
def grpo_loss_for_group(
    logprob_sums: torch.Tensor,
    rewards: torch.Tensor,
    logprob_sums_ref: Optional[torch.Tensor] = None,
    kl_coef: float = 0.0,
    gen_lens: torch.Tensor =None,
    eps: float = 1e-5,
    pos_thr: float = POS_REWARD_THR,
sample_weights: Optional[torch.Tensor] = None,
):
    r_max = rewards.max().detach()
    r_min = rewards.min().detach()
    reward_span = (r_max - r_min)

    use_kl = (
        (logprob_sums_ref is not None)
        and (kl_coef > 0.0)
        and (r_max > pos_thr)
        and (reward_span > REWARD_SPAN_EPS)
    )

    # ---- adv: 只用 raw rewards，组内标准化 + clip，然后 detach ----
    adv = rewards - rewards.mean()
    std = adv.std(unbiased=False)
    std = torch.clamp(std, min=eps)
    adv = adv / std
    ADV_CLIP = 5.0
    adv = adv.clamp(-ADV_CLIP, ADV_CLIP).detach()
    pg_lp = logprob_sums / (gen_lens + 1e-6)
    if sample_weights is None:
        w = torch.ones_like(rewards)
    else:
        w = sample_weights.to(device=rewards.device, dtype=rewards.dtype)

    pg_loss = -((adv * pg_lp) * w).sum() / (w.sum() + 1e-6)
    # ---- KL: 作为正则项加入 loss，必须保留梯度 ----
    KL_EST_MAX = 5.0
    if use_kl:
        log_ratio = (logprob_sums - logprob_sums_ref) / (gen_lens + 1e-6)
        log_ratio = log_ratio.clamp(min=-20.0, max=20.0)

        kl_est = torch.expm1(log_ratio) - log_ratio  # exp(x)-1-x
        kl_est = kl_est.clamp(max=KL_EST_MAX)

        kl_loss = (kl_coef * kl_est).mean()

        # 仅用于 logging / 可视化
        kl_term_detached = (kl_coef * kl_est.detach())
    else:
        kl_loss = torch.zeros((), device=logprob_sums.device)
        kl_term_detached = torch.zeros_like(rewards)

    loss = pg_loss + kl_loss
    shaped_rewards = (rewards - kl_term_detached).detach()
    return loss, adv, shaped_rewards, kl_term_detached


def grpo_step_for_batch(
    policy_model: TinyLLM,
    ref_model: TinyLLM,
    tokenizer,
    batch: Dict[str, List[str]],
    system_prompt: Optional[str],
    num_generations: int = GRPO_NUM_GENERATIONS,
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
    per_question_active = []  # ✅补
    active_losses = []
    debug_infos = []

    for q_idx, (user_prompt, gt_answer_str) in enumerate(zip(prompts, answers)):
        # 1) 采样：无梯度
        # 1) 采样：无梯度
        B = num_generations
        greedy_mask = [False] * B
        temps = [0.9] * B
        ps = [0.95] * B

        greedy_mask[0] = True
        temps[0] = None
        ps[0] = None

        if B > 1:
            temps[1] = 0.25
            ps[1] = 1.0
        if B > 2:
            temps[2] = 0.35
            ps[2] = 1.0
        if B > 3:
            temps[3] = 0.50
            ps[3] = 0.95
        if B > 4:
            temps[4] = 0.65
            ps[4] = 0.95
        if B > 5:
            temps[5] = 0.75
            ps[5] = 0.95
        if B > 6:
            temps[6] = 0.80
            ps[6] = 0.95
        if B > 7:
            temps[7] = 0.85
            ps[7] = 0.95
        if B>8:
            temlen = 8 + (B - 8) // 2
            for i in range(8, temlen):
                temps[i] = 0.85
                ps[i] = 0.95
        samples = sample_for_grpo_manual(
            model=policy_model,
            tokenizer=tokenizer,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            num_generations=B,
            max_new_tokens=GRPO_MAX_NEW_TOKENS_TRAIN,
            temperatures=temps,
            top_ps=ps,
            greedy_mask=greedy_mask,
            stop_on_punct=False,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
        )

        texts = samples["texts"]
        token_ids = samples["token_ids"]

        # 2) 先算 rewards（便宜）=> 决定要不要算 ref（贵）
        with torch.no_grad():
            rewards, parsed_list = compute_rewards_for_group(
                gt_answer_str=gt_answer_str,
                candidate_texts=texts,
                tokenizer=tokenizer,
            )

        r_max = float(rewards.max().item()) if rewards.numel() > 0 else -1e9
        r_min = float(rewards.min().item()) if rewards.numel() > 0 else  1e9
        reward_span = r_max - r_min

        # 你要的判定：只要没有任何 >0.05 的 reward，就视为“全错/无正例”
        has_pos = any(p.get("is_correct_and_box", False) for p in parsed_list)

        # 是否需要 KL：必须有正例 + KL>0 + rewards 有区分度
        need_kl = (kl_coef > 0.0) and has_pos and (reward_span > REWARD_SPAN_EPS)

        # ✅（可选，但强烈建议）全错且 rewards 完全没区分度 => 这题 adv=0，直接跳过整题 logprob 计算省算力
        skip_question = SKIP_ZERO_SIGNAL_QUESTION and ((not has_pos) or (reward_span <= REWARD_SPAN_EPS))

        if skip_question:
            # 造一个“带 grad 的 0 loss”，避免整个 micro-batch 都跳过时 backward 报错
            loss_q = None
            for p in policy_model.parameters():
                if p.requires_grad:
                    loss_q = p.sum() * 0.0
                    break
            if loss_q is None:
                loss_q = torch.tensor(0.0, device=DEVICE, requires_grad=True)

            adv_q = torch.zeros_like(rewards)
            shaped_rewards_q = rewards.detach()
            kl_term_q = torch.zeros_like(rewards)

            # debug 里需要 list，给占位
            logprob_sums_policy = torch.zeros_like(rewards)
            logprob_sums_ref = torch.zeros_like(rewards)

        else:
            # 3) ref logprob：只有 need_kl 才算（省算力）
            logprob_sums_ref = None
            if need_kl:
                with torch.no_grad():
                    logprob_sums_ref = compute_logprob_sums_for_model(
                        model=ref_model,
                        tokenizer=tokenizer,
                        user_prompt=user_prompt,
                        system_prompt=system_prompt,
                        sampled_token_ids=token_ids,
                        use_cache=False,
                    )

            # 4) policy logprob：有梯度（必须算，除非你上面 skip 了）
            logprob_sums_policy = compute_logprob_sums_for_model(
                model=policy_model,
                tokenizer=tokenizer,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                sampled_token_ids=token_ids,
                use_cache=False,
                force_checkpoint=True,
            )

            gen_lens = torch.tensor(
                [len(x) for x in token_ids],
                dtype=torch.float32,
                device=DEVICE,
            )
            temp_vals = [0.0 if (t is None) else float(t) for t in temps]
            correct_mask = [bool(p.get("is_correct_and_box", False)) for p in parsed_list]

            T_LOW = 0.60
            T_MID = 0.65
            T_HIGH = 0.80

            has_low_correct = any(c and (temp_vals[i] <= T_LOW) for i, c in enumerate(correct_mask))

            w_list = [1.0] * len(temp_vals)
            if has_low_correct:
                # 只有“同组里已经有低温正确”时，才把高温正确降权（避免唯一正确来自高温时被你误伤）
                for i, (c, t) in enumerate(zip(correct_mask, temp_vals)):
                    if c and t >= T_HIGH:
                        w_list[i] = 0.3  # 强降权：幸运高温正确
                    elif c and t >= T_MID:
                        w_list[i] = 0.60  # 中温正确：轻降权（可选）
                    # 低温正确 & 错误样本：保持 1.0

            sample_weights = torch.tensor(w_list, device=DEVICE, dtype=torch.float32)

            # 5) GRPO loss：如果 need_kl=False，就把 kl_coef 置 0，ref 也可以为 None
            loss_q, adv_q, shaped_rewards_q, kl_term_q = grpo_loss_for_group(
                logprob_sums=logprob_sums_policy,
                rewards=rewards,
                logprob_sums_ref=logprob_sums_ref,
                kl_coef=(kl_coef if need_kl else 0.0),
                gen_lens=gen_lens,
                sample_weights=sample_weights,
            )

        per_question_losses.append(loss_q)
        per_question_active.append(not skip_question)
        if not skip_question:  # ✅补：只有有信号的题才参与 batch_loss 平均
            active_losses.append(loss_q)
        # debug_infos 里别再对 None 做 detach：统一转成 list（None 就用 0 占位）
        if logprob_sums_ref is None:
            logprob_sums_ref_list = [0.0] * len(token_ids)
        else:
            logprob_sums_ref_list = logprob_sums_ref.detach().cpu().tolist()

        debug_infos.append({
            "question_idx": q_idx,
            "user_prompt": user_prompt,
            "gt": gt_answer_str,
            "texts": texts,
            "rewards": rewards.detach().cpu().tolist(),
            "shaped_rewards": shaped_rewards_q.detach().cpu().tolist(),
            "adv": adv_q.detach().cpu().tolist(),
            "logprob_sums_policy": logprob_sums_policy.detach().cpu().tolist(),
            "logprob_sums_ref": logprob_sums_ref_list,
            "parsed": parsed_list,
            "loss": float(loss_q.item()),
            "kl_term": kl_term_q.detach().cpu().tolist(),
            # 额外给你看一下 gating 是否触发：
            "need_kl": bool(need_kl),
            "r_max": float(r_max),
            "reward_span": float(reward_span),
            "skip_question": bool(skip_question),
            "has_pos": bool(has_pos),
            "temps": [0.0 if (temps[i] is None) else float(temps[i]) for i in range(len(texts))],
            "top_ps": [None if (ps[i] is None) else float(ps[i]) for i in range(len(texts))],
            "greedy_mask": [bool(greedy_mask[i]) for i in range(len(texts))],

        })
    if len(active_losses) > 0:
        # frac = len(active_losses) / len(per_question_losses)
        # scale = max(frac, 0.25)  # 下限别让它太小
        batch_loss = torch.stack(active_losses).mean() #* scale
    else:
        batch_loss = torch.stack(per_question_losses).mean()  # 全是 0 的占位 loss
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

        gt_val = extract_gsm8k_gt(gt_answer)      # 'A'/'B'/...
        pred_val = info["final_answer"]           # 'A'/'B'/...

        is_correct = (
            (gt_val is not None)
            and (pred_val is not None)
            and (pred_val == gt_val)
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
            gt_val   = extract_gsm8k_gt(gt_answer)     # 'A'/'B'/...
            pred_val = info["final_answer"]            # 'A'/'B'/...

            is_correct = (
                (gt_val is not None)
                and (pred_val is not None)
                and (pred_val == gt_val)
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
                    print(info["raw"])
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
    lr: float = 1e-5,
    kl_final: float = 0.04,
):
    loader = make_gsm8k_dataloader(
        split=GSM8K_SPLIT,
        batch_size=GSM8K_BATCH_SIZE_QUESTIONS,
        max_examples=GSM8K_MAX_EXAMPLES,
        seed=42,
    )
    replay_buffer: List[Dict[str, str]] = []
    global_batch_idx = 0
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
        configs=["ARC-Easy"],  # <<< 关键
    )
    system_prompt = SYSTEM_PROMPT_FOR_COT if USE_SYSTEM_PROMPT_FOR_COT else None
    print("[eval @ step 0] running baseline evaluation...")
    eval_acc0, eval_avg_reward0 = evaluate_on_gsm8k_batched(
        policy_model=policy_model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        max_new_tokens=700,
        batch_size=GSM8K_EVAL_BATCH_SIZE_QUESTIONS,
        debug=True,  # 开启打印
        debug_max=30,  # 只看前 30 题
        debug_only_wrong=False,  # 也可以改成 True 只看错题
    )
    print(
        f"[eval @ step 0] gsm8k_acc={eval_acc0:.3f}  "
        f"avg_reward={eval_avg_reward0:.3f}"
    )
    writer.add_scalar("eval/arc_acc", eval_acc0, 0)
    writer.add_scalar("eval/avg_reward", eval_avg_reward0, 0)
    optimizer = torch.optim.AdamW(
        (p for p in policy_model.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=0.0,
    )

    num_batches_per_epoch = len(loader)
    # 每个 step = GRPO_GRAD_ACCUM_STEPS 个 micro step
    max_train_steps = 500
    # 10% warmup，你可以改成 0.05 等
    warmup_steps = 10
    def lr_lambda(current_step: int):
        """
        current_step 从 0 开始：
        - [0, warmup_steps): 线性升到 1.0
        - [warmup_steps, max_train_steps): 线性降到 0.2
        """
        step = float(current_step)
        if step < warmup_steps:
            return (step + 1.0) / float(warmup_steps)
        # 进入 decay 阶段
        progress = (step - warmup_steps) / max(1.0, max_train_steps - warmup_steps)
        # 从 1.0 线性到 0.2
        return 1.0 - 0.8 * min(max(progress, 0.0), 1.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    step_idx = 0          # 优化器 update 次数
    micro_step = 0        # 累积用的 micro step 计数
    accum_loss = 0.0
    accum_has_signal = False
    accum_kl_term_on: List[float] = []
    accum_need_kl_q = 0
    accum_total_q = 0
    accum_raw_rewards_for_step: List[float] = []
    accum_shaped_rewards_for_step: List[float] = []
    accum_kl_for_step: List[float] = []
    KL_STEP1 = 150   # 在第 100 个 update 时到 0.05
    KL_STEP2 = 300   # 在第 200 个 update 时到 0.1

    def get_kl_coef(current_step: int) -> float:
        """
        current_step = 已经完成的 optimizer update 次数（step_idx）
        - 0   → KL ~ 0.0
        - 100 → KL ~ 0.05
        - 200 → KL ~ 0.10
        - 之后维持在 kl_final（默认为 0.10）
        """
        if current_step <= 0:
            return 0.0

        if current_step < KL_STEP1:
            # 0 → 0.5 * kl_final（即 0 → 0.05）
            return (0.5 * kl_final) * float(current_step) / float(KL_STEP1)

        if current_step < KL_STEP2:
            # 0.5*kl_final → kl_final（即 0.05 → 0.10）
            frac = float(current_step - KL_STEP1) / float(max(1, KL_STEP2 - KL_STEP1))
            return 0.5 * kl_final + 0.5 * kl_final * min(max(frac, 0.0), 1.0)

        # 200 之后一直维持在 kl_final（0.10）
        return kl_final
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        for batch in loader:
            global_batch_idx += 1
            micro_step += 1

            # 这里用当前的 step_idx 来取 KL（一个 step_idx 对应 GRPO_GRAD_ACCUM_STEPS 个 micro step）
            current_kl = get_kl_coef(step_idx)

            loss, debug_infos = grpo_step_for_batch(
                policy_model=policy_model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                batch=batch,
                system_prompt=system_prompt,
                num_generations=GRPO_NUM_GENERATIONS,
                kl_coef=current_kl,
            )
            batch_has_signal = any((not info["skip_question"]) for info in debug_infos)
            accum_has_signal = accum_has_signal or batch_has_signal
            if batch_has_signal:
                (loss / GRPO_GRAD_ACCUM_STEPS).backward()

            for info in debug_infos:
                accum_total_q += 1
                if info["need_kl"]:
                    accum_need_kl_q += 1
                    accum_kl_term_on.extend(info["kl_term"])
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

            # 累积当前 loss/reward，方便做 step 级别的 logging
            maybe_add_to_replay_buffer(replay_buffer, debug_infos, tokenizer)

            # ====== 2) 每 REPLAY_EVERY_N_BATCHES 个 batch，插入一轮 replay GRPO ======
            # ====== 2) 每 REPLAY_EVERY_N_BATCHES 个 batch，插入一轮 replay CE（自证流 / SFT） ======
            replay_n=REPLAY_EVERY_N_BATCHES

            if (
                    REPLAY_EVERY_N_BATCHES > 0
                    and (global_batch_idx % replay_n == 0)
                    and len(replay_buffer) > REPLAY_MIN_BUF
            ):
                sft_batch = build_sft_batch_from_replay_buffer(
                    replay_buffer,
                    batch_size=GSM8K_BATCH_SIZE_QUESTIONS,
                )

                loss_replay_ce, n_tok = sft_ce_loss_on_completions(
                    model=policy_model,
                    tokenizer=tokenizer,
                    user_prompts=sft_batch["prompts"],
                    completions=sft_batch["completions"],
                    system_prompt=system_prompt,
                )

                scaled_loss_replay = loss_replay_ce * REPLAY_CE_SCALE
                (scaled_loss_replay / GRPO_GRAD_ACCUM_STEPS).backward()

                # CE 一定有梯度信号（除非 n_tok==0），所以这么写就行
                if n_tok > 0:
                    accum_has_signal = True
                    accum_loss += float(scaled_loss_replay.item())
                    writer.add_scalar("train/replay_ce_loss", float(loss_replay_ce.item()), global_batch_idx)
                    writer.add_scalar("train/replay_ce_tokens", float(n_tok), global_batch_idx)


            # ====== 3) 累积当前 loss/reward，方便做 step 级别的 logging ======
            accum_loss += float(loss.item())
            for info in debug_infos:
                accum_raw_rewards_for_step.extend(info["rewards"])
                accum_shaped_rewards_for_step.extend(info["shaped_rewards"])
                accum_kl_for_step.extend(info["kl_term"])
            # 到了一个完整的累计步数，才真正更新一次参数 & 记录一次 TB
            if micro_step % GRPO_GRAD_ACCUM_STEPS == 0:
                if not accum_has_signal:
                    # 这一整个累积窗口完全没信号：不更新、不走 scheduler、不涨 step_idx
                    optimizer.zero_grad(set_to_none=True)
                    accum_has_signal = False

                    # 把 accum_* 清掉，避免污染下一次
                    accum_loss = 0.0
                    accum_raw_rewards_for_step.clear()
                    accum_shaped_rewards_for_step.clear()
                    accum_kl_for_step.clear()
                    accum_total_q = 0
                    accum_need_kl_q = 0
                    accum_kl_term_on.clear()
                    # 可选：用 global_batch_idx 记一下发生次数（别用 step_idx，避免同 x 重复写）
                    writer.add_scalar("train/skip_update", 1.0, global_batch_idx)
                    continue

                step_idx += 1

                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                avg_loss = accum_loss / GRPO_GRAD_ACCUM_STEPS
                if accum_raw_rewards_for_step:
                    avg_raw_reward = float(
                        torch.tensor(accum_raw_rewards_for_step, dtype=torch.float32).mean().item()
                    )
                else:
                    avg_raw_reward = 0.0

                if accum_shaped_rewards_for_step:
                    avg_shaped_reward = float(
                        torch.tensor(accum_shaped_rewards_for_step, dtype=torch.float32).mean().item()
                    )
                else:
                    avg_shaped_reward = 0.0
                if accum_kl_for_step:
                    avg_kl_term = float(
                        torch.tensor(accum_kl_for_step, dtype=torch.float32).mean().item()
                    )
                else:
                    avg_kl_term = 0.0

                writer.add_scalar("train/loss", avg_loss, step_idx)
                writer.add_scalar("train/avg_reward_raw", avg_raw_reward, step_idx)
                writer.add_scalar("train/avg_reward_shaped", avg_shaped_reward, step_idx)
                writer.add_scalar("train/kl_coef", float(current_kl), step_idx)
                writer.add_scalar("train/kl_q_trigger_rate", accum_need_kl_q / max(1, accum_total_q), step_idx)

                avg_kl_term_on = float(torch.tensor(accum_kl_term_on).mean().item()) if accum_kl_term_on else 0.0
                writer.add_scalar("train/avg_kl_term_on", avg_kl_term_on, step_idx)
                writer.add_scalar("train/avg_kl_term", avg_kl_term, step_idx)
                if current_kl > 0:
                    writer.add_scalar("train/avg_kl_est_on", avg_kl_term_on / current_kl, step_idx)

                if step_idx % 10 == 0:
                    print(
                        f"[step {step_idx}] "
                        f"loss={avg_loss:.4f}  "
                        f"raw_reward={avg_raw_reward:.3f}  "
                        f"shaped_reward={avg_shaped_reward:.3f}  "
                        f"kl={current_kl:.4f}"
                    )
                # eval
                if step_idx % GRPO_EVAL_EVERY_STEPS == 0:
                    eval_acc, eval_avg_reward = evaluate_on_gsm8k_batched(
                        policy_model=policy_model,
                        tokenizer=tokenizer,
                        eval_dataset=eval_dataset,
                        system_prompt=system_prompt,
                        max_new_tokens=700,
                        batch_size=GSM8K_EVAL_BATCH_SIZE_QUESTIONS,
                        debug=False,  # 训练中就先关掉打印，避免太吵
                    )
                    print(
                        f"[eval @ step {step_idx}] "
                        f"gsm8k_acc={eval_acc:.3f}  avg_reward={eval_avg_reward:.3f}"
                    )
                    writer.add_scalar("eval/arc_acc", eval_acc, step_idx)
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
                                "GRPO on AI2 ARC (train: Easy, eval: Easy)."
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
                accum_raw_rewards_for_step = []
                accum_shaped_rewards_for_step = []
                accum_kl_for_step = []
                accum_kl_term_on.clear()
                accum_need_kl_q = 0
                accum_total_q = 0
                accum_has_signal = False
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
        num_epochs=3,
        lr=5e-6,  # 更稳一点
        kl_final=10.0,  # 最终 KL 目标值
    )
#kl_final=0.04
if __name__ == "__main__":
    main()
