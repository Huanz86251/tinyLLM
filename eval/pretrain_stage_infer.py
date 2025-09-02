# infer.py
import os, re, torch
from transformers import AutoTokenizer
from safetensors.torch import load_file as safe_load  # 不存在时也能继续
from model.config import Config
from model.model import TinyLLM

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "train/runs/tinyllm_3")
MODEL_DIR_FOR_TOKENIZER = os.path.join(os.path.dirname(__file__), "..", "model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16


def latest_ckpt_dir(output_dir: str) -> str:
    pats = []
    for d in os.listdir(output_dir):
        m = re.match(r"checkpoint-(\d+)$", d)
        if m:
            pats.append((int(m.group(1)), os.path.join(output_dir, d)))
    if not pats:
        return output_dir
    pats.sort()
    return pats[-1][1]


tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR_FOR_TOKENIZER)


def build_model_and_load():
    cfg = Config(
        vocab_size=tokenizer.vocab_size,
        train_maxlength=512,
    )
    cfg.use_checkpoint = False
    model = TinyLLM(cfg)
    ckpt = latest_ckpt_dir(OUTPUT_DIR)
    sf_path = os.path.join(ckpt, "model.safetensors")
    state = safe_load(sf_path, device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[WARN] Missing keys:", missing[:10], "..." if len(missing) > 10 else "")
    if unexpected:
        print("[WARN] Unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")
    model.to(DEVICE, dtype=DTYPE)
    model.eval()
    return model


@torch.no_grad()
def generate(
    model, tokenizer, prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 3,
):
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    generated = input_ids.clone()
    eos = tokenizer.eos_token_id
    unk = getattr(tokenizer, "unk_token_id", None)

    kv_cache = None
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else torch.cpu.amp.autocast(dtype=torch.bfloat16)
    )

    def _penalize_repetition(logits, seq, penalty):
        if penalty is None or penalty <= 1.0:  # <=1 不生效
            return logits
        unique, counts = seq.unique(return_counts=True)
        logits[:, unique] /= penalty ** counts.float().to(logits.dtype)
        return logits

    def _apply_no_repeat_ngram(seq, logits, n):
        if n <= 1 or seq.size(1) < n - 1:
            return logits
        bsz = seq.size(0)
        for b in range(bsz):
            prev = seq[b].tolist()
            if len(prev) < n - 1:
                continue
            ngram_prefix = tuple(prev[-(n-1):])
            # 找所有已出现的 ngram，屏蔽其后续 token
            for i in range(len(prev) - (n - 1)):
                if tuple(prev[i:i+n-1]) == ngram_prefix:
                    next_tok = prev[i+n-1]
                    logits[b, next_tok] = -float("inf")
        return logits

    with autocast_ctx:
        cur_input = input_ids
        for _ in range(max_new_tokens):
            out = model(input_ids=cur_input, use_cache=True, past_key_value=kv_cache)
            logits = out["logits"][:, -1, :]  # [B,V]
            kv_cache = out["past_key_values"]

            # 屏蔽 <unk>
            if unk is not None and 0 <= unk < logits.size(-1):
                logits[:, unk] = -float("inf")

            logits = _penalize_repetition(logits, generated, repetition_penalty)
            logits = _apply_no_repeat_ngram(generated, logits, no_repeat_ngram_size)


            if temperature and temperature > 0:
                logits = logits / temperature

            if top_k is not None and top_k > 0:
                topk_vals, topk_idx = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
                filt = torch.full_like(logits, -float("inf"))
                filt.scatter_(1, topk_idx, topk_vals)
                logits = filt

            if top_p is not None and 0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs = torch.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(probs, dim=-1)
                mask = cumprobs > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                sorted_logits = sorted_logits.masked_fill(mask, -float("inf"))
                logits = torch.full_like(logits, -float("inf"))
                logits.scatter_(1, sorted_idx, sorted_logits)

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # 采样而不是 argmax
            generated = torch.cat([generated, next_token], dim=-1)

            if eos is not None and next_token.item() == eos:
                break
            cur_input = next_token

    text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=False)
    return text

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    model = build_model_and_load()
    prompts = [
        "在漫长的历史长河中，普通人的命运常常被时代裹挟，但每一次微小的坚持",
    ]
    for p in prompts:
        print("\n=== Prompt ===")
        print(p)
        for L in [10, 15, 30]:
            out = generate(model, tokenizer, p, max_new_tokens=L, temperature=0.5)
            added = out[len(p):]
            print(f"\n-- 续写 {L} tokens --")
            print(added.strip())
