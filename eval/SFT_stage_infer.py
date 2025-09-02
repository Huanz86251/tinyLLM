import os, re, torch
from transformers import AutoTokenizer
from safetensors.torch import load_file as safe_load
from model.config import Config
from model.model import TinyLLM

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "train/runs/tinyllm_SFT")
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
        train_maxlength=2048,
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
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    min_new_tokens: int = 20,
    stop_on_eos: bool = True,
):
    device = next(model.parameters()).device
    messages = [{"role": "user", "content": prompt}]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    generated = input_ids.clone()
    eos = tokenizer.eos_token_id
    unk = getattr(tokenizer, "unk_token_id", None)

    try:
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, list):
            im_end_id = im_end_id[0] if im_end_id else None
    except Exception:
        im_end_id = None

    stop_ids = [tok for tok in [eos, im_end_id] if stop_on_eos and tok is not None and tok >= 0]

    kv_cache = None
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else torch.cpu.amp.autocast(dtype=torch.bfloat16)
    )

    new_tokens_len = 0

    with autocast_ctx:
        cur_input = input_ids
        for _ in range(max_new_tokens):
            out = model(input_ids=cur_input, use_cache=True, past_key_value=kv_cache)
            logits = out["logits"][:, -1, :]  # [B,V]
            kv_cache = out["past_key_values"]

            if unk is not None and 0 <= unk < logits.size(-1):
                logits[:, unk] = -float("inf")

            if new_tokens_len < max(min_new_tokens, 0):
                for sid in stop_ids:
                    logits[:, sid] = -float("inf")

            if temperature and temperature > 0:
                logits = logits / temperature

            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                topk_vals, topk_idx = torch.topk(logits, k=k, dim=-1)
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
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)
            new_tokens_len += 1

            if new_tokens_len >= max(min_new_tokens, 0) and stop_ids and int(next_token.item()) in stop_ids:
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
        "程序设计的本质并不在于语法本身，你觉得呢？"
    ]
    for p in prompts:
        print("\n=== Prompt ===")
        print(p)
        out = generate(
            model, tokenizer,
            "程序设计的本质并不在于语法本身，你觉得呢？",
            max_new_tokens=128,
            temperature=0.2,
            top_p=0.85,
            top_k=0,
            min_new_tokens=40,
            stop_on_eos=True,
        )
        print("\n-- FULL --")
        print(out)
