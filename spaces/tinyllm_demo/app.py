"""Small CPU-friendly Hugging Face Space for the public tinyLLM checkpoints."""
from __future__ import annotations

import threading

import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


BASE_REPO = "chris0809/tinyLLM-0.51B-SFT"

_model = None
_tokenizer = None
_lock = threading.RLock()


def load_runtime():
    global _model, _tokenizer
    with _lock:
        if _model is None:
            _tokenizer = AutoTokenizer.from_pretrained(BASE_REPO)
            _model = AutoModelForCausalLM.from_pretrained(
                BASE_REPO,
                trust_remote_code=True,
                dtype=torch.bfloat16,
            ).to("cuda").eval()
        return _model, _tokenizer


@spaces.GPU(duration=45)
def respond(message: str, history: list[dict], max_new_tokens: int):
    if not message.strip():
        yield "请输入一个问题。"
        return
    with _lock:
        model, tokenizer = load_runtime()
        messages = [
            {"role": row["role"], "content": row["content"]}
            for row in history
            if row.get("role") in {"user", "assistant"} and isinstance(row.get("content"), str)
        ]
        messages.append({"role": "user", "content": message.strip()})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = {
            key: value.to(next(model.parameters()).device)
            for key, value in tokenizer(prompt, return_tensors="pt").items()
        }
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
            "repetition_penalty": 1.10,
            "no_repeat_ngram_size": 12,
            "use_cache": True,
        }
        worker = threading.Thread(target=model.generate, kwargs=kwargs, daemon=True)
        worker.start()
        text = ""
        for piece in streamer:
            text += piece
            yield text
        worker.join()


# ZeroGPU provides a CUDA emulation context during startup, so the weights can
# stay GPU-resident and are ready when the decorated function receives a slot.
load_runtime()


with gr.Blocks(title="tinyLLM 0.51B") as demo:
    gr.Markdown(
        "# tinyLLM 0.51B\n"
        "从零训练的中英双语小模型。这里在线运行公开的 0.51B SFT checkpoint。"
    )
    max_tokens = gr.Slider(32, 256, value=128, step=32, label="Maximum new tokens")
    gr.ChatInterface(
        fn=respond,
        additional_inputs=[max_tokens],
        type="messages",
        examples=[
            ["用两句话介绍一下你自己。", 128],
            ["What is 12 minus 5? Give a short answer.", 96],
            ["Write three bullet points about small language models.", 160],
        ],
        cache_examples=False,
        concurrency_limit=1,
    )
    gr.Markdown(
        "多模态权重已发布在 [tinyLLM-0.51B-VLM]"
        "(https://huggingface.co/chris0809/tinyLLM-0.51B-VLM)。"
        "[ARC GRPO](https://huggingface.co/chris0809/tinyLLM-0.51B-ARC-GRPO) 和 "
        "[IFEval OPD](https://huggingface.co/chris0809/tinyLLM-0.51B-IFEval-OPD) LoRA 也可单独下载。"
        "完整图片演示建议使用 GitHub 中的本地 Demo；这个免费 Space 主要用于快速验证文本基座。"
    )


if __name__ == "__main__":
    demo.launch()
