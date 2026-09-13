# Hugging Face usage

Hosted demo: <https://huggingface.co/spaces/chris0809/tinyLLM-Demo>. It runs the
SFT text checkpoint on free ZeroGPU. The local web app remains the full demo for
five-view images and multi-adapter switching.

## Text model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "chris0809/tinyLLM-0.51B-SFT"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype="auto",
    device_map="auto",
)
```

### Reasoning mode

The SFT checkpoint uses ordinary chat mode by default. For mathematics or other
multi-step tasks, pass `enable_thinking=True` to the bundled chat template. It
automatically inserts the same reasoning protocol used during training and
evaluation.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "chris0809/tinyLLM-0.51B-SFT"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype="auto",
    device_map="auto",
).eval()

messages = [{
    "role": "user",
    "content": (
        "A box contains 12 red balls and 8 blue balls. How many balls are there?\n"
        "Put your final answer in LaTeX boxed form like $\\boxed{answer}$."
    ),
}]
inputs = tokenizer.apply_chat_template(
    messages,
    enable_thinking=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)
with torch.inference_mode():
    output = model.generate(
        inputs,
        max_new_tokens=640,
        do_sample=False,
        repetition_penalty=1.08,
        no_repeat_ngram_size=16,
    )
print(tokenizer.decode(output[0, inputs.shape[-1]:], skip_special_tokens=True))
```

Omit `enable_thinking` or set it to `False` for ordinary chat. Internally, the
template inserts the trained `<|thought_start|> ... <|thought_end|>` protocol.
Those boundaries are encoded as multiple existing tokenizer pieces, so do not
register new special tokens or resize the embedding matrix. The boxed-answer
suffix is useful for GSM8K/ARC-style evaluation and is optional in normal chat.

tinyLLM is a custom architecture, so `trust_remote_code=True` is required. The
repository contains its configuration and model implementation; callers do not
need to clone the GitHub project first.

## Task LoRA

ARC and IFEval use the project's native multi-adapter LoRA implementation. The
files use `safetensors`, but they are not PEFT adapters.

```python
model.load_lora_pretrained("chris0809/tinyLLM-0.51B-ARC-GRPO")

# Switch later without reloading the 0.51B base.
model.load_lora_pretrained(
    "chris0809/tinyLLM-0.51B-IFEval-OPD",
    activate=False,
)
model.activate_single_lora("opd_ifeval_rank64")
```

The two public task adapters correspond to the reported results:

| Adapter | Evaluation | Base | Adapter |
| --- | --- | ---: | ---: |
| ARC GRPO | ARC-Easy accuracy | 30.09% | 35.98% |
| IFEval OPD | strict instruction accuracy | 23.74% | 25.18% |

## Vision-language model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "chris0809/tinyLLM-0.51B-VLM"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype="auto",
)
model.load_lora_pretrained(model_id, subfolder="adapter")
```

The VLM repository contains the language model, Q-Former, projector and visual
LoRA. The frozen image encoder is
`OpenGVLab/InternViT-300M-448px-V2_5`, loaded separately so the same 608 MB
tower is not duplicated in every checkpoint.

The current VLM interface accepts precomputed `vision_feats`, `vision_mask`,
`global_pos` and `global_off`. The complete five-view image preprocessing and
chat path are implemented in `inference/vision.py` and `inference/runtime.py`.
It is compatible with Transformers model loading, but is not presented as a
generic `AutoProcessor`/`pipeline` because the project uses its own five-view
image protocol.

## Exporting another checkpoint

```powershell
python tools/export_huggingface.py model `
  --checkpoint E:\path\to\checkpoint `
  --tokenizer E:\path\to\tokenizer `
  --output E:\path\to\hub-folder `
  --repo-id owner/repository

python tools/export_huggingface.py adapter `
  --weights E:\path\to\adapter.pt `
  --output E:\path\to\adapter-folder `
  --repo-id owner/adapter-repository `
  --name demo --rank 64 --alpha 64 --target all `
  --base-model owner/base-repository
```

The exporter copies the tokenizer, adds the Transformers `auto_map`, bundles
the custom Python implementation and converts native LoRA tensors to
`adapter_model.safetensors`.
