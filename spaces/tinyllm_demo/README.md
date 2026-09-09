---
title: tinyLLM 0.51B Demo
emoji: 🧩
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: other
models:
- chris0809/tinyLLM-0.51B-SFT
- chris0809/tinyLLM-0.51B-ARC-GRPO
- chris0809/tinyLLM-0.51B-IFEval-OPD
- chris0809/tinyLLM-0.51B-VLM
---

# tinyLLM 0.51B Demo

This free ZeroGPU Space loads the public tinyLLM 0.51B SFT checkpoint and offers
a small streaming chat demo.

The first start is slower while the 0.51B checkpoint is downloaded. The full
multimodal demo remains available in the GitHub repository; this small hosted
Space focuses on the stable text base. ARC GRPO and IFEval OPD adapters are
linked from the app and can be loaded locally with `load_lora_pretrained()`.
