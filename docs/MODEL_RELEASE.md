# Model release layout

Keep source code in GitHub and publish weights in a separate model repository.
The demo expects the following layout after downloading the files:

```text
models/
├── text/sft_base_50000/
├── adapters/arc_best/
├── adapters/opd_ifeval/                 # optional; original cloud artifact is unavailable
├── vision/InternViT-300M-448px-V2_5/
└── vision_language/vlm_continual_v5_best_step18000/
    ├── base/
    └── lora/
```

Prepared release artifacts (2026-09-08):

| Archive | Bytes | SHA-256 | Purpose |
| --- | ---: | --- | --- |
| `tinyllm-sft-base-0.51b.zip` | 2,047,652,348 | `4c733698ce207acab31dffbf5af1da094dc7257a892657d20f5fcb68ce5879b7` | Default 0.510B text model and tokenizer |
| `tinyllm-arc-grpo-best.zip` | 58,446,630 | `1549521ebdcbc4b16c287e54e5bd45e0d80bdd4f827b2c6af2ffb7ef86169001` | ARC GRPO best adapter, step 200 |
| `tinyllm-gsm8k-opd-experimental.zip` | 4,572,327 | `08bfa0b69e2c758c08500e1c6418d8f7e35ac1c389dc13f049ddc7663ac7c750` | Small local GSM8K OPD experiment; not the historical 51.40% adapter |
| `tinyllm-internvit-300m.zip` | 608,104,056 | `0b608a7906fc54ed3e5131afd9d360f3b154ddc6743b12f7d28fc7bd08d4cd36` | InternViT tower required by the VLM demo |
| `tinyllm-vlm-step18000.zip` | 2,648,110,041 | `1e5490b68c79bb4c7c94476457544cf32fef348fd866faacb292a86cacde8f64` | VLM step 18000 base delta, bridge, Q-Former and LoRA |

Publish a SHA-256 checksum next to every archive. Hugging Face or ModelScope
should be the canonical versioned repository; a Baidu Netdisk share can mirror
the same immutable archives for users in mainland China.

`release_manifest.json` is shipped next to the archives and mirrored as
[`MODEL_FILES.json`](MODEL_FILES.json). The current architecture uses the
project's custom loader. Hosting these files in a model repository is supported,
but direct Transformers `AutoModel.from_pretrained` loading requires a future
`PreTrainedModel` wrapper and `auto_map` metadata.
