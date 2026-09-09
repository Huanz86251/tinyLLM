# Model release layout

Source code lives on GitHub. The canonical model repositories are:

- <https://huggingface.co/chris0809/tinyLLM-0.51B-SFT>
- <https://huggingface.co/chris0809/tinyLLM-0.51B-ARC-GRPO>
- <https://huggingface.co/chris0809/tinyLLM-0.51B-IFEval-OPD>
- <https://huggingface.co/chris0809/tinyLLM-0.51B-VLM>

The local demo expects the following layout after downloading the files:

```text
models/
├── text/sft_base_50000/
├── adapters/arc_best/
├── adapters/opd_ifeval/
├── vision/InternViT-300M-448px-V2_5/
└── vision_language/vlm_continual_v5_best_step18000/
    ├── base/
    └── lora/
```

Prepared release artifacts (updated 2026-09-09):

| Archive | Bytes | SHA-256 | Purpose |
| --- | ---: | --- | --- |
| `tinyllm-sft-base-0.51b.zip` | 2,047,652,348 | `4c733698ce207acab31dffbf5af1da094dc7257a892657d20f5fcb68ce5879b7` | Default 0.510B text model and tokenizer |
| `tinyllm-arc-grpo-best.zip` | 58,446,630 | `1549521ebdcbc4b16c287e54e5bd45e0d80bdd4f827b2c6af2ffb7ef86169001` | ARC GRPO best adapter, step 200 |
| `tinyllm-gsm8k-opd-experimental.zip` | 4,572,327 | `08bfa0b69e2c758c08500e1c6418d8f7e35ac1c389dc13f049ddc7663ac7c750` | Archived GSM8K OPD control; not used for the public GSM8K result |
| `tinyllm-ifeval-opd-epoch3.zip` | 58,450,184 | `dbc1fba242057ef27adf409ab0c964c44d31b24d26b62543e36cbf7424d3976b` | IFEval OPD epoch 3 adapter and evaluation metadata |
| `tinyllm-internvit-300m.zip` | 608,104,056 | `0b608a7906fc54ed3e5131afd9d360f3b154ddc6743b12f7d28fc7bd08d4cd36` | InternViT tower required by the VLM demo |
| `tinyllm-vlm-step18000.zip` | 2,648,110,041 | `1e5490b68c79bb4c7c94476457544cf32fef348fd866faacb292a86cacde8f64` | VLM step 18000 base delta, bridge, Q-Former and LoRA |

Publish a SHA-256 checksum next to every archive. Hugging Face or ModelScope
should be the canonical versioned repository; a Baidu Netdisk share can mirror
the same immutable archives for users in mainland China.

Baidu Netdisk mirror:

- URL: <https://pan.baidu.com/s/1rpM9mtMbtyq01GluLk72NA?pwd=tjda>
- Extraction code: `tjda`

`release_manifest.json` is shipped next to the archives and mirrored as
[`MODEL_FILES.json`](MODEL_FILES.json). Hugging Face exports include the custom
`PreTrainedModel` implementation and `auto_map` metadata, so the base model can
be loaded with `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`.
See [`HUGGINGFACE.md`](HUGGINGFACE.md) for adapter and VLM examples.
