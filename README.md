# tinyLLM

tinyLLM 是一套从零搭建并训练的 0.51B 中英双语小模型。项目从 decoder-only Transformer 开始，完整做了预训练、继续预训练、SFT、GRPO、On-Policy Distillation（OPD）和视觉语言训练，最后做成了一个可以在 Windows 16GB 显卡上运行的网页 Demo。

最初只是想把一个小模型真正训出来，后来陆续补了数学、代码、长上下文、强化学习和视觉。中间踩过不少坑，比如小模型循环解码、教师和学生思维格式不一致、视觉数据偏英文和 OCR、混合 loss 被通用回放压住等。这些问题最后都落到了代码里，而不是只留一个训练结果。

> **在线体验：[Hugging Face Demo](https://huggingface.co/spaces/chris0809/tinyLLM-Demo)**<br>
> 免费 ZeroGPU 第一次打开可能需要排队，适合快速体验 SFT 文本模型。图片输入和多 LoRA 切换请使用本地 Demo。

## 结果

| 任务 | 方法 | 基线 | 最终结果 | 提升 |
| --- | --- | ---: | ---: | ---: |
| GSM8K | 通用SFT后，1,319 题 greedy | — | **50.64%** | — |
| ARC-Easy | GRPO | 30.09% | **35.98%** | **+5.89 pp** |
| Google IFEval | MiniCPM3-4B OPD | 23.74% | **25.18%** | **+1.44 pp** |

IFEval 这里写的是 strict instruction accuracy。其余三项也有提升：strict prompt 14.79% → 15.71%，loose prompt 17.19% → 17.74%，loose instruction 26.86% → 27.70%。评测使用 Google IFEval 的 541 个 prompt、834 条 instruction。

GSM8K 展示的是 SFT 基座成绩：668 / 1319。数学强化学习版本也跑过，但结果一般，没有保留这版权重。

## 模型结构

文本模型是自己写的 `nn.Module + GenerationMixin`，不是直接套一个现成的 Transformers 模型。

| 配置 | 数值 |
| --- | --- |
| 参数量 | 约 0.510B |
| Transformer 层数 | 24 |
| hidden size | 1280 |
| attention | 20 query heads / 4 KV heads |
| MLP 宽度 | 3.5× → 4.0× → 4.5×，最后三层 5.0× |
| attention 稳定性 | QK RMSNorm + 可学习 attention temperature |
| 训练上下文 | 2K、8K、16K 分阶段混合 |
| 训练精度 | BF16 |
| 视觉塔 | InternViT-300M-448px-V2_5 |
| 视觉连接 | projector + Q-Former + language LoRA |

LoRA 按任务分别保存，本地 Demo 可以在普通 SFT、ARC、OPD 和 VLM 之间切换。

## 训练中几个特殊设计

下面不重复 Transformer、LoRA 或 GRPO 的通用原理，只记录这个项目在训练中遇到问题后留下来的处理。

### 0.51B 的参数没有平均铺在每一层

模型只有 24 层，平均加宽会很快吃完参数预算。前 8 层的 MLP ratio 设为 3.5，中间 8 层设为 4.0，第 17～21 层设为 4.5，最后三层再提高到 5.0。这不是照搬某篇论文的固定配方，而是沿用了 [MobileLLM](https://arxiv.org/abs/2402.14905) 在亚十亿参数模型上“深而窄、共享 embedding、使用 GQA”的预算思路，再把后几层的 FFN 留得更宽一些。tinyLLM 主要做中英双语、数学和少量代码，不准备覆盖完整多语言和大型代码模型的能力，因此更愿意把有限参数放在后段的语义组合与回答生成上。

Attention 使用 20 个 query head、4 个 KV head，在 16GB 显卡推理时可以明显压低 KV cache；Q、K 投影后再做 RMSNorm，并给 attention temperature 留一个可学习标量。每层 attention 和 MLP 的残差分支还有限制在固定范围内的独立 gate，避免某个分支在训练早期突然放大。

实现见 [`model/config.py`](model/config.py) 和 [`model/model.py`](model/model.py)。

### 为什么教师选 MiniCPM3-4B，而没有沿用 Qwen tokenizer

该项目最早开始于 2025 年，早期实验实际以 Qwen2.5 为参照，后来也没有为了 Qwen3 重新设计底层词表和模型结构。（放到 2026 年看，这套架构确实有一丢丢的落后...）因为想尝试蒸馏，早期也试过 Qwen2.5-1.5B；Qwen 的模型规格和生态更丰富，但其 tokenizer 对 0.51B 自研模型有点重。tinyLLM hidden size 是 1280，MiniCPM3 tokenizer 有 73,448 个 token，共享输入、输出 embedding 后约占 **94.0M** 参数，也就是整模的 **18.4%**。如果换成 Qwen2.5 的 151,936 词表，同一块会变成约 **194.5M**；模型总量会从 510M 涨到约 611M，其中词表一项就占 **31.8%**。即使输入输出权重已经绑定，仍会白白多出约 100.5M 参数，而当时更想把这多余的参数用来增加模型的深度和宽度。

最终选择 MiniCPM3-4B，一方面是它的中英词表更适合当前预算，另一方面是学生从一开始就使用同一套 tokenizer，教师保存的 top-k token id 可以直接和学生 logits 对齐，不需要再做跨词表映射。4B 的教师也能在单卡上通过离线 logits、量化生成和分阶段加载完成工作，能力差距够用，工程成本又没有大到失控。

### 蒸馏没有用教师答案替换原始标签

一开始如果让 4B 教师和 0.51B 学生在线同时跑，显存和吞吐都不合适。实际做法是提前保存 MiniCPM3-4B 每个位置的 top-16 logits，并按学生 `input_ids` 的 hash 写入分片 LMDB；训练时只读取命中的位置。学生始终保留原始 next-token CE，再额外加入温度 1.5、权重 0.2 的 soft-target loss。也就是说，gold token 一直参与训练，教师 top-16 只是补充分布信息，并没有把整道题的最终答案提前塞给学生。代码里还保留了 shortlist 版本：即使 gold 不在教师 top-16，也会被强制放到候选第 0 列。

这样处理还有一个实际好处：教师 logits 只生成一次，后面的预训练可以继续用 DDP，不必常驻第二个模型。对应代码是 [`train/pretrain/build_teacher_logits.py`](train/pretrain/build_teacher_logits.py)、[`train/pretrain/kd_reader.py`](train/pretrain/kd_reader.py) 和 [`train/pretrain/base.py`](train/pretrain/base.py)。

### 数学 token 单独加权，但没有把所有公式符号都抬高

数学数据混入大规模自然语言后，数字和运算符在总 token 中很稀疏，100个 token 里面真正决定计算结果的且模型出错的只有几个token，普通 token mean 容易把这部分梯度盖住。CPT 2 和 SFT 因此给数字以及单字符 `+ - * / = ^ % × ÷` 乘 1.2 的 loss 权重。这个 mask 会同时检查 tokenizer 原始 token 和实际 decode 结果，兼容半角、全角和 Unicode 数字。

这里特意没有给所有“像数学”的字符加权：括号只保留 1.05 的轻权重，逗号、句号不加权，`////`、`***`、带字母或 markup 的 token 也会过滤。数学增强只在 reasoning batch 打开，语言回放和验证集关闭，避免为了数学把正常中文标点和文本流畅度一起拉偏。实现见 [`train/pretrain/math_mask.py`](train/pretrain/math_mask.py)。

### 推理和语言回放使用相反的分层学习率

CPT 2 跑到后面时，数学数据的难点不只是 loss 高低。很多解题文本已经很模板化，真正决定答案的信号集中在数字、运算符和少量关键推理位置上；如果把 reasoning 和普通语言直接拼成一个 batch 再取平均，这些稀疏信号很容易被大量自然语言 token 盖住。只把数学 loss 整体放大也不太稳，碰到难题或长推导时梯度会突然跳高。

最后把一个短周期拆成了两步：先累计 2 个 reasoning micro-batch，完成一次 AdamW update；清空梯度后，再用 1 个 language micro-batch 单独更新。这样数学梯度每个周期都有一次完整的更新方向，不会先在混合 batch 里被平均掉；语言步则继续负责双语表达和文本流畅度。

Reasoning 更新的前、中、后层学习率比例是 `0.5 / 0.6 / 1.0`。这里的想法很直接：数学继续训练不需要把已经学到的中英文表示从底层重写一遍，更多改动留给负责组合推理过程和组织答案的后层，也希望通过降低前中层的学习率降低前中层的遗忘。语言更新反过来使用 `1.0 / 0.6 / 0.5`，用回放稳住词法和双语表达，同时尽量少冲掉后层刚学到的解题模式。它是项目里的工程偏置，不把“某一层只负责某种能力”当成严格结论。两类更新仍共用同一个 AdamW 和它的动量状态，没有维护两套优化器。

每个 batch 的题目难度和有效 token 数差别很大，单步梯度范数抖得很厉害，所以控制器记录的是两类梯度范数的 EMA。每 10 个短周期再根据这个平滑值微调 language loss scale，让 reasoning 的有效梯度占比大致维持在 70%；EMA 在这里负责稳定配比判断，并不是用来抵消 AdamW 动量。代码还给未缩放的语言梯度加了一道 guard，避免语言步长期撞上 gradient clipping 上限。

旧 checkpoint 曾在索引 `4、11、19`，也就是第 5、12、20 个 block 中使用 SSM，希望降低长上下文的计算和缓存开销，顺带增强一些长文本能力。

**这里需要先说明：从理论复杂度上看，SSM 在处理长序列时应该比 Attention 更节省计算和缓存。下面描述的只是当时这套 Mamba2 实现、软件环境和训练配置下的实际结果，并不代表 Attention 在一般情况下比 SSM 更省显存。**

实际接入的是 Mamba2，但当时 fused kernel 快路径一直编译失败，只能退回普通 PyTorch 路径；加入 8K 训练数据后，**普通 PyTorch 路径出现了显存溢出，而在相同训练配置下换回 Attention，实际显存占用反而更低**，Mamba2 原本期待的效率优势也没有体现出来。再考虑到 hybrid 架构需要同时维护 SSM state 和 KV cache，以及后续 Windows 部署的成本……非常心痛，也知道它理论上不应该是这个结果，但它当时确实爆显存了，最后只能换回 Attention。

替换时没有直接解冻整个模型，而是先只训练三个新 Attention 分支、相关 norm/gate、final norm 和输出 bias 2,000 step。期间单独检查了新分支的梯度 L2 范数：**梯度持续非零，也没有出现明显的梯度爆炸或消失**。随后解冻全模型继续训练，loss 保持稳定。至少从训练信号来看，这三个新分支已经正常接入模型，并没有成为摆设。

这套调度在 [`train/pretrain/cpt_stage2.py`](train/pretrain/cpt_stage2.py) 里，8K 和 16K 样本还分别有 1.10 和 1.20 的短期 loss boost。

### ARC 的 GRPO 重点处理“没有训练信号”和“幸运答对”

最初只靠当前 batch 做 GRPO 时，0.51B 模型在 ARC 上经常 16 条候选全部答错，一组里没有正样本就没有可用的相对优势；偶尔找到一条正确路径，后面的更新又很容易把它冲掉。后来尝试加入 CE replay：ARC 每题生成 16 条候选，里面同时放 greedy、低温和高温采样。低温轨迹已经答对时，高温碰巧答对的样本会降权，避免模型把偶然探索当成稳定策略。只有低温、答案正确、格式完整且长度合适的轨迹才进入最多 256 条的 replay buffer；每 12 个数据 batch 插入一次小权重 CE 回放，用来反复巩固模型自己已经找到的稳定路径。

如果一组候选全部答错，或者 reward 完全没有差异，这道题不会继续计算昂贵的 reference logprob，也不会让空梯度推动 optimizer 和 scheduler。KL 系数则从 0 缓慢升高，到 300 update 才达到设定值。这里解决的是小模型常见的两个问题：大量题目没有正样本，以及高温采样偶然命中后把更新方向带偏。实现见 [`train/grpo/arc.py`](train/grpo/arc.py)。

### OPD 沿学生轨迹纠偏，再用教师完整轨迹补上正确走法

IFEval OPD 每次先让当前学生模型在线生成 10 条回答，无论回答是否通过 verifier，都会在学生实际生成的 token 路径上计算教师与学生之间的 JSD。这样教师面对的是学生真正会到达的状态，可以逐 token 调整学生在错误路径、犹豫位置和中间步骤上的概率分布；如果学生轨迹刚好通过 verifier，再额外加入权重为 `0.05` 的 self-CE，巩固它自己已经找到的成功路径。

但学生基线只有 20% 多，很多轨迹很早就已经走偏。只做 on-policy JSD 时，教师始终是在学生产生的 prefix 上给出下一步分布，学生不一定能够完整看到教师如何从头完成一条合格回答。因此每次更新又加入 3 条通过 verifier 的教师完整轨迹，在这些轨迹上同时计算 JSD 和 CE，让它们作为稳定的正确路径锚点。

剩下 3 条是普通中英文 SFT 回放，只计算 assistant CE。三部分虽然是 `10 + 3 + 3` 条样本，但会根据 loss EMA 调整 scale，使学生轨迹、教师轨迹和双语回放的实际贡献接近 `65% / 20% / 15%`。

教师和学生使用的思维边界格式并不完全相同。训练时会定位人工加入的完整边界 token span，只把这些边界位置从 JSD 中排除，正文和推理内容仍然正常反向传播。这样教师可以纠正学生的回答分布，又不会顺手把学生原有的思维边界格式一起改掉。实现见 [`train/opd/ifeval.py`](train/opd/ifeval.py) 和 [`configs/opd_ifeval_shared_65_20_15.json`](configs/opd_ifeval_shared_65_20_15.json)。

### 视觉训练把视觉样本和文本回放分成两次 forward

VLM 继续训练时，纯文本回放不能伪造一个空图片前缀，否则 Q-Former 和 bridge 也会收到没有意义的梯度。混合 batch 会按是否含图拆成两次 forward：视觉样本更新 language LoRA、Q-Former 和 projector，文本样本只经过语言模型，最后再按样本数合并两个 loss。

第一轮 VLM 用的是普通 token mean。一条很长的图片描述会比“图里有什么”这种短问答产生更多梯度，训练久了以后，模型逐渐偏向输出很长、但依据不足的描述；有时即使没有图片，也会带上这种视觉描述的语气。表格、OCR 和学术问答堆得太多后，0.51B 模型还容易先学会一套“看起来像答案”的句式，却没有真正读准图里的细节。

第二轮因此没有继续追复杂 OCR，而是把目标收回到日常物体、场景、中文短问答和围绕同一张图的多轮对话，同时把归约改成 75% sample mean + 25% token mean。这样每条样本先拥有更接近的基础权重，又保留一部分 token 级统计。实际试用时，回答长度和重复更容易控制，模型也较少为了拉长回答而补充图片中不存在的细节。

InternViT 特征没有一次性全部落盘。训练只预计算下一个 chunk 需要的五视图特征，随后卸载视觉塔再开始反向传播，到 chunk 边界轮换 LMDB；这样在单卡训练时能同时控制显存和缓存体积。实现见 [`train/vlm/continual.py`](train/vlm/continual.py)。

## 训练路线

```mermaid
flowchart LR
    A[Base pretraining<br/>23.8B] --> B[CPT 1<br/>2K / 8K]
    B --> C[CPT 2<br/>reasoning + 2K / 8K / 16K]
    C --> D[General SFT]
    D --> E[ARC GRPO]
    D --> F[IFEval OPD]
    D --> G[Initial VLM SFT]
    H[InternViT-300M] --> G
    G --> I[Chinese/English VLM continual SFT]
```

整个文本模型大约看过 38.7B token：初始预训练约 23.8B，两段继续预训练约 5.3B 和 9.6B。

### 1. 初始预训练

入口：[`scripts/pretrain.py`](scripts/pretrain.py)

初始预训练用 2048 上下文，per-device batch 3、gradient accumulation 3、4 卡 DDP，峰值学习率 `2.5e-4`，warmup 后 cosine decay。

数据以中文为主，再加入英文、代码和数学：

- `Mxode/Chinese-Instruct` 和清洗后的中文高质量文本；
- `opencsg/chinese-cosmopedia`；
- `ajibawa-2023/Children-Stories-Collection`；
- `nampdn-ai/tiny-codes`；
- `nvidia/OpenCodeInstruct`；
- `nvidia/OpenMathInstruct-2`。

这一阶段同时使用 MiniCPM3-4B 的离线 top-16 logits；gold CE、LMDB 对齐和蒸馏权重的处理见上面的“蒸馏没有用教师答案替换原始标签”。

### 2. 两段继续预训练

- [`scripts/train_cpt_stage1.py`](scripts/train_cpt_stage1.py)：第一段 2K / 8K 混合训练；
- [`scripts/train_cpt_stage2.py`](scripts/train_cpt_stage2.py)：第二段推理、语言和长上下文混合训练。

第一段从 `checkpoint-323180` 开始，每 20 个 2K optimizer step 插入 1 个 8K step，短序列继续使用教师 logits，学习率降到 `1.8e-4`，最后训练到 `checkpoint-396000`。

第二段把短序列拆成两份推理数据和一份语言数据。每 20 个短周期加入一次 8K，每 5 个长序列周期再加入一次 16K；学习率降到 `6e-5`，前、中、后层使用不同的学习率缩放，最后训练到约 `checkpoint-651195`。

数学与推理部分主要用了 OpenMathInstruct-2、NuminaMath-CoT、MetaMathQA、MathInstruct、Orca Math、OpenR1-Math、AceMath、Natural Reasoning，以及多组中文数学和 DeepSeek-R1 蒸馏数据。代码和工具部分用了 OpenCodeInstruct、tiny-codes、When2Call。语言和长文本部分用了 Chinese Cosmopedia、Fineweb-Edu-Chinese、FineWeb-Edu、FineMath、OpenStax、Khan Academy、GovReport、BookSum 和 Research-14K。

更完整的数据对应关系放在 [`docs/TRAINING_DATA.md`](docs/TRAINING_DATA.md)。

### 3. 通用 SFT

入口：[`scripts/train_sft.py`](scripts/train_sft.py)

SFT 从 CPT 最终模型继续训练。短样本最长 2048，长样本最长 16K，每 50 个短序列 step 插入一个长序列 step。训练使用 token-budget 动态 batch、BF16、峰值学习率 `1.5e-5`、warmup 500 step 和 cosine decay。

主要数据包括：

- 数学：OpenMathInstruct-2、NuminaMath-CoT、MetaMathQA、MathInstruct、Orca Math、Natural Reasoning、OpenO1-SFT、OpenThoughts；
- 通用中英文：UltraChat 200K、Helpful Instructions、Dolly 15K、中文 SmolTalk、BiST 翻译；
- 代码与工具：Ling-Coder-SFT、When2Call、Hermes reasoning tool use；
- 长上下文：LongAlign-10k。

早期 SFT 还试过混入不到 1% 的专门拒答样本。比例虽然很小，但“我不知道”“无法回答”这类低变化句式对 0.51B 模型成了很容易学会的捷径，普通问题也开始出现过度拒答。最终训练去掉了这批专门构造的拒答数据，只保留正常指令数据里自然出现的边界回答。这个实验也说明，小模型的数据占比很低，不等于行为影响一定很小。

后续实验都从 `sft_base_50000` 开始，ARC、GSM8K、IFEval 和 VLM 共用同一个文本底座。

### 4. ARC-Easy GRPO

入口：[`scripts/train_grpo_arc.py`](scripts/train_grpo_arc.py)

ARC 用规则直接验证最终选项，再配合答案格式奖励和 KL 约束更新 LoRA。完整评测从 30.09% 提高到 35.98%，最佳点在 step 200。

[`scripts/train_grpo_gsm8k.py`](scripts/train_grpo_gsm8k.py) 是 GSM8K 数值奖励版本，完整训练流程还在。因为这一轮结果不理想，Demo 最后还是用了 50.64% 的 SFT 版本。

### 5. IFEval On-Policy Distillation

入口：

- [`scripts/prepare_opd_data.py`](scripts/prepare_opd_data.py)
- [`scripts/build_opd_teacher_pool.py`](scripts/build_opd_teacher_pool.py)
- [`scripts/train_opd_ifeval.py`](scripts/train_opd_ifeval.py)
- [`scripts/evaluate_opd_ifeval.py`](scripts/evaluate_opd_ifeval.py)

任务数据是 `allenai/RLVR-IFeval`。14,973 条原始记录清洗后保留 14,690 条，其中 14,210 条训练、480 条开发集，覆盖 24 类指令约束。

教师使用 MiniCPM3-4B。教师先生成候选，只有通过 IFEval verifier、没有循环、没有撞到输出上限的回答才进入教师池。最后得到 6,583 条合格教师轨迹。

训练使用上面介绍的 65% 学生轨迹、20% 合格教师轨迹和 15% 中英文回放。中文回放来自 `m-a-p/COIG-CQIA`，英文来自 `HuggingFaceTB/smoltalk2` 的 everyday conversations。LoRA 为 rank 64 / alpha 64，覆盖 attention 和 MLP，跳过最前四层；峰值学习率 `1.2e-6`，warmup 60 step，共三轮。

最终用 Google IFEval 做全量评测，strict instruction accuracy 从 23.74% 提高到 25.18%。

### 6. VLM：先做视觉对齐，再补中文日常看图

初始入口：[`scripts/train_vlm_initial.py`](scripts/train_vlm_initial.py)

第一轮视觉 SFT 打包了 845,755 条样本和约 243.8M 文本 token。数据主要来自 LLaVA-OneVision 的 DocVQA、ChartQA、DVQA、FigureQA、GeoQA、ScienceQA 等子集，也加入了 LLaVA-CoT-100k 和 LLaVA instruct mix。

这批数据把视觉前缀和语言模型接通了，但它太偏英文、图表、OCR 和学术题。实际演示时英文回答还可以，中文日常看图和普通物体描述容易幻觉。所以第二轮没有继续堆表格 OCR，而是重新补中文短问答、图片描述和多轮视觉对话。

| 数据 | 训练条数 | 用途 |
| --- | ---: | --- |
| FM-IQA | 81,148 | 中文短问答、物体和场景识别 |
| M3IT COCO-CN | 17,865 | 中文图片描述 |
| M3IT Flickr8k-CN | 5,789 | 中文日常场景描述 |
| Pangea 中文多轮 | 44,019 | 围绕同一张图连续对话 |
| VQAv2 | 18,998 | 英文短问答 |
| CogVLM detail/multi/single，中英 | 33,929 | 细节描述、单轮和多轮补充 |

清洗后共有 **201,748 条视觉训练样本和 4,730 条视觉验证样本**。每条只保留一张源图，处理成四张有重叠的局部图和一张全局缩略图，统一为 448×448。上游的各种图片占位符会先删掉，只在第一轮 user 消息开头放一个规范 `<img>`。超过 2048 token、多张源图、角色顺序错误或图片路径失效的样本会被过滤。

继续训练沿用原来的视觉 LoRA，rank / alpha 调到 128 / 128。InternViT 保持冻结，训练 language LoRA、Q-Former 和 projector：

| 模块 | 学习率 |
| --- | ---: |
| Language LoRA | `8e-6` |
| Q-Former | `1.2e-5` |
| Projector / bridge | `1.5e-5` |

视觉数据之外加入 20% 纯文本回放，中文和英文约 2:1；具体的分路 forward 在上面已经说明。训练两轮，3% warmup、cosine decay、weight decay 0.01、gradient clip 1.0。最终选择 step 18000，visual eval loss `2.8359`，text replay loss `3.1927`。

```powershell
python scripts/prepare_vlm_data.py --start-preparation
python scripts/audit_vlm_data.py
python scripts/train_vlm_continual.py --start-training
```

## 本地 Demo

按 [`docs/MODEL_RELEASE.md`](docs/MODEL_RELEASE.md) 放好权重，然后复制 `configs/models.example.json` 为 `configs/models.json`，填写本机路径。Windows 可以直接双击 `start_demo.cmd`，也可以运行：

```powershell
python chat_server.py --port 8501
```

打开 `http://127.0.0.1:8501`。网页支持流式输出、KaTeX 公式渲染、图片上传和模型切换。默认解码加入 repetition penalty、no-repeat n-gram、循环片段检测和提前停止，主要用来防止 0.5B 模型在演示时反复生成同一句话。

上传新图片时会开启新的图片上下文，不把上一张图一起送进模型。这与训练时的单图格式一致，也能避免旧图片污染新回答。

页面顶部的 [Hugging Face 在线 Demo](https://huggingface.co/spaces/chris0809/tinyLLM-Demo) 使用免费 ZeroGPU，主要展示 SFT 文本基座。打开后等状态从 Sleeping/Starting 变成 Running 就可以提问；冷启动和排队慢一点属于正常情况。未登录用户每天约 2 分钟 GPU 时间，登录后的免费账户每天约 5 分钟，单次生成限制为 256 token，更适合快速试两三个问题。完整的图片和多 LoRA 切换仍以本地 Demo 为准。

## 权重下载

Hugging Face 保存可版本化加载的模型和 LoRA，百度网盘提供国内镜像：

- [tinyLLM-0.51B-SFT](https://huggingface.co/chris0809/tinyLLM-0.51B-SFT)：文本基座；
- [tinyLLM-0.51B-ARC-GRPO](https://huggingface.co/chris0809/tinyLLM-0.51B-ARC-GRPO)：ARC-Easy GRPO LoRA；
- [tinyLLM-0.51B-IFEval-OPD](https://huggingface.co/chris0809/tinyLLM-0.51B-IFEval-OPD)：IFEval OPD LoRA；
- [tinyLLM-0.51B-VLM](https://huggingface.co/chris0809/tinyLLM-0.51B-VLM)：语言模型、Q-Former、projector 和视觉 LoRA。冻结的视觉塔直接引用 [InternViT-300M-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5)。

先安装 Hugging Face 命令行工具：

```powershell
pip install -U huggingface_hub
```

只运行普通文本模型，下载 SFT 基座即可：

```powershell
hf download chris0809/tinyLLM-0.51B-SFT `
  --local-dir models\text\sft_base_50000
```

需要 ARC 和 IFEval 模式时，再下载两套 LoRA。它们共用上面的 SFT 基座，不会重复占用 2GB：

```powershell
hf download chris0809/tinyLLM-0.51B-ARC-GRPO `
  --local-dir models\adapters\arc_best

hf download chris0809/tinyLLM-0.51B-IFEval-OPD `
  --local-dir models\adapters\opd_ifeval
```

运行多模态版本需要下载 VLM checkpoint 和 InternViT 视觉塔：

```powershell
hf download chris0809/tinyLLM-0.51B-VLM `
  --local-dir models\vision_language\tinyLLM-0.51B-VLM

hf download OpenGVLab/InternViT-300M-448px-V2_5 `
  --local-dir models\vision\InternViT-300M-448px-V2_5
```

全部下载大约占 5.4GB。然后使用准备好的 Hugging Face 路径配置启动网页：

```powershell
Copy-Item configs\models.huggingface.example.json configs\models.json
python chat_server.py --port 8501
```

浏览器打开 `http://127.0.0.1:8501`，右上角可以切换 SFT、ARC、IFEval 和 VLM。只想在 Python 中加载模型时，可以使用下面的方式。

文本基座已经注册到 Transformers 的 `AutoModelForCausalLM`：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "chris0809/tinyLLM-0.51B-SFT"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype="auto",
)

# 可选：热加载某一个任务 LoRA
model.load_lora_pretrained("chris0809/tinyLLM-0.51B-ARC-GRPO")
```

完整的导出、LoRA 切换和 VLM 加载方法见 [`docs/HUGGINGFACE.md`](docs/HUGGINGFACE.md)。

国内镜像：

- [下载 tinyLLM-release-20260908](https://pan.baidu.com/s/1rpM9mtMbtyq01GluLk72NA?pwd=tjda)
- 提取码：`tjda`

其中包含 SFT 基座、ARC GRPO LoRA、GSM8K OPD 实验 LoRA、VLM step 18000、InternViT-300M 和发布清单。文件大小、SHA-256 和放置目录见 [`docs/MODEL_RELEASE.md`](docs/MODEL_RELEASE.md)。

## 目录

```text
configs/          训练、模型和解码配置
data_preprocess/  文本数据构建、视觉清洗与五视图构建
eval/             ARC、GSM8K 和视觉评测
inference/        文本、多 LoRA、视觉推理与解码保护
model/            tinyLLM、LoRA、Q-Former 和视觉连接层
scripts/          每个训练阶段的入口
third_party/      固定版本的 IFEval verifier
tools/            OPD 教师池、数据审计与权重导出
train/            预训练、CPT、SFT、GRPO、OPD 和 VLM 核心代码
ui/               本地网页界面
```

两段 CPT、KD LMDB、文本 packer、第一轮 VLM SFT 和视觉特征缓存代码都已经整理到 `train/`。临时启动器、调试探针和机器恢复脚本没有放进公开仓库，目录里只保留训练和复现真正会用到的部分。

## 环境

推荐 Python 3.10、PyTorch 2.x 和支持 BF16 的 NVIDIA GPU。Windows 验证环境见 [`docs/windows_verified.json`](docs/windows_verified.json)。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m compileall -q .
python -m unittest tests.test_decoding_safety tests.test_grpo_reward_regression -v
```

0.51B 的容量更适合短数学题、格式约束、常见物体和场景描述。密集表格 OCR、很长的复杂推理和多图联合理解不是这版模型的重点，Demo 主要展示从训练、对齐到本地部署的完整流程。

VLM 版本还有一点小模型常见的能力遗忘。20% 的中英文文本回放可以保住日常聊天，但没有完全保住代码这类低频能力：例如 SFT 基座可以直接写出 Python 的二叉树前序遍历，VLM 版本偶尔只会解释思路，代码不够完整。受训练时间和预算限制，第二轮视觉训练使用了相对积极的学习率，也没有再补一段低学习率的纯文本巩固。后续如果继续训练，会增加代码和推理回放，并在视觉收敛后加一轮更低学习率的短阶段。
