# 训练数据清单

## 初始预训练

初始预训练一共使用约 23.8B token，打包为 11,635,935 个 2048-token block。

| 数据集 | 内容 |
| --- | --- |
| Mxode/Chinese-Instruct | 中文指令与通用文本 |
| 本地 pretrain_hq | 清洗后的中文高质量文本 |
| opencsg/chinese-cosmopedia | 中文教育与知识文本 |
| ajibawa-2023/Children-Stories-Collection | 英文故事 |
| nampdn-ai/tiny-codes | 短代码 |
| nvidia/OpenCodeInstruct | 代码指令与生成过程 |
| nvidia/OpenMathInstruct-2 | 数学指令与解题过程 |

## 继续预训练

第一段 CPT 使用 2K / 8K 混合序列，训练约 5.3B token。第二段加入推理池、语言锚定池、8K 与 16K 长文本，训练约 9.6B token。

### 数学和推理

- Mxode/Math-Chinese-DeepSeek-R1-10K
- supinyu/goat-chinese
- Jackrong/Chinese-Qwen3-235B-Thinking-2507-Distill-100k
- Congliu/Chinese-DeepSeek-R1-Distill-data-110k
- AI-MO/NuminaMath-CoT
- meta-math/MetaMathQA
- TIGER-Lab/MathInstruct 与 MATH-plus
- allenai/math_qa
- microsoft/orca-math-word-problems-200k
- hkust-nlp/dart-math-hard、dart-math-uniform、dart-math-pool
- MathLLMs/MathCodeInstruct
- nvidia/AceMath-Instruct-Training-Data
- open-r1/OpenR1-Math-220k
- facebook/natural_reasoning
- fnlp/Ultra-Innerthought

### 代码、工具和通用指令

- nvidia/OpenCodeInstruct
- nampdn-ai/tiny-codes
- nvidia/When2Call 与函数调用数据
- Open-Orca/OpenOrca
- Mxode/Chinese-Instruct
- FradSer/DeepSeek-R1-Distilled-Translate-en-zh_CN-39k
- 中文 SmolTalk

### 语言和长上下文

- opencsg/chinese-cosmopedia
- HuggingFaceTB/cosmopedia：khanacademy、openstax、wikihow、automathtext、web_samples_v1
- opencsg/Fineweb-Edu-Chinese-V2.1
- HuggingFaceFW/fineweb-edu
- HuggingFaceTB/finemath
- ccdv/govreport-summarization
- WestlakeNLP/Research-14K
- ubaada/booksum-complete-cleaned

## 通用 SFT

| 方向 | 主要数据集 |
| --- | --- |
| 数学和推理 | OpenMathInstruct-2、NuminaMath-CoT、MetaMathQA、MathInstruct、Orca Math Word Problems、Natural Reasoning、Chinese DeepSeek-R1 Distill、OpenO1-SFT、OpenThoughts-114k、Problemathic |
| 通用中英文 | UltraChat 200K、Helpful Instructions、Databricks Dolly 15K、中文 SmolTalk、Mxode/BiST |
| 代码和工具 | Ling-Coder-SFT、When2Call、Hermes reasoning tool use |
| 长上下文 | LongAlign-10k |

## IFEval OPD

| 部分 | 数据 |
| --- | --- |
| 任务训练集 | allenai/RLVR-IFeval：14,210 train + 480 dev |
| 教师模型 | openbmb/MiniCPM3-4B |
| 合格教师池 | 6,583 条 verifier 通过轨迹 |
| 中文回放 | m-a-p/COIG-CQIA |
| 英文回放 | HuggingFaceTB/smoltalk2 everyday conversations no-think |
| loss 配比 | student on-policy 65% / teacher 20% / replay 15% |

## 第一轮 VLM SFT

第一轮视觉训练共有 845,755 条样本和约 243.8M 文本 token。

| 子集 | 行数 |
| --- | ---: |
| LLaVA-OneVision tqa | 106,375 |
| Xkev/LLaVA-CoT-100k | 98,250 |
| LLaVA-OneVision dvqa | 94,998 |
| geo170k_qa | 90,000 |
| FigureQA | 70,000 |
| llava_instruct | 54,999 |
| mapqa | 50,000 |
| llava-instruct-mix-vsft | 30,000 |
| GeoQA+ | 26,810 |
| IconQA | 25,000 |
| docvqa_train | 25,000 |
| tabmwp | 24,999 |
| Geometry3K | 20,000 |
| unigeo | 18,819 |
| ChartQA | 18,202 |
| ScienceQA | 21,140 |
| vistext | 15,447 |
| textocr_gpt4v | 15,000 |
| 其他 PlotQA、AI2D、InterGPS、GEOS 等 | 40,714 |

这一版数据整体偏英文、图表、OCR 和学术问答，其中约 103,920 条带 CoT。

## VLM continual SFT v5

| 数据集 | train | eval | 作用 |
| --- | ---: | ---: | --- |
| FM-IQA | 81,148 | 990 | 中文短 VQA |
| M3IT COCO-CN | 17,865 | 476 | 中文图片描述 |
| M3IT Flickr8k-CN | 5,789 | 211 | 中文日常场景描述 |
| Pangea zh multi-turn | 44,019 | 981 | 中文多轮视觉对话 |
| VQAv2 | 18,998 | 1,002 | 英文短 VQA |
| CogVLM detail zh/en | 14,549 | 451 | 中英文细节描述 |
| CogVLM multi zh/en | 12,596 | 404 | 中英文多轮视觉对话 |
| CogVLM single zh/en | 6,784 | 215 | 中英文单轮问答 |
| 合计 | **201,748** | **4,730** |  |

视觉训练额外加入 20% 纯文本回放，中文和英文约 2:1。训练使用 COIG-CQIA 中文样本与 SmolTalk2 everyday conversations。纯文本样本不带 `<img>`，单独 forward，只更新 language LoRA。
