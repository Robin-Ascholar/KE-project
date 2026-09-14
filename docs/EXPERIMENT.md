# 七册端到端实验记录

本文记录 Harry Potter 七册数据在 LinearRAG 上完成清洗、切分、建图、检索、生成和评测的当前实验快照。

## 实验范围

本实验使用七本英文原著作为自定义语料，验证 LinearRAG 在长篇小说上的完整工作流。它复用了论文公开代码的 Tri-Graph 构建与两阶段检索方法，但数据集、passage 切分、问题集、Top-k 和生成模型与论文官方 benchmark 不同，因此结果不能与论文表格中的指标直接比较。

## 运行环境

```text
OS: Ubuntu 24.04
Conda environment: linearrag
Python: 3.9
PyTorch: 2.1.2+cu121
GPU: NVIDIA GeForce RTX 3090
spaCy: en_core_web_trf 3.6.1
Embedding: all-mpnet-base-v2, 768 dimensions, max_seq_length 384
Generation: Ollama qwen2.5:7b through OpenAI-compatible API
```

环境验证命令：

```bash
conda activate linearrag
python scripts/check_environment.py
```

### 当前 DeepSeek-OCR 问答配置

在上述历史实验之后，服务器 Ollama 已新增 `deepseek-ocr:latest`，代码默认问答模型也已切换到该模型：

```bash
ollama pull deepseek-ocr:latest
bash scripts/run.sh
```

`run.py` 默认使用 `--llm_model deepseek-ocr:latest`、`--llm_max_tokens 128` 和 `--llm_timeout 240`。`src/LinearRAG.py` 会识别 `deepseek-ocr` 模型并使用单轮直接回答提示，因为原有的 `Thought:` 链式提示会使该 OCR 模型在长文本上重复问题。使用真实 Top-8 检索段落的烟雾测试输出为 `The Triwizard Cup.`，参考答案为 `The Triwizard Cup`。

DeepSeek-OCR 是视觉 OCR 模型，纯文本 RAG 问答不属于其主要训练用途。以下已有指标仍对应 qwen2.5:7b 历史运行；切换模型后需要重新运行完整问题集（1,000 题），才能得到新的可比较结果。

## 数据清洗

原始文件位于 `原始数据/`。执行：

```bash
python cleanUP/clean_pdf.py --overwrite
```

清洗器使用 Poppler 的文字坐标，保留章节标题和正文，过滤首章前内容、重复页眉页脚、页码、可识别脚注和纯图片内容。输出位于 `cleanUP/txt/`，包括七个 TXT 文件与七个 `report.json` 文件。

七册识别章数依次为：

```text
17, 18, 22, 37, 38, 30, 37
```

总计 199 章。

## Passage 切分

执行：

```bash
python scripts/prepare_all_books.py
```

当前参数：

```text
target_tokens: 300
max_tokens: 360
min_tokens: 80
overlap_sentences: 2
```

切分使用本地 MPNet tokenizer 精确计数，遵守章节边界，并优先在完整句子之间切分。每个 chunk 都带有书号、章号和章节标题前缀。最终数据：

```text
books: 7
chapters: 199
chunks: 6,865
minimum chunk tokens: 80
average chunk tokens: 282.41
maximum chunk tokens: 360
```

`chunks.json` 是字符串数组，`manifest.json` 记录输入哈希、章节信息和每个 chunk 的 token 数。问答问题集位于 `dataset/question/book1.json`–`book7.json`（1,000 题），示例问题见 `dataset/harrypotter/questions.example.json`。

## 建图结果

索引目录：

```text
import/harrypotter/
```

核心文件：

| 文件 | 作用 |
|---|---|
| `LinearRAG.graphml` | passage/entity 图，可用 Gephi 或 Cytoscape 打开 |
| `ner_results.json` | passage 与 sentence 的实体识别结果 |
| `passage_embedding.parquet` | passage embedding |
| `sentence_embedding.parquet` | sentence embedding |
| `entity_embedding.parquet` | entity embedding |
| `chapter_embedding.parquet` | 章节级 embedding，创新 3 的 coarse retrieval 索引 |

当前规模：

```text
passage nodes: 6,865
entity nodes: 4,701
GraphML vertices: 11,566
GraphML edges: 63,517
sentence records: 53,580
```

GraphML 中保存 passage/entity 图；sentence 记录用于实体激活阶段的 mention 矩阵和向量检索。

## 创新 3：章节粗到细检索分析

本项目在原始 passage/entity/sentence 检索之前增加章节级 coarse retrieval。系统首先将 6,865 个 passage 按 `Book N | Chapter M: title` 归属到 199 个章节，对每章 passage embedding 求均值形成章节向量；问题到达时先从章节向量中选取 Top-K 章节，再在这些章节的 passage 中运行原有 Dense、Entity 和 PPR fine retrieval。同时保留少量全局 Dense fallback，避免章节初筛错误造成完全漏召回。

这一设计的核心假设是：长篇小说中的问题通常具有较强的章节局部性，章节级筛选可以减少无关 passage 进入细粒度检索和问答上下文。实验需要同时观察答案质量、Evidence Recall、Context Precision、候选 passage 数量和运行时间，不能只看最终答案准确率。

详细方法、参数解释、七册 1,000 题实验命令、基线对照、消融实验和输出文件说明见 [docs/CHAPTER_RETRIEVAL.md](CHAPTER_RETRIEVAL.md)。

推荐实验至少包含：

| 实验 | 章节检索 | `chapter_top_k` | `chapter_global_fallback` |
|---|---|---:|---:|
| 原始基线 | 关闭 | - | - |
| 创新 3 | 开启 | 16 | 4 |
| 消融实验 | 开启 | 4 或 16 | 0 或 4 |

所有实验应使用不同的输出目录，并固定问题集、问答模型、`retrieval_top_k`、embedding 模型、并发数和评测模型。

## 端到端运行参数

```bash
export OPENAI_API_KEY="ollama"
export OPENAI_BASE_URL="http://127.0.0.1:11434/v1"

python run.py \
  --spacy_model en_core_web_trf \
  --embedding_model model/all-mpnet-base-v2 \
  --dataset_name harrypotter \
  --llm_model qwen2.5:7b \
  --max_workers 2 \
  --max_iterations 3 \
  --iteration_threshold 0.4 \
  --passage_ratio 2 \
  --top_k_sentence 3 \
  --retrieval_top_k 20 \
  --use_vectorized_retrieval
```

## 评测结果

完整实验（1,000 题）结果目录：

```text
results/harrypotter/batch_2026-09-07_qwen2.5-7b_full-1000/
```

其中 `combined_predictions.json` 为合并预测，`combined_predictions.scored.json` 为逐题评测（`local_judge_accuracy`/`local_judge_label` 为人工语义判定），`evaluation_results.deterministic.json` 为人工评测汇总。

人工逐题语义评测结果（模型 `qwen2.5:7b`）：

| 分册 | 答对 | 正确率 |
|---|---:|---:|
| book1 | 77/143 | 53.8% |
| book2 | 72/143 | 50.3% |
| book3 | 85/143 | 59.4% |
| book4 | 72/143 | 50.3% |
| book5 | 84/143 | 58.7% |
| book6 | 53/143 | 37.1% |
| book7 | 49/142 | 34.5% |
| **合计** | **492/1,000** | **49.2%** |

判定口径：模型答案与参考答案语义一致即算对；"上下文未提及"式回避明确事实、答错人物/数字/情节或答非所问均判错。早期 14 题实验、其产物目录及自动打分结果已删除。错误模式与代表性示例见 [error_analysis/2026-09-08_qwen2.5-7b_full-1000_manual_review.md](../error_analysis/2026-09-08_qwen2.5-7b_full-1000_manual_review.md)。

### 论文式本地评分器

论文公开代码的 `src/evaluate.py` 已包含两种指标的实现：Contain-Acc 对规范化后的预测与参考答案做包含检查；GPT-Acc 把参考答案和模型回答交给 LLM，要求同时满足“包含关键信息、事实准确且一致、没有矛盾信息”。代码只调用外部推理接口，没有附带 GPT-4o-mini 的模型权重。

本项目新增 `src/paper_evaluator.py` 和 `scripts/evaluate_predictions.py`。默认通过 Ollama 原生接口关闭思考输出，以免推理 token 截断最终标签；也支持 `--provider openai` 调用 OpenAI 兼容服务。评分结果保留裁判原始返回和解析错误，便于审计。

```bash
conda activate linearrag
python scripts/evaluate_predictions.py \
  --provider ollama \
  --judge-model qwen3.6:27b \
  --base-url http://127.0.0.1:11434/v1 \
  --workers 1 \
  --predictions results/harrypotter/batch_2026-09-07_qwen2.5-7b_full-1000/combined_predictions.json \
  --output results/harrypotter/batch_2026-09-07_qwen2.5-7b_full-1000/combined_predictions.scored.json \
  --summary results/harrypotter/batch_2026-09-07_qwen2.5-7b_full-1000/evaluation_results.json
```

本地 Qwen 裁判遵循论文判分内容，可用于离线端到端验证和结果检查；其绝对分数不能当作论文 GPT-4o-mini 的 GPT-Acc。若要严格复现论文数值，需要用论文相同的 GPT-4o-mini 裁判重新运行，并记录模型版本与日期。

> 最终结论指标来自人工语义评测（见上表与 error_analysis/ 文件）；自动评分器运行受裁判中断与字符串假阴性影响，仅作为过程诊断。

## 与论文设置的关系

论文将输入定义为 passage 集合，并在每个 passage 内用 spaCy 划分 sentence。官方代码直接读取准备好的 `chunks.json`，运行时不会重新按 `chunk_token_size` 切块。

官方发布的 HotpotQA、2WikiMultiHopQA 和 MuSiQue 数据以约 1,000 token 为主，相邻内部块通常重叠 100 token；论文使用 `all-mpnet-base-v2`、Top-5 和 GPT-4o-mini。本实验使用 300-token 章节感知切分、2 句重叠、Top-20 和本地 Qwen。因此当前结果属于方法复现与自定义数据应用实验，不属于论文官方数值的严格复现。

若需要严格对照，应另建数据集目录，保留官方 `chunks.json`、1,000 个验证问题、Top-5、GPT-4o-mini 和论文对应超参数，避免覆盖当前七册索引。
