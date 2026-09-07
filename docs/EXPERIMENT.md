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
questions: 14
minimum chunk tokens: 80
average chunk tokens: 282.41
maximum chunk tokens: 360
```

`chunks.json` 是字符串数组，`questions.json` 是包含 `question` 与 `answer` 的对象数组，`manifest.json` 记录输入哈希、章节信息和每个 chunk 的 token 数。

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

当前规模：

```text
passage nodes: 6,865
entity nodes: 4,701
GraphML vertices: 11,566
GraphML edges: 63,517
sentence records: 53,580
```

GraphML 中保存 passage/entity 图；sentence 记录用于实体激活阶段的 mention 矩阵和向量检索。

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

结果目录：

```text
results/harrypotter/2026-09-07_15-06-11/
```

当前指标：

| 指标 | 结果 |
|---|---:|
| LLM Accuracy | 35.71%（5/14） |
| Contain Accuracy | 42.86%（6/14） |

补充召回检查显示：Top-20 passage 中有 9/14 个问题包含答案证据，扩展到 Top-100 后为 12/14；其余两个问题的参考答案文本没有以预期形式出现在清洗语料中。

## 与论文设置的关系

论文将输入定义为 passage 集合，并在每个 passage 内用 spaCy 划分 sentence。官方代码直接读取准备好的 `chunks.json`，运行时不会重新按 `chunk_token_size` 切块。

官方发布的 HotpotQA、2WikiMultiHopQA 和 MuSiQue 数据以约 1,000 token 为主，相邻内部块通常重叠 100 token；论文使用 `all-mpnet-base-v2`、Top-5 和 GPT-4o-mini。本实验使用 300-token 章节感知切分、2 句重叠、Top-20 和本地 Qwen。因此当前结果属于方法复现与自定义数据应用实验，不属于论文官方数值的严格复现。

若需要严格对照，应另建数据集目录，保留官方 `chunks.json`、1,000 个验证问题、Top-5、GPT-4o-mini 和论文对应超参数，避免覆盖当前七册索引。
