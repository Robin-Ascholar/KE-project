# Harry Potter LinearRAG

基于 [LinearRAG](https://github.com/DEEP-PolyU/LinearRAG) 的课程项目，使用 Harry Potter 七册英文原著构建关系无关图检索增强生成（Graph RAG）系统。

仓库包含完整的端到端实验快照：原始 PDF、正文清洗脚本与清洗结果、JSON 数据集、本地 `all-mpnet-base-v2` 模型、LinearRAG 图与向量索引、问答预测和评测结果。大型模型及 Parquet 文件通过 Git LFS 管理；API 密钥和 Python 缓存不会提交。

## 当前状态

- 七本 PDF 已完成正文清洗，移除了图片、页码、重复页眉页脚和可识别脚注
- 共识别 199 章，生成 6,865 个 passage chunk
- chunk 以 MPNet tokenizer 计数：目标 300 token、最大 360 token、最小 80 token，相邻块重叠 2 个完整句子
- `all-mpnet-base-v2` passage、sentence、entity 向量已生成
- LinearRAG 图已生成：11,566 个节点、63,517 条边
- 七册端到端问答已完成第一次完整实验：`dataset/question` 七个 `bookN.json` 问题集共 1,000 道题，问答模型 `qwen2.5:7b`，人工语义评测正确率 49.2%（492/1,000）
- 人工语义评测的逐题判定、错误模式与代表性示例见 [error_analysis/2026-09-08_qwen2.5-7b_full-1000_manual_review.md](error_analysis/2026-09-08_qwen2.5-7b_full-1000_manual_review.md)
- 当前环境已在 NVIDIA GeForce RTX 3090、PyTorch 2.1.2+cu121 上验证

## 目录结构

```text
.
├── 原始数据/                     # 七本原始 PDF
├── cleanUP/                      # PDF 正文清洗脚本、TXT 和清洗报告
├── dataset/harrypotter/          # 6,865 个 chunk 和数据清单（示例问题见 questions.example.json）
├── dataset/question/             # 1,000 道题（book1–book7 七个问题集，含参考答案）
├── import/harrypotter/           # GraphML、NER 结果与三类向量索引
├── results/harrypotter/          # 预测、日志与评测结果
├── model/all-mpnet-base-v2/      # 本地 embedding 模型
├── docs/                         # 数据格式、实验记录与上游说明
├── figure/                       # LinearRAG 上游示意图
├── scripts/                      # 环境、数据准备和运行脚本
├── src/                          # LinearRAG 核心代码
├── environment.yml              # Conda 基础环境
├── requirements.txt             # Python 依赖
└── run.py                        # 主入口
```

工作目录中的 LinearRAG 论文 PDF 也随实验快照保存。

## 克隆完整仓库

仓库包含 Git LFS 文件。克隆前先安装 Git LFS：

```bash
sudo apt-get install git-lfs
git lfs install
git clone https://github.com/Robin-Ascholar/KE-project.git
cd KE-project
git lfs pull
```

如果只需要源码，可跳过 `git lfs pull`，此时模型权重和 Parquet 文件只是 LFS 指针。

## 配置运行环境

Linux + NVIDIA GPU 推荐运行：

```bash
bash scripts/setup_environment.sh
conda activate linearrag
python scripts/check_environment.py
```

脚本会创建 Python 3.9 环境，安装 CUDA 12.1 版 PyTorch、项目依赖、`en_core_web_trf` 和 `all-mpnet-base-v2`。仓库已经包含 embedding 模型；下载脚本会复用完整文件或补齐缺失文件。

当前验证环境：

```text
Python: 3.9
PyTorch: 2.1.2+cu121
GPU: NVIDIA GeForce RTX 3090
spaCy model: en_core_web_trf 3.6.1
Embedding model: sentence-transformers/all-mpnet-base-v2
Embedding dimension: 768
SentenceTransformer max_seq_length: 384
```

## 从 PDF 重新生成数据

### 1. 清洗七本 PDF

```bash
python cleanUP/clean_pdf.py --overwrite
```

默认从 `原始数据/` 读取 PDF，输出到 `cleanUP/txt/`。每本书生成一个 UTF-8 TXT 和一个清洗报告 JSON。详细规则见 [cleanUP/README.md](cleanUP/README.md)。

### 2. 合并并切分七本书

```bash
python scripts/prepare_all_books.py
```

脚本读取 `cleanUP/txt/*.txt` 和对应的 `*.report.json`，按章节和完整句子切分，输出：

```text
dataset/harrypotter/chunks.json
dataset/harrypotter/manifest.json
```

切分参数可以通过 `--target-tokens`、`--max-tokens`、`--min-tokens` 和 `--overlap-sentences` 调整。当前数据集使用 `300 / 360 / 80 / 2`。每册问答问题集在 `dataset/question/bookN.json`，格式示例见 `dataset/harrypotter/questions.example.json`，完整说明见 [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md)。

## 运行端到端实验

### OpenAI 或兼容接口

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://your-compatible-endpoint/v1"  # 官方 OpenAI 可不设置

python run.py \
  --spacy_model en_core_web_trf \
  --embedding_model model/all-mpnet-base-v2 \
  --dataset_name harrypotter \
  --llm_model gpt-4o-mini \
  --max_workers 4 \
  --retrieval_top_k 5
```

### 当前本地 Ollama 问答模型

服务器已部署 `deepseek-ocr:latest`。该模型是支持文本与图片输入的 3B 视觉语言模型，Ollama 包大小约 6.7 GB、上下文窗口 8K。安装与确认命令：

```bash
ollama pull deepseek-ocr:latest
ollama list
```

项目已将它设为默认问答模型，并为它使用单轮、仅返回答案的兼容提示。直接运行：

```bash
bash scripts/run.sh
```

等价的显式命令为：

```bash
export OPENAI_API_KEY="ollama"
export OPENAI_BASE_URL="http://127.0.0.1:11434/v1"

python run.py \
  --spacy_model en_core_web_trf \
  --embedding_model model/all-mpnet-base-v2 \
  --dataset_name harrypotter \
  --llm_model deepseek-ocr:latest \
  --llm_max_tokens 128 \
  --llm_timeout 240 \
  --max_workers 2 \
  --retrieval_top_k 8 \
  --use_vectorized_retrieval
```

DeepSeek-OCR 的主要用途是文档 OCR，当前纯文本问答属于实验性用法。真实 Top-8 召回烟雾测试已通过；若追求问答质量，应将它与通用指令模型的结果单独比较。本仓库保留的正式问答结果使用 `qwen2.5:7b` 生成。

### 批量测试 `dataset/question`

七个 `bookN.json` 问题集共 1,000 道题。批量入口会逐套运行、每 10 题保存检查点、合并预测，并自动调用综合评分器：

```bash
python scripts/run_question_sets.py
```

中断后给出原 `--output-dir` 即可从检查点继续。每套预测、合并文件、评分明细、指标汇总和进度清单都保存在同一个 `results/harrypotter/batch_*` 目录；完整参数与输出结构见 [docs/EVALUATION.md](docs/EVALUATION.md)。

当前 Qwen2.5:7b 全量问答的错误来源、检索条件分析和改进建议见 [error_analysis/2026-09-08_qwen2.5-7b_full-1000_manual_review.md](error_analysis/2026-09-08_qwen2.5-7b_full-1000_manual_review.md)。

首次建图或语料改变后，LinearRAG 会在 `import/harrypotter/` 写入索引。当前仓库已经包含完成的索引，未改变数据时会复用缓存。不要把不同语料放在相同的 `dataset_name` 下复用旧索引。

## 图谱与实验结果

主图谱：

```text
import/harrypotter/LinearRAG.graphml
```

该文件可用 Gephi、Cytoscape 或支持 GraphML 的 Python 图工具打开。相关索引包括：

```text
import/harrypotter/passage_embedding.parquet
import/harrypotter/sentence_embedding.parquet
import/harrypotter/entity_embedding.parquet
import/harrypotter/ner_results.json
```

本次 1,000 题完整实验的结果保存在：

```text
results/harrypotter/batch_2026-09-07_qwen2.5-7b_full-1000/
```

其中 `combined_predictions.json` 保存合并后的问题、召回 passage、参考答案与模型回答，`combined_predictions.scored.json` 保存逐题评测字段（含人工语义判定的 `local_judge_accuracy`/`local_judge_label`），`evaluation_results.deterministic.json` 保存人工评测汇总（总正确率 0.492，分册明细见 [error_analysis/](error_analysis/)）。早期 14 题实验与自动打分结果已删除，不再保留。

论文公开代码包含 Contain-Acc 和 GPT-Acc 的判分逻辑与提示词，但 GPT-4o-mini 是外部 API 模型，仓库不含模型权重。本项目还提供可审计的本地论文式评分器：

```bash
python scripts/evaluate_predictions.py \
  --provider ollama \
  --judge-model qwen3.6:27b \
  --predictions results/harrypotter/batch_2026-09-07_qwen2.5-7b_full-1000/combined_predictions.json
```

评分器会计算论文定义的规范化 Contain-Acc，并让本地 LLM 按“包含参考答案关键信息、事实一致、无矛盾内容”三项规则判分。详细说明见 [docs/EXPERIMENT.md](docs/EXPERIMENT.md)。

## 实体别名扩展实验

本分支提供默认关闭的 query-time alias activation：问题命中 `Tom Riddle / Voldemort / You-Know-Who`、`Scabbers / Peter Pettigrew / Wormtail` 等别名后，以受控的额外权重激活同组实体节点，无需重建 embedding、NER 或图索引。

固定划分的初步检索实验中，开发集触发题的 Contain@20 从 34.38% 提升至 37.50%（新增 1、丢失 0）；测试划分触发题从 25.60% 提升至 28.80%（新增 5、丢失 1）。这是值得继续验证的小幅正向结果，尚不能视为正式结论。实现限制、案例分析和复现命令见 [docs/ALIAS_EXPANSION_PRELIMINARY_RESULTS.md](docs/ALIAS_EXPANSION_PRELIMINARY_RESULTS.md)。

```bash
conda activate linearrag
python scripts/run_alias_expansion_experiment.py \
  --split dev \
  --seed 20260913 \
  --alias-expansion-budget 0.75 \
  --retrieval-top-k 20
```

## 与论文官方实验的差异

当前实验验证了 LinearRAG 在七本小说上的完整流程，但不是论文四个官方 benchmark 的严格指标复现。官方发布的多跳数据通常采用约 1,000-token passage 和 100-token overlap，并使用 Top-5 与 GPT-4o-mini；本项目采用不跨章节的 300-token passage、2 句重叠、Top-20 与本地 `qwen2.5:7b`。这些差异应在实验报告中明确说明。

## 上游项目与许可

本项目基于 DEEP-PolyU 的 LinearRAG，导入基线为提交 `bcc94e66c221f798801255effba09311d6fbcd8d6`。原始说明保存在 [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md)，修改与归属见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

由于上游代码采用 GPL-3.0，本仓库中的派生代码同样按 [GPL-3.0](LICENSE) 分发。
