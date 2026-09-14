# 创新 3：章节粗到细检索

## 1. 创新内容分析

### 1.1 问题背景

原始 LinearRAG 在 passage、entity 和 sentence 层面直接进行检索。对于 Harry Potter 七册长篇语料，全部语料共有 6,865 个 passage 和 199 个章节。每个问题都在全量 passage 上进行细粒度匹配，会带来两个问题：

1. 检索候选范围较大，增加检索时间和图搜索开销。
2. 相似但不属于答案所在章节的 passage 可能进入候选集，降低上下文的集中度。

### 1.2 方法设计

创新 3 采用章节级 coarse retrieval 与 passage 级 fine retrieval 的两阶段结构：

1. 根据 passage 前缀 `Book N | Chapter M: title` 恢复每个 passage 的章节归属。
2. 对同一章节的 passage embedding 求均值并归一化，构造章节表示向量。
3. 将问题 embedding 与 199 个章节表示进行相似度计算，选取 Top-K 个章节。
4. 只在选中章节的 passage 中执行原有 Dense、Entity 和 PPR 检索。
5. 从全量 passage 的 Dense 检索结果中保留少量 global fallback，降低章节粗召回错误造成完全漏召回的风险。

该设计不改变原有图结构和细粒度排序方法，只在细粒度检索之前增加章节范围筛选。因此它既可以作为效率优化，也可以作为检索上下文的去噪机制。

### 1.3 预期效果与实验假设

实验需要验证以下假设：

- **H1：检索效率提升。** 章节筛选后，进入 fine retrieval 的 passage 数量明显少于 6,865 个，平均检索耗时下降。
- **H2：检索质量保持或提升。** 合理的 chapter Top-K 和 global fallback 能保持 Evidence Recall、Hit Rate 和 MRR，甚至提高 Context Precision。
- **H3：端到端回答质量改善。** 更集中的上下文能够提升模型答案正确率，但如果 Top-K 过小，也可能因章节误召回造成证据丢失。

因此，章节检索不是无条件地增大 Top-K，而是需要在召回率、上下文纯度和运行效率之间进行参数权衡。

### 1.4 可调参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `enable_chapter_retrieval` | `False` | 是否启用章节粗到细检索；关闭时为原始基线 |
| `chapter_top_k` | `8` | coarse 阶段保留的章节数 |
| `chapter_global_fallback` | `2` | 从全量 Dense 检索中额外保留的 passage 数 |
| `retrieval_top_k` | `8` | 最终交给问答模型的 passage 数 |

默认关闭章节检索是为了保证原始 LinearRAG 基线行为不变。创新 3 实验必须显式添加 `--enable-chapter-retrieval`。

## 2. 实现位置

- `src/LinearRAG.py`：章节元数据解析、章节向量构建、章节候选筛选和受限 fine retrieval。
- `src/config.py`：章节检索配置字段。
- `src/embedding_store.py`：保存预计算章节向量。
- `scripts/run_question_sets.py`：批量问答实验参数和断点续跑。
- `scripts/index_and_test.py`：无 LLM 的索引与检索 smoke test。
- `tests/test_chapter_retrieval.py`：章节解析和候选筛选单元测试。

章节向量缓存位于：

```text
import/harrypotter/chapter_embedding.parquet
```

## 3. 七册 1000 题全量实验

### 3.1 数据范围

批量脚本默认读取：

```text
dataset/question/book1.json
dataset/question/book2.json
dataset/question/book3.json
dataset/question/book4.json
dataset/question/book5.json
dataset/question/book6.json
dataset/question/book7.json
```

七个文件共 1,000 道题：`book1` 至 `book6` 各 143 题，`book7` 为 142 题。

### 3.2 运行前检查

先进入 Conda 环境，并确认 Ollama 已启动且问答模型存在：

```bash
conda activate linearrag
ollama list
```

当前脚本默认通过 Ollama 的 OpenAI 兼容接口访问问答模型。若使用 `deepseek-ocr:latest`：

```bash
ollama pull deepseek-ocr:latest
```

### 3.3 推荐的创新 3 实验命令

下面命令运行七册全部 1,000 题，启用章节粗到细检索，并使用 `qwen3.6:27b` 做自动综合评测：

```bash
conda activate linearrag

python scripts/run_question_sets.py \
  --question-dir dataset/question \
  --dataset-name harrypotter \
  --embedding-model model/all-mpnet-base-v2 \
  --embedding-device cuda \
  --llm-model deepseek-ocr:latest \
  --llm-max-tokens 128 \
  --llm-timeout 240 \
  --max-workers 2 \
  --question-batch-size 10 \
  --retrieval-top-k 8 \
  --enable-chapter-retrieval \
  --chapter-top-k 8 \
  --chapter-global-fallback 2 \
  --judge-model qwen3.6:27b \
  --judge-workers 1 \
  --judge-timeout 300 \
  --judge-checkpoint-every 10
```

如果当前机器没有可用 CUDA，将 `--embedding-device cuda` 改为 `cpu`。问答模型和评测模型必须已经在 Ollama 中安装；否则可以先使用 `--no-evaluate` 只生成回答，之后单独执行评测。

### 3.4 推荐的基线对照命令

为了判断创新 3 是否有效，使用相同的数据、问答模型、Top-K、并发数和评测模型，只去掉章节检索参数：

```bash
conda activate linearrag

python scripts/run_question_sets.py \
  --question-dir dataset/question \
  --dataset-name harrypotter \
  --embedding-model model/all-mpnet-base-v2 \
  --embedding-device cuda \
  --llm-model deepseek-ocr:latest \
  --llm-max-tokens 128 \
  --llm-timeout 240 \
  --max-workers 2 \
  --question-batch-size 10 \
  --retrieval-top-k 8 \
  --judge-model qwen3.6:27b \
  --judge-workers 1 \
  --judge-timeout 300 \
  --judge-checkpoint-every 10
```

两次实验应使用不同的 `--output-dir`，避免断点文件互相复用。例如：

```bash
--output-dir results/harrypotter/innovation3_chapter_top8_fallback2
```

基线可使用：

```bash
--output-dir results/harrypotter/baseline_dense_graph
```

### 3.5 参数敏感性实验

在完成基线和推荐配置后，可固定其他参数，分别测试：

```text
chapter_top_k: 4, 8, 16
chapter_global_fallback: 0, 2, 4
```

建议至少保留以下三组：

| 实验 | 章节 Top-K | Global fallback |
|---|---:|---:|
| Baseline | 不启用 | 不适用 |
| Innovation-3 | 8 | 2 |
| Ablation-small | 4 | 0 |
| Ablation-large | 16 | 4 |

## 4. 实验数据保存位置

如果没有指定 `--output-dir`，脚本会自动创建：

```text
results/harrypotter/batch_YYYY-MM-DD_HH-MM-SS/
```

推荐显式指定目录，便于区分实验。每个实验目录包含：

```text
run_manifest.json                         # 参数、数据集哈希和进度
log.txt                                   # 运行日志
sets/book1/predictions.json               # book1 逐题结果
sets/book2/predictions.json               # book2 逐题结果
...
sets/book7/predictions.json               # book7 逐题结果
combined_predictions.json                 # 七册合并后的 1000 题结果
combined_predictions.scored.json          # 逐题评测结果
evaluation_results.json                   # 汇总指标
```

章节检索的诊断字段保存在每条 `combined_predictions.json` 记录中：

- `retrieval_mode`：应为 `chapter_coarse_to_fine`。
- `selected_chapters`：coarse 阶段选中的章节及相似度。
- `chapter_candidate_passages`：章节筛选后的 passage 数量。
- `chapter_global_fallback`：实际加入的全局 fallback 数量。

这些字段可用于统计平均候选规模、章节命中情况和检索效率。

## 5. 中断后继续运行

脚本每 10 道题保存一次 checkpoint。运行中断后，使用原来的输出目录重新执行同一命令即可：

```bash
python scripts/run_question_sets.py \
  --output-dir results/harrypotter/innovation3_chapter_top8_fallback2 \
  --question-dir dataset/question \
  --dataset-name harrypotter \
  --embedding-model model/all-mpnet-base-v2 \
  --embedding-device cuda \
  --llm-model deepseek-ocr:latest \
  --enable-chapter-retrieval \
  --chapter-top-k 8 \
  --chapter-global-fallback 2
```

不要修改同一实验目录对应的问题集；脚本会通过 SHA256 检查问题集是否发生变化。

## 6. 重点比较指标

优先比较以下结果：

1. `evaluation_results.json` 中的答案正确率和生成质量指标。
2. `context_relevance`、`evidence_recall`、`retrieval_hit`、`retrieval_mrr` 和 `retrieval_ndcg`。
3. `context_precision` 和 `retrieval_f1`，衡量章节筛选是否减少无关上下文。
4. `chapter_candidate_passages` 的平均值，衡量 fine retrieval 的候选规模变化。
5. `log.txt` 中的总运行时间和分阶段耗时。

实验报告中应同时报告基线和创新 3 的配置，不能只报告创新 3 的单次准确率。若创新 3 的答案准确率没有提升，但 Evidence Recall 保持不变且候选规模、运行时间下降，也可以说明它主要带来了效率收益；若 Recall 下降，应增大 `chapter_top_k` 或 `chapter_global_fallback`。
