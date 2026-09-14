# Task 4：Baseline 与统一对照实验分析

> 本文是基于当前仓库、task3 文档和历史正式实验产物生成的结果汇总。缺失的逐题产物不会被用其他实验推断，统一记为 `N/A`。

## 1. 结论摘要

在 1,000 道问题上，Closed-book 的人工语义正确率为 **3.20%**，Dense RAG 为 **18.30%**，Original LinearRAG 为 **49.20%**。Dense RAG 相比 Closed-book 提升 **15.10 个百分点**；Original LinearRAG 比 Dense RAG 高 **30.90 个百分点**。

但三者当前历史结果**不能直接作为严格公平的最终对照结论**：Closed-book/Dense RAG 使用 `qwen2.5:0.5b`、Top-5，Original LinearRAG 使用 `qwen2.5:7b`、Top-20。模型规模和上下文数量同时变化，因此 Original LinearRAG 的优势不能全部归因于检索方法。

## 2. 三个 Baseline 结果

### 2.1 核心指标

| Method | N | Answer Accuracy | Contain-Acc | Exact Match | Token F1 | ROUGE-L F1 | Evidence Recall@5 | LLM RAG Judge Evidence Recall | Retrieval Avg (ms) | Total Avg (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Closed-book | 1000 | 3.20% | 2.20% | 0.30% | 4.73% | 4.55% | N/A | N/A | 0.00 | 111.44 |
| Dense RAG | 1000 | 18.30% | 12.80% | 7.50% | 17.13% | 16.87% | 11.93% | N/A | 43.27 | 313.12 |
| Original LinearRAG | 1000 | 49.20% | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

说明：Exact Match、Token F1、ROUGE-L 和 Dense RAG 的 passage-id 检索指标均由历史正式预测文件逐题重算；Closed-book 没有检索过程，因此 Evidence Recall@5 和 retrieval metrics 不适用；Original LinearRAG 的历史 1,000 题逐题结果未保留，不能补算这些指标。

### 2.2 Dense RAG 检索诊断

| Metric | Value |
|---|---:|
| Evidence Precision@5 | 7.16% |
| Hit@5 | 29.30% |
| MRR@5 | 18.24% |
| nDCG@5 | 20.83% |
| Retrieval F1@5 | 8.95% |

这些值是针对自动词法候选 gold evidence 的诊断结果，不是人工确认 gold evidence 上的最终结论。

### 2.3 分册人工答案正确率

| Book | Closed-book | Dense RAG | Original LinearRAG |
|---|---:|---:|---:|
| book1 | 6/143 (4.2%) | 33/143 (23.1%) | 77/143 (53.8%) |
| book2 | 6/143 (4.2%) | 31/143 (21.7%) | 72/143 (50.3%) |
| book3 | 7/143 (4.9%) | 30/143 (21.0%) | 85/143 (59.4%) |
| book4 | 2/143 (1.4%) | 25/143 (17.5%) | 72/143 (50.3%) |
| book5 | 5/143 (3.5%) | 24/143 (16.8%) | 84/143 (58.7%) |
| book6 | 2/143 (1.4%) | 19/143 (13.3%) | 53/143 (37.1%) |
| book7 | 4/142 (2.8%) | 21/142 (14.8%) | 49/142 (34.5%) |
| **Total** | **32/1000 (3.2%)** | **183/1000 (18.3%)** | **492/1000 (49.2%)** |

Dense RAG 在七册上均高于 Closed-book，说明引入文本检索确实提供了有效信息。Original LinearRAG 在所有分册也高于 Dense RAG，但由于 generator 不同，不能把该差异视作纯检索增益。

## 3. 实验控制变量

| Variable | Closed-book | Dense RAG | Original LinearRAG | 统一性判断 |
|---|---|---|---|---|
| Question set | 7 books, 1,000 questions | 同左 | 同左 | 一致 |
| Corpus | 6,865 passages, 199 chapters | 同左 | 同左 | 一致 |
| Chunking | 300 target / 360 max / 80 min / 2 sentence overlap | 同左 | 同左 | 一致 |
| Generator | `qwen2.5:0.5b` | `qwen2.5:0.5b` | `qwen2.5:7b` | 不一致 |
| Prompt | concise answer-only | context + concise answer | LinearRAG QA prompt | 不一致 |
| Top-k | 0 context | 5 | 20 | 不一致 |
| Embedding | 不使用 | `all-mpnet-base-v2` | `all-mpnet-base-v2` | 部分一致 |
| Temperature | 0 | 0 | 文档未完整记录，历史 run 使用本地固定配置 | 需补记录 |
| Chapter retrieval | 不适用 | 不适用 | 关闭，属于 Original baseline | 一致 |

任务 4 的严格重跑应把三种方法都固定到同一个生成模型、同一 prompt 规范、同一 Top-k、同一 generation 参数。当前报告保留历史结果，不把不同配置伪装成严格公平实验。

## 4. Evidence Recall 的两种定义

### 4.1 Evidence Recall@k：passage-id 检索指标

定义：

```text
Recall@k = |Gold passage IDs ∩ Top-k retrieved passage IDs| / |Gold passage IDs|
```

Dense RAG 的历史正式结果为 **11.93%**。但其 gold evidence 来源是 `auto_lexical_candidate`，只能作为流程验证和弱监督候选，不能称为人工确认的最终证据标注。

Closed-book 没有检索结果，所以该指标为 `N/A`。Original LinearRAG 当前缺少逐题 `gold_evidence` 和 retrieved passage artifact，因此为 `N/A`。

### 4.2 LLM RAG Judge Evidence Recall：语义支撑指标

该指标由 `scripts/evaluate_predictions.py` / `src/rag_metrics.py` 让裁判模型拆分 reference facts，并判断这些事实是否能被 Top-k 上下文支持：

```text
LLM RAG Judge Evidence Recall = supported reference facts / all reference facts
```

它与 passage-id Recall@k 不是同一个指标，不能互换。当前 Task 4 正式 Closed-book/Dense RAG 运行只生成了 passage-id Recall@5，没有执行这套 JSON RAG judge；Original LinearRAG 也没有保留对应逐题文件，因此三个 baseline 的该项均为 `N/A`。

## 5. 延迟分析

Closed-book 平均总延迟为 **111.44 ms**，Dense RAG 平均总延迟为 **313.12 ms**，增加 **201.68 ms**。Dense RAG 的平均检索延迟为 **43.27 ms**，约占总延迟的 **13.8%**；主要成本仍然来自 LLM generation。

Original LinearRAG 的历史日志记录了图规模和运行配置，但没有保留逐题 retrieval/total latency，因此不报告伪造的平均值。后续需要在 `retrieve()` 和 `qa()` 周围直接记录计时，并区分 index、retrieval、generation、total 四个阶段。

## 6. 指标覆盖矩阵

下表中的 `N/A` 表示当前没有可审计的输入或该方法不适用，不表示分数为 0。

| Metric | Closed-book | Dense RAG | Original LinearRAG | 当前状态 |
|---|---:|---:|---:|---|
| Answer Accuracy | 已有 | 已有 | 已有 | 人工语义评测 |
| Contain-Acc | 已有 | 已有 | N/A | Original 缺逐题答案产物 |
| Evidence Recall@5 | N/A | 已有 | N/A | Closed 不适用；Original 缺证据 ID |
| Evidence Precision/Hit/MRR/nDCG/Retrieval F1@5 | N/A | 已补齐 | N/A | 从 Dense RAG 逐题 Top-5 重算 |
| LLM RAG Judge Evidence Recall | N/A | N/A | N/A | 未执行 JSON RAG judge |
| Context Relevance/Precision | N/A | N/A | N/A | 未执行 RAG judge |
| Answer Fact Coverage/Faithfulness | N/A | N/A | N/A | 未执行 RAG judge |
| Exact Match/Token F1/ROUGE-L | 已补齐 | 已补齐 | N/A | 从 Closed/Dense 逐题预测重算 |
| Semantic Similarity | N/A | N/A | N/A | 当前环境缺少 sentence-transformers |
| Retrieval/Total Latency | 部分 | 已有 | N/A | Original 缺逐题计时 |

## 7. Gold 标注与数据泄露检查

- `gold_answer`、`gold_evidence`、`gold_book`、`gold_chapter` 可以保存在问题记录中，但只能用于评测。
- `run_task4_minimal_baselines.py` 的正式逻辑没有把 `gold_book` 或 `gold_chapter` 作为 Dense RAG 的检索过滤条件。
- Original LinearRAG 的 `retrieve()` 只使用问题 embedding、NER seed、图传播和可选配置；task 4 baseline 必须保持 `enable_chapter_retrieval=False`。
- Dense RAG 的 `sentence_transformer` backend 已在历史正式 run 中使用；`tfidf_fallback` 只出现在早期 70 题 smoke test，不应作为正式 Dense RAG 结论。
- gold evidence 当前是自动词法候选，正式论文结论前仍需要逐题人工审计或建立独立标注文件。

## 8. 复现与后续补齐

当前正式历史数据来源：

- Closed-book/Dense RAG：Git commit `95ae490` 的 `results/task4_formal/`。
- Original LinearRAG：task3 分支的 `README.md`、`docs/EXPERIMENT.md`、`docs/CHAPTER_RETRIEVAL.md` 和 `错误分析/2026-09-08_qwen2.5-7b_full-1000_manual_review.md`。
- 统一评测实现：`scripts/evaluate_predictions.py`、`src/paper_evaluator.py`、`src/rag_metrics.py`。

建议严格最终版统一为：同一 `qwen2.5:7b`、同一 answer-only prompt、同一 Top-5、同一 `all-mpnet-base-v2`、temperature 0、同一 1,000 题顺序；Original LinearRAG 关闭 chapter retrieval，Dense RAG 只做向量 Top-5，Closed-book 不注入上下文。然后保存三套逐题 JSON，再执行统一评测器，才能补齐上表所有 N/A 指标。

## 9. 产物

- `results/task4_formal/baseline_comparison.json`：结构化结果汇总。
- `results/task4_formal/baseline_comparison.csv`：表格导出。
- `results/task4_formal/baseline_comparison.md`：简表。
- `results/task4_formal/TASK4_BASELINE_ANALYSIS.md`：本文完整分析。
