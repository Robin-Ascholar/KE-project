# Task 4：Baseline 与统一对照实验分析

> 本文基于仓库现有的 Closed-book、Dense RAG 和 Original LinearRAG 1000 题实验文件整理。它是历史实验结果汇总，不代表三个方法已经完成同模型、同 prompt、同 Top-k 的严格重跑。

## 1. 结论摘要

| Method | Questions | Model | Answer Accuracy | Contain-Acc | Exact Match | Token-F1 | ROUGE-L-F1 |
|---|---:|---|---:|---:|---:|---:|---:|
| Closed-book | 1000 | `qwen2.5:0.5b` | 3.20% | 2.20% | 0.30% | 4.73% | 4.55% |
| Dense RAG | 1000 | `qwen2.5:0.5b` | 18.30% | 12.80% | 7.50% | 17.13% | 16.87% |
| Original LinearRAG | 1000 | `qwen2.5:7b` | 49.20% | 26.60% | 20.50% | 32.18% | 31.67% |

在已有人工语义评测口径下，Dense RAG 比 Closed-book 高 15.10 个百分点，Original LinearRAG 比 Dense RAG 高 30.90 个百分点。但 Original LinearRAG 使用更大的生成模型，不能把全部差异归因于检索方法。

## 2. 检索指标

检索指标定义为 passage ID 级别的 Top-k 结果与 gold passage ID 的重叠。当前 gold evidence 来自 `auto_lexical_candidate`，需要人工审计，不能直接视为最终人工 gold。

| Method | Native Top-k | Recall@5 | Recall@8 | Precision@5 | Hit@5 | MRR@5 | NDCG@5 | Retrieval F1@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Closed-book | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Dense RAG | 5 | 11.93% | 11.93% | 7.16% | 29.30% | 18.24% | 20.83% | 8.95% |
| Original LinearRAG | 8 | 8.17% | 12.23% | 4.90% | 19.60% | 11.41% | 13.32% | 6.13% |

Closed-book 没有检索过程，因此检索指标不适用。Original LinearRAG 的 Recall@5/8 是根据原始 1000 条预测中的 passage 前缀 ID，以及现有 1000 题自动候选证据补算得到的。

## 3. Original LinearRAG 的 LLM RAG Judge

Original LinearRAG 的 `combined_predictions.scored.json` 只有前 230/1000 题包含 RAG judge 输出，下面的数值只代表这 230 题。它们与 passage-id Recall@k 不是同一个指标。

| Metric | Value | Evaluable N |
|---|---:|---:|
| Context Relevance | 0.5870 | 230 |
| Context Precision | 0.1435 | 230 |
| LLM RAG Judge Evidence Recall | 0.5565 | 230 |
| Answer Fact Coverage | 0.5522 | 230 |
| Answer Faithfulness | 0.7182 | 230 |
| Retrieval Hit | 0.5957 | 230 |
| Retrieval MRR | 0.3448 | 230 |
| Retrieval NDCG | 0.3939 | 230 |
| Retrieval F1 | 0.2074 | 230 |

这里的 LLM RAG Judge Evidence Recall 表示“参考事实是否被上下文支持”；Evidence Recall@k 表示“gold passage ID 是否进入 Top-k”。两者不能混用。

## 4. 延迟

| Method | Retrieval Avg (ms) | Total Avg (ms) |
|---|---:|---:|
| Closed-book | 0.00 | 111.44 |
| Dense RAG | 43.27 | 313.12 |
| Original LinearRAG | N/A | N/A |

Original LinearRAG 的归档预测和日志没有逐题 retrieval/total latency 字段，不能从现有文件可靠反推。Dense RAG 的检索平均耗时约占总耗时 13.8%。

## 5. 分册人工答案准确率

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

## 6. 控制变量检查

| Variable | Closed-book | Dense RAG | Original LinearRAG | Status |
|---|---|---|---|---|
| Question set | 1000 questions | Same | Same | Consistent |
| Generator | `qwen2.5:0.5b` | `qwen2.5:0.5b` | `qwen2.5:7b` | Not consistent |
| Prompt | Answer-only | Context + answer | Historical LinearRAG prompt | Not consistent |
| Top-k | 0 context | 5 | 8 | Not consistent |
| Chunk setting | Not recorded in formal artifact | Not recorded in formal artifact | 1000 token / 100 overlap | Incomplete |
| Gold evidence | Auto lexical candidate | Auto lexical candidate | Same candidate set used for recomputation | Needs audit |
| Chapter retrieval | N/A | N/A | Disabled for Original baseline | Consistent in intent |

因此当前结果适合做已有实验对比和问题定位，不应作为严格公平的最终 ablation 结论。严格版需要统一生成模型、prompt、Top-k、chunk 参数和 generation 参数，并保存三套逐题输出及逐题计时。

## 7. 指标覆盖结论

- 三种 baseline 的 1000 题答案类指标已经补齐：Answer Accuracy、Contain-Acc、Exact Match、Token-F1、ROUGE-L-F1。
- Dense RAG 和 Original LinearRAG 的 passage-id 检索指标已经补齐，但依赖自动候选 evidence。
- Closed-book 的检索指标为 `N/A`，这是方法不适用，不是得分为 0。
- Original LinearRAG 的 LLM RAG Judge 只有 230 题，不能代表完整 1000 题。
- Original LinearRAG 的延迟指标缺少原始计时，保留为 `N/A`。

## 8. 结果文件

- `results/task4_formal/baseline_comparison.json`
- `results/task4_formal/baseline_comparison.csv`
- `results/task4_formal/baseline_comparison.md`
- `results/task4_formal/original_linearrag/predictions.task4_scored.json`
- `scripts/complete_task4_linear_rag_metrics.py`
