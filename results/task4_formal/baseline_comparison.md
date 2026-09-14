# Task 4 Baseline Comparison

本报告基于仓库现有实验文件补全指标，没有重新调用 LLM。三种 baseline 的原始模型并不一致：Closed-book/Dense RAG 使用 `qwen2.5:0.5b`，Original LinearRAG 使用 `qwen2.5:7b`，因此表中结果可以作为已有实验汇总，但不能直接宣称为严格同模型对照。

## 1. 统一结果

`Answer Accuracy` 使用已有人工语义复核；`Contain-Acc`、Exact Match、Token-F1、ROUGE-L-F1 为逐题确定性计算。检索 ID 指标使用 `results/task4_formal/questions_1000.json` 中已有的 `auto_lexical_candidate` 证据标注，必须人工审计后才能作为最终 benchmark gold。

| Method | N | Model | Native Top-k | Answer Acc | Contain-Acc | Exact | Token-F1 | ROUGE-L-F1 | Recall@5 | Recall@8 | Hit@5 | MRR@5 | NDCG@5 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| closed_book | 1000 | qwen2.5:0.5b | N/A | 0.0320 | 0.0220 | 0.0030 | 0.0473 | 0.0455 | N/A | N/A | N/A | N/A | N/A |
| dense_rag | 1000 | qwen2.5:0.5b | 5 | 0.1830 | 0.1280 | 0.0750 | 0.1713 | 0.1687 | 0.1193 | 0.1193 | 0.2930 | 0.1824 | 0.2083 |
| original_linearrag | 1000 | qwen2.5:7b | 8 | 0.4920 | 0.2660 | 0.2050 | 0.3218 | 0.3167 | 0.0817 | 0.1223 | 0.1960 | 0.1141 | 0.1332 |

## 2. 延迟

| Method | Retrieval Avg (ms) | Retrieval P50 (ms) | Retrieval P95 (ms) | Total Avg (ms) | Total P50 (ms) | Total P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| closed_book | 0.00 | 0.00 | 0.00 | 111.44 | 93.47 | 281.16 |
| dense_rag | 43.27 | 40.46 | 60.49 | 313.12 | 297.53 | 468.99 |
| original_linearrag | N/A | N/A | N/A | N/A | N/A | N/A |

Original LinearRAG 的归档预测没有保存 retrieval/total latency 字段，因此延迟不能从已有数据反推，不能用 0 填充。

## 3. LLM RAG Judge 指标

Original LinearRAG 的 `combined_predictions.scored.json` 只有前 230/1000 题包含 RAG judge 输出；下面明确标注为部分样本，不能与完整 1000 题结果混用。这里的 `LLM RAG judge evidence recall` 是“参考事实是否被上下文支持”，不是 passage id 是否进入 Top-k。

| Metric | Original LinearRAG | Evaluable N |
|---|---:|---:|
| Context Relevance | 0.5870 | 230 |
| Context Precision | 0.1435 | 230 |
| LLM RAG judge Evidence Recall | 0.5565 | 230 |
| Answer Fact Coverage | 0.5522 | 230 |
| Answer Faithfulness | 0.7182 | 230 |
| LLM judge Retrieval Hit | 0.5957 | 230 |
| LLM judge Retrieval MRR | 0.3448 | 230 |
| LLM judge Retrieval NDCG | 0.3939 | 230 |
| LLM judge Retrieval F1 | 0.2074 | 230 |

## 4. Original LinearRAG 结果来源

- 原始目录：`原始数据/baseline_qwen2.5-7b_full-1000`
- Question count: 1000；原始 retrieval top-k: 8；chunk token size: 1000；chunk overlap: 100。
- 人工语义准确率：492/1000 = 0.4920。
- Relaxed local judge：478/1000 = 0.4780；Contain-Acc = 0.2660；Exact = 0.2050；Token-F1 = 0.3218；ROUGE-L-F1 = 0.3167。
- 检索 Recall@k 使用 passage id 计算；gold evidence 来源为自动词法候选，不是人工最终标注。

## 5. Original LinearRAG 按书准确率

| Book | Correct | Total | Manual Accuracy |
|---|---:|---:|---:|
| book1 | 77 | 143 | 0.5385 |
| book2 | 72 | 143 | 0.5035 |
| book3 | 85 | 143 | 0.5944 |
| book4 | 72 | 143 | 0.5035 |
| book5 | 84 | 143 | 0.5874 |
| book6 | 53 | 143 | 0.3706 |
| book7 | 49 | 142 | 0.3451 |

## 6. 结论与限制

1. 在已有人工语义口径下，Original LinearRAG 的 Answer Accuracy 为 49.20%，高于 Closed-book 的 3.20% 和 Dense RAG 的 18.30%。
2. Original LinearRAG 的 Recall@5 为 8.17%，Recall@8 为 12.23%；这两个数依赖自动词法候选证据，不能替代人工 gold evidence 评测。
3. Original LinearRAG 的 LLM RAG judge 结果只有 230 题，不能代表完整 1000 题；延迟数据完全缺失。
4. 三个 baseline 使用的生成模型不同，且 Dense RAG 与 Original LinearRAG 的 top-k 分别为 5 和 8，所以当前表是“现有实验数据汇总”，不是已经完成的严格控制变量实验。
