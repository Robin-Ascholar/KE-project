# LinearRAG 综合评测说明

## 论文正式使用的四项质量指标

LinearRAG 论文在第 4.1 节明确将质量评测分成两类，共四项指标：

| 类别 | 指标 | 本项目字段 | 定义 |
|---|---|---|---|
| 生成质量 | Contain-Acc | `paper_contain_accuracy` | 规范化后参考答案是否出现在模型回答中 |
| 生成质量 | GPT-Acc | `local_judge_accuracy` | LLM 判断回答是否包含参考答案关键信息、事实一致且无矛盾 |
| 检索质量 | Context Relevance | `context_relevance` | 问题与检索上下文的语义相关程度，裁判原始等级 0/1/2 再归一化到 0–1 |
| 检索质量 | Evidence Recall | `evidence_recall` | 参考答案事实中有多少能由检索上下文支持 |

论文使用 GPT-4o-mini 进行生成和 LLM 评测。本项目默认用本地 Ollama `qwen3.6:27b` 作为裁判，因此字段名称保留 `local_judge`，其绝对分数不能直接冒充论文的 GPT-4o-mini 数值。

## 扩展诊断指标

论文没有把“多跳路径覆盖率”作为单独的表格指标。本项目按照实验需要增加了路径、上下文纯度、答案支撑和排序诊断：

| 指标 | 字段 | 计算方式 |
|---|---|---|
| Context Precision | `context_precision` | 对每个 Top-k passage 判定是否贡献必要证据，相关 passage 数除以 k |
| Retrieval F1 | `retrieval_f1` | Context Precision 与 Evidence Recall 的调和平均 |
| Evidence Path Coverage | `evidence_path_coverage` | 已被上下文支持的证据链步骤数除以全部步骤数 |
| Complete Path Rate | `evidence_path_complete` | 全部路径步骤和参考答案事实都被覆盖时为 1 |
| Multi-hop Path Coverage | `multi_hop_path_coverage` | 仅对两步及以上证据链计算路径覆盖率 |
| Hit Rate | `retrieval_hit` | Top-k 是否至少有一个相关 passage |
| MRR | `retrieval_mrr` | 第一个相关 passage 排名的倒数 |
| nDCG | `retrieval_ndcg` | 按 passage 排名计算的二值归一化折损累计增益 |
| Answer Fact Coverage | `answer_fact_coverage` | 参考答案事实中有多少出现在模型回答中 |
| Faithfulness | `answer_faithfulness` | 模型回答中的原子事实有多少能由检索上下文直接支持 |
| Exact Match | `answer_exact_match` | 规范化后的回答与参考答案是否完全相同 |
| Token F1 | `answer_token_f1` | 规范化 token 的 precision/recall 调和平均 |
| ROUGE-L F1 | `answer_rouge_l_f1` | 基于最长公共子序列的 F1 |
| Semantic Similarity | `answer_semantic_similarity` | `all-mpnet-base-v2` 余弦相似度映射到 0–1 |

Context Relevance 是整组上下文的总体相关性；Context Precision 是逐 passage 的有效证据比例，二者不能互换。Evidence Recall 检查参考答案事实，Path Coverage 检查得到答案所需的有序证据步骤，二者也分别保留。

## 证据链标注

评分器优先使用问题记录中的人工标注，支持以下任一字段：`reasoning_path`、`supporting_facts`、`gold_evidence`、`evidence`。推荐格式：

```json
{
  "question": "Which object transported Harry and Cedric, and why?",
  "answer": "The Triwizard Cup, because it was a Portkey",
  "reasoning_path": [
    "Harry and Cedric touched the Triwizard Cup together",
    "Touching the Cup transported them away",
    "The Triwizard Cup was a Portkey that took them to the graveyard"
  ]
}
```

`run.py` 会把这些标注复制到 `predictions.json`。如果没有人工标注，裁判会根据问题、参考答案和检索上下文生成最短证据链，结果中的 `path_source` 为 `judge_derived`。派生路径适合诊断，严格的多跳 benchmark 对比应使用人工 gold path。

路径计分强制最后一步对应参考答案事实；只检索到前置线索但没有检索到答案关系时，完整路径不能得满分。

## 运行命令

```bash
conda activate linearrag
python scripts/evaluate_predictions.py \
  --predictions results/harrypotter/2026-09-07_15-06-11/predictions.json \
  --provider ollama \
  --judge-model qwen3.6:27b \
  --workers 1 \
  --num-ctx 32768 \
  --embedding-model model/all-mpnet-base-v2 \
  --embedding-device cpu \
  --output results/harrypotter/2026-09-07_15-06-11/predictions.comprehensive-scored.qwen3.6-27b.json \
  --summary results/harrypotter/2026-09-07_15-06-11/evaluation_results.comprehensive.qwen3.6-27b.json
```

逐题文件保留裁判原始 JSON、passage 标签、参考事实、回答事实、证据链和解析错误；汇总文件只平均完整解析的 RAG 记录，并单独报告失败数。

## 当前 14 题结果

这些回答来自历史 `qwen2.5:7b` 运行，裁判使用本地 `qwen3.6:27b`：

| 指标 | 结果 |
|---|---:|
| Contain-Acc | 42.86% |
| Local Judge Accuracy | 42.86% |
| Context Relevance | 64.29% |
| Evidence Recall | 50.00% |
| Context Precision | 7.50% |
| Retrieval F1 | 10.68% |
| Evidence Path Coverage | 58.93% |
| Complete Path Rate | 50.00% |
| Multi-hop Path Coverage | 75.00%（7 条裁判派生多步路径） |
| Hit Rate | 64.29% |
| MRR | 16.56% |
| nDCG | 26.49% |
| Answer Fact Coverage | 42.86% |
| Faithfulness | 79.05% |
| Exact Match | 0.00% |
| Token F1 | 15.49% |
| ROUGE-L F1 | 15.49% |
| Semantic Similarity | 68.90% |

Exact Match 为零的主要原因是参考答案很短，而正确预测通常是包含答案的完整句子。Context Precision 较低说明 Top-20 带入了大量不直接贡献答案的段落；它与 50% 的 Evidence Recall 一起表明当前瓶颈主要在检索证据的准确性和完整性。

## 论文中的效率指标

论文还报告索引时间、平均检索时间、prompt token、completion token、图规模和复杂度。这些属于运行效率与可扩展性指标，不能从已有 `predictions.json` 准确反推，因此综合评分器不会伪造数值。后续严格效率实验应在运行时记录这些数据，并固定硬件、并发、Top-k、模型版本和语料规模。
