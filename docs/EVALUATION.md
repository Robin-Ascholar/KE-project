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
  --predictions results/harrypotter/batch_2026-09-07_qwen2.5-7b_full-1000/combined_predictions.json \
  --provider ollama \
  --judge-model qwen3.6:27b \
  --workers 1 \
  --num-ctx 32768 \
  --embedding-model model/all-mpnet-base-v2 \
  --embedding-device cpu \
  --output results/harrypotter/batch_2026-09-07_qwen2.5-7b_full-1000/combined_predictions.scored.json \
  --summary results/harrypotter/batch_2026-09-07_qwen2.5-7b_full-1000/evaluation_results.json
```

逐题文件保留裁判原始 JSON、passage 标签、参考事实、回答事实、证据链和解析错误；汇总文件只平均完整解析的 RAG 记录，并单独报告失败数。

> 说明：本项目最终采用的正确率来自**人工逐题语义评测**（回答与参考答案语义一致即判对），已写入 `combined_predictions.scored.json` 的 `local_judge_accuracy`/`local_judge_label` 与 `evaluation_results.deterministic.json`（总正确率 0.492）。自动打分（字符串包含与本地 LLM 裁判）结果存在偏差且已中断，不再作为结论指标；人工评测的错误模式分析见 [error_analysis/2026-09-08_qwen2.5-7b_full-1000_manual_review.md](../error_analysis/2026-09-08_qwen2.5-7b_full-1000_manual_review.md)。

## 当前 1,000 题人工评测结果

完整实验为 `dataset/question/book1.json`–`book7.json` 的 1,000 道题，问答模型 `qwen2.5:7b`。早期 14 题 run 与其自动打分已删除。

人工逐题语义评测（每题一个 C/W 判定，无评测脚本参与打分）：

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

判定口径：模型答案与参考答案语义一致即算对（含同义改写、别名、部分关键信息正确）；"上下文未提及/未说明"式回避明确事实、答错人物/数字/情节或答非所问均判错。错误模式与代表性逐题示例见 [error_analysis/2026-09-08_qwen2.5-7b_full-1000_manual_review.md](../error_analysis/2026-09-08_qwen2.5-7b_full-1000_manual_review.md)。

## 论文中的效率指标

论文还报告索引时间、平均检索时间、prompt token、completion token、图规模和复杂度。这些属于运行效率与可扩展性指标，不能从已有 `predictions.json` 准确反推，因此综合评分器不会伪造数值。后续严格效率实验应在运行时记录这些数据，并固定硬件、并发、Top-k、模型版本和语料规模。

## 批量运行七套问题

`dataset/question/book1.json` 到 `book7.json` 共包含 1,000 条 `question`/`answer` 记录。以下命令只初始化一次 embedding 模型和 `import/harrypotter` 图索引，随后按文件顺序逐套问答，最后合并并执行综合评分：

```bash
conda activate linearrag
python scripts/run_question_sets.py \
  --question-dir dataset/question \
  --dataset-name harrypotter \
  --llm-model deepseek-ocr:latest \
  --judge-model qwen3.6:27b
```

每 10 道题会保存一次问答检查点。输出目录形如 `results/harrypotter/batch_YYYY-MM-DD_HH-MM-SS/`，其中每套结果位于 `sets/bookN/predictions.json`，合并结果为 `combined_predictions.json`，评分明细和汇总分别为 `combined_predictions.scored.json` 与 `evaluation_results.json`。

如运行中断，使用原输出目录即可续跑问答和评分：

```bash
python scripts/run_question_sets.py \
  --output-dir results/harrypotter/batch_YYYY-MM-DD_HH-MM-SS
```

`run_manifest.json` 的 `status` 和各问题集的 `completed` 字段记录当前进度。仅做快速流程检查时可使用 `--sets book1 --limit-per-set 1`；只生成和合并回答可添加 `--no-evaluate`。
