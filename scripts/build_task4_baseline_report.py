#!/usr/bin/env python3
"""Build the Task 4 baseline comparison and analysis report.

The repository intentionally keeps this report's source values explicit.  The
large prediction artifacts from the formal Closed-book/Dense RAG run live in
historical commit 95ae490, while the Original LinearRAG 1,000-question run is
documented by task3's experiment notes and manual review.  Missing metrics are
reported as null/N/A instead of being inferred from unrelated runs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "task4_formal"


BASELINES: list[dict[str, Any]] = [
    {
        "method": "Closed-book",
        "method_id": "closed_book",
        "question_count": 1000,
        "answer_accuracy": 0.032,
        "answer_accuracy_correct": 32,
        "answer_accuracy_source": "lenient_manual_semantic_review",
        "automatic_answer_accuracy": 0.009,
        "contain_accuracy": 0.022,
        "evidence_recall_at_5": None,
        "evidence_precision_at_5": None,
        "llm_rag_judge_evidence_recall": None,
        "retrieval_hit_at_5": None,
        "retrieval_mrr_at_5": None,
        "retrieval_ndcg_at_5": None,
        "retrieval_f1_at_5": None,
        "context_relevance": None,
        "context_precision": None,
        "answer_fact_coverage": None,
        "answer_faithfulness": None,
        "exact_match": 0.003,
        "token_f1": 0.04729639976620908,
        "rouge_l_f1": 0.04552885968539348,
        "semantic_similarity": None,
        "retrieval_latency_ms_avg": 0.0,
        "retrieval_latency_ms_p50": 0.0,
        "retrieval_latency_ms_p95": 0.0,
        "total_latency_ms_avg": 111.44387562200245,
        "total_latency_ms_p50": 93.4729470000093,
        "total_latency_ms_p95": 281.1647536502731,
        "generator": "qwen2.5:0.5b",
        "embedding": None,
        "dense_backend": None,
        "top_k": None,
        "retrieval_top_k": 5,
        "notes": "No retrieval context; retrieval metrics are not applicable.",
        "source": "95ae490:results/task4_formal/closed_book + manual_review_summary.json",
        "by_book": {
            "book1": [6, 143],
            "book2": [6, 143],
            "book3": [7, 143],
            "book4": [2, 143],
            "book5": [5, 143],
            "book6": [2, 143],
            "book7": [4, 142],
        },
    },
    {
        "method": "Dense RAG",
        "method_id": "dense_rag",
        "question_count": 1000,
        "answer_accuracy": 0.183,
        "answer_accuracy_correct": 183,
        "answer_accuracy_source": "lenient_manual_semantic_review",
        "automatic_answer_accuracy": 0.031,
        "contain_accuracy": 0.128,
        "evidence_recall_at_5": 0.11933333333333299,
        "evidence_precision_at_5": 0.0716,
        "llm_rag_judge_evidence_recall": None,
        "retrieval_hit_at_5": 0.293,
        "retrieval_mrr_at_5": 0.18241666666666664,
        "retrieval_ndcg_at_5": 0.2082933797087101,
        "retrieval_f1_at_5": 0.0895,
        "context_relevance": None,
        "context_precision": None,
        "answer_fact_coverage": None,
        "answer_faithfulness": None,
        "exact_match": 0.075,
        "token_f1": 0.17134867748509947,
        "rouge_l_f1": 0.1687471298859308,
        "semantic_similarity": None,
        "retrieval_latency_ms_avg": 43.26670132700201,
        "retrieval_latency_ms_p50": 40.46323600005053,
        "retrieval_latency_ms_p95": 60.493705200724435,
        "total_latency_ms_avg": 313.1215288910207,
        "total_latency_ms_p50": 297.52942399909443,
        "total_latency_ms_p95": 468.9858695494875,
        "generator": "qwen2.5:0.5b",
        "embedding": "model/all-mpnet-base-v2",
        "dense_backend": "sentence_transformer",
        "top_k": 5,
        "retrieval_top_k": 5,
        "notes": "Gold evidence is an automatically generated lexical candidate annotation.",
        "source": "95ae490:results/task4_formal/dense_rag + manual_review_summary.json",
        "by_book": {
            "book1": [33, 143],
            "book2": [31, 143],
            "book3": [30, 143],
            "book4": [25, 143],
            "book5": [24, 143],
            "book6": [19, 143],
            "book7": [21, 142],
        },
    },
    {
        "method": "Original LinearRAG",
        "method_id": "original_linearrag",
        "question_count": 1000,
        "answer_accuracy": 0.492,
        "answer_accuracy_correct": 492,
        "answer_accuracy_source": "manual_semantic_review_task3",
        "automatic_answer_accuracy": None,
        "contain_accuracy": None,
        "evidence_recall_at_5": None,
        "evidence_precision_at_5": None,
        "llm_rag_judge_evidence_recall": None,
        "retrieval_hit_at_5": None,
        "retrieval_mrr_at_5": None,
        "retrieval_ndcg_at_5": None,
        "retrieval_f1_at_5": None,
        "context_relevance": None,
        "context_precision": None,
        "answer_fact_coverage": None,
        "answer_faithfulness": None,
        "exact_match": None,
        "token_f1": None,
        "rouge_l_f1": None,
        "semantic_similarity": None,
        "retrieval_latency_ms_avg": None,
        "retrieval_latency_ms_p50": None,
        "retrieval_latency_ms_p95": None,
        "total_latency_ms_avg": None,
        "total_latency_ms_p50": None,
        "total_latency_ms_p95": None,
        "generator": "qwen2.5:7b",
        "embedding": "model/all-mpnet-base-v2",
        "dense_backend": "LinearRAG vectorized graph retrieval",
        "top_k": 20,
        "retrieval_top_k": 20,
        "notes": "Task3 retained the manual answer review and run configuration, but not the 1,000-row prediction/evidence/timing artifacts.",
        "source": "task3 README.md, docs/EXPERIMENT.md, error_analysis/*",
        "by_book": {
            "book1": [77, 143],
            "book2": [72, 143],
            "book3": [85, 143],
            "book4": [72, 143],
            "book5": [84, 143],
            "book6": [53, 143],
            "book7": [49, 142],
        },
    },
]


def pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.2f}%"


def ms(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def write_json() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "baseline_comparison.json").write_text(
        json.dumps(BASELINES, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_manifest() -> None:
    manifest = {
        "task": "task4_baseline_comparison",
        "status": "historical_result_synthesis",
        "question_count": 1000,
        "methods": ["closed_book", "dense_rag", "original_linearrag"],
        "formal_closed_dense_source": "git:95ae490:results/task4_formal",
        "original_linearrag_source": [
            "task3:README.md",
            "task3:docs/EXPERIMENT.md",
            "task3:docs/CHAPTER_RETRIEVAL.md",
            "task3:错误分析/2026-09-08_qwen2.5-7b_full-1000_manual_review.md",
        ],
        "warning": (
            "Original LinearRAG 1,000-row predictions and per-question timing were not retained; "
            "unavailable metrics are null/N/A in the report."
        ),
        "gold_evidence_status": "auto_lexical_candidate_for_dense_rag; not human-audited",
        "generated_by": "scripts/build_task4_baseline_report.py",
    }
    (OUT / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_csv() -> None:
    fields = [
        "method_id", "question_count", "answer_accuracy", "answer_accuracy_source",
        "automatic_answer_accuracy", "contain_accuracy", "evidence_recall_at_5",
        "evidence_precision_at_5",
        "llm_rag_judge_evidence_recall", "retrieval_hit_at_5", "retrieval_mrr_at_5",
        "retrieval_ndcg_at_5", "retrieval_f1_at_5", "context_relevance",
        "context_precision", "answer_fact_coverage", "answer_faithfulness", "exact_match",
        "token_f1", "rouge_l_f1", "semantic_similarity", "retrieval_latency_ms_avg",
        "retrieval_latency_ms_p50", "retrieval_latency_ms_p95", "total_latency_ms_avg",
        "total_latency_ms_p50", "total_latency_ms_p95", "generator", "embedding",
        "dense_backend", "retrieval_top_k", "source",
    ]
    with (OUT / "baseline_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in BASELINES)


def write_report() -> None:
    dense = BASELINES[1]
    closed = BASELINES[0]
    original = BASELINES[2]
    lines = [
        "# Task 4：Baseline 与统一对照实验分析",
        "",
        "> 本文是基于当前仓库、task3 文档和历史正式实验产物生成的结果汇总。缺失的逐题产物不会被用其他实验推断，统一记为 `N/A`。",
        "",
        "## 1. 结论摘要",
        "",
        f"在 1,000 道问题上，Closed-book 的人工语义正确率为 **{pct(closed['answer_accuracy'])}**，Dense RAG 为 **{pct(dense['answer_accuracy'])}**，Original LinearRAG 为 **{pct(original['answer_accuracy'])}**。Dense RAG 相比 Closed-book 提升 **{(dense['answer_accuracy'] - closed['answer_accuracy']) * 100:.2f} 个百分点**；Original LinearRAG 比 Dense RAG 高 **{(original['answer_accuracy'] - dense['answer_accuracy']) * 100:.2f} 个百分点**。",
        "",
        "但三者当前历史结果**不能直接作为严格公平的最终对照结论**：Closed-book/Dense RAG 使用 `qwen2.5:0.5b`、Top-5，Original LinearRAG 使用 `qwen2.5:7b`、Top-20。模型规模和上下文数量同时变化，因此 Original LinearRAG 的优势不能全部归因于检索方法。",
        "",
        "## 2. 三个 Baseline 结果",
        "",
        "### 2.1 核心指标",
        "",
        "| Method | N | Answer Accuracy | Contain-Acc | Exact Match | Token F1 | ROUGE-L F1 | Evidence Recall@5 | LLM RAG Judge Evidence Recall | Retrieval Avg (ms) | Total Avg (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in BASELINES:
        lines.append(
            f"| {row['method']} | {row['question_count']} | {pct(row['answer_accuracy'])} | {pct(row['contain_accuracy'])} | {pct(row['exact_match'])} | {pct(row['token_f1'])} | {pct(row['rouge_l_f1'])} | {pct(row['evidence_recall_at_5'])} | {pct(row['llm_rag_judge_evidence_recall'])} | {ms(row['retrieval_latency_ms_avg'])} | {ms(row['total_latency_ms_avg'])} |"
        )
    lines.extend([
        "",
        "说明：Exact Match、Token F1、ROUGE-L 和 Dense RAG 的 passage-id 检索指标均由历史正式预测文件逐题重算；Closed-book 没有检索过程，因此 Evidence Recall@5 和 retrieval metrics 不适用；Original LinearRAG 的历史 1,000 题逐题结果未保留，不能补算这些指标。",
        "",
        "### 2.2 Dense RAG 检索诊断",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Evidence Precision@5 | {pct(dense['evidence_precision_at_5'])} |",
        f"| Hit@5 | {pct(dense['retrieval_hit_at_5'])} |",
        f"| MRR@5 | {pct(dense['retrieval_mrr_at_5'])} |",
        f"| nDCG@5 | {pct(dense['retrieval_ndcg_at_5'])} |",
        f"| Retrieval F1@5 | {pct(dense['retrieval_f1_at_5'])} |",
        "",
        "这些值是针对自动词法候选 gold evidence 的诊断结果，不是人工确认 gold evidence 上的最终结论。",
        "",
        "### 2.3 分册人工答案正确率",
        "",
        "| Book | Closed-book | Dense RAG | Original LinearRAG |",
        "|---|---:|---:|---:|",
    ])
    for book in ("book1", "book2", "book3", "book4", "book5", "book6", "book7"):
        values = []
        for row in BASELINES:
            correct, total = row["by_book"][book]
            values.append(f"{correct}/{total} ({correct / total * 100:.1f}%)")
        lines.append(f"| {book} | {values[0]} | {values[1]} | {values[2]} |")
    lines.extend([
        "| **Total** | **32/1000 (3.2%)** | **183/1000 (18.3%)** | **492/1000 (49.2%)** |",
        "",
        "Dense RAG 在七册上均高于 Closed-book，说明引入文本检索确实提供了有效信息。Original LinearRAG 在所有分册也高于 Dense RAG，但由于 generator 不同，不能把该差异视作纯检索增益。",
        "",
        "## 3. 实验控制变量",
        "",
        "| Variable | Closed-book | Dense RAG | Original LinearRAG | 统一性判断 |",
        "|---|---|---|---|---|",
        "| Question set | 7 books, 1,000 questions | 同左 | 同左 | 一致 |",
        "| Corpus | 6,865 passages, 199 chapters | 同左 | 同左 | 一致 |",
        "| Chunking | 300 target / 360 max / 80 min / 2 sentence overlap | 同左 | 同左 | 一致 |",
        "| Generator | `qwen2.5:0.5b` | `qwen2.5:0.5b` | `qwen2.5:7b` | 不一致 |",
        "| Prompt | concise answer-only | context + concise answer | LinearRAG QA prompt | 不一致 |",
        "| Top-k | 0 context | 5 | 20 | 不一致 |",
        "| Embedding | 不使用 | `all-mpnet-base-v2` | `all-mpnet-base-v2` | 部分一致 |",
        "| Temperature | 0 | 0 | 文档未完整记录，历史 run 使用本地固定配置 | 需补记录 |",
        "| Chapter retrieval | 不适用 | 不适用 | 关闭，属于 Original baseline | 一致 |",
        "",
        "任务 4 的严格重跑应把三种方法都固定到同一个生成模型、同一 prompt 规范、同一 Top-k、同一 generation 参数。当前报告保留历史结果，不把不同配置伪装成严格公平实验。",
        "",
        "## 4. Evidence Recall 的两种定义",
        "",
        "### 4.1 Evidence Recall@k：passage-id 检索指标",
        "",
        "定义：",
        "",
        "```text",
        "Recall@k = |Gold passage IDs ∩ Top-k retrieved passage IDs| / |Gold passage IDs|",
        "```",
        "",
        f"Dense RAG 的历史正式结果为 **{pct(dense['evidence_recall_at_5'])}**。但其 gold evidence 来源是 `auto_lexical_candidate`，只能作为流程验证和弱监督候选，不能称为人工确认的最终证据标注。",
        "",
        "Closed-book 没有检索结果，所以该指标为 `N/A`。Original LinearRAG 当前缺少逐题 `gold_evidence` 和 retrieved passage artifact，因此为 `N/A`。",
        "",
        "### 4.2 LLM RAG Judge Evidence Recall：语义支撑指标",
        "",
        "该指标由 `scripts/evaluate_predictions.py` / `src/rag_metrics.py` 让裁判模型拆分 reference facts，并判断这些事实是否能被 Top-k 上下文支持：",
        "",
        "```text",
        "LLM RAG Judge Evidence Recall = supported reference facts / all reference facts",
        "```",
        "",
        "它与 passage-id Recall@k 不是同一个指标，不能互换。当前 Task 4 正式 Closed-book/Dense RAG 运行只生成了 passage-id Recall@5，没有执行这套 JSON RAG judge；Original LinearRAG 也没有保留对应逐题文件，因此三个 baseline 的该项均为 `N/A`。",
        "",
        "## 5. 延迟分析",
        "",
        f"Closed-book 平均总延迟为 **{ms(closed['total_latency_ms_avg'])} ms**，Dense RAG 平均总延迟为 **{ms(dense['total_latency_ms_avg'])} ms**，增加 **{dense['total_latency_ms_avg'] - closed['total_latency_ms_avg']:.2f} ms**。Dense RAG 的平均检索延迟为 **{ms(dense['retrieval_latency_ms_avg'])} ms**，约占总延迟的 **{dense['retrieval_latency_ms_avg'] / dense['total_latency_ms_avg'] * 100:.1f}%**；主要成本仍然来自 LLM generation。",
        "",
        "Original LinearRAG 的历史日志记录了图规模和运行配置，但没有保留逐题 retrieval/total latency，因此不报告伪造的平均值。后续需要在 `retrieve()` 和 `qa()` 周围直接记录计时，并区分 index、retrieval、generation、total 四个阶段。",
        "",
        "## 6. 指标覆盖矩阵",
        "",
        "下表中的 `N/A` 表示当前没有可审计的输入或该方法不适用，不表示分数为 0。",
        "",
        "| Metric | Closed-book | Dense RAG | Original LinearRAG | 当前状态 |",
        "|---|---:|---:|---:|---|",
        "| Answer Accuracy | 已有 | 已有 | 已有 | 人工语义评测 |",
        "| Contain-Acc | 已有 | 已有 | N/A | Original 缺逐题答案产物 |",
        "| Evidence Recall@5 | N/A | 已有 | N/A | Closed 不适用；Original 缺证据 ID |",
        "| Evidence Precision/Hit/MRR/nDCG/Retrieval F1@5 | N/A | 已补齐 | N/A | 从 Dense RAG 逐题 Top-5 重算 |",
        "| LLM RAG Judge Evidence Recall | N/A | N/A | N/A | 未执行 JSON RAG judge |",
        "| Context Relevance/Precision | N/A | N/A | N/A | 未执行 RAG judge |",
        "| Answer Fact Coverage/Faithfulness | N/A | N/A | N/A | 未执行 RAG judge |",
        "| Exact Match/Token F1/ROUGE-L | 已补齐 | 已补齐 | N/A | 从 Closed/Dense 逐题预测重算 |",
        "| Semantic Similarity | N/A | N/A | N/A | 当前环境缺少 sentence-transformers |",
        "| Retrieval/Total Latency | 部分 | 已有 | N/A | Original 缺逐题计时 |",
        "",
        "## 7. Gold 标注与数据泄露检查",
        "",
        "- `gold_answer`、`gold_evidence`、`gold_book`、`gold_chapter` 可以保存在问题记录中，但只能用于评测。",
        "- `run_task4_minimal_baselines.py` 的正式逻辑没有把 `gold_book` 或 `gold_chapter` 作为 Dense RAG 的检索过滤条件。",
        "- Original LinearRAG 的 `retrieve()` 只使用问题 embedding、NER seed、图传播和可选配置；task 4 baseline 必须保持 `enable_chapter_retrieval=False`。",
        "- Dense RAG 的 `sentence_transformer` backend 已在历史正式 run 中使用；`tfidf_fallback` 只出现在早期 70 题 smoke test，不应作为正式 Dense RAG 结论。",
        "- gold evidence 当前是自动词法候选，正式论文结论前仍需要逐题人工审计或建立独立标注文件。",
        "",
        "## 8. 复现与后续补齐",
        "",
        "当前正式历史数据来源：",
        "",
        "- Closed-book/Dense RAG：Git commit `95ae490` 的 `results/task4_formal/`。",
        "- Original LinearRAG：task3 分支的 `README.md`、`docs/EXPERIMENT.md`、`docs/CHAPTER_RETRIEVAL.md` 和 `错误分析/2026-09-08_qwen2.5-7b_full-1000_manual_review.md`。",
        "- 统一评测实现：`scripts/evaluate_predictions.py`、`src/paper_evaluator.py`、`src/rag_metrics.py`。",
        "",
        "建议严格最终版统一为：同一 `qwen2.5:7b`、同一 answer-only prompt、同一 Top-5、同一 `all-mpnet-base-v2`、temperature 0、同一 1,000 题顺序；Original LinearRAG 关闭 chapter retrieval，Dense RAG 只做向量 Top-5，Closed-book 不注入上下文。然后保存三套逐题 JSON，再执行统一评测器，才能补齐上表所有 N/A 指标。",
        "",
        "## 9. 产物",
        "",
        "- `results/task4_formal/baseline_comparison.json`：结构化结果汇总。",
        "- `results/task4_formal/baseline_comparison.csv`：表格导出。",
        "- `results/task4_formal/baseline_comparison.md`：简表。",
        "- `results/task4_formal/TASK4_BASELINE_ANALYSIS.md`：本文完整分析。",
    ])
    (OUT / "TASK4_BASELINE_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    short = [
        "# Task 4 baseline results",
        "",
        "| Method | Questions | Answer Accuracy | Contain-Acc | Evidence Recall@5 | Retrieval Avg(ms) | Total Avg(ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in BASELINES:
        short.append(
            f"| {row['method']} | {row['question_count']} | {pct(row['answer_accuracy'])} | {pct(row['contain_accuracy'])} | {pct(row['evidence_recall_at_5'])} | {ms(row['retrieval_latency_ms_avg'])} | {ms(row['total_latency_ms_avg'])} |"
        )
    short.extend(["", "See `TASK4_BASELINE_ANALYSIS.md` for configuration, metric definitions, limitations, and source provenance."])
    (OUT / "baseline_comparison.md").write_text("\n".join(short) + "\n", encoding="utf-8")


def main() -> None:
    write_json()
    write_manifest()
    write_csv()
    write_report()
    print(f"wrote {OUT / 'TASK4_BASELINE_ANALYSIS.md'}")
    print(f"wrote {OUT / 'baseline_comparison.json'}")
    print(f"wrote {OUT / 'baseline_comparison.csv'}")
    print(f"wrote {OUT / 'baseline_comparison.md'}")


if __name__ == "__main__":
    main()
