#!/usr/bin/env python3
"""Complete Task 4 baseline metrics from the existing 1000-question runs.

This script does not rerun an LLM.  It combines the existing Closed-book and
Dense RAG artifacts with the archived Original LinearRAG run, computes the
deterministic answer/retrieval metrics that the files support, and records
partial RAG-judge metrics separately from full-dataset retrieval metrics.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import string
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "results" / "task4_formal"
LINEAR_SOURCE = ROOT / "原始数据" / "baseline_qwen2.5-7b_full-1000"
LINEAR_OUT = FORMAL / "original_linearrag"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def normalize(value: object) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(character for character in text if character not in set(string.punctuation))
    return " ".join(text.split())


def contain(predicted: object, gold: object) -> float:
    pred = normalize(predicted)
    ref = normalize(gold)
    return float(bool(pred and ref and ref in pred))


def exact(predicted: object, gold: object) -> float:
    return float(normalize(predicted) == normalize(gold))


def token_f1(predicted: object, gold: object) -> float:
    pred = normalize(predicted).split()
    ref = normalize(gold).split()
    if not pred or not ref:
        return float(pred == ref)
    pred_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for token in pred:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    overlap = sum(min(count, ref_counts.get(token, 0)) for token, count in pred_counts.items())
    if not overlap:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(predicted: object, gold: object) -> float:
    pred = normalize(predicted).split()
    ref = normalize(gold).split()
    if not pred or not ref:
        return float(pred == ref)
    previous = [0] * (len(ref) + 1)
    for token in pred:
        current = [0]
        for index, ref_token in enumerate(ref, start=1):
            if token == ref_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    lcs = previous[-1]
    if not lcs:
        return 0.0
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def retrieval_metrics(retrieved: list[int], gold: set[int], k: int) -> dict[str, float | None]:
    if not gold:
        return {key: None for key in ("recall", "precision", "hit", "mrr", "ndcg", "f1")}
    ranked = retrieved[:k]
    labels = [int(passage_id in gold) for passage_id in ranked]
    relevant = sum(labels)
    recall = relevant / len(gold)
    precision = relevant / len(labels) if labels else 0.0
    hit = float(bool(relevant))
    first = next((rank for rank, label in enumerate(labels, start=1) if label), None)
    mrr = 1.0 / first if first else 0.0
    dcg = sum(label / math.log2(rank + 1) for rank, label in enumerate(labels, start=1))
    ideal = sorted(labels, reverse=True)
    idcg = sum(label / math.log2(rank + 1) for rank, label in enumerate(ideal, start=1))
    ndcg = dcg / idcg if idcg else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"recall": recall, "precision": precision, "hit": hit, "mrr": mrr, "ndcg": ndcg, "f1": f1}


def parse_ids(passages: list[Any]) -> list[int]:
    ids: list[int] = []
    for passage in passages:
        if isinstance(passage, dict) and passage.get("passage_id") is not None:
            ids.append(int(passage["passage_id"]))
            continue
        match = re.match(r"^\s*(\d+)\s*:", str(passage))
        if match:
            ids.append(int(match.group(1)))
    return ids


def summarize_rows(
    method: str,
    rows: list[dict[str, Any]],
    top_k: int,
    *,
    model: str,
    dense_backend: str | None,
    chunk_token_size: int | None,
    answer_accuracy_source: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scored: list[dict[str, Any]] = []
    answer_accuracy = []
    automatic_accuracy = []
    contains = []
    exact_scores = []
    f1_scores = []
    rouge_scores = []
    retrieval_by_k: dict[int, dict[str, list[float]]] = {
        k: {metric: [] for metric in ("recall", "precision", "hit", "mrr", "ndcg", "f1")}
        for k in (5, 8)
    }
    retrieval_latencies: list[float] = []
    total_latencies: list[float] = []

    for row in rows:
        gold = row.get("gold_answer", row.get("answer", ""))
        predicted = row.get("pred_answer", row.get("generated_answer", ""))
        contains.append(contain(predicted, gold))
        exact_scores.append(exact(predicted, gold))
        f1_scores.append(token_f1(predicted, gold))
        rouge_scores.append(rouge_l_f1(predicted, gold))
        if row.get("answer_accuracy") is not None:
            answer_accuracy.append(float(row["answer_accuracy"]))
        if row.get("automatic_answer_accuracy") is not None:
            automatic_accuracy.append(float(row["automatic_answer_accuracy"]))

        retrieved = parse_ids(row.get("retrieved_passages", row.get("sorted_passage", [])))
        gold_evidence = row.get("gold_evidence", [])
        gold_ids = {int(item["passage_id"]) for item in gold_evidence if item.get("passage_id") is not None}
        if method != "closed_book":
            for k in (5, 8):
                metrics = retrieval_metrics(retrieved, gold_ids, k)
                for metric, value in metrics.items():
                    if value is not None:
                        retrieval_by_k[k][metric].append(value)

        if row.get("retrieval_latency_ms") is not None:
            retrieval_latencies.append(float(row["retrieval_latency_ms"]))
        if row.get("total_latency_ms") is not None:
            total_latencies.append(float(row["total_latency_ms"]))

        enriched = dict(row)
        enriched["retrieved_passage_ids"] = retrieved
        enriched["paper_contain_accuracy"] = contains[-1]
        enriched["answer_exact_match"] = exact_scores[-1]
        enriched["answer_token_f1"] = f1_scores[-1]
        enriched["answer_rouge_l_f1"] = rouge_scores[-1]
        for k in (5, 8):
            metrics = retrieval_metrics(retrieved, gold_ids, k) if method != "closed_book" else {
                metric: None for metric in ("recall", "precision", "hit", "mrr", "ndcg", "f1")
            }
            for metric, value in metrics.items():
                enriched[f"retrieval_{metric}_at_{k}"] = value
        scored.append(enriched)

    def avg(values: list[float]) -> float | None:
        return mean(values) if values else None

    summary: dict[str, Any] = {
        "method": method,
        "model": model,
        "dense_backend": dense_backend,
        "question_count": len(rows),
        "top_k_native": top_k if method != "closed_book" else None,
        "chunk_token_size": chunk_token_size,
        "answer_accuracy": avg(answer_accuracy),
        "answer_accuracy_source": answer_accuracy_source,
        "automatic_answer_accuracy": avg(automatic_accuracy),
        "contain_accuracy": avg(contains),
        "answer_exact_match": avg(exact_scores),
        "answer_token_f1": avg(f1_scores),
        "answer_rouge_l_f1": avg(rouge_scores),
        "evidence_annotation_source": "auto_lexical_candidate" if method != "closed_book" else "auto_lexical_candidate_unused",
        "evidence_recall_evaluable_count": len(rows) if method != "closed_book" else 0,
        "retrieval_latency_ms_avg": avg(retrieval_latencies),
        "retrieval_latency_ms_p50": percentile(retrieval_latencies, 0.50),
        "retrieval_latency_ms_p95": percentile(retrieval_latencies, 0.95),
        "total_latency_ms_avg": avg(total_latencies),
        "total_latency_ms_p50": percentile(total_latencies, 0.50),
        "total_latency_ms_p95": percentile(total_latencies, 0.95),
    }
    for k in (5, 8):
        for metric, values in retrieval_by_k[k].items():
            summary[f"retrieval_{metric}_at_{k}"] = avg(values)
    return summary, scored


def add_partial_rag_judge_metrics(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    keys = (
        "context_relevance",
        "context_precision",
        "evidence_recall",
        "answer_fact_coverage",
        "answer_faithfulness",
        "evidence_path_coverage",
        "evidence_path_complete",
        "retrieval_hit",
        "retrieval_mrr",
        "retrieval_ndcg",
        "retrieval_f1",
    )
    valid = [row for row in rows if row.get("rag_judge_details") is not None]
    summary["llm_rag_judge_evaluable_count"] = len(valid)
    for key in keys:
        values = [float(row[key]) for row in valid if row.get(key) is not None]
        summary[f"llm_rag_{key}"] = mean(values) if values else None


def copy_linear_artifacts() -> None:
    LINEAR_OUT.mkdir(parents=True, exist_ok=True)
    for name in (
        "README.md",
        "run_manifest.json",
        "evaluation_results.deterministic.json",
        "evaluation_results.relaxed.json",
        "combined_predictions.json",
        "combined_predictions.deterministic-scored.json",
        "combined_predictions.relaxed_scored.json",
        "combined_predictions.scored.json",
    ):
        source = LINEAR_SOURCE / name
        if source.exists():
            shutil.copy2(source, LINEAR_OUT / name)


def main() -> None:
    questions = load(FORMAL / "questions_1000.json")
    closed_rows = load(FORMAL / "closed_book" / "predictions.json")
    dense_rows = load(FORMAL / "dense_rag" / "predictions.json")
    linear_rows = load(LINEAR_SOURCE / "combined_predictions.json")
    linear_scored_rows = load(LINEAR_SOURCE / "combined_predictions.scored.json")
    linear_relaxed_scored_rows = load(LINEAR_SOURCE / "combined_predictions.relaxed_scored.json")
    deterministic = load(LINEAR_SOURCE / "evaluation_results.deterministic.json")
    relaxed = load(LINEAR_SOURCE / "evaluation_results.relaxed.json")

    if len(linear_rows) != len(questions) or any(
        row.get("question") != question.get("question") or row.get("gold_answer") != question.get("answer")
        for row, question in zip(linear_rows, questions)
    ):
        raise ValueError("Original LinearRAG rows do not align with Task 4 question set")

    # Attach the existing Task 4 gold annotations to Original LinearRAG rows.
    linear_for_metrics = []
    for row, question in zip(linear_rows, questions):
        item = dict(row)
        item["gold_evidence"] = question.get("gold_evidence", [])
        item["gold_book"] = question.get("gold_book")
        item["gold_chapter"] = question.get("gold_chapter")
        linear_for_metrics.append(item)

    closed_summary, closed_scored = summarize_rows(
        "closed_book", closed_rows, 0, model="qwen2.5:0.5b", dense_backend=None,
        chunk_token_size=None, answer_accuracy_source="lenient_manual_semantic_review",
    )
    dense_summary, dense_scored = summarize_rows(
        "dense_rag", dense_rows, 5, model="qwen2.5:0.5b", dense_backend="sentence_transformer",
        chunk_token_size=None, answer_accuracy_source="lenient_manual_semantic_review",
    )
    linear_summary, linear_scored = summarize_rows(
        "original_linearrag", linear_for_metrics, 8, model="qwen2.5:7b",
        dense_backend="sentence_transformer", chunk_token_size=1000,
        answer_accuracy_source="human_manual_semantic_review",
    )
    linear_summary["answer_accuracy"] = float(deterministic["manual_semantic_accuracy"])
    linear_summary["manual_review_correct"] = int(deterministic["correct"])
    linear_summary["manual_review_question_count"] = int(deterministic["sample_count"])
    linear_summary["relaxed_local_judge_accuracy"] = float(relaxed["metrics"]["local_judge_accuracy"])
    # The archived relaxed evaluation is the authoritative aggregate snapshot.
    # It differs from a fresh recomputation on one row, so retain its exact
    # published values in the formal comparison table.
    for field, metric_name in (
        ("contain_accuracy", "paper_contain_accuracy"),
        ("answer_exact_match", "answer_exact_match"),
        ("answer_token_f1", "answer_token_f1"),
        ("answer_rouge_l_f1", "answer_rouge_l_f1"),
    ):
        linear_summary[field] = float(relaxed["metrics"][metric_name])
    for target, source in zip(linear_scored, linear_relaxed_scored_rows):
        for field in ("paper_contain_accuracy", "answer_exact_match", "answer_token_f1", "answer_rouge_l_f1"):
            target[field] = source.get(field)
    linear_summary["rag_judge_source_note"] = "230-row prefix only; remaining 770 rows have no RAG judge output"
    add_partial_rag_judge_metrics(linear_summary, linear_scored_rows)
    linear_summary["source_data_dir"] = str(LINEAR_SOURCE.relative_to(ROOT))
    linear_summary["retrieval_latency_note"] = "Unavailable: source predictions contain no retrieval timestamps"

    # Preserve the existing two baseline summaries as much as possible, but add
    # the same deterministic answer/retrieval fields used for LinearRAG.
    summaries = [closed_summary, dense_summary, linear_summary]
    for summary in summaries:
        summary["task4_gold_evidence_note"] = (
            "gold_evidence is auto_lexical_candidate and requires manual audit; "
            "it is not a human final annotation"
        )
    copy_linear_artifacts()
    write_json(LINEAR_OUT / "predictions.task4_scored.json", linear_scored)
    write_json(LINEAR_OUT / "metrics.json", linear_summary)

    manifest = load(FORMAL / "run_manifest.json")
    manifest["methods"] = ["closed_book", "dense_rag", "original_linearrag"]
    manifest["excluded_methods"] = []
    manifest["original_linearrag_source"] = str(LINEAR_SOURCE.relative_to(ROOT))
    manifest["original_linearrag_manifest"] = "original_linearrag/run_manifest.json"
    manifest["metric_completion_script"] = "scripts/complete_task4_linear_rag_metrics.py"
    manifest["metric_completion_note"] = (
        "Original LinearRAG answer metrics come from existing deterministic/relaxed files; "
        "retrieval ID metrics use the existing auto lexical candidate annotations; "
        "latency is unavailable for the archived run."
    )
    write_json(FORMAL / "run_manifest.json", manifest)

    write_json(FORMAL / "baseline_comparison.json", summaries)
    fields = sorted({key for row in summaries for key in row})
    with (FORMAL / "baseline_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)

    lines = [
        "# Task 4 Baseline Comparison",
        "",
        "本报告基于仓库现有实验文件补全指标，没有重新调用 LLM。三种 baseline 的原始模型并不一致：Closed-book/Dense RAG 使用 `qwen2.5:0.5b`，Original LinearRAG 使用 `qwen2.5:7b`，因此表中结果可以作为已有实验汇总，但不能直接宣称为严格同模型对照。",
        "",
        "## 1. 统一结果",
        "",
        "`Answer Accuracy` 使用已有人工语义复核；`Contain-Acc`、Exact Match、Token-F1、ROUGE-L-F1 为逐题确定性计算。检索 ID 指标使用 `results/task4_formal/questions_1000.json` 中已有的 `auto_lexical_candidate` 证据标注，必须人工审计后才能作为最终 benchmark gold。",
        "",
        "| Method | N | Model | Native Top-k | Answer Acc | Contain-Acc | Exact | Token-F1 | ROUGE-L-F1 | Recall@5 | Recall@8 | Hit@5 | MRR@5 | NDCG@5 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        fmt = lambda key: "N/A" if row.get(key) is None else f"{row[key]:.4f}"
        top_k = "N/A" if row.get("top_k_native") is None else str(row["top_k_native"])
        lines.append(
            f"| {row['method']} | {row['question_count']} | {row['model']} | {top_k} | "
            f"{fmt('answer_accuracy')} | {fmt('contain_accuracy')} | {fmt('answer_exact_match')} | "
            f"{fmt('answer_token_f1')} | {fmt('answer_rouge_l_f1')} | {fmt('retrieval_recall_at_5')} | "
            f"{fmt('retrieval_recall_at_8')} | {fmt('retrieval_hit_at_5')} | {fmt('retrieval_mrr_at_5')} | "
            f"{fmt('retrieval_ndcg_at_5')} |"
        )
    lines += [
        "",
        "## 2. 延迟",
        "",
        "| Method | Retrieval Avg (ms) | Retrieval P50 (ms) | Retrieval P95 (ms) | Total Avg (ms) | Total P50 (ms) | Total P95 (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        fmt = lambda key: "N/A" if row.get(key) is None else f"{row[key]:.2f}"
        lines.append(
            f"| {row['method']} | {fmt('retrieval_latency_ms_avg')} | {fmt('retrieval_latency_ms_p50')} | "
            f"{fmt('retrieval_latency_ms_p95')} | {fmt('total_latency_ms_avg')} | "
            f"{fmt('total_latency_ms_p50')} | {fmt('total_latency_ms_p95')} |"
        )
    lines += [
        "",
        "Original LinearRAG 的归档预测没有保存 retrieval/total latency 字段，因此延迟不能从已有数据反推，不能用 0 填充。",
        "",
        "## 3. LLM RAG Judge 指标",
        "",
        "Original LinearRAG 的 `combined_predictions.scored.json` 只有前 230/1000 题包含 RAG judge 输出；下面明确标注为部分样本，不能与完整 1000 题结果混用。这里的 `LLM RAG judge evidence recall` 是“参考事实是否被上下文支持”，不是 passage id 是否进入 Top-k。",
        "",
        "| Metric | Original LinearRAG | Evaluable N |",
        "|---|---:|---:|",
    ]
    for key, label in (
        ("llm_rag_context_relevance", "Context Relevance"),
        ("llm_rag_context_precision", "Context Precision"),
        ("llm_rag_evidence_recall", "LLM RAG judge Evidence Recall"),
        ("llm_rag_answer_fact_coverage", "Answer Fact Coverage"),
        ("llm_rag_answer_faithfulness", "Answer Faithfulness"),
        ("llm_rag_retrieval_hit", "LLM judge Retrieval Hit"),
        ("llm_rag_retrieval_mrr", "LLM judge Retrieval MRR"),
        ("llm_rag_retrieval_ndcg", "LLM judge Retrieval NDCG"),
        ("llm_rag_retrieval_f1", "LLM judge Retrieval F1"),
    ):
        value = linear_summary.get(key)
        shown = "N/A" if value is None else f"{value:.4f}"
        lines.append(f"| {label} | {shown} | {linear_summary['llm_rag_judge_evaluable_count']} |")
    lines += [
        "",
        "## 4. Original LinearRAG 结果来源",
        "",
        f"- 原始目录：`{LINEAR_SOURCE.relative_to(ROOT)}`",
        "- Question count: 1000；原始 retrieval top-k: 8；chunk token size: 1000；chunk overlap: 100。",
        "- 人工语义准确率：492/1000 = 0.4920。",
        "- Relaxed local judge：478/1000 = 0.4780；Contain-Acc = 0.2660；Exact = 0.2050；Token-F1 = 0.3218；ROUGE-L-F1 = 0.3167。",
        "- 检索 Recall@k 使用 passage id 计算；gold evidence 来源为自动词法候选，不是人工最终标注。",
        "",
        "## 5. Original LinearRAG 按书准确率",
        "",
        "| Book | Correct | Total | Manual Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for book, values in deterministic.get("per_book", {}).items():
        lines.append(f"| {book} | {values['correct']} | {values['total']} | {values['accuracy']:.4f} |")
    lines += [
        "",
        "## 6. 结论与限制",
        "",
        "1. 在已有人工语义口径下，Original LinearRAG 的 Answer Accuracy 为 49.20%，高于 Closed-book 的 3.20% 和 Dense RAG 的 18.30%。",
        "2. Original LinearRAG 的 Recall@5 为 8.17%，Recall@8 为 12.23%；这两个数依赖自动词法候选证据，不能替代人工 gold evidence 评测。",
        "3. Original LinearRAG 的 LLM RAG judge 结果只有 230 题，不能代表完整 1000 题；延迟数据完全缺失。",
        "4. 三个 baseline 使用的生成模型不同，且 Dense RAG 与 Original LinearRAG 的 top-k 分别为 5 和 8，所以当前表是“现有实验数据汇总”，不是已经完成的严格控制变量实验。",
    ]
    (FORMAL / "baseline_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"updated {FORMAL / 'baseline_comparison.md'}")


if __name__ == "__main__":
    main()
