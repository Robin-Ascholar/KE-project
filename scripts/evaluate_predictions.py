#!/usr/bin/env python3
"""Evaluate LinearRAG answers, retrieved context, evidence, and reasoning paths."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
from openai import OpenAI
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paper_evaluator import contain_match, score_with_judge  # noqa: E402
from src.rag_metrics import (  # noqa: E402
    exact_match,
    rouge_l_f1,
    score_rag_record,
    token_f1,
)


class JudgeClient:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.openai_client = None
        self.http_client = httpx.Client(timeout=args.timeout, trust_env=False)
        if args.provider == "openai":
            self.openai_client = OpenAI(
                api_key=args.api_key,
                base_url=args.base_url,
                timeout=httpx.Timeout(args.timeout),
                max_retries=max(0, args.max_retries),
                http_client=self.http_client,
            )
        else:
            self.ollama_url = args.base_url.removesuffix("/v1").rstrip("/")

    def infer_label(self, messages: List[Dict[str, str]]) -> str:
        if self.args.provider == "openai":
            response = self.openai_client.chat.completions.create(
                model=self.args.judge_model,
                messages=messages,
                temperature=0,
                max_tokens=8,
            )
            return response.choices[0].message.content or ""
        response = self.http_client.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": self.args.judge_model,
                "messages": messages,
                "stream": False,
                "think": False,
                "options": {
                    "temperature": 0,
                    "num_ctx": self.args.num_ctx,
                    "num_predict": 8,
                },
            },
        )
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "")

    def infer_json(self, messages: List[Dict[str, str]]) -> str:
        if self.args.provider == "openai":
            response = self.openai_client.chat.completions.create(
                model=self.args.judge_model,
                messages=messages,
                temperature=0,
                max_tokens=self.args.judge_max_tokens,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content or ""
        response = self.http_client.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": self.args.judge_model,
                "messages": messages,
                "stream": False,
                "think": False,
                "format": "json",
                "options": {
                    "temperature": 0,
                    "num_ctx": self.args.num_ctx,
                    "num_predict": self.args.judge_max_tokens,
                },
            },
        )
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "")


def latest_predictions() -> Path:
    candidates = sorted(
        (PROJECT_ROOT / "results" / "harrypotter").glob("*/predictions.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No results/harrypotter/*/predictions.json found")
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, help="Prediction JSON; defaults to latest Harry Potter run")
    parser.add_argument("--judge-model", default="qwen3.6:27b", help="Judge model name")
    parser.add_argument("--provider", choices=("ollama", "openai"), default="ollama")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "ollama"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--num-ctx", type=int, default=32768, help="Ollama judge context window")
    parser.add_argument("--judge-max-tokens", type=int, default=4096, help="Maximum JSON judge output tokens")
    parser.add_argument("--contain-only", action="store_true", help="Compute deterministic answer metrics only")
    parser.add_argument("--skip-rag-metrics", action="store_true", help="Skip retrieval, evidence, and path metrics")
    parser.add_argument("--skip-semantic-similarity", action="store_true")
    parser.add_argument("--embedding-model", default="model/all-mpnet-base-v2")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--output", type=Path, help="Scored prediction JSON output path")
    parser.add_argument("--summary", type=Path, help="Evaluation summary JSON output path")
    return parser.parse_args()


def _mean(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def _add_semantic_similarity(
    rows: List[Dict[str, Any]],
    model_path: str,
    device: str,
) -> Optional[str]:
    try:
        from sentence_transformers import SentenceTransformer

        path = Path(model_path)
        resolved = path if path.is_absolute() else PROJECT_ROOT / path
        model = SentenceTransformer(str(resolved), device=device)
        predicted = [str(row.get("pred_answer", row.get("generated_answer", "")) or "") for row in rows]
        gold = [str(row.get("gold_answer", row.get("answer", "")) or "") for row in rows]
        embeddings = model.encode(
            predicted + gold,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        size = len(rows)
        for index, row in enumerate(rows):
            cosine = float(np.dot(embeddings[index], embeddings[index + size]))
            row["answer_semantic_similarity"] = max(0.0, min(1.0, (cosine + 1.0) / 2.0))
        return None
    except Exception as exc:
        for row in rows:
            row["answer_semantic_similarity"] = None
        return f"{type(exc).__name__}: {exc}"


def main() -> None:
    args = parse_args()
    predictions_path = (args.predictions or latest_predictions()).resolve()
    records = json.loads(predictions_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Predictions must be a non-empty JSON array")

    output_path = args.output or predictions_path.with_name("predictions.comprehensive-scored.json")
    summary_path = args.summary or output_path.with_name("evaluation_results.comprehensive.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = None if args.contain_only else JudgeClient(args)

    def evaluate_one(index: int, record: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
        row = dict(record)
        predicted = record.get("pred_answer", record.get("generated_answer"))
        gold = record.get("gold_answer", record.get("answer"))
        row["paper_contain_accuracy"] = contain_match(predicted, gold)
        row["answer_exact_match"] = exact_match(predicted, gold)
        row["answer_token_f1"] = token_f1(predicted, gold)
        row["answer_rouge_l_f1"] = rouge_l_f1(predicted, gold)
        if args.contain_only:
            return index, row

        try:
            judge = score_with_judge(record.get("question"), predicted, gold, client.infer_label)
            row["local_judge_accuracy"] = judge.score
            row["local_judge_label"] = judge.label
            row["local_judge_raw_response"] = judge.raw_response
            row["local_judge_parse_error"] = judge.parse_error
        except Exception as exc:
            row["local_judge_accuracy"] = 0.0
            row["local_judge_label"] = "incorrect"
            row["local_judge_error"] = f"{type(exc).__name__}: {exc}"

        if not args.skip_rag_metrics:
            try:
                rag_result = score_rag_record(record, client.infer_json)
                details = rag_result.to_dict()
                for key in (
                    "context_relevance",
                    "context_precision",
                    "evidence_recall",
                    "answer_fact_coverage",
                    "answer_faithfulness",
                    "evidence_path_coverage",
                    "evidence_path_complete",
                    "multi_hop_path_coverage",
                    "retrieval_hit",
                    "retrieval_mrr",
                    "retrieval_ndcg",
                    "retrieval_f1",
                    "path_length",
                    "is_multi_hop",
                    "path_source",
                ):
                    row[key] = details[key]
                row["rag_judge_details"] = {
                    "passage_relevance": details["passage_relevance"],
                    "reference_facts": details["reference_facts"],
                    "answer_claims": details["answer_claims"],
                    "reasoning_path": details["reasoning_path"],
                }
                row["rag_judge_raw_response"] = details["raw_response"]
                row["rag_judge_parse_error"] = details["parse_error"]
            except Exception as exc:
                row["rag_judge_error"] = f"{type(exc).__name__}: {exc}"
        return index, row

    scored_records: List[Optional[Dict[str, Any]]] = [None] * len(records)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(evaluate_one, index, row): index for index, row in enumerate(records)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Comprehensive RAG evaluation"):
            index = futures[future]
            try:
                _, scored_records[index] = future.result()
            except Exception as exc:
                row = dict(records[index])
                row["evaluation_error"] = f"{type(exc).__name__}: {exc}"
                scored_records[index] = row

    rows: List[Dict[str, Any]] = [row for row in scored_records if row is not None]
    semantic_error = None
    if not args.skip_semantic_similarity:
        semantic_error = _add_semantic_similarity(rows, args.embedding_model, args.embedding_device)

    rag_valid_rows = [
        row for row in rows
        if "context_relevance" in row and not row.get("rag_judge_parse_error") and not row.get("rag_judge_error")
    ]
    multi_hop_rows = [row for row in rag_valid_rows if row.get("is_multi_hop")]
    answer_parse_failures = sum(bool(row.get("local_judge_parse_error") or row.get("local_judge_error")) for row in rows)
    rag_failures = sum(bool(row.get("rag_judge_parse_error") or row.get("rag_judge_error")) for row in rows)

    metric_keys = (
        "paper_contain_accuracy",
        "local_judge_accuracy",
        "answer_exact_match",
        "answer_token_f1",
        "answer_rouge_l_f1",
        "answer_semantic_similarity",
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
    summary_metrics = {key: _mean(rows if key not in {
        "context_relevance", "context_precision", "evidence_recall", "answer_fact_coverage",
        "answer_faithfulness", "evidence_path_coverage", "evidence_path_complete", "retrieval_hit", "retrieval_mrr",
        "retrieval_ndcg", "retrieval_f1"
    } else rag_valid_rows, key) for key in metric_keys}
    summary_metrics["multi_hop_path_coverage"] = _mean(multi_hop_rows, "multi_hop_path_coverage")

    summary = {
        "metric_definition": "LinearRAG paper metrics plus comprehensive RAG diagnostics",
        "predictions": str(predictions_path),
        "sample_count": len(rows),
        "judge_enabled": not args.contain_only,
        "judge_model": None if args.contain_only else args.judge_model,
        "judge_provider": None if args.contain_only else args.provider,
        "judge_base_url": None if args.contain_only else args.base_url,
        "paper_reported_metrics": [
            "paper_contain_accuracy",
            "local_judge_accuracy (local substitute for GPT-Acc)",
            "context_relevance",
            "evidence_recall",
        ],
        "additional_diagnostics": [
            "answer_exact_match", "answer_token_f1", "answer_rouge_l_f1",
            "answer_semantic_similarity", "context_precision", "answer_fact_coverage",
            "answer_faithfulness", "evidence_path_coverage", "evidence_path_complete",
            "multi_hop_path_coverage",
            "retrieval_hit", "retrieval_mrr", "retrieval_ndcg", "retrieval_f1",
        ],
        "metrics": summary_metrics,
        "answer_judge_failures": answer_parse_failures,
        "rag_judge_failures": rag_failures,
        "rag_valid_sample_count": len(rag_valid_rows),
        "multi_hop_sample_count": len(multi_hop_rows),
        "path_annotation_note": "reasoning_path/supporting_facts/gold_evidence are preferred; otherwise the judge derives a minimal path",
        "semantic_similarity_error": semantic_error,
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Scored predictions: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
