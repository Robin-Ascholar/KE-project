#!/usr/bin/env python3
"""Compare baseline and book-aware LinearRAG retrieval without calling an LLM."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.LinearRAG import LinearRAG  # noqa: E402
from src.config import LinearRAGConfig  # noqa: E402
from src.utils import normalize_answer, setup_logging  # noqa: E402


DEFAULT_SETS = [f"book{book_number}" for book_number in range(1, 8)]
VARIANT_CONFIG = {
    "baseline": ("off", False, False),
    "oracle": ("oracle", False, False),
    "predicted": ("predicted", False, False),
    "confidence": ("predicted", True, False),
    "confidence_boost": ("predicted", True, True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="harrypotter")
    parser.add_argument("--question-dir", type=Path, default=Path("dataset/harrypotter"))
    parser.add_argument("--sets", nargs="+", default=DEFAULT_SETS)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANT_CONFIG),
        default=["baseline", "oracle", "predicted", "confidence"],
    )
    parser.add_argument("--split", choices=("all", "dev", "test"), default="all")
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--limit-per-book", type=int, help="Small retrieval-only smoke test")
    parser.add_argument("--embedding-model", default="model/all-mpnet-base-v2")
    parser.add_argument("--embedding-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--spacy-model", default="en_core_web_trf")
    parser.add_argument("--working-dir", default="./import")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--retrieval-top-k", type=int, default=20)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--iteration-threshold", type=float, default=0.4)
    parser.add_argument("--passage-ratio", type=float, default=2.0)
    parser.add_argument("--top-k-sentence", type=int, default=3)
    parser.add_argument("--use-vectorized-retrieval", action="store_true")
    parser.add_argument("--routing-strength", type=float, default=0.5)
    parser.add_argument("--routing-temperature", type=float, default=0.05)
    parser.add_argument("--routing-top-m", type=int, default=3)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def resolve_project_path(path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def select_indices(
    record_count: int,
    *,
    book_number: int,
    split: str,
    dev_ratio: float,
    seed: int,
    limit: int | None,
) -> list[int]:
    indices = list(range(record_count))
    random.Random(seed + book_number).shuffle(indices)
    dev_count = round(record_count * dev_ratio)
    if split == "dev":
        indices = indices[:dev_count]
    elif split == "test":
        indices = indices[dev_count:]
    if limit is not None:
        indices = indices[:limit]
    return indices


def load_questions(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not 0.0 < args.dev_ratio < 1.0:
        raise ValueError("dev-ratio must be between 0 and 1")
    if args.limit_per_book is not None and args.limit_per_book < 1:
        raise ValueError("limit-per-book must be positive")

    question_dir = resolve_project_path(args.question_dir)
    questions: list[dict[str, Any]] = []
    for set_name in args.sets:
        match = re.fullmatch(r"book([1-7])", set_name)
        if not match:
            raise ValueError(f"Question set must be named book1..book7, got {set_name!r}")
        book_number = int(match.group(1))
        path = question_dir / f"{set_name}.json"
        records = json.loads(path.read_text(encoding="utf-8"))
        indices = select_indices(
            len(records),
            book_number=book_number,
            split=args.split,
            dev_ratio=args.dev_ratio,
            seed=args.seed,
            limit=args.limit_per_book,
        )
        for question_index in indices:
            source = records[question_index]
            questions.append(
                {
                    **source,
                    "book_number": book_number,
                    "question_set": set_name,
                    "question_index": question_index,
                    "split": args.split,
                }
            )
    if not questions:
        raise ValueError("No questions selected")
    return questions


def passage_contains_answer(passage: str, gold_answer: str) -> bool:
    answer = normalize_answer(gold_answer)
    return bool(answer) and answer in normalize_answer(passage)


def summarize_records(records: list[dict[str, Any]], top_ks: tuple[int, ...]) -> dict[str, Any]:
    max_k = max(top_ks)
    hit_by_k = {top_k: [] for top_k in top_ks}
    reciprocal_ranks: list[float] = []
    gold_book_shares = {top_k: [] for top_k in top_ks}
    routing_correct: list[float] = []
    routing_confidence: list[float] = []

    for record in records:
        relevance = [
            passage_contains_answer(passage, record["gold_answer"])
            for passage in record["sorted_passage"][:max_k]
        ]
        first_rank = next(
            (rank for rank, relevant in enumerate(relevance, start=1) if relevant),
            None,
        )
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        retrieved_books = record.get("retrieved_passage_books", [])
        gold_book = int(record["book_number"])
        for top_k in top_ks:
            hit_by_k[top_k].append(float(any(relevance[:top_k])))
            books_at_k = retrieved_books[:top_k]
            gold_book_shares[top_k].append(
                sum(book == gold_book for book in books_at_k) / len(books_at_k)
                if books_at_k else 0.0
            )
        routing = record.get("book_routing")
        if routing and routing["mode"] == "predicted":
            routing_correct.append(float(routing["predicted_book"] == gold_book))
            routing_confidence.append(float(routing["confidence"]))

    summary: dict[str, Any] = {
        "question_count": len(records),
        f"mrr_at_{max_k}": mean(reciprocal_ranks),
    }
    for top_k in top_ks:
        summary[f"contain_at_{top_k}"] = mean(hit_by_k[top_k])
        summary[f"gold_book_share_at_{top_k}"] = mean(gold_book_shares[top_k])
    if routing_correct:
        summary["routing_top1_accuracy"] = mean(routing_correct)
        summary["mean_routing_confidence"] = mean(routing_confidence)
    return summary


def summarize_variant(records: list[dict[str, Any]], top_ks: tuple[int, ...]) -> dict[str, Any]:
    by_book = {
        str(book_number): summarize_records(
            [record for record in records if record["book_number"] == book_number],
            top_ks,
        )
        for book_number in range(1, 8)
        if any(record["book_number"] == book_number for record in records)
    }
    return {"overall": summarize_records(records, top_ks), "by_book": by_book}


def compare_to_baseline(
    baseline: list[dict[str, Any]],
    variant: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    if len(baseline) != len(variant):
        raise ValueError("Baseline and variant record counts differ")
    gained = lost = unchanged_correct = unchanged_incorrect = rank_changed = 0
    for base_record, variant_record in zip(baseline, variant):
        identity = (base_record["question_set"], base_record["question_index"])
        if identity != (variant_record["question_set"], variant_record["question_index"]):
            raise ValueError("Baseline and variant question order differs")
        base_hit = any(
            passage_contains_answer(passage, base_record["gold_answer"])
            for passage in base_record["sorted_passage"][:top_k]
        )
        variant_hit = any(
            passage_contains_answer(passage, variant_record["gold_answer"])
            for passage in variant_record["sorted_passage"][:top_k]
        )
        if not base_hit and variant_hit:
            gained += 1
        elif base_hit and not variant_hit:
            lost += 1
        elif base_hit:
            unchanged_correct += 1
        else:
            unchanged_incorrect += 1
        if base_record["sorted_passage"][:top_k] != variant_record["sorted_passage"][:top_k]:
            rank_changed += 1
    return {
        "top_k": top_k,
        "gained_answer_containment": gained,
        "lost_answer_containment": lost,
        "unchanged_contained": unchanged_correct,
        "unchanged_not_contained": unchanged_incorrect,
        "rank_changed": rank_changed,
    }


def main() -> None:
    args = parse_args()
    if "baseline" not in args.variants:
        raise ValueError("variants must include baseline for paired comparison")
    if args.retrieval_top_k < 1:
        raise ValueError("retrieval-top-k must be positive")

    questions = load_questions(args)
    chunks_path = PROJECT_ROOT / "dataset" / args.dataset_name / "chunks.json"
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    passages = [f"{index}:{chunk}" for index, chunk in enumerate(chunks)]

    output_dir = (
        resolve_project_path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT
        / "results"
        / "book_routing"
        / datetime.now().strftime("retrieval_%Y-%m-%d_%H-%M-%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(output_dir / "log.txt"))

    device = args.embedding_device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_path = resolve_project_path(args.embedding_model)
    embedding_model = SentenceTransformer(str(embedding_path), device=device)
    config = LinearRAGConfig(
        dataset_name=args.dataset_name,
        embedding_model=embedding_model,
        llm_model=None,
        spacy_model=args.spacy_model,
        working_dir=args.working_dir,
        max_workers=args.max_workers,
        retrieval_top_k=args.retrieval_top_k,
        max_iterations=args.max_iterations,
        iteration_threshold=args.iteration_threshold,
        passage_ratio=args.passage_ratio,
        top_k_sentence=args.top_k_sentence,
        use_vectorized_retrieval=args.use_vectorized_retrieval,
        save_graph_index=False,
        book_routing_strength=args.routing_strength,
        book_routing_temperature=args.routing_temperature,
        book_routing_top_m=args.routing_top_m,
    )
    rag = LinearRAG(global_config=config)
    rag.index(passages)

    top_ks = tuple(
        top_k
        for top_k in sorted({5, 10, args.retrieval_top_k})
        if top_k <= args.retrieval_top_k
    )
    variant_records: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": args.dataset_name,
        "split": args.split,
        "seed": args.seed,
        "dev_ratio": args.dev_ratio,
        "question_count": len(questions),
        "retrieval_top_k": args.retrieval_top_k,
        "routing_strength": args.routing_strength,
        "routing_temperature": args.routing_temperature,
        "routing_top_m": args.routing_top_m,
        "variants": {},
        "comparisons": {},
    }

    for variant in args.variants:
        mode, confidence_aware, boost_only = VARIANT_CONFIG[variant]
        rag.config.book_routing_mode = mode
        rag.config.book_routing_confidence_aware = confidence_aware
        rag.config.book_routing_boost_only = boost_only
        started = time.perf_counter()
        records = rag.retrieve(questions)
        elapsed = time.perf_counter() - started
        variant_records[variant] = records
        atomic_json(output_dir / "variants" / f"{variant}.json", records)
        variant_summary = summarize_variant(records, top_ks)
        variant_summary["elapsed_seconds"] = elapsed
        variant_summary["seconds_per_question"] = elapsed / len(records)
        summary["variants"][variant] = variant_summary
        overall = variant_summary["overall"]
        print(
            f"{variant}: contain@{args.retrieval_top_k}="
            f"{overall[f'contain_at_{args.retrieval_top_k}']:.3f} "
            f"gold-book-share@{args.retrieval_top_k}="
            f"{overall[f'gold_book_share_at_{args.retrieval_top_k}']:.3f} "
            f"seconds/question={variant_summary['seconds_per_question']:.3f}",
            flush=True,
        )

    baseline = variant_records["baseline"]
    for variant, records in variant_records.items():
        if variant != "baseline":
            summary["comparisons"][variant] = compare_to_baseline(
                baseline, records, args.retrieval_top_k
            )
    config_payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config_payload["output_dir"] = str(output_dir)
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(output_dir / "experiment_config.json", config_payload)
    print(f"Wrote experiment results to {output_dir}")


if __name__ == "__main__":
    main()
