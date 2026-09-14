#!/usr/bin/env python3
"""Compare baseline and conservative alias activation without calling an LLM."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.LinearRAG import LinearRAG  # noqa: E402
from src.config import LinearRAGConfig  # noqa: E402
from src.entity_alias import EntityAliasResolver  # noqa: E402
from src.utils import normalize_answer, setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="harrypotter")
    parser.add_argument("--question-dir", type=Path, default=Path("dataset/harrypotter"))
    parser.add_argument(
        "--alias-file",
        type=Path,
        default=Path("dataset/harrypotter/entity_aliases.conservative.json"),
    )
    parser.add_argument("--split", choices=("all", "dev", "test"), default="dev")
    parser.add_argument(
        "--only-alias-triggered",
        action="store_true",
        help="Evaluate only questions that lexically match the frozen alias file",
    )
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260913)
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
    parser.add_argument("--alias-expansion-budget", type=float, default=0.75)
    parser.add_argument("--max-alias-expansions", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def project_path(path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_questions(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not 0.0 < args.dev_ratio < 1.0:
        raise ValueError("--dev-ratio must be between 0 and 1")
    question_dir = project_path(args.question_dir)
    questions = []
    for book_number in range(1, 8):
        path = question_dir / f"book{book_number}.json"
        records = json.loads(path.read_text(encoding="utf-8"))
        indices = list(range(len(records)))
        random.Random(args.seed + book_number).shuffle(indices)
        dev_count = round(len(records) * args.dev_ratio)
        if args.split == "dev":
            indices = indices[:dev_count]
        elif args.split == "test":
            indices = indices[dev_count:]
        for index in indices:
            questions.append(
                {
                    **records[index],
                    "question_set": f"book{book_number}",
                    "question_index": index,
                }
            )
    return questions


def contains_answer(record: dict[str, Any], top_k: int) -> bool:
    answer = normalize_answer(record["gold_answer"])
    return bool(answer) and any(
        answer in normalize_answer(passage)
        for passage in record["sorted_passage"][:top_k]
    )


def summarize(records: list[dict[str, Any]], top_ks: tuple[int, ...]) -> dict[str, Any]:
    if not records:
        return {"count": 0}
    max_k = max(top_ks)
    reciprocal_ranks = []
    for record in records:
        answer = normalize_answer(record["gold_answer"])
        rank = next(
            (
                index
                for index, passage in enumerate(record["sorted_passage"][:max_k], start=1)
                if answer and answer in normalize_answer(passage)
            ),
            None,
        )
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    result = {"count": len(records), f"mrr_at_{max_k}": mean(reciprocal_ranks)}
    for top_k in top_ks:
        result[f"contain_at_{top_k}"] = mean(
            float(contains_answer(record, top_k)) for record in records
        )
    return result


def paired_counts(
    baseline: list[dict[str, Any]],
    alias: list[dict[str, Any]],
    top_k: int,
) -> dict[str, int]:
    counts = Counter(gained=0, lost=0, rank_changed=0)
    for base_record, alias_record in zip(baseline, alias):
        base_hit = contains_answer(base_record, top_k)
        alias_hit = contains_answer(alias_record, top_k)
        if alias_hit and not base_hit:
            counts["gained"] += 1
        elif base_hit and not alias_hit:
            counts["lost"] += 1
        if base_record["sorted_passage"][:top_k] != alias_record["sorted_passage"][:top_k]:
            counts["rank_changed"] += 1
    return dict(counts)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alias_expansion_budget <= 1.0:
        raise ValueError("--alias-expansion-budget must be in [0, 1]")
    if args.max_alias_expansions < 0:
        raise ValueError("--max-alias-expansions must be non-negative")

    questions = load_questions(args)
    if args.only_alias_triggered:
        selector = EntityAliasResolver(str(project_path(args.alias_file)))
        questions = [
            record for record in questions if selector.find_mentions(record["question"])
        ]
        if not questions:
            raise ValueError("No questions matched the alias file")
    chunks = json.loads(
        (PROJECT_ROOT / "dataset" / args.dataset_name / "chunks.json").read_text(encoding="utf-8")
    )
    passages = [f"{index}:{chunk}" for index, chunk in enumerate(chunks)]
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(output_dir / "log.txt"))

    device = args.embedding_device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(str(project_path(args.embedding_model)), device=device)
    config = LinearRAGConfig(
        dataset_name=args.dataset_name,
        embedding_model=model,
        llm_model=None,
        spacy_model=args.spacy_model,
        working_dir=args.working_dir,
        max_workers=args.max_workers,
        retrieval_top_k=args.retrieval_top_k,
        max_iterations=args.max_iterations,
        iteration_threshold=args.iteration_threshold,
        passage_ratio=args.passage_ratio,
        top_k_sentence=args.top_k_sentence,
        save_graph_index=False,
        entity_alias_path=str(project_path(args.alias_file)),
        enable_alias_activation=True,
        alias_expansion_budget=args.alias_expansion_budget,
        max_alias_expansions=args.max_alias_expansions,
    )
    rag = LinearRAG(global_config=config)
    rag.index(passages)

    variants = {}
    elapsed = {}
    for name, enabled in (("baseline", False), ("alias", True)):
        rag.entity_alias_resolver.enabled = enabled
        started = time.perf_counter()
        variants[name] = rag.retrieve(questions)
        elapsed[name] = time.perf_counter() - started
        atomic_json(output_dir / f"{name}.json", variants[name])

    triggered_indices = [
        index for index, record in enumerate(variants["alias"]) if record.get("alias_expansions")
    ]
    baseline_triggered = [variants["baseline"][index] for index in triggered_indices]
    alias_triggered = [variants["alias"][index] for index in triggered_indices]
    top_ks = tuple(sorted({5, 10, args.retrieval_top_k}))
    summary = {
        "question_count": len(questions),
        "triggered_count": len(triggered_indices),
        "only_alias_triggered": args.only_alias_triggered,
        "alias_file": str(args.alias_file),
        "alias_expansion_budget": args.alias_expansion_budget,
        "max_alias_expansions": args.max_alias_expansions,
        "elapsed_seconds": elapsed,
        "overall": {
            "baseline": summarize(variants["baseline"], top_ks),
            "alias": summarize(variants["alias"], top_ks),
            "paired": {
                str(top_k): paired_counts(variants["baseline"], variants["alias"], top_k)
                for top_k in top_ks
            },
        },
        "triggered": {
            "baseline": summarize(baseline_triggered, top_ks),
            "alias": summarize(alias_triggered, top_ks),
            "paired": {
                str(top_k): paired_counts(baseline_triggered, alias_triggered, top_k)
                for top_k in top_ks
            },
        },
    }
    atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
