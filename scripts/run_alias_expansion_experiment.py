#!/usr/bin/env python3
"""Compare baseline and query-time entity alias activation without an LLM."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
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
from src.entity_aliases import EntityAliasResolver  # noqa: E402
from src.utils import normalize_answer, setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="harrypotter")
    parser.add_argument("--question-dir", type=Path, default=Path("dataset/harrypotter"))
    parser.add_argument(
        "--alias-file",
        type=Path,
        default=Path("config/harry_potter_entity_aliases.json"),
    )
    parser.add_argument("--split", choices=("all", "dev", "test"), default="dev")
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--limit-per-book", type=int)
    parser.add_argument(
        "--only-alias-triggered",
        action="store_true",
        help="Evaluate only questions that lexically match a configured alias group",
    )
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
    parser.add_argument("--output-dir", type=Path)
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
        raise ValueError("dev-ratio must be between 0 and 1")
    question_dir = project_path(args.question_dir)
    questions = []
    for book_number in range(1, 8):
        records = json.loads((question_dir / f"book{book_number}.json").read_text())
        indices = list(range(len(records)))
        random.Random(args.seed + book_number).shuffle(indices)
        dev_count = round(len(records) * args.dev_ratio)
        if args.split == "dev":
            indices = indices[:dev_count]
        elif args.split == "test":
            indices = indices[dev_count:]
        if args.limit_per_book is not None:
            indices = indices[: args.limit_per_book]
        for index in indices:
            questions.append(
                {
                    **records[index],
                    "book_number": book_number,
                    "question_set": f"book{book_number}",
                    "question_index": index,
                    "split": args.split,
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
        return {"question_count": 0}
    max_k = max(top_ks)
    reciprocal_ranks = []
    for record in records:
        answer = normalize_answer(record["gold_answer"])
        first_rank = next(
            (
                rank
                for rank, passage in enumerate(record["sorted_passage"][:max_k], start=1)
                if answer and answer in normalize_answer(passage)
            ),
            None,
        )
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
    result = {
        "question_count": len(records),
        f"mrr_at_{max_k}": mean(reciprocal_ranks),
    }
    for top_k in top_ks:
        result[f"contain_at_{top_k}"] = mean(
            float(contains_answer(record, top_k)) for record in records
        )
    return result


def compare(
    baseline: list[dict[str, Any]],
    alias: list[dict[str, Any]],
    top_k: int,
    selected: set[tuple[str, int]] | None = None,
) -> dict[str, int]:
    counts = Counter()
    for base_record, alias_record in zip(baseline, alias):
        identity = (base_record["question_set"], base_record["question_index"])
        if selected is not None and identity not in selected:
            continue
        base_hit = contains_answer(base_record, top_k)
        alias_hit = contains_answer(alias_record, top_k)
        if not base_hit and alias_hit:
            counts["gained"] += 1
        elif base_hit and not alias_hit:
            counts["lost"] += 1
        elif base_hit:
            counts["unchanged_contained"] += 1
        else:
            counts["unchanged_not_contained"] += 1
        if base_record["sorted_passage"][:top_k] != alias_record["sorted_passage"][:top_k]:
            counts["rank_changed"] += 1
    return dict(counts)


def main() -> None:
    args = parse_args()
    if args.retrieval_top_k < 5:
        raise ValueError("retrieval-top-k must be at least 5")
    if not 0.0 <= args.alias_expansion_budget <= 1.0:
        raise ValueError("alias-expansion-budget must be in [0, 1]")

    alias_groups = json.loads(project_path(args.alias_file).read_text(encoding="utf-8"))
    questions = load_questions(args)
    if args.only_alias_triggered:
        resolver = EntityAliasResolver(alias_groups)
        questions = [
            question for question in questions if resolver.matched_groups(question["question"])
        ]
        if not questions:
            raise ValueError("No alias-triggered questions selected")
    chunks = json.loads(
        (PROJECT_ROOT / "dataset" / args.dataset_name / "chunks.json").read_text(encoding="utf-8")
    )
    passages = [f"{index}:{chunk}" for index, chunk in enumerate(chunks)]
    output_dir = (
        project_path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT
        / "results"
        / "alias_expansion"
        / datetime.now().strftime("retrieval_%Y-%m-%d_%H-%M-%S")
    )
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
        entity_alias_groups=alias_groups,
        alias_expansion_budget=args.alias_expansion_budget,
    )
    rag = LinearRAG(global_config=config)
    rag.index(passages)

    variants = {}
    elapsed = {}
    for name, enabled in (("baseline", False), ("alias_expansion", True)):
        rag.config.enable_entity_alias_expansion = enabled
        started = time.perf_counter()
        variants[name] = rag.retrieve(questions)
        elapsed[name] = time.perf_counter() - started
        atomic_json(output_dir / "variants" / f"{name}.json", variants[name])

    alias_records = variants["alias_expansion"]
    triggered_ids = {
        (record["question_set"], record["question_index"])
        for record in alias_records
        if record.get("alias_activation", {}).get("matched_groups")
    }
    triggered_groups = Counter(
        group
        for record in alias_records
        for group in record.get("alias_activation", {}).get("matched_groups", [])
    )
    baseline_triggered = [
        record
        for record in variants["baseline"]
        if (record["question_set"], record["question_index"]) in triggered_ids
    ]
    alias_triggered = [
        record
        for record in alias_records
        if (record["question_set"], record["question_index"]) in triggered_ids
    ]
    top_ks = tuple(sorted({5, 10, args.retrieval_top_k}))
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "seed": args.seed,
        "question_count": len(questions),
        "only_alias_triggered": args.only_alias_triggered,
        "triggered_question_count": len(triggered_ids),
        "alias_group_count": len(alias_groups),
        "triggered_groups": dict(triggered_groups),
        "alias_expansion_budget": args.alias_expansion_budget,
        "overall": {
            "baseline": summarize(variants["baseline"], top_ks),
            "alias_expansion": summarize(alias_records, top_ks),
            "paired": compare(variants["baseline"], alias_records, args.retrieval_top_k),
        },
        "triggered_subset": {
            "baseline": summarize(baseline_triggered, top_ks),
            "alias_expansion": summarize(alias_triggered, top_ks),
            "paired": compare(
                variants["baseline"], alias_records, args.retrieval_top_k, triggered_ids
            ),
        },
        "elapsed_seconds": elapsed,
    }
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(
        output_dir / "experiment_config.json",
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    )
    for scope in ("overall", "triggered_subset"):
        base = summary[scope]["baseline"]
        alias = summary[scope]["alias_expansion"]
        print(
            f"{scope}: n={base['question_count']} "
            f"baseline@{args.retrieval_top_k}={base[f'contain_at_{args.retrieval_top_k}']:.3f} "
            f"alias@{args.retrieval_top_k}={alias[f'contain_at_{args.retrieval_top_k}']:.3f} "
            f"paired={summary[scope]['paired']}",
            flush=True,
        )
    print(f"Wrote results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
