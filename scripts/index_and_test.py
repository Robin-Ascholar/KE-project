"""Build a LinearRAG index and run retrieval-only checks without an LLM API."""

import argparse
import json
import sys
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.LinearRAG import LinearRAG  # noqa: E402
from src.config import LinearRAGConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="harrypotter")
    parser.add_argument(
        "--checks",
        type=Path,
        help="Optional JSON retrieval check file. Omit to run indexing without checks.",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--enable-chapter-retrieval", action="store_true")
    parser.add_argument("--chapter-top-k", type=int, default=8)
    parser.add_argument("--chapter-global-fallback", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = PROJECT_ROOT / "dataset" / args.dataset_name
    chunks_path = dataset_dir / "chunks.json"
    if not chunks_path.is_file():
        raise FileNotFoundError(f"Run scripts/prepare_harry_potter.py first: {chunks_path}")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    passages = [f"{index}:{chunk}" for index, chunk in enumerate(chunks)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer(
        str(PROJECT_ROOT / "model" / "all-mpnet-base-v2"),
        device=device,
    )
    config = LinearRAGConfig(
        dataset_name=args.dataset_name,
        embedding_model=embedding_model,
        spacy_model="en_core_web_trf",
        working_dir=str(PROJECT_ROOT / "import"),
        max_workers=max(1, args.max_workers),
        retrieval_top_k=max(1, args.top_k),
        max_iterations=3,
        iteration_threshold=0.4,
        passage_ratio=2.0,
        top_k_sentence=3,
        enable_chapter_retrieval=args.enable_chapter_retrieval,
        chapter_top_k=args.chapter_top_k,
        chapter_global_fallback=args.chapter_global_fallback,
    )
    rag = LinearRAG(global_config=config)
    rag.index(passages)

    if args.checks is None:
        print("Retrieval checks: skipped (no --checks file provided)")
        print(f"Graph: vertices={rag.graph.vcount()} edges={rag.graph.ecount()}")
        return

    if not args.checks.is_file():
        raise FileNotFoundError(f"Retrieval check file does not exist: {args.checks}")

    checks = json.loads(args.checks.read_text(encoding="utf-8"))
    questions = [
        {"question": check["question"], "answer": ", ".join(check["expected_terms"])}
        for check in checks
    ]
    results = rag.retrieve(questions)

    failures = 0
    for check, result in zip(checks, results):
        expected_terms = [term.casefold() for term in check["expected_terms"]]
        rank = next(
            (
                index
                for index, passage in enumerate(result["sorted_passage"], start=1)
                if all(term in passage.casefold() for term in expected_terms)
            ),
            None,
        )
        status = "PASS" if rank is not None else "FAIL"
        failures += rank is None
        print(f"{status} rank={rank or '-'} question={check['question']}")

    print(f"Graph: vertices={rag.graph.vcount()} edges={rag.graph.ecount()}")
    print(f"Retrieval checks: passed={len(checks) - failures} failed={failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
