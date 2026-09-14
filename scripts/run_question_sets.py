#!/usr/bin/env python3
"""Run dataset/question/*.json sequentially, merge predictions, and evaluate them."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LinearRAGConfig  # noqa: E402
from src.LinearRAG import LinearRAG  # noqa: E402
from src.utils import LLM_Model, setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question-dir", type=Path, default=Path("dataset/question"))
    parser.add_argument("--sets", nargs="*", help="Set stems to run, e.g. book1 book2; default: all JSON files")
    parser.add_argument("--dataset-name", default="harrypotter")
    parser.add_argument("--embedding-model", default="model/all-mpnet-base-v2")
    parser.add_argument("--embedding-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--spacy-model", default="en_core_web_trf")
    parser.add_argument("--llm-model", default="deepseek-ocr:latest")
    parser.add_argument("--llm-max-tokens", type=int, default=128)
    parser.add_argument("--llm-timeout", type=float, default=240.0)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--question-batch-size", type=int, default=10)
    parser.add_argument("--retrieval-top-k", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--iteration-threshold", type=float, default=0.4)
    parser.add_argument("--passage-ratio", type=float, default=2.0)
    parser.add_argument("--top-k-sentence", type=int, default=3)
    parser.add_argument("--use-vectorized-retrieval", action="store_true")
    parser.add_argument("--enable-chapter-retrieval", action="store_true", help="Use chapter coarse-to-fine retrieval")
    parser.add_argument("--chapter-top-k", type=int, default=16)
    parser.add_argument("--chapter-global-fallback", type=int, default=4)
    parser.add_argument("--entity-alias-path", type=Path, help="Optional entity alias JSON path")
    alias_group = parser.add_mutually_exclusive_group()
    alias_group.add_argument("--enable-alias-activation", dest="enable_alias_activation", action="store_true", help="Enable query-time entity alias expansion")
    alias_group.add_argument("--disable-alias-activation", dest="enable_alias_activation", action="store_false", help="Disable query-time entity alias expansion")
    parser.set_defaults(enable_alias_activation=False)
    parser.add_argument("--alias-expansion-budget", type=float, default=0.75, help="Total score shared by expanded aliases in one entity group")
    parser.add_argument("--max-alias-expansions", type=int, default=8, help="Maximum expanded surface forms per logical entity")
    parser.add_argument("--output-dir", type=Path, help="Existing directory resumes automatically")
    parser.add_argument("--limit-per-set", type=int, help="Development/smoke-test limit")
    parser.add_argument("--no-evaluate", action="store_true")
    parser.add_argument("--judge-model", default="qwen3.6:27b")
    parser.add_argument("--judge-workers", type=int, default=1)
    parser.add_argument("--judge-timeout", type=float, default=300.0)
    parser.add_argument("--judge-checkpoint-every", type=int, default=10)
    parser.add_argument("--contain-only", action="store_true", help="Skip LLM judge and compute deterministic metrics")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def natural_key(path: Path) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.stem)]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_question_sets(args: argparse.Namespace) -> list[tuple[Path, list[dict[str, Any]]]]:
    question_dir = (PROJECT_ROOT / args.question_dir).resolve() if not args.question_dir.is_absolute() else args.question_dir
    files = sorted(question_dir.glob("*.json"), key=natural_key)
    if args.sets:
        requested = set(args.sets)
        files = [path for path in files if path.stem in requested or path.name in requested]
        missing = requested - {path.stem for path in files} - {path.name for path in files}
        if missing:
            raise ValueError(f"Question sets not found: {sorted(missing)}")
    if not files:
        raise FileNotFoundError(f"No question JSON files found in {question_dir}")

    loaded = []
    for path in files:
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list) or not records:
            raise ValueError(f"{path} must be a non-empty JSON array")
        for index, record in enumerate(records):
            if not isinstance(record, dict) or not str(record.get("question", "")).strip() or "answer" not in record:
                raise ValueError(f"Invalid question record at {path}:{index}")
        if args.limit_per_set is not None:
            records = records[: max(0, args.limit_per_set)]
        loaded.append((path, records))
    return loaded


def load_existing_predictions(path: Path, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) > len(questions):
        raise ValueError(f"Invalid checkpoint: {path}")
    for index, row in enumerate(rows):
        if row.get("question") != questions[index].get("question") or row.get("gold_answer") != questions[index].get("answer"):
            raise ValueError(f"Checkpoint no longer matches question set at {path}:{index}")
    return rows


def update_manifest(path: Path, manifest: dict[str, Any], status: str | None = None) -> None:
    if status:
        manifest["status"] = status
    manifest["updated_at_utc"] = utc_now()
    atomic_json(path, manifest)


def main() -> None:
    args = parse_args()
    if args.question_batch_size < 1:
        raise ValueError("--question-batch-size must be positive")
    if not 0.0 <= args.alias_expansion_budget <= 1.0:
        raise ValueError("--alias-expansion-budget must be in [0, 1]")
    if args.max_alias_expansions < 0:
        raise ValueError("--max-alias-expansions must be non-negative")
    if args.chapter_top_k < 1:
        raise ValueError("--chapter-top-k must be positive")
    if args.chapter_global_fallback < 0:
        raise ValueError("--chapter-global-fallback must be non-negative")

    os.environ.setdefault("OPENAI_API_KEY", "ollama")
    os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
    question_sets = load_question_sets(args)

    if args.output_dir:
        output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    else:
        stamp = datetime.now().strftime("batch_%Y-%m-%d_%H-%M-%S")
        output_dir = PROJECT_ROOT / "results" / args.dataset_name / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    setup_logging(str(output_dir / "log.txt"))

    set_meta = [
        {
            "name": path.stem,
            "source": str(path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(path),
            "question_count": len(records),
            "completed": 0,
        }
        for path, records in question_sets
    ]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        old_rows = {row["name"]: row for row in manifest.get("question_sets", [])}
        for row in set_meta:
            previous = old_rows.get(row["name"], {})
            if previous.get("sha256") not in (None, row["sha256"]):
                raise ValueError(f"Question set changed since checkpoint: {row['name']}")
            row["completed"] = min(int(previous.get("completed", 0)), row["question_count"])
        manifest["entity_alias_path"] = str(args.entity_alias_path) if args.entity_alias_path else None
        manifest["alias_activation_enabled"] = args.enable_alias_activation
        manifest["alias_expansion_budget"] = args.alias_expansion_budget
        manifest["max_alias_expansions"] = args.max_alias_expansions
        manifest["enable_chapter_retrieval"] = args.enable_chapter_retrieval
        manifest["chapter_top_k"] = args.chapter_top_k
        manifest["chapter_global_fallback"] = args.chapter_global_fallback
        manifest["question_sets"] = set_meta
    else:
        manifest = {
            "status": "initializing",
            "created_at_utc": utc_now(),
            "dataset_name": args.dataset_name,
            "question_dir": str(args.question_dir),
            "qa_model": args.llm_model,
            "judge_model": None if args.no_evaluate or args.contain_only else args.judge_model,
            "retrieval_top_k": args.retrieval_top_k,
            "enable_chapter_retrieval": args.enable_chapter_retrieval,
            "chapter_top_k": args.chapter_top_k,
            "chapter_global_fallback": args.chapter_global_fallback,
            "question_batch_size": args.question_batch_size,
            "entity_alias_path": str(args.entity_alias_path) if args.entity_alias_path else None,
            "alias_activation_enabled": args.enable_alias_activation,
            "alias_expansion_budget": args.alias_expansion_budget,
            "max_alias_expansions": args.max_alias_expansions,
            "question_sets": set_meta,
        }
    update_manifest(manifest_path, manifest, "indexing")

    chunks_path = PROJECT_ROOT / "dataset" / args.dataset_name / "chunks.json"
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    passages = [f"{index}:{chunk}" for index, chunk in enumerate(chunks)]
    device = args.embedding_device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_path = Path(args.embedding_model)
    if not embedding_path.is_absolute() and (PROJECT_ROOT / embedding_path).exists():
        embedding_path = PROJECT_ROOT / embedding_path
    embedding_model = SentenceTransformer(str(embedding_path), device=device)
    llm_model = LLM_Model(args.llm_model, max_tokens=args.llm_max_tokens, timeout=args.llm_timeout)
    config = LinearRAGConfig(
        dataset_name=args.dataset_name,
        embedding_model=embedding_model,
        spacy_model=args.spacy_model,
        max_workers=args.max_workers,
        llm_model=llm_model,
        max_iterations=args.max_iterations,
        iteration_threshold=args.iteration_threshold,
        passage_ratio=args.passage_ratio,
        top_k_sentence=args.top_k_sentence,
        retrieval_top_k=args.retrieval_top_k,
        use_vectorized_retrieval=args.use_vectorized_retrieval,
        enable_chapter_retrieval=args.enable_chapter_retrieval,
        chapter_top_k=args.chapter_top_k,
        chapter_global_fallback=args.chapter_global_fallback,
        save_graph_index=False,
        entity_alias_path=str(args.entity_alias_path) if args.entity_alias_path else None,
        enable_alias_activation=args.enable_alias_activation,
        alias_expansion_budget=args.alias_expansion_budget,
        max_alias_expansions=args.max_alias_expansions,
    )
    rag = LinearRAG(global_config=config)
    rag.index(passages)
    update_manifest(manifest_path, manifest, "answering")

    global_offset = 0
    for set_position, (question_path, questions) in enumerate(question_sets):
        set_name = question_path.stem
        predictions_path = output_dir / "sets" / set_name / "predictions.json"
        predictions = load_existing_predictions(predictions_path, questions)
        print(f"\n=== {set_name}: resuming at {len(predictions)}/{len(questions)} ===", flush=True)

        for start in range(len(predictions), len(questions), args.question_batch_size):
            stop = min(start + args.question_batch_size, len(questions))
            batch_results = rag.qa(questions[start:stop])
            for local_index, result in enumerate(batch_results, start=start):
                result["question_set"] = set_name
                result["question_index"] = local_index
                result["global_index"] = global_offset + local_index
            predictions.extend(batch_results)
            atomic_json(predictions_path, predictions)
            manifest["question_sets"][set_position]["completed"] = len(predictions)
            update_manifest(manifest_path, manifest)
            print(f"Checkpoint {set_name}: {len(predictions)}/{len(questions)}", flush=True)
        global_offset += len(questions)

    combined = []
    for question_path, questions in question_sets:
        predictions_path = output_dir / "sets" / question_path.stem / "predictions.json"
        predictions = load_existing_predictions(predictions_path, questions)
        if len(predictions) != len(questions):
            raise RuntimeError(f"Incomplete question set: {question_path.stem}")
        combined.extend(predictions)
    combined_path = output_dir / "combined_predictions.json"
    atomic_json(combined_path, combined)
    manifest["combined_prediction_count"] = len(combined)
    manifest["combined_predictions"] = str(combined_path)
    update_manifest(manifest_path, manifest, "answers_complete")
    print(f"Combined {len(combined)} predictions into {combined_path}", flush=True)

    if args.no_evaluate:
        update_manifest(manifest_path, manifest, "complete")
        return

    scored_path = output_dir / "combined_predictions.scored.json"
    summary_path = output_dir / "evaluation_results.json"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_predictions.py"),
        "--predictions", str(combined_path),
        "--output", str(scored_path),
        "--summary", str(summary_path),
        "--judge-model", args.judge_model,
        "--workers", str(args.judge_workers),
        "--timeout", str(args.judge_timeout),
        "--checkpoint-every", str(args.judge_checkpoint_every),
        "--resume",
        "--embedding-model", str(embedding_path),
        "--embedding-device", device,
    ]
    if args.contain_only:
        command.append("--contain-only")
    update_manifest(manifest_path, manifest, "evaluating")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    manifest["scored_predictions"] = str(scored_path)
    manifest["evaluation_summary"] = str(summary_path)
    update_manifest(manifest_path, manifest, "complete")


if __name__ == "__main__":
    main()
