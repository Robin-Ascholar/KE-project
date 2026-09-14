# -*- coding: utf-8 -*-
"""Web API and static-file server for the Harry Potter LinearRAG demo."""

from __future__ import annotations

import gc
import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

try:
    from flask_cors import CORS
except ImportError:
    CORS = None

FRONTEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FRONTEND_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(FRONTEND_DIR / ".env", override=True)
os.environ.setdefault("OPENAI_API_KEY", "ollama")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")

DATASET_NAME = os.getenv("RAG_DATASET", "harrypotter")
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "model/all-mpnet-base-v2")
QA_MODEL = os.getenv("RAG_QA_MODEL", "qwen2.5:7b")
SPACY_MODEL = os.getenv("RAG_SPACY_MODEL", "en_core_web_trf")
EMBEDDING_DEVICE = os.getenv("RAG_EMBEDDING_DEVICE", "cuda")
REQUEST_TIMEOUT = float(os.getenv("RAG_REQUEST_TIMEOUT", "240"))
DEFAULT_TOP_K = max(1, min(20, int(os.getenv("RAG_TOP_K", "8"))))
CHAPTER_TOP_K = max(1, int(os.getenv("RAG_CHAPTER_TOP_K", "8")))
CHAPTER_GLOBAL_FALLBACK = max(0, int(os.getenv("RAG_CHAPTER_GLOBAL_FALLBACK", "2")))
ENABLE_CHAPTER_RETRIEVAL = os.getenv("RAG_ENABLE_CHAPTER_RETRIEVAL", "1").lower() in {
    "1", "true", "yes", "on"
}
AUTO_LOAD = os.getenv("RAG_AUTO_LOAD", "1").lower() in {"1", "true", "yes", "on"}
FRONTEND_DEBUG = os.getenv("FRONTEND_DEBUG", "0").lower() in {"1", "true", "yes", "on"}

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
if CORS is not None:
    CORS(app)

logging.basicConfig(
    level=os.getenv("RAG_LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("rag_frontend")

_runtime: dict[str, Any] | None = None
_runtime_lock = threading.RLock()
_runtime_state: dict[str, Any] = {
    "status": "idle",
    "detail": "等待加载 LinearRAG 索引",
    "error": None,
    "started_at": None,
    "loaded_at": None,
}


def _problem(title: str, detail: str, status: int):
    return jsonify({
        "ok": False,
        "type": "about:blank",
        "title": title,
        "detail": detail,
        "status": status,
    }), status


def _dataset_paths(dataset: str) -> tuple[Path, Path]:
    return (
        PROJECT_ROOT / "dataset" / dataset / "chunks.json",
        PROJECT_ROOT / "import" / dataset / "LinearRAG.graphml",
    )


def _validate_dataset(dataset: str) -> str:
    dataset = str(dataset or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", dataset):
        raise ValueError("数据集名称只能包含字母、数字、下划线和连字符")
    chunks_path, _ = _dataset_paths(dataset)
    if not chunks_path.is_file():
        raise FileNotFoundError(f"未找到数据集文件：{chunks_path}")
    return dataset


def _set_state(status: str, detail: str, error: str | None = None) -> None:
    _runtime_state.update({"status": status, "detail": detail, "error": error})
    if status == "loading":
        _runtime_state["started_at"] = time.time()
    if status == "ready":
        _runtime_state["loaded_at"] = time.time()


def _load_runtime(dataset: str | None = None, force: bool = False) -> dict[str, Any]:
    """Load embeddings, cached graph and the QA client once for interactive use."""
    global _runtime, DATASET_NAME

    dataset = _validate_dataset(dataset or DATASET_NAME)
    with _runtime_lock:
        if _runtime and _runtime["dataset"] == dataset and not force:
            return _runtime

        _set_state("loading", f"正在加载 {dataset} 的图谱与向量索引")
        started = time.perf_counter()
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            from src.config import LinearRAGConfig
            from src.LinearRAG import LinearRAG
            from src.utils import LLM_Model

            device = (
                "cuda"
                if EMBEDDING_DEVICE.lower().startswith("cuda") and torch.cuda.is_available()
                else "cpu"
            )
            embedding_path = Path(EMBEDDING_MODEL)
            if not embedding_path.is_absolute():
                embedding_path = PROJECT_ROOT / embedding_path

            chunks_path, graph_path = _dataset_paths(dataset)
            chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            passages = [f"{index}:{chunk}" for index, chunk in enumerate(chunks)]

            embedding_model = SentenceTransformer(str(embedding_path), device=device)
            llm_model = LLM_Model(
                QA_MODEL,
                max_tokens=int(os.getenv("RAG_LLM_MAX_TOKENS", "128")),
                timeout=REQUEST_TIMEOUT,
            )
            config = LinearRAGConfig(
                dataset_name=dataset,
                embedding_model=embedding_model,
                spacy_model=SPACY_MODEL,
                working_dir=str(PROJECT_ROOT / "import"),
                max_workers=1,
                llm_model=llm_model,
                retrieval_top_k=DEFAULT_TOP_K,
                max_iterations=3,
                iteration_threshold=0.4,
                passage_ratio=2,
                top_k_sentence=3,
                use_vectorized_retrieval=True,
                enable_chapter_retrieval=ENABLE_CHAPTER_RETRIEVAL,
                chapter_top_k=CHAPTER_TOP_K,
                chapter_global_fallback=CHAPTER_GLOBAL_FALLBACK,
            )
            rag = LinearRAG(global_config=config)
            rag.index(passages)

            elapsed_ms = int((time.perf_counter() - started) * 1000)
            DATASET_NAME = dataset
            _runtime = {
                "dataset": dataset,
                "rag": rag,
                "chunks": len(chunks),
                "chapters": len(getattr(rag, "chapter_keys", [])),
                "entities": len(rag.entity_embedding_store.text_to_hash_id),
                "sentences": len(rag.sentence_embedding_store.text_to_hash_id),
                "nodes": rag.graph.vcount(),
                "edges": rag.graph.ecount(),
                "device": device,
                "graph_path": str(graph_path.relative_to(PROJECT_ROOT)),
                "loaded_ms": elapsed_ms,
            }
            _set_state("ready", f"{dataset} 索引已加载")
            logger.info("LinearRAG runtime ready in %d ms", elapsed_ms)
            return _runtime
        except Exception as exc:
            _runtime = None
            _set_state("error", "LinearRAG 索引加载失败", str(exc))
            logger.exception("Failed to initialize LinearRAG runtime")
            raise


def _load_runtime_background(dataset: str | None = None, force: bool = False) -> None:
    try:
        _load_runtime(dataset, force=force)
    except Exception:
        pass


def _runtime_status() -> dict[str, Any]:
    chunks_path, graph_path = _dataset_paths(DATASET_NAME)
    runtime = _runtime
    stats = {
        "dataset": DATASET_NAME,
        "chunks": runtime.get("chunks") if runtime else None,
        "chapters": runtime.get("chapters") if runtime else None,
        "entities": runtime.get("entities") if runtime else None,
        "sentences": runtime.get("sentences") if runtime else None,
        "nodes": runtime.get("nodes") if runtime else None,
        "edges": runtime.get("edges") if runtime else None,
        "device": runtime.get("device") if runtime else None,
        "graphPath": runtime.get("graph_path") if runtime else str(graph_path.relative_to(PROJECT_ROOT)),
        "chunksPath": str(chunks_path.relative_to(PROJECT_ROOT)),
        "indexExists": graph_path.is_file(),
        "datasetExists": chunks_path.is_file(),
    }
    return {
        "ok": True,
        "app": "Harry Potter LinearRAG",
        "backend": {
            "reachable": _runtime_state["status"] == "ready",
            "status": _runtime_state["status"],
            "detail": _runtime_state["detail"],
            "error": _runtime_state["error"],
        },
        "dataset": DATASET_NAME,
        "defaultDataset": DATASET_NAME,
        "defaultCsvPath": DATASET_NAME,
        "apiBase": "/api",
        "qaModel": QA_MODEL,
        "embeddingModel": EMBEDDING_MODEL,
        "topK": runtime["rag"].config.retrieval_top_k if runtime else DEFAULT_TOP_K,
        "chapterRetrieval": ENABLE_CHAPTER_RETRIEVAL,
        "timeoutSeconds": REQUEST_TIMEOUT,
        "stats": stats,
    }


def _last_user_message(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages or []):
        if message.get("role") in {"user", "human"} and str(message.get("content", "")).strip():
            return str(message["content"]).strip()
    return ""


def _passage_payload(passage: str, score: Any, rank: int) -> dict[str, Any]:
    clean = re.sub(r"^\d+:\s*", "", str(passage), count=1)
    match = re.match(r"Book\s+(\d+)\s+\|\s+Chapter\s+(\d+):\s*([^\n]*)", clean, re.I)
    item: dict[str, Any] = {"rank": rank, "score": float(score), "passage": clean}
    if match:
        item.update({
            "book": int(match.group(1)),
            "chapter": int(match.group(2)),
            "chapter_title": match.group(3).strip(),
        })
    return item


def _graph_payload(
    runtime: dict[str, Any],
    result: dict[str, Any],
    question: str,
    max_passages: int = 8,
    max_entities: int = 120,
) -> dict[str, Any]:
    """Build a query-local graph with real mentions and one-hop co-occurrences."""
    rag = runtime["rag"]
    graph = rag.graph
    passage_rows: list[tuple[str, str, float, int]] = []
    scores = result.get("sorted_passage_scores", [])

    for rank, passage in enumerate(result.get("sorted_passage", [])[:max_passages], 1):
        passage_id = rag.passage_embedding_store.text_to_hash_id.get(passage)
        if not passage_id or passage_id not in rag.node_name_to_vertex_idx:
            continue
        score = float(scores[rank - 1]) if rank - 1 < len(scores) else 0.0
        passage_rows.append((passage_id, passage, score, rank))

    seed_ids: set[str] = set()
    try:
        _, _seed_texts, seed_hash_ids, _seed_scores = rag.get_seed_entities(question)
        seed_ids.update(seed_hash_ids)
    except Exception:
        logger.debug("Could not recompute graph seed entities", exc_info=True)

    candidates: dict[str, dict[str, Any]] = {}
    mentions: list[tuple[str, str, float]] = []
    for passage_id, _passage, passage_score, _rank in passage_rows:
        vertex_index = rag.node_name_to_vertex_idx[passage_id]
        for neighbor_index in graph.neighbors(vertex_index):
            vertex = graph.vs[neighbor_index]
            entity_id = vertex["name"]
            if not str(entity_id).startswith("entity-"):
                continue
            edge_id = graph.get_eid(vertex_index, neighbor_index, directed=False, error=False)
            edge_weight = float(graph.es[edge_id]["weight"]) if edge_id >= 0 else 0.0
            item = candidates.setdefault(entity_id, {
                "id": entity_id,
                "label": str(vertex["content"]),
                "score": 0.0,
                "degree": 0,
                "is_direct": True,
            })
            item["score"] += max(passage_score, 0.000001) * max(edge_weight, 0.000001)
            item["degree"] += 1
            item["is_direct"] = True
            mentions.append((entity_id, passage_id, edge_weight))

    # Expand through passages shared with the strongest directly retrieved entities.
    # These links are marked as co-occurrence instead of being presented as authored relations.
    direct_ids = list(candidates)
    cooccurrences: dict[tuple[str, str], int] = {}
    for anchor_id in direct_ids:
        if len(candidates) >= max_entities * 2:
            break
        anchor_index = rag.node_name_to_vertex_idx.get(anchor_id)
        if anchor_index is None:
            continue
        context_passages = [
            index for index in graph.neighbors(anchor_index)
            if str(graph.vs[index]["name"]).startswith("passage-")
        ][:4]
        for context_index in context_passages:
            for neighbor_index in graph.neighbors(context_index):
                entity_id = graph.vs[neighbor_index]["name"]
                if not str(entity_id).startswith("entity-") or entity_id == anchor_id:
                    continue
                item = candidates.setdefault(entity_id, {
                    "id": entity_id,
                    "label": str(graph.vs[neighbor_index]["content"]),
                    "score": candidates[anchor_id]["score"] * 0.25,
                    "degree": 1,
                    "is_direct": False,
                })
                if not item.get("is_direct"):
                    item["degree"] += 1
                pair = tuple(sorted((anchor_id, entity_id)))
                cooccurrences[pair] = cooccurrences.get(pair, 0) + 1
                if len(candidates) >= max_entities * 2:
                    break
            if len(candidates) >= max_entities * 2:
                break

    # Keep detected query entities visible even when they are not attached to a top passage.
    for seed_id in seed_ids:
        if seed_id not in rag.node_name_to_vertex_idx:
            continue
        vertex = graph.vs[rag.node_name_to_vertex_idx[seed_id]]
        candidates.setdefault(seed_id, {
            "id": seed_id,
            "label": str(vertex["content"]),
            "score": 1.0,
            "degree": 0,
            "is_direct": True,
        })

    question_lower = question.casefold()
    ranked_entities = sorted(
        candidates.values(),
        key=lambda item: (
            item["id"] in seed_ids,
            item["label"].casefold() in question_lower,
            item.get("is_direct", False),
            item["degree"],
            item["score"],
        ),
        reverse=True,
    )[:max_entities]
    selected_entities = {item["id"] for item in ranked_entities}

    nodes: list[dict[str, Any]] = []
    for item in ranked_entities:
        nodes.append({
            **item,
            "type": "entity",
            "is_seed": item["id"] in seed_ids or item["label"].casefold() in question_lower,
        })

    for passage_id, passage, score, rank in passage_rows:
        clean = re.sub(r"^\d+:\s*", "", str(passage), count=1)
        match = re.match(r"Book\s+(\d+)\s+\|\s+Chapter\s+(\d+):\s*([^\n]*)", clean, re.I)
        label = f"Evidence {rank}"
        metadata: dict[str, Any] = {"rank": rank, "score": score, "passage": clean}
        if match:
            label = f"Book {match.group(1)} · Ch {match.group(2)}"
            metadata.update({
                "book": int(match.group(1)),
                "chapter": int(match.group(2)),
                "chapter_title": match.group(3).strip(),
            })
        nodes.append({"id": passage_id, "label": label, "type": "passage", **metadata})

    edges = [
        {"source": entity_id, "target": passage_id, "label": "appears in", "weight": weight}
        for entity_id, passage_id, weight in mentions
        if entity_id in selected_entities
    ]
    remaining_edge_budget = max(0, 260 - len(edges))
    cooccurrence_edges = [
        (pair, weight)
        for pair, weight in sorted(cooccurrences.items(), key=lambda item: item[1], reverse=True)
        if pair[0] in selected_entities and pair[1] in selected_entities
    ]
    covered_entities = {edge["source"] for edge in edges if edge["source"] in selected_entities}
    added_pairs: set[tuple[str, str]] = set()

    # First give every expanded entity a visible connection, then use the rest
    # of the budget for the strongest repeated co-occurrences.
    for (source, target), weight in cooccurrence_edges:
        if remaining_edge_budget <= 0:
            break
        if source in covered_entities and target in covered_entities:
            continue
        edges.append({"source": source, "target": target, "label": "co-occurs", "weight": weight})
        added_pairs.add((source, target))
        covered_entities.update((source, target))
        remaining_edge_budget -= 1
    for (source, target), weight in cooccurrence_edges:
        if remaining_edge_budget <= 0:
            break
        if (source, target) in added_pairs:
            continue
        edges.append({"source": source, "target": target, "label": "co-occurs", "weight": weight})
        remaining_edge_budget -= 1

    return {
        "nodes": nodes,
        "edges": edges,
        "legend": {
            "seed": "查询种子实体",
            "entity": "相关实体",
            "passage": "检索段落",
            "mention": "实体出现在段落中",
            "cooccurrence": "实体在其他段落中共现",
        },
        "stats": {
            "entities": len(ranked_entities),
            "passages": len(passage_rows),
            "relations": len(edges),
            "seed_entities": sum(bool(node.get("is_seed")) for node in nodes),
        },
    }


@app.after_request
def _cors_headers(response):
    if CORS is None:
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/css/<path:filename>")
def serve_css(filename: str):
    return send_from_directory(FRONTEND_DIR / "css", filename)


@app.route("/js/<path:filename>")
def serve_js(filename: str):
    return send_from_directory(FRONTEND_DIR / "js", filename)


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify(_runtime_status())


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = str(data.get("question") or "").strip() or _last_user_message(data.get("messages", []))
    if not question:
        return _problem("请求参数错误", "缺少字段：question", 400)

    try:
        top_k = max(1, min(20, int(data.get("top_k", DEFAULT_TOP_K))))
    except (TypeError, ValueError):
        return _problem("请求参数错误", "top_k 必须是 1–20 的整数", 400)

    started = time.perf_counter()
    try:
        runtime = _load_runtime()
        with _runtime_lock:
            runtime["rag"].config.retrieval_top_k = top_k
            result = runtime["rag"].qa([{"question": question, "answer": ""}])[0]
    except Exception as exc:
        logger.exception("Question answering failed")
        return _problem("LinearRAG 问答失败", str(exc), 500)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    passages = result.get("sorted_passage", [])
    scores = result.get("sorted_passage_scores", [])
    evidence = [
        _passage_payload(passage, scores[index] if index < len(scores) else 0.0, index + 1)
        for index, passage in enumerate(passages)
    ]
    retrieval_mode = result.get("retrieval_mode", "linearrag_graph")
    diagnostics = {
        "dataset": runtime["dataset"],
        "qa_model": QA_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "device": runtime["device"],
        "top_k": top_k,
        "retrieval_mode": retrieval_mode,
        "selected_chapters": result.get("selected_chapters", []),
        "chapter_candidate_passages": result.get("chapter_candidate_passages"),
        "chapter_global_fallback": result.get("chapter_global_fallback"),
    }
    graph_payload = _graph_payload(runtime, result, question)
    return jsonify({
        "ok": True,
        "answer": result.get("pred_answer", ""),
        "content": result.get("pred_answer", ""),
        "context": evidence,
        "context_raw": [],
        "elapsed_ms": elapsed_ms,
        "route": retrieval_mode,
        "route_confidence": len(result.get("selected_chapters", [])),
        "missing_slots": [f"top_k={top_k}"],
        "answer_file": QA_MODEL,
        "query_ir": diagnostics,
        "diagnostics": diagnostics,
        "graph": graph_payload,
        "raw": {"question": question, "answer": result.get("pred_answer", ""), "evidence": evidence, "graph": graph_payload, **diagnostics},
    })


@app.route("/api/top-k", methods=["POST"])
def set_top_k():
    data = request.get_json(silent=True) or {}
    try:
        top_k = max(1, min(20, int(data.get("top_k"))))
    except (TypeError, ValueError):
        return _problem("请求参数错误", "top_k 必须是 1–20 的整数", 400)
    with _runtime_lock:
        if _runtime:
            _runtime["rag"].config.retrieval_top_k = top_k
    return jsonify({"ok": True, "top_k": top_k})


@app.route("/api/load-index", methods=["POST"])
@app.route("/api/create-graph", methods=["POST"])
def load_index():
    data = request.get_json(silent=True) or {}
    dataset = data.get("dataset") or data.get("path") or DATASET_NAME
    try:
        runtime = _load_runtime(str(dataset), force=True)
    except (ValueError, FileNotFoundError) as exc:
        return _problem("数据集不可用", str(exc), 400)
    except Exception as exc:
        return _problem("索引加载失败", str(exc), 500)
    return jsonify({
        "ok": True,
        "dataset": runtime["dataset"],
        "elapsed_ms": runtime["loaded_ms"],
        "stats": _runtime_status()["stats"],
    })


@app.route("/api/unload", methods=["POST"])
@app.route("/api/clear-graph", methods=["POST"])
def unload_runtime():
    global _runtime
    data = request.get_json(silent=True) or {}
    if data.get("confirm") is not True:
        return _problem("请求参数错误", "卸载运行时需要 confirm=true", 400)
    with _runtime_lock:
        was_loaded = _runtime is not None
        _runtime = None
        _set_state("idle", "运行时已卸载，索引文件仍保留")
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    return jsonify({"ok": True, "result": {"unloaded": was_loaded}})


if __name__ == "__main__":
    frontend_host = os.getenv("FRONTEND_HOST", "0.0.0.0")
    frontend_port = int(os.getenv("FRONTEND_PORT", "5000"))
    if AUTO_LOAD:
        threading.Thread(target=_load_runtime_background, name="linearrag-loader", daemon=True).start()

    print("=" * 64)
    print("Harry Potter LinearRAG")
    print(f"Frontend: http://{frontend_host}:{frontend_port}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"QA model: {QA_MODEL}")
    print(f"Embedding: {EMBEDDING_MODEL}")
    print("=" * 64)
    app.run(
        host=frontend_host,
        port=frontend_port,
        debug=FRONTEND_DEBUG,
        use_reloader=False,
        threaded=True,
    )
