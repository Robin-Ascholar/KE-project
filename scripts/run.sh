#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export OPENAI_API_KEY="${OPENAI_API_KEY:-ollama}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:11434/v1}"

python run.py \
    --spacy_model en_core_web_trf \
    --embedding_model model/all-mpnet-base-v2 \
    --dataset_name harrypotter \
    --llm_model "${LLM_MODEL:-deepseek-ocr:latest}" \
    --llm_max_tokens "${LLM_MAX_TOKENS:-128}" \
    --llm_timeout "${LLM_TIMEOUT:-240}" \
    --max_workers "${MAX_WORKERS:-2}" \
    --max_iterations "${MAX_ITERATIONS:-3}" \
    --iteration_threshold "${ITERATION_THRESHOLD:-0.4}" \
    --passage_ratio "${PASSAGE_RATIO:-2.0}" \
    --top_k_sentence "${TOP_K_SENTENCE:-3}" \
    --retrieval_top_k "${RETRIEVAL_TOP_K:-8}"
