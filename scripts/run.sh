#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python run.py \
    --spacy_model en_core_web_trf \
    --embedding_model model/all-mpnet-base-v2 \
    --dataset_name harrypotter \
    --llm_model "${LLM_MODEL:-gpt-4o-mini}" \
    --max_workers "${MAX_WORKERS:-4}" \
    --max_iterations "${MAX_ITERATIONS:-3}" \
    --iteration_threshold "${ITERATION_THRESHOLD:-0.4}" \
    --passage_ratio "${PASSAGE_RATIO:-2.0}" \
    --top_k_sentence "${TOP_K_SENTENCE:-3}"
