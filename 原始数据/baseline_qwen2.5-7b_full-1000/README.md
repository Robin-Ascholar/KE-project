# Baseline qwen2.5:7b Full 1000-Question Run

This directory stores the original LinearRAG baseline outputs before task-specific innovations were added.

## Scope

- Dataset: Harry Potter seven-book question set, 1,000 questions
- QA model: `qwen2.5:7b`
- Retrieval: original LinearRAG baseline, without task1 alias activation and without task3 chapter retrieval
- Original local source directory: `results/harrypotter/batch_2026-09-07_qwen2.5-7b_full-1000/`

## Files

- `combined_predictions.json`: merged 1,000-question predictions.
- `combined_predictions.scored.json`: per-question scored predictions from the original review snapshot.
- `combined_predictions.deterministic-scored.json`: deterministic/manual-label scored snapshot.
- `combined_predictions.relaxed_scored.json`: later relaxed semantic review snapshot.
- `evaluation_results.deterministic.json`: original human semantic review summary, 492/1000 = 49.2%.
- `evaluation_results.relaxed.json`: relaxed semantic review summary, 478/1000 = 47.8%.
- `run_manifest.json`: run metadata and per-book completion counts.
- `sets/book*/predictions.json`: per-book prediction files.

