# Task 4 baseline results

Answer Accuracy now uses the lenient manual semantic review. The previous same-model automatic judge values are retained in `automatic_answer_accuracy` and in each method's `predictions.auto_judge.json` backup.
Contain-Acc, Evidence Recall@5, and latency are preserved from the original run.

| Method | Questions | Answer Accuracy (manual) | Contain-Acc | Evidence Recall@5 | Retrieval Latency Avg(ms) | Total Latency Avg(ms) |
|---|---:|---:|---:|---:|---:|---:|
| closed_book | 1000 | 0.0320 | 0.0220 | N/A | 0.00 | 111.44 |
| dense_rag | 1000 | 0.1830 | 0.1280 | 0.1193 | 43.27 | 313.12 |

## Review counts

| Method | Reviewed correct | Total | Manual accuracy | Previous automatic accuracy |
|---|---:|---:|---:|---:|
| closed_book | 32 | 1000 | 0.0320 | 0.0090 |
| dense_rag | 183 | 1000 | 0.1830 | 0.0310 |
