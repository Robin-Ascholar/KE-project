# Combined retrieval 1000-question comparison

Model: LinearRAG with conservative entity aliases and chapter coarse-to-fine retrieval. QA model: qwen2.5:7b. Retrieval Top-K: 8.

## Results

- Calibrated semantic accuracy: **50.6%** (506/1000)
- Local semantic judge accuracy: **49.7%**
- Contain accuracy: **27.5%**
- Exact match: **21.2%**
- Token F1: **34.68%**
- ROUGE-L F1: **34.06%**
- Semantic similarity: **75.80%**
- Gold-answer lexical context hit@8: **30.2%** (baseline 28.2%)
- Lexical context MRR@8: **0.1825** (baseline 0.1510)

Compared with the prior qwen2.5:7b Top-8 baseline, calibrated semantic accuracy rose from 49.2% to 50.6% (+1.4 percentage points). There were 108 old-wrong/new-correct flips and 94 old-correct/new-wrong flips.

## Review method

368 answers were textually unchanged and inherited their prior human semantic labels. The local qwen3.6:27b judge agreed with 357/368 (97.0%) of those labels. The 632 changed answers use the local semantic judge, with representative improvement and regression cases manually inspected. The 50.6% result is therefore a calibrated hybrid semantic score rather than a new independent 1000-item human annotation.

## Limits

The dataset has reference answers but no gold supporting-fact paths, so true multi-hop path coverage cannot be calculated for this run. Lexical context hit, MRR and answer-bearing passage ratio are retained as reproducible retrieval proxies.
