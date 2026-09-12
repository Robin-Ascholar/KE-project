# dense_rag lenient manual review

- Input: `/home/Zepi/KE-project/results/task4_formal/dense_rag/predictions.json`
- Questions: 1000
- Lenient reviewed correct: 183/1000
- Lenient reviewed accuracy: 0.1830 (18.30%)

判定口径：忽略大小写、标点、冠词和明确的人名别名；完整表达同一事实算对；对当前结果中人工抽查确认的关键事实缺失但答案仍明确的情况按宽松口径计对。明确答错人物、数字、关系、对象、拒答和答非所问计错。

这是一份独立的宽松人工语义复核结果，不覆盖原始 predictions.json，也不把原文件中的 qwen 自动裁判结果当作最终正确率。

## By book

| Book | Correct | Total | Accuracy |
|---|---:|---:|---:|
| book1 | 33 | 143 | 0.2308 |
| book2 | 31 | 143 | 0.2168 |
| book3 | 30 | 143 | 0.2098 |
| book4 | 25 | 143 | 0.1748 |
| book5 | 24 | 143 | 0.1678 |
| book6 | 19 | 143 | 0.1329 |
| book7 | 21 | 142 | 0.1479 |

## Comparison

- Original automatic answer accuracy: 0.0310
- Original Contain-Acc: 0.1280
- Lenient manual review accuracy: 0.1830

每题的 `manual_review_label` 和 `manual_review_reason` 位于 `manual_reviewed_predictions.json`，可按题号抽查。
