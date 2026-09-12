# closed_book lenient manual review

- Input: `/home/Zepi/KE-project/results/task4_formal/closed_book/predictions.json`
- Questions: 1000
- Lenient reviewed correct: 32/1000
- Lenient reviewed accuracy: 0.0320 (3.20%)

判定口径：忽略大小写、标点、冠词和明确的人名别名；完整表达同一事实算对；对当前结果中人工抽查确认的关键事实缺失但答案仍明确的情况按宽松口径计对。明确答错人物、数字、关系、对象、拒答和答非所问计错。

这是一份独立的宽松人工语义复核结果，不覆盖原始 predictions.json，也不把原文件中的 qwen 自动裁判结果当作最终正确率。

## By book

| Book | Correct | Total | Accuracy |
|---|---:|---:|---:|
| book1 | 6 | 143 | 0.0420 |
| book2 | 6 | 143 | 0.0420 |
| book3 | 7 | 143 | 0.0490 |
| book4 | 2 | 143 | 0.0140 |
| book5 | 5 | 143 | 0.0350 |
| book6 | 2 | 143 | 0.0140 |
| book7 | 4 | 142 | 0.0282 |

## Comparison

- Original automatic answer accuracy: 0.0090
- Original Contain-Acc: 0.0220
- Lenient manual review accuracy: 0.0320

每题的 `manual_review_label` 和 `manual_review_reason` 位于 `manual_reviewed_predictions.json`，可按题号抽查。
