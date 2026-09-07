# Qwen2.5:7b 问答错误分析

## 结论

主要瓶颈是检索，其次是问题集与语料之间的表达差异和跨书歧义；Qwen2.5 在证据确实进入 Top-8 时通常能作答，但在证据缺失时会从错误上下文复制实体、依赖参数记忆或生成冗长的“不确定”回答。

## 全量结果

- 样本数：1,000
- Contain-Acc：26.60%
- Exact Match：20.50%
- Token F1：32.18%
- ROUGE-L F1：31.67%
- 空回答：0

采用项目与论文相同的规范化字符串包含规则检查证据：

| Top-8 中是否出现规范化参考答案 | 回答正确 | 回答错误 | 条件正确率 |
|---|---:|---:|---:|
| 出现 | 214 | 67 | 76.16% |
| 未出现 | 52 | 667 | 7.23% |

只有 281/1,000 个问题的参考答案字面形式进入 Top-8。参考答案字面形式在全部清洗语料中出现 594/1,000；其中大量未出现项属于摘要式、推理式或同义改写答案，不能简单解释为清洗缺失。

## 问题来源

1. **召回不足**：Top-8 没有参考答案时，正确率从 76.16% 降到 7.23%，说明错误首先由检索决定。
2. **七册混合检索产生歧义**：问题没有附带 `Book N` 条件。Top-8 平均只有 3.64 条来自问题所属书册，22.8% 的问题一条同册 passage 都没有。较大的第 5、7 册在全局召回中占比更高。
3. **问题集包含跨书同名问题**：例如 `What happened to Harry on the Hogwarts Express?` 在 book3 和 book6 有不同参考答案；不附书册信息时查询本身不唯一。
4. **参考答案经常不是原文连续片段**：如 `A warning letter about underage magic`、`He transformed into a dog`，原文可能表达同一事实但不包含完整参考字符串，导致 Contain-Acc 和字面证据召回偏低。
5. **生成模型在无证据时选错相关实体**：例如将 Harry 的昵称答为 `You-Know-Who`，将 McGonagall 安排的扫帚答为 `Malfoy`，将 Snape 误判为 Quirrell。这些回答常能在召回上下文中找到，属于“有上下文依据但选错事实”。
6. **输出仍会变长并被截断**：225 条回答超过 20 个规范化词，24 条超过 50 个词；部分回答虽然使用 answer-only 提示，仍解释“上下文未提供信息”，并在 128-token 上限处截断。
7. **Contain-Acc 存在假阴性**：在已完成的前 60 条 LLM 判分快照中，Judge Accuracy 为 58.33%，Contain-Acc 为 43.33%；有 9 条语义正确但字符串包含判错，例如 `Uncle Vernon` 对 `Vernon Dursley`、`Voldemort` 对 `Lord Voldemort`、`motorcycle` 对 `flying motorbike`。

## 分组现象

- `why` 问题：61 条，Contain-Acc 约 1.6%。这类问题需要事件因果证据，短答案字面匹配和当前检索都很弱。
- `how` 问题：70 条，Contain-Acc 约 8.6%。方法、过程和数量问题更容易需要多个句子。
- `which` 问题：46 条，Contain-Acc 约 45.7%，表现最好。
- 人物/名称问题：361 条，Contain-Acc 约 35.5%。
- 地点问题：59 条，Contain-Acc 约 39.0%。
- 时间/数量问题：39 条，Contain-Acc 约 17.9%。

## 代表性错误

- 问：`What nickname is Harry Potter known by?`；参考：`The Boy Who Lived`；预测：`You-Know-Who`。Top-8 未召回目标答案，模型选中了错误称谓。
- 问：`Who arranged for Harry to get the Nimbus Two Thousand?`；参考：`Professor McGonagall`；预测：`Malfoy`。目标答案进入上下文，但模型选择错误人物。
- 问：`How many birthday presents did Dudley count?`；参考：`Thirty-six`；预测：`37`。属于相邻数字混淆。
- 问：`What creature was being attacked in the Forbidden Forest?`；参考：`Unicorns`；预测：`unicorn`。语义正确但 Contain-Acc 因单复数判错。

## 下一步实验建议

1. 将 `book_number` 加入问题查询，并在检索阶段优先或强制过滤到对应书册。
2. 分别统计“目标书册内检索”和“七册全局检索”，验证跨书干扰。
3. 对 `why/how` 问题提高候选范围或使用重排器，再将重排后的 Top-5 交给模型。
4. 把问题集补充 `supporting_facts` 或 gold passage 标注，避免用参考答案字符串近似证据召回。
5. 最终报告同时给出 Contain-Acc 和 LLM Judge Accuracy，并单独报告同义词、别名、单复数导致的字符串假阴性。
