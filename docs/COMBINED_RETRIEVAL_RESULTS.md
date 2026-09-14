# 章节粗检索与保守实体别名组合实验

## 结论

`task3` 的章节粗到细检索和保守实体别名激活可以组合运行。组合方案在固定开发集和冻结测试集上都明显提高 Top-5、Top-10 和 MRR，并将平均候选 passage 从 6,865 个缩小到约 540 个；但冻结测试集 Contain@20 小幅下降。因此两个功能均以显式开关进入主线，默认关闭，不改变原始 LinearRAG 行为。

适合在汇报中表述为：

> 章节路由显著提高了高排名证据的集中度，并缩小约 92% 的候选范围；保守实体别名激活在此基础上继续提供小幅补充。当前代价是 Top-20 长尾召回略有下降。

## 集成内容

- `origin/main`：包含任务4基线和现有项目代码。
- `origin/task3`：章节元数据解析、章节均值向量、Top-K 章节筛选和全局 fallback。
- `experiment/yyq-conservative-alias-budget`：逻辑实体归组、重复表面节点去重和组内共享扩展预算。

合并时保留双方独立开关和诊断字段。章节功能只在启用时解析 Harry Potter 章节前缀，避免破坏其他数据集。任务3提交中与检索创新无关的 GraphML 重写和宽松裁判提示词未带入最终集成结果。

## 固定设置

- 七册共 1,000 题，按书册划分 202 题开发集和 798 题测试集。
- 随机种子：`20260913`。
- 检索深度：Top-20，同时统计 Contain@5、Contain@10、Contain@20 和 MRR@20。
- 保守别名：10 组，组内扩展共享 `0.75` 总预算，最多 8 个扩展名称。
- 章节参数只在开发集选择：`chapter_top_k=16`，`chapter_global_fallback=4`。
- 参数冻结后在 798 题测试集上运行一次，不依据测试集继续调参。

## 开发集结果（202题）

| 方案 | Contain@5 | Contain@10 | Contain@20 | MRR@20 |
|---|---:|---:|---:|---:|
| 原始基线 | 26.73% | 35.15% | 43.56% | 0.1809 |
| 仅保守别名 | 28.22% | 36.14% | 45.05% | 0.1951 |
| 仅章节路由 | 33.17% | 38.12% | 43.07% | 0.2275 |
| 章节 + 别名 | **34.16%** | **38.61%** | **44.06%** | **0.2340** |

组合方案平均保留 539 个候选 passage。相对基线，Contain@5 新增 19、丢失 4；Contain@10 新增 15、丢失 8；Contain@20 新增 4、丢失 3。

## 冻结测试集结果（798题）

| 方案 | Contain@5 | Contain@10 | Contain@20 | MRR@20 |
|---|---:|---:|---:|---:|
| 原始基线 | 21.30% | 29.82% | **38.72%** | 0.1521 |
| 仅保守别名 | 21.55% | 30.45% | **39.22%** | 0.1548 |
| 仅章节路由 | 25.94% | 31.58% | 36.97% | 0.1904 |
| 章节 + 别名 | **26.32%** | **31.83%** | 36.97% | **0.1943** |

组合方案平均保留 541 个候选 passage，相对全量候选减少约 92.1%。相对基线的配对变化：

- Contain@5：新增 59、丢失 19，净增 40。
- Contain@10：新增 44、丢失 28，净增 16。
- Contain@20：新增 21、丢失 35，净减 14。

所以该方案改善的是高排名证据和候选集中度，不是所有检索深度上的无条件召回提升。

## 30题端到端回答抽样

使用 `.env` 中相同的回答模型，对 Book 3 前30题运行组合方案，并与既有原始基线比较：

| 指标 | 原始基线 | 章节 + 别名 |
|---|---:|---:|
| Contain-Acc | 66.67% | 70.00% |
| Exact Match | 50.00% | 43.33% |
| Token F1 | 0.7109 | 0.6893 |
| Semantic Similarity | 0.9182 | 0.9197 |

逐题复核发现，6 个变化回答的核心语义均未恶化；Exact Match 和 Token F1 下降主要来自答案更长，例如 `A stag` 变为 `A silver stag`。该样本只支持“回答语义基本保持”，不能证明端到端准确率显著提升。

## 使用方法

原始基线仍是默认行为：

```bash
python scripts/run_question_sets.py
```

启用两个实验功能：

```bash
python scripts/run_question_sets.py \
  --enable-chapter-retrieval \
  --chapter-top-k 16 \
  --chapter-global-fallback 4 \
  --enable-alias-activation \
  --entity-alias-path dataset/harrypotter/entity_aliases.conservative.json \
  --alias-expansion-budget 0.75 \
  --max-alias-expansions 8
```

复现四方案无 LLM 检索对照：

```bash
python scripts/evaluate_entity_alias_retrieval.py \
  --split dev \
  --seed 20260913 \
  --compare-chapter-variants \
  --chapter-top-k 16 \
  --chapter-global-fallback 4 \
  --retrieval-top-k 20 \
  --output-dir results/integration/dev_202
```

## 限制与下一步

1. 当前章节表示是章内 passage embedding 的均值，可能稀释只占一小段的关键情节。
2. 当前实现仍计算全量 Dense 相似度和 PPR，候选减少不等同于端到端运行时间减少；主要收益是最终候选范围和上下文集中度。
3. 下一步可比较 max-pooling 或标题与正文的加权章节表示，并使用独立验证集，不能继续依据本次冻结测试集调参。
4. 若最终系统固定向 LLM 提供20段上下文，应优先使用“仅保守别名”；若更关注 Top-5 证据集中度和可解释的章节路径，可以启用组合方案。
