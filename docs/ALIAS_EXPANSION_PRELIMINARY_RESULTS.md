# Alias-aware Entity Activation 初步实验（2026-09-13）

## 结论

实体别名扩展在开发划分和固定测试划分上都产生了小幅正向结果，且测试划分的 Top-20 配对结果为新增 5 个命中、丢失 1 个。与书册路由不同，这个方向目前**值得继续验证**，但增益较小，且人工词典可能带来评测偏好，暂时不能写成已经成立的创新结论。

## 最小实现

- 使用 10 组人工核验的 Harry Potter 人物别名，例如：
  - `Lord Voldemort / Tom Riddle / You-Know-Who / the Dark Lord`
  - `Peter Pettigrew / Wormtail / Scabbers`
  - `Sirius Black / Padfoot`
  - `Severus Snape / the Half-Blood Prince`
- 在查询中匹配到任一别名后，同步激活该组其他实体节点。
- 别名扩展总预算为原始种子权重的 `0.75`，并在新增节点之间均分，避免别名较多的人物支配 Personalized PageRank。
- 只做查询时扩展，不重建 embedding、NER 或图索引，也不调用回答模型。
- 默认关闭，因此不影响原始 LinearRAG。

当前版本属于 **query-time alias activation**，还没有真正把图中的多个实体节点合并成统一 canonical ID。这样更适合先验证假设，也避免直接破坏现有索引。

## 实验设置

- 数据：七册共 1,000 道问题，按书册分层随机划分 20% 开发集和 80% 测试集
- 随机种子：`20260913`
- 检索：LinearRAG，Top-20
- 指标：规范化答案字符串是否出现在检索 passage 中（Contain-Acc）和 MRR@20
- 参数只在开发集确定；测试时保持别名词典和 expansion budget 不变
- 测试划分只运行 125 道会触发别名的题；其余 673 道不会发生任何代码路径或排名变化

## 开发集结果

开发集共 202 题，其中 32 题触发别名扩展。

### 全部 202 题

| 方案 | Contain@5 | Contain@10 | Contain@20 | MRR@20 |
|---|---:|---:|---:|---:|
| Baseline | 26.73% | 35.15% | 43.56% | 0.1809 |
| Alias expansion | 27.72% | 35.64% | 44.06% | 0.1893 |

### 32 道触发题

| 方案 | Contain@5 | Contain@10 | Contain@20 | MRR@20 |
|---|---:|---:|---:|---:|
| Baseline | 31.25% | 34.38% | 34.38% | 0.1990 |
| Alias expansion | 37.50% | 37.50% | 37.50% | 0.2516 |

Top-20 配对结果为新增 1、丢失 0。代表性改善：

- “Who was Tom Riddle's mother?”：答案 `Merope Gaunt` 从 Top-20 外提升到第 1。
- “What is Scabbers's true identity?”：答案 `Peter Pettigrew` 从第 2 提升到第 1。
- “What did James, Sirius, Remus and Peter create at Hogwarts?”：答案从第 12 提升到第 5。

## 固定测试划分结果

剩余测试划分中有 125 道题触发别名。没有运行其余 673 道题，因为关闭与开启别名扩展时，它们的种子实体和排名应完全相同；开发集也验证了排名变化只发生在触发题上。

| 方案 | Contain@5 | Contain@10 | Contain@20 | MRR@20 |
|---|---:|---:|---:|---:|
| Baseline | 15.20% | 19.20% | 25.60% | 0.1132 |
| Alias expansion | 17.60% | 21.60% | 28.80% | 0.1150 |

配对变化：

| 指标 | 新增命中 | 丢失命中 | 净变化 |
|---|---:|---:|---:|
| Contain@5 | 5 | 2 | +3 |
| Contain@10 | 6 | 3 | +3 |
| Contain@20 | 5 | 1 | +4 |

Top-20 新增命中包括 Tom Riddle 全名、Sirius Black 被谁杀死、Padfoot、James Potter 的 Animagus 形态，以及谁摧毁日记 Horcrux。唯一丢失的是 Harry 与 Voldemort 预言内容问题，原答案由第 17 位跌出 Top-20。

若把未触发题视为严格不变，开发集与测试集合计后，别名扩展在全部 1,000 题上的净变化约为：Contain@5 `+0.5`、Contain@10 `+0.4`、Contain@20 `+0.5` 个百分点，MRR@20 约 `+0.0019`。这是可复现的小幅改善，不是大幅提升。

## 限制

1. **人工词典存在数据泄漏风险。** 当前词典由项目成员依据领域知识和现有语料整理，部分问题本身直接询问别名关系，例如 Tom Riddle 的全名或 Sirius 的昵称。该固定测试只能称为初步复现，不能当作严格盲测。
2. **Contain-Acc 只检查答案字符串。** 它不能证明检索 passage 完整支持答案，也不能替代最终问答正确率。
3. **触发范围较窄。** 只有 157/1,000 道题触发词典，因此即使触发子集改善明显，总体提升仍有限。
4. **不是实体消歧。** 当前方法能够扩展同义名称，但没有根据上下文处理同名人物、假扮身份或时间变化。
5. **排名并非单向改善。** 测试集仍有 1 个 Top-20 命中丢失，说明扩展权重还需要更细粒度控制。

## 下一步建议

如果继续这个方向，优先级应为：

1. 从小说正文或训练划分自动抽取/核验别名，冻结词典后再建立真正的盲测集，避免使用测试答案整理词典。
2. 单独标注 alias-sensitive 问题，并区分“问题直接询问别名”和“别名只是检索入口”的问题。
3. 在固定词典上比较 `0.25 / 0.50 / 0.75` 的扩展预算，仅在开发集选一次参数。
4. 参数确定后，再做 Qwen2.5:7B 的配对回答实验，并人工复核所有新增与丢失案例。
5. 只有回答准确率和证据质量也改善后，再考虑把查询时激活升级为真正的 canonical entity ID 合并。

## 复现命令

```bash
conda activate linearrag

# 开发集
python scripts/run_alias_expansion_experiment.py \
  --split dev \
  --seed 20260913 \
  --alias-expansion-budget 0.75 \
  --retrieval-top-k 20

# 固定测试划分中实际触发别名的题
python scripts/run_alias_expansion_experiment.py \
  --split test \
  --seed 20260913 \
  --only-alias-triggered \
  --alias-expansion-budget 0.75 \
  --retrieval-top-k 20
```

机器可读结果位于本地：

```text
results/alias_expansion/dev_budget075_2026-09-13/
results/alias_expansion/test_triggered_budget075_2026-09-13/
```
