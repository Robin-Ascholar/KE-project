# 保守实体别名激活实验

## 结论

实体别名归组本身有效，但需要限制别名带入图检索的总权重。原始实现给每个别名节点固定 `0.85` 分，别名越多的实体会获得越大的总激活质量；在 202 题开发集上，Contain@20 因此从 43.56% 降到 38.61%。

本实验改为“一个逻辑实体共享一个扩展预算”，并仅使用 10 组含身份变化或明显不同称呼的人工核验别名。固定开发集和测试集均取得小幅正向结果，Top-10 和 Top-20 没有出现原命中丢失。因此，这个方向值得保留，但现阶段应表述为“小幅改善检索召回”，不能宣称大幅提升最终问答准确率。

## 保守策略

1. 问题中直接出现的名称作为主种子，权重为 `1.0`。
2. 同一逻辑实体的其他别名共享总预算 `0.75`，而不是每个节点都获得 `0.85`。
3. 大小写和标点差异形成的重复表面节点只激活一个代表。
4. 同一问题中由 NER 和词面匹配重复发现的逻辑实体只处理一次。
5. 每组最多扩展 8 个名称。
6. 使用单独的保守词典，排除普通名字缩写，仅保留 Voldemort/Tom Riddle、Scabbers/Peter Pettigrew、Padfoot/Sirius Black 等高价值身份别名。

当前实现仍然是查询时激活，没有破坏或重建原图中的实体节点。

## 检索实验

固定设置：七册共 1,000 题；随机种子 `20260913`；按书册划分 20% 开发集和 80% 测试集；LinearRAG Top-20；Contain-Acc 检查标准答案字符串是否出现在检索段落中。

### 开发集（全部 202 题，33 题触发）

| 指标 | Baseline | 保守别名激活 | 配对变化 |
|---|---:|---:|---:|
| Contain@5 | 26.73% | 27.72% | +2 / -0 |
| Contain@10 | 35.15% | 36.14% | +2 / -0 |
| Contain@20 | 43.56% | 45.05% | +3 / -0 |
| MRR@20 | 0.1807 | 0.1947 | +0.0140 |

Top-20 新增命中包括：

- “Whom did Sirius Black escape Azkaban to find?”：`Peter Pettigrew`，升至第 4。
- “What curse did Snape use to kill Dumbledore?”：`Avada Kedavra`，升至第 14。
- “Who was Tom Riddle's mother?”：`Merope Gaunt`，升至第 1。

### 冻结测试集（126 道触发题）

参数和词典在开发集结果产生后不再修改，只运行测试划分中会触发别名的 126 题；其余 672 题不会进入别名分支，排名保持不变。

| 指标 | Baseline | 保守别名激活 | 配对变化 |
|---|---:|---:|---:|
| Contain@5 | 15.08% | 16.67% | +5 / -3 |
| Contain@10 | 18.25% | 23.02% | +6 / -0 |
| Contain@20 | 25.40% | 28.57% | +4 / -0 |
| MRR@20 | 0.1132 | 0.1294 | +0.0162 |

Top-20 新增命中包括 Padfoot、James Potter 的 Animagus 形态、预言的提出者，以及 Snape 偷听预言后的行为。没有 Top-20 命中丢失。

折算到全部 1,000 题，Contain@20 净新增 7 个命中，即约 `+0.7` 个百分点。增益不大，但开发集和冻结测试集方向一致。

## 30 题回答抽样

使用 `.env` 配置的同一回答模型，对 Book 3 前 30 题与关闭别名的既有结果比较：

| 指标 | Baseline | 保守别名激活 |
|---|---:|---:|
| Contain-Acc | 66.67% | 73.33% |
| Exact Match | 50.00% | 50.00% |
| Token F1 | 0.7109 | 0.7170 |
| Semantic Similarity | 0.9182 | 0.9307 |

13/30 题触发别名，6 个回答发生措辞变化。Contain-Acc 新增 2、丢失 0，但新增项主要是把已经语义正确的回答改写得更接近标准答案，因此不能把 `+6.66` 个百分点全部解释为真实准确率提升。检索层的大样本配对结果是目前更可靠的证据。

## 复现

```bash
conda activate linearrag

# 开发集全部 202 题
python scripts/evaluate_entity_alias_retrieval.py \
  --split dev \
  --seed 20260913 \
  --alias-file dataset/harrypotter/entity_aliases.conservative.json \
  --alias-expansion-budget 0.75 \
  --max-alias-expansions 8 \
  --retrieval-top-k 20 \
  --output-dir results/alias_conservative/dev_202

# 冻结测试集中的触发题
python scripts/evaluate_entity_alias_retrieval.py \
  --split test \
  --only-alias-triggered \
  --seed 20260913 \
  --alias-file dataset/harrypotter/entity_aliases.conservative.json \
  --alias-expansion-budget 0.75 \
  --max-alias-expansions 8 \
  --retrieval-top-k 20 \
  --output-dir results/alias_conservative/test_triggered
```

评测脚本不会调用 LLM，也不会覆盖 `LinearRAG.graphml`。运行问答时先传入 `--enable-alias-activation`，再通过 `run_question_sets.py` 的同名参数选择词典、预算和最大扩展数。

## 局限

- 别名词典仍由人工整理，后续应从训练语料构造并在独立问题集上验证。
- Contain-Acc 依赖答案字符串，只衡量检索召回，不等同于证据完整性或回答正确率。
- 当前是查询时的逻辑归组，不是真正将索引中的实体节点合并为 canonical ID。
- Top-5 在测试触发题中仍有 3 个原命中丢失，说明权重和排序仍有改进空间。
