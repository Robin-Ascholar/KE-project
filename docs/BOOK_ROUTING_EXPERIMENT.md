# Book-aware Soft Routing 实验

该实验只调整检索阶段的 passage 权重，不重新生成 embedding、NER 或图索引，也不调用回答模型。实现默认关闭，因此原始 LinearRAG 行为不受影响。

## 五个实验变体

| 变体 | 书册信息来源 | 用途 |
|---|---|---|
| `baseline` | 无 | 原始全局 LinearRAG 基线 |
| `oracle` | 问题集所属书册 | 只作为理论上限，不可作为正式结果 |
| `predicted` | 问题与各册 Top-m passage 的相似度 | 固定强度的正式软路由 |
| `confidence` | 与 `predicted` 相同 | 按路由置信度自动减弱书册先验 |
| `confidence_boost` | 与 `confidence` 相同 | 只增益高概率书册，不压低其余书册证据 |

正式的 `predicted` 和 `confidence` 变体不会读取问题的真实书号来调整检索。真实书号只用于计算路由准确率、目标书册占比，以及 `oracle` 上限。

## 快速验证

每册抽取一道题，共七道题：

```bash
conda activate linearrag
python scripts/run_book_routing_experiment.py \
  --limit-per-book 1 \
  --variants baseline oracle predicted confidence \
  --retrieval-top-k 20
```

结果默认写入 `results/book_routing/retrieval_<timestamp>/`：

```text
experiment_config.json
summary.json
variants/baseline.json
variants/oracle.json
variants/predicted.json
variants/confidence.json
log.txt
```

`summary.json` 包含 Contain-Acc@5/@10/@20、MRR@20、Top-k 目标书册占比、预测书册准确率、耗时，以及各改进方案相对基线的 gained/lost 配对统计。

## 开发集调参

问题按书册分层并使用固定随机种子划分，避免因为原文件按情节顺序排列而直接截取前 20%。建议只在 `dev` 上选择参数：

```bash
python scripts/run_book_routing_experiment.py \
  --split dev \
  --seed 20260913 \
  --routing-top-m 3 \
  --routing-temperature 0.05 \
  --routing-strength 0.50 \
  --variants baseline oracle predicted confidence \
  --retrieval-top-k 20
```

建议搜索范围：

```text
routing-top-m:       1, 3, 5
routing-temperature: 0.03, 0.05, 0.10
routing-strength:    0.25, 0.50, 0.75
```

选择配置时以 `contain_at_20` 和相对基线的 gained/lost 为主，`gold_book_share_at_20` 只作为噪声诊断，不能单独证明检索质量改善。

## 测试集确认

选定唯一配置后，使用完全相同的随机种子运行测试集：

```bash
python scripts/run_book_routing_experiment.py \
  --split test \
  --seed 20260913 \
  --routing-top-m 3 \
  --routing-temperature 0.05 \
  --routing-strength 0.50 \
  --variants baseline oracle predicted confidence \
  --retrieval-top-k 20
```

只有测试集上的最佳正式变体稳定优于同轮 `baseline` 后，才进入 Qwen2.5:7B 的配对回答实验。历史 49.2% 没有完整预测产物，不能直接作为新方案的严格对照。
