# 实体别名扩展

任务1以独立模块实现，不改写现有图中的实体节点，也不改变 `get_seed_entities()` 之外的检索流程。

## 文件

- `src/entity_alias.py`：别名归一化、规范实体解析、查询短语匹配和候选实体扩展。
- `dataset/harrypotter/entity_aliases.json`：人工复核后的别名组数据。
- `dataset/harrypotter/entity_aliases.conservative.json`：用于正式对照实验的 10 组高价值身份别名。
- `tests/test_entity_alias.py`：标准库单元测试。
- `scripts/validate_entity_aliases.py`：检查别名组与现有 NER 实体索引的覆盖情况。

别名数据采用以下结构：

```json
{
  "canonical_name": "Voldemort",
  "aliases": ["Lord Voldemort", "Tom Riddle", "You-Know-Who"],
  "entity_id": "alias-..."
}
```

`entity_id` 不必写入 JSON，模块会依据规范名稳定生成。现有图仍使用原始实体字符串；逻辑实体 ID 只用于别名层的归组和审计。

## 查询阶段行为

1. 使用 spaCy NER 提取问题中的实体。
2. 在问题原文中补充匹配配置别名，覆盖 NER 漏识别的称呼。
3. 将别名规范化到 `canonical_name`，并激活当前实体索引中同组的可用表面形式。
4. 原有实体未命中别名组时，继续使用原来的 embedding 最近实体路径。

问题中直接出现的名称保留 `1.0` 主种子权重，同一逻辑实体的其他名称共享默认总预算 `0.75`，每组最多扩展 8 个名称。大小写和标点差异形成的重复节点只选一个代表，避免别名多的实体仅凭名称数量支配 PPR。不同实体共享的歧义别名会被跳过，避免错误激活。

典型覆盖包括：

- `Voldemort` / `Tom Riddle` / `You-Know-Who` / `Dark Lord`
- `Harry Potter` / `The Boy Who Lived` / `The Chosen One`
- `Sirius Black` / `Padfoot` / `Snuffles`
- `Peter Pettigrew` / `Wormtail` / `Scabbers`
- `Severus Snape` / `The Half-Blood Prince`

## 验证与对照

验证别名文件和现有 NER 索引：

```bash
python scripts/validate_entity_aliases.py
```

别名扩展默认关闭，保证原始 LinearRAG 行为不变。显式启用经过验证的保守词典：

```bash
python scripts/index_and_test.py --enable-alias-activation
```

完整问题集可用同样的开关进行对照：

```bash
python scripts/run_question_sets.py
python scripts/run_question_sets.py \
  --enable-alias-activation \
  --entity-alias-path dataset/harrypotter/entity_aliases.conservative.json \
  --alias-expansion-budget 0.75 \
  --max-alias-expansions 8
```

两次运行应使用不同的 `--output-dir`，再比较 evidence recall、答案准确率和带有 `alias_expansions` 字段的逐题结果。固定划分的检索对照结果与复现方式见 [CONSERVATIVE_ALIAS_ACTIVATION_RESULTS.md](CONSERVATIVE_ALIAS_ACTIVATION_RESULTS.md)。
