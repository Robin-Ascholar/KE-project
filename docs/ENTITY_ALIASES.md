# 实体别名扩展

任务1以独立模块实现，不改写现有图中的实体节点，也不改变 `get_seed_entities()` 之外的检索流程。

## 文件

- `src/entity_alias.py`：别名归一化、规范实体解析、查询短语匹配和候选实体扩展。
- `dataset/harrypotter/entity_aliases.json`：人工复核后的别名组数据。
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

默认别名激活分数为 `0.85`，高于当前实体迭代阈值 `0.4`，因此同组实体可以参与后续图传播。不同实体共享的歧义别名会被跳过，避免错误激活。

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

运行原有检索检查时，默认启用别名扩展；要运行 Original 基线，可使用：

```bash
python scripts/index_and_test.py --disable-alias-activation
```

完整问题集可用同样的开关进行对照：

```bash
python scripts/run_question_sets.py --disable-alias-activation
python scripts/run_question_sets.py
```

两次运行应使用不同的 `--output-dir`，再比较 evidence recall、答案准确率和带有 `alias_expansions` 字段的逐题结果。当前代码只提供实验入口和审计信息，不预先填写尚未运行的对照指标。
