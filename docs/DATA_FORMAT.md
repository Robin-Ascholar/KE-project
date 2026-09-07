# Harry Potter 数据格式

`run.py` 从 `dataset/<dataset_name>/` 读取 `chunks.json` 和 `questions.json`。当前七册数据集名称为 `harrypotter`。

## 生成数据

从已有清洗文本重新生成：

```bash
python scripts/prepare_all_books.py
```

默认输入与输出：

```text
cleanUP/txt/*.txt                         # 七册清洗正文
cleanUP/txt/*.report.json                 # 清洗和章节报告
dataset/harrypotter/chunks.json           # 可检索 passage
dataset/harrypotter/questions.json        # 问题与参考答案
dataset/harrypotter/manifest.json         # 数据来源和切分清单
```

## `chunks.json`

顶层是 JSON 字符串数组，每个字符串会被 `run.py` 当作一个 passage：

```json
[
  "Book 1 | Chapter 1: THE BOY WHO LIVED\nMr. and Mrs. Dursley ...",
  "Book 1 | Chapter 1: THE BOY WHO LIVED\nThey didn’t think ..."
]
```

`run.py` 会在加载时自动添加递增编号，因此文件中的字符串不需要手工添加数字 ID。

当前切分规则：

- 不跨越书籍或章节边界
- 每个 chunk 重复书号、章号和标题
- 使用 `all-mpnet-base-v2` tokenizer 计数
- 目标 300 token，最大 360 token，最小 80 token
- 相邻 chunk 重叠两个完整句子
- 超长单句优先按词切分，任何 chunk 均不超过 384-token 模型上限

当前文件包含 6,865 个字符串。

## `questions.json`

顶层是对象数组，每项至少包含 `question` 和 `answer`：

```json
[
  {
    "question": "Who are Harry Potter's close friends?",
    "answer": "Hermione Granger and Ron Weasley"
  }
]
```

当前问题集包含 14 个覆盖七册情节的事实问题。`run.py` 会把 `answer` 作为评测参考答案，并将模型输出写入结果目录。

## `manifest.json`

清单记录：

- embedding 模型路径和切分参数
- 书籍、章节、chunk 和问题数量
- 每本清洗 TXT 的 SHA256
- 每章标题与 chunk 序号
- 每个 chunk 的实际 token 数

它用于审计和复现实验，不会被 `run.py` 直接读取。

## 索引与结果

```text
import/harrypotter/LinearRAG.graphml       # passage/entity 图
import/harrypotter/ner_results.json        # NER 结果
import/harrypotter/*_embedding.parquet     # passage/sentence/entity 向量
results/harrypotter/<timestamp>/           # 日志、预测与评测指标
```

如果修改 `chunks.json`，不要直接复用旧索引；应清理对应缓存或使用新的 `dataset_name`，否则哈希缓存与图结构可能混合不同版本的数据。
