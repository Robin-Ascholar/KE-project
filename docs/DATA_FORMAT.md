# Harry Potter 数据格式

项目当前通过 `run.py` 读取两个 JSON 文件。

第一册的 `chunks.json` 可直接生成：

```bash
python scripts/prepare_harry_potter.py
```

生成后可在不调用 LLM API 的情况下建立索引并检查检索：

```bash
python scripts/index_and_test.py
```

## `chunks.json`

顶层是字符串数组，每个字符串是一个可检索文本块：

```json
[
  "Chapter 1 ...",
  "Chapter 1 continued ..."
]
```

`run.py` 会自动为每个文本块添加递增编号。准备脚本按章节和句子边界切块，并控制文本不超过本地 MPNet 模型的 384-token 上限。

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

仓库提供 `dataset/harrypotter/questions.example.json` 作为结构示例。正式问题集和从书籍生成的文本块默认不提交。

## 本地路径

```text
Book/                                  # 原始 PDF
dataset/harrypotter/chunks.json        # 清洗和切块后的语料
dataset/harrypotter/questions.json     # 问题与参考答案
import/harrypotter/                    # 自动生成的图与向量索引
results/harrypotter/                   # 预测与评测结果
```
