# Dataset directory

当前数据集位于 `dataset/harrypotter/`，包含 Harry Potter 七册清洗语料与 14 个端到端问答问题。

```text
chunks.json       # 6,865 个 passage 字符串
questions.json    # 14 个 question/answer 对象
manifest.json     # 数据来源、SHA256、章节与 token 统计
```

从 `cleanUP/txt/` 重新生成：

```bash
python scripts/prepare_all_books.py
```

当前切分参数为目标 300 token、最大 360 token、最小 80 token、重叠两个完整句子，并严格遵守书籍与章节边界。详细格式见 [`docs/DATA_FORMAT.md`](../docs/DATA_FORMAT.md)。

这些文件已作为当前完整实验快照提交。以后新增的其他数据集默认仍受 `.gitignore` 约束；若需要提交，应先确认体积并根据需要使用 Git LFS。
