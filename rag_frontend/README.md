# Harry Potter LinearRAG 前端

该目录提供面向本项目的网页问答界面。Flask 服务会直接加载
`dataset/harrypotter/chunks.json`、`import/harrypotter/LinearRAG.graphml`
及现有向量索引，并通过 Ollama `qwen2.5:7b` 完成回答。

## 启动

```bash
cd /home/lizhaoxuan/KE-project
conda activate linearrag
pip install -r rag_frontend/requirements.txt
cp rag_frontend/.env.example rag_frontend/.env
python rag_frontend/server.py
```

浏览器访问 `http://服务器地址:5000`。首次启动需要加载 embedding、GraphML 和稀疏检索矩阵，
状态变为“LinearRAG 已就绪”后即可提问。

## 接口

- `GET /api/status`：运行状态、模型配置和图谱规模。
- `POST /api/chat`：请求体为 `{"question": "...", "top_k": 8}`。
- `POST /api/top-k`：修改当前检索结果数量。
- `POST /api/load-index`：加载或重新加载指定数据集。
- `POST /api/unload`：只释放内存/GPU 运行时，不删除磁盘图谱与索引。

问答响应包含最终答案、Top-K passage、检索分数、书号、章节号、章节候选和运行诊断。
