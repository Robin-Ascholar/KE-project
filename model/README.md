# Embedding model directory

当前实验使用 `model/all-mpnet-base-v2/`：

```text
source: sentence-transformers/all-mpnet-base-v2
embedding dimension: 768
SentenceTransformer max_seq_length: 384
runtime device: CUDA when available, otherwise CPU
```

模型权重已通过 Git LFS 纳入当前实验快照。完整克隆后执行：

```bash
git lfs pull
```

如果模型文件缺失，也可以从 Hugging Face 重新下载：

```bash
python scripts/download_embedding_model.py
```

下载脚本只获取 SentenceTransformer 推理所需文件，不下载未使用的 ONNX/OpenVINO 版本。
