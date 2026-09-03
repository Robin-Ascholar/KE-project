# Embedding model directory

Run the following command from the repository root to download the required model without pulling unused ONNX/OpenVINO artifacts:

```bash
python scripts/download_embedding_model.py
```

The resulting `model/all-mpnet-base-v2/` directory is intentionally ignored by Git.
