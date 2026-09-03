"""Download only the files needed by the PyTorch sentence-transformers model."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model" / "all-mpnet-base-v2"
REQUIRED_FILES = (
    "1_Pooling/config.json",
    "config.json",
    "model.safetensors",
    "modules.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Download again even if the model exists")
    args = parser.parse_args()

    if not args.force and all((MODEL_DIR / path).is_file() for path in REQUIRED_FILES):
        print(f"Embedding model already exists at {MODEL_DIR}")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="sentence-transformers/all-mpnet-base-v2",
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False,
        allow_patterns=["*.json", "*.txt", "*.safetensors"],
    )
    print(f"Embedding model is ready at {MODEL_DIR}")


if __name__ == "__main__":
    main()
