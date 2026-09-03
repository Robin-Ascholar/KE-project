"""Run a lightweight validation of the local LinearRAG environment."""

from pathlib import Path
import sys

import spacy
import torch
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model" / "all-mpnet-base-v2"
sys.path.insert(0, str(PROJECT_ROOT))

from src.LinearRAG import LinearRAG  # noqa: E402, F401


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nlp = spacy.load("en_core_web_trf")
    entities = [(ent.text, ent.label_) for ent in nlp("Harry Potter studies at Hogwarts.").ents]

    model = SentenceTransformer(str(MODEL_DIR), device=device)
    embeddings = model.encode(
        ["Harry Potter studies at Hogwarts."],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"spaCy entities: {entities}")
    print("LinearRAG environment check passed.")


if __name__ == "__main__":
    main()
