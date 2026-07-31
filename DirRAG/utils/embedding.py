"""
embedding.py
────────────
Thin wrapper around SentenceTransformer for L2-normalised dense embeddings.
Exposes a global singleton `embed_model` and the `EmbeddingModel` class.

Matches the snippet in the design doc (document 2, section 一).
"""
import torch
torch.cuda.empty_cache()

import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

# ── Config ──────────────────────────────────────────────────────────────────
EMBED_MODEL_PATH = "/root/dir-sem-rag/gte-Qwen2-1.5B-instruct"
EMBEDDING_DIM    = 1536


class EmbeddingModel:
    """L2-normalised text embedding model backed by SentenceTransformer."""

    def __init__(self, model_name: str = EMBED_MODEL_PATH):
        print(f"📦 Loading Embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("✅ Embedding model ready.")

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode a list of strings into L2-normalised vectors.
        Returns ndarray of shape (len(texts), EMBEDDING_DIM).
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors (both should be normalised)."""
        return float(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        )


# Global singleton — load once, reuse everywhere
# embed_model = EmbeddingModel()


# if __name__ == "__main__":
#     text = "Hello, this is a test."
#     emb  = embed_model.encode([text])[0]
#     print(f"Embedding shape: {emb.shape}")

#     texts = ["I like AI", "Machine learning is fun", "Deep learning rocks"]
#     embs  = embed_model.encode(texts)
#     print(f"Batch shape: {embs.shape}")
#     print(f"Cosine sim(0,1): {embed_model.cosine_sim(embs[0], embs[1]):.4f}")
