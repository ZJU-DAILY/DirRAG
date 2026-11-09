import torch
torch.cuda.empty_cache()

import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

                                                                              
EMBED_MODEL_PATH = "/root/dir-sem-rag/gte-Qwen2-1.5B-instruct"
EMBEDDING_DIM    = 1536


def select_best_gpu() -> int:
    if not torch.cuda.is_available():
        return 0
    best_idx = 0
    best_free = 0
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        if free > best_free:
            best_free = free
            best_idx = i
    return best_idx


class EmbeddingModel:

    def __init__(self, model_name: str = EMBED_MODEL_PATH):
                                             
        self._model_name = model_name
        self.model = None
        self._load()

    def _load(self):
        if self.model is not None:
            return
        best_gpu = select_best_gpu()
        free_gb = torch.cuda.mem_get_info(best_gpu)[0] / 1024**3
        print(f"📦 Loading Embedding model: {self._model_name} "
              f"(选择 GPU {best_gpu}, 空闲 {free_gb:.2f}GB)")
        self.model = SentenceTransformer(self._model_name, device=f"cuda:{best_gpu}")
        print("✅ Embedding model ready.")

    def destroy(self):
        if self.model is not None:
            del self.model
            self.model = None
                                      
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("🔥 Embedding model destroyed, GPU cache cleared.")

    def ensure_loaded(self):
        if self.model is None:
            self._load()

    def is_loaded(self) -> bool:
        return self.model is not None

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        self.ensure_loaded()
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        )


