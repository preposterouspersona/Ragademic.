import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer


class EmbeddingManager:

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("Embedding model loaded.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def create_embeddings(self, text: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            text,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return embeddings
