"""
SentenceTransformers embedding function provider for SvectraDB.
Requires `sentence-transformers` package.
"""

from typing import List, Optional
import numpy as np

from svectra.embeddings.base import BaseEmbeddingFunction


class SentenceTransformerEmbedding(BaseEmbeddingFunction):
    """
    Embedding function powered by Hugging Face Sentence Transformers.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with: pip install 'svectradb[embeddings]' or pip install sentence-transformers"
            )

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self._dim = self.model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, documents: List[str]) -> np.ndarray:
        if not documents:
            return np.empty((0, self._dim), dtype=np.float32)
        embeddings = self.model.encode(documents, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
        return embedding.astype(np.float32)
