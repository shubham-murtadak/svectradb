"""
Abstract base class for SvectraDB embedding functions.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseEmbeddingFunction(ABC):
    """Abstract interface for text embedding functions in SvectraDB."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension produced by this model."""
        pass

    @abstractmethod
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """Embed a batch of document texts into a 2D float32 numpy array (N, D)."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string into a 1D float32 numpy array (D,)."""
        pass

    def __call__(self, texts: List[str]) -> np.ndarray:
        return self.embed_documents(texts)
