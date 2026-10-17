"""
Zero-dependency fast semantic hash embedding function for SvectraDB.
Uses token + character n-gram hashing projected onto a unit hypersphere.
Fast, deterministic, and lightweight.
"""

import hashlib
import re
from typing import List
import numpy as np

from svectra.embeddings.base import BaseEmbeddingFunction
from svectra.engine.distance import normalize_matrix, normalize_vector


class FastHashEmbedding(BaseEmbeddingFunction):
    """
    Lightweight, high-speed n-gram feature hashing embedder.
    Requires only standard library and NumPy.
    """

    def __init__(self, dimension: int = 128):
        self._dim = int(dimension)

    @property
    def dimension(self) -> int:
        return self._dim

    def _text_to_vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        if not text:
            return vec

        text = text.lower()
        words = re.findall(r"\w+", text)

        # 1. Word unigrams and bigrams
        tokens = list(words)
        for i in range(len(words) - 1):
            tokens.append(f"{words[i]}_{words[i+1]}")

        # 2. Character 3-grams
        for i in range(len(text) - 2):
            tokens.append(text[i:i+3])

        for tok in tokens:
            # Deterministic MD5 hash to index and sign
            h = hashlib.md5(tok.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "little") % self._dim
            sign = 1.0 if (h[4] & 1) == 0 else -1.0
            vec[idx] += sign

        return normalize_vector(vec)

    def embed_documents(self, documents: List[str]) -> np.ndarray:
        if not documents:
            return np.empty((0, self._dim), dtype=np.float32)

        mat = np.zeros((len(documents), self._dim), dtype=np.float32)
        for i, doc in enumerate(documents):
            mat[i] = self._text_to_vector(doc)

        return mat

    def embed_query(self, query: str) -> np.ndarray:
        return self._text_to_vector(query)
