"""
Embeddings package for SvectraDB.
"""

from typing import Optional, Union
from svectra.embeddings.base import BaseEmbeddingFunction
from svectra.embeddings.fast_hash import FastHashEmbedding


def get_default_embedding_function(dimension: int = 128) -> BaseEmbeddingFunction:
    """Return default lightweight fast hash embedding function."""
    return FastHashEmbedding(dimension=dimension)


__all__ = [
    "BaseEmbeddingFunction",
    "FastHashEmbedding",
    "get_default_embedding_function",
]
