"""
Abstract Base Class for all SvectraDB vector index structures.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Set, Tuple, Union
import numpy as np
from svectra.core.models import MetricType


class BaseIndex(ABC):
    """Abstract interface for all vector indexing algorithms in SvectraDB."""

    def __init__(self, dimension: int, metric: Union[MetricType, str] = MetricType.COSINE):
        self.dimension = int(dimension)
        self.metric = MetricType(metric)

    @abstractmethod
    def add(self, ids: List[str], vectors: np.ndarray) -> None:
        """Add batch of vectors with corresponding unique IDs."""
        pass

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Search for top-k nearest neighbors.
        Returns list of (id, distance) tuples sorted in ascending distance order.
        """
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> int:
        """Delete vectors by IDs. Returns number of removed items."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total number of vectors in index."""
        pass

    @abstractmethod
    def get_vector(self, id: str) -> Optional[np.ndarray]:
        """Retrieve original vector for a given ID."""
        pass

    @abstractmethod
    def save(self, filepath_prefix: str) -> None:
        """Persist index state to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath_prefix: str) -> "BaseIndex":
        """Load index state from disk."""
        pass
