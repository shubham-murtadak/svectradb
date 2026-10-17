"""
Flat (Brute-Force) Vector Index.
Provides 100% exact recall using SIMD-accelerated matrix operations.
"""

import json
import os
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np

from svectra.core.models import MetricType
from svectra.engine.distance import compute_batch_distances
from svectra.engine.indices.base import BaseIndex


class FlatIndex(BaseIndex):
    """
    Exact nearest-neighbor index storing vectors in a contiguous NumPy array.
    """

    def __init__(self, dimension: int, metric: Union[MetricType, str] = MetricType.COSINE):
        super().__init__(dimension=dimension, metric=metric)
        self.ids: List[str] = []
        self.id_to_idx: Dict[str, int] = {}
        self.vectors: np.ndarray = np.empty((0, self.dimension), dtype=np.float32)

    def add(self, ids: List[str], vectors: np.ndarray) -> None:
        if len(ids) == 0:
            return

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} does not match index dimension {self.dimension}")

        # Check existing IDs to handle updates cleanly
        existing_indices = []
        new_ids = []
        new_vectors = []

        for i, doc_id in enumerate(ids):
            if doc_id in self.id_to_idx:
                idx = self.id_to_idx[doc_id]
                self.vectors[idx] = vectors[i]
            else:
                new_ids.append(doc_id)
                new_vectors.append(vectors[i])

        if new_ids:
            start_idx = len(self.ids)
            for offset, doc_id in enumerate(new_ids):
                self.id_to_idx[doc_id] = start_idx + offset
            self.ids.extend(new_ids)

            new_vectors_arr = np.array(new_vectors, dtype=np.float32)
            if self.vectors.shape[0] == 0:
                self.vectors = new_vectors_arr
            else:
                self.vectors = np.vstack([self.vectors, new_vectors_arr])

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        if len(self.ids) == 0 or k <= 0:
            return []

        query = np.ascontiguousarray(query_vector, dtype=np.float32).flatten()
        if query.shape[0] != self.dimension:
            raise ValueError(f"Query vector dimension {query.shape[0]} != index dimension {self.dimension}")

        # When filter_ids is specified
        if filter_ids is not None:
            valid_indices = [self.id_to_idx[fid] for fid in filter_ids if fid in self.id_to_idx]
            if not valid_indices:
                return []
            candidate_matrix = self.vectors[valid_indices]
            distances = compute_batch_distances(query, candidate_matrix, self.metric)
            candidate_ids = [self.ids[idx] for idx in valid_indices]

            top_k_count = min(k, len(candidate_ids))
            if top_k_count < len(candidate_ids):
                partition_idx = np.argpartition(distances, top_k_count)[:top_k_count]
                sorted_rel_idx = partition_idx[np.argsort(distances[partition_idx])]
            else:
                sorted_rel_idx = np.argsort(distances)

            return [(candidate_ids[i], float(distances[i])) for i in sorted_rel_idx]

        # Full matrix search
        distances = compute_batch_distances(query, self.vectors, self.metric)
        n = len(self.ids)
        top_k_count = min(k, n)

        if top_k_count < n:
            partition_idx = np.argpartition(distances, top_k_count)[:top_k_count]
            sorted_idx = partition_idx[np.argsort(distances[partition_idx])]
        else:
            sorted_idx = np.argsort(distances)

        return [(self.ids[i], float(distances[i])) for i in sorted_idx]

    def delete(self, ids: List[str]) -> int:
        ids_to_remove = set(ids)
        keep_indices = []
        new_ids = []
        deleted_count = 0

        for idx, doc_id in enumerate(self.ids):
            if doc_id in ids_to_remove:
                deleted_count += 1
            else:
                keep_indices.append(idx)
                new_ids.append(doc_id)

        if deleted_count > 0:
            if keep_indices:
                self.vectors = self.vectors[keep_indices]
            else:
                self.vectors = np.empty((0, self.dimension), dtype=np.float32)

            self.ids = new_ids
            self.id_to_idx = {doc_id: i for i, doc_id in enumerate(new_ids)}

        return deleted_count

    def count(self) -> int:
        return len(self.ids)

    def get_vector(self, id: str) -> Optional[np.ndarray]:
        idx = self.id_to_idx.get(id)
        if idx is not None:
            return self.vectors[idx].copy()
        return None

    def save(self, filepath_prefix: str) -> None:
        os.makedirs(os.path.dirname(filepath_prefix), exist_ok=True)
        meta = {
            "dimension": self.dimension,
            "metric": self.metric.value,
            "ids": self.ids,
        }
        with open(f"{filepath_prefix}.meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)
        np.save(f"{filepath_prefix}.vec.npy", self.vectors)

    @classmethod
    def load(cls, filepath_prefix: str) -> "FlatIndex":
        with open(f"{filepath_prefix}.meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        index = cls(dimension=meta["dimension"], metric=meta["metric"])
        index.ids = meta["ids"]
        index.id_to_idx = {doc_id: i for i, doc_id in enumerate(index.ids)}
        index.vectors = np.load(f"{filepath_prefix}.vec.npy")
        return index
