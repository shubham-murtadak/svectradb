"""
IVF-Flat (Inverted File Index) for SvectraDB.
Partitions the vector space into Voronoi cells using K-Means clustering,
speeding up queries by inspecting only the most relevant posting lists.
"""

import json
import os
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np

from svectra.core.models import MetricType
from svectra.engine.distance import compute_batch_distances
from svectra.engine.indices.base import BaseIndex


class IVFFlatIndex(BaseIndex):
    """
    Inverted File Index with K-Means centroids and inverted posting lists.
    """

    def __init__(
        self,
        dimension: int,
        metric: Union[MetricType, str] = MetricType.COSINE,
        nlist: int = 64,
        nprobe: int = 8,
        max_iter: int = 15,
        random_seed: int = 42,
    ):
        super().__init__(dimension=dimension, metric=metric)
        self.nlist = int(nlist)
        self.nprobe = min(int(nprobe), self.nlist)
        self.max_iter = int(max_iter)
        self.rng = np.random.RandomState(random_seed)

        self.centroids: Optional[np.ndarray] = None  # (nlist, dimension)
        self.is_trained: bool = False

        # Inverted lists: centroid_id -> list of internal indices
        self.invlists: Dict[int, List[int]] = {i: [] for i in range(self.nlist)}

        # Node storage
        self.id_to_int: Dict[str, int] = {}
        self.int_to_id: List[str] = []
        self.vectors: np.ndarray = np.empty((0, self.dimension), dtype=np.float32)

    def _train_kmeans(self, data: np.ndarray) -> None:
        """Lightweight and robust K-Means clustering."""
        n_samples = data.shape[0]
        k = min(self.nlist, n_samples)
        if k <= 0:
            return

        # Initialize centroids randomly from samples
        init_indices = self.rng.choice(n_samples, size=k, replace=False)
        centroids = data[init_indices].copy()

        for _ in range(self.max_iter):
            # Assign each vector to closest centroid
            dists = np.zeros((n_samples, k), dtype=np.float32)
            for c_idx in range(k):
                dists[:, c_idx] = compute_batch_distances(centroids[c_idx], data, self.metric)

            assignments = np.argmin(dists, axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for c_idx in range(k):
                mask = (assignments == c_idx)
                if np.any(mask):
                    new_centroids[c_idx] = np.mean(data[mask], axis=0)
                else:
                    new_centroids[c_idx] = data[self.rng.choice(n_samples)]

            if np.allclose(centroids, new_centroids, atol=1e-4):
                break
            centroids = new_centroids

        self.centroids = centroids
        self.is_trained = True

    def add(self, ids: List[str], vectors: np.ndarray) -> None:
        if len(ids) == 0:
            return

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dim {vectors.shape[1]} != index dim {self.dimension}")

        # Train if not yet trained and we have enough data
        if not self.is_trained:
            train_pool = vectors if self.vectors.shape[0] == 0 else np.vstack([self.vectors, vectors])
            if train_pool.shape[0] >= min(self.nlist, 10):
                self._train_kmeans(train_pool)

        start_idx = len(self.int_to_id)
        new_int_ids = []
        for i, doc_id in enumerate(ids):
            if doc_id in self.id_to_int:
                old_idx = self.id_to_int[doc_id]
                self.vectors[old_idx] = vectors[i]
                # Reassign to centroid
                if self.is_trained and self.centroids is not None:
                    c_dists = compute_batch_distances(vectors[i], self.centroids, self.metric)
                    best_c = int(np.argmin(c_dists))
                    for clist in self.invlists.values():
                        if old_idx in clist:
                            clist.remove(old_idx)
                    self.invlists[best_c].append(old_idx)
            else:
                curr_idx = start_idx + len(new_int_ids)
                self.id_to_int[doc_id] = curr_idx
                self.int_to_id.append(doc_id)
                new_int_ids.append(curr_idx)

        new_vectors = [vectors[i] for i, doc_id in enumerate(ids) if doc_id in self.id_to_int and self.id_to_int[doc_id] in new_int_ids]
        if new_vectors:
            new_vectors_arr = np.array(new_vectors, dtype=np.float32)
            if self.vectors.shape[0] == 0:
                self.vectors = new_vectors_arr
            else:
                self.vectors = np.vstack([self.vectors, new_vectors_arr])

            # Assign new vectors to centroids
            if self.is_trained and self.centroids is not None:
                for idx in new_int_ids:
                    c_dists = compute_batch_distances(self.vectors[idx], self.centroids, self.metric)
                    best_c = int(np.argmin(c_dists))
                    self.invlists[best_c].append(idx)
            else:
                # Fallback to bucket 0 before training
                self.invlists[0].extend(new_int_ids)

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        if len(self.int_to_id) == 0 or k <= 0:
            return []

        query = np.ascontiguousarray(query_vector, dtype=np.float32).flatten()

        # If not trained yet, fallback to brute force
        if not self.is_trained or self.centroids is None:
            distances = compute_batch_distances(query, self.vectors, self.metric)
            sorted_indices = np.argsort(distances)
            results = []
            for idx in sorted_indices:
                doc_id = self.int_to_id[idx]
                if filter_ids is None or doc_id in filter_ids:
                    results.append((doc_id, float(distances[idx])))
                    if len(results) >= k:
                        break
            return results

        # 1. Probe top centroids
        c_dists = compute_batch_distances(query, self.centroids, self.metric)
        probed_centroids = np.argsort(c_dists)[: min(self.nprobe, len(c_dists))]

        # 2. Gather candidate vectors from probed posting lists
        candidate_indices = []
        for c in probed_centroids:
            candidate_indices.extend(self.invlists.get(int(c), []))

        if not candidate_indices:
            return []

        # Remove duplicates
        candidate_indices = list(set(candidate_indices))

        if filter_ids is not None:
            candidate_indices = [
                idx for idx in candidate_indices if self.int_to_id[idx] in filter_ids
            ]
            if not candidate_indices:
                return []

        candidate_matrix = self.vectors[candidate_indices]
        distances = compute_batch_distances(query, candidate_matrix, self.metric)

        top_count = min(k, len(candidate_indices))
        if top_count < len(candidate_indices):
            part_idx = np.argpartition(distances, top_count)[:top_count]
            sorted_rel = part_idx[np.argsort(distances[part_idx])]
        else:
            sorted_rel = np.argsort(distances)

        return [
            (self.int_to_id[candidate_indices[i]], float(distances[i]))
            for i in sorted_rel
        ]

    def delete(self, ids: List[str]) -> int:
        ids_set = set(ids)
        deleted = 0
        for doc_id in ids_set:
            if doc_id not in self.id_to_int:
                continue
            idx = self.id_to_int[doc_id]
            for clist in self.invlists.values():
                if idx in clist:
                    clist.remove(idx)
            deleted += 1
        return deleted

    def count(self) -> int:
        return len(self.int_to_id)

    def get_vector(self, id: str) -> Optional[np.ndarray]:
        idx = self.id_to_int.get(id)
        if idx is not None and idx < self.vectors.shape[0]:
            return self.vectors[idx].copy()
        return None

    def save(self, filepath_prefix: str) -> None:
        os.makedirs(os.path.dirname(filepath_prefix), exist_ok=True)
        meta = {
            "dimension": self.dimension,
            "metric": self.metric.value,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "is_trained": self.is_trained,
            "int_to_id": self.int_to_id,
            "invlists": {str(k): v for k, v in self.invlists.items()},
        }
        with open(f"{filepath_prefix}.ivf.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)
        np.save(f"{filepath_prefix}.ivf.vec.npy", self.vectors)
        if self.centroids is not None:
            np.save(f"{filepath_prefix}.ivf.centroids.npy", self.centroids)

    @classmethod
    def load(cls, filepath_prefix: str) -> "IVFFlatIndex":
        with open(f"{filepath_prefix}.ivf.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        index = cls(
            dimension=meta["dimension"],
            metric=meta["metric"],
            nlist=meta["nlist"],
            nprobe=meta["nprobe"],
        )
        index.is_trained = meta["is_trained"]
        index.int_to_id = meta["int_to_id"]
        index.id_to_int = {doc_id: i for i, doc_id in enumerate(index.int_to_id)}
        index.invlists = {int(k): v for k, v in meta["invlists"].items()}
        index.vectors = np.load(f"{filepath_prefix}.ivf.vec.npy")

        centroids_path = f"{filepath_prefix}.ivf.centroids.npy"
        if os.path.exists(centroids_path):
            index.centroids = np.load(centroids_path)

        return index
