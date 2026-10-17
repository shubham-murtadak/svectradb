"""
Hierarchical Navigable Small World (HNSW) Vector Index.
Provides ultra-fast logarithmic O(log N) approximate nearest neighbor search
with tunable recall and query latency.
"""

import heapq
import json
import math
import os
import random
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np

from svectra.core.models import MetricType
from svectra.engine.distance import compute_distance
from svectra.engine.indices.base import BaseIndex


class HNSWIndex(BaseIndex):
    """
    Hierarchical Navigable Small World graph index for fast ANN search.
    """

    def __init__(
        self,
        dimension: int,
        metric: Union[MetricType, str] = MetricType.COSINE,
        M: int = 16,
        ef_construction: int = 100,
        ef_search: int = 64,
        random_seed: int = 42,
    ):
        super().__init__(dimension=dimension, metric=metric)
        self.M = int(M)
        self.M0 = int(2 * M)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)
        self.mL = 1.0 / math.log(self.M)
        self.rng = random.Random(random_seed)

        # Node storage
        self.id_to_int: Dict[str, int] = {}
        self.int_to_id: List[str] = []
        self.vectors: np.ndarray = np.empty((0, self.dimension), dtype=np.float32)
        self.node_levels: List[int] = []

        # Graph layers: list of dicts: [ {node_int: [neighbor_ints]} for each layer ]
        self.graphs: List[Dict[int, List[int]]] = []
        self.enter_node: Optional[int] = None
        self.max_level: int = -1

    def _dist(self, u_idx: int, v_idx: int) -> float:
        return compute_distance(self.vectors[u_idx], self.vectors[v_idx], self.metric)

    def _dist_vec(self, query: np.ndarray, v_idx: int) -> float:
        return compute_distance(query, self.vectors[v_idx], self.metric)

    def _sample_level(self) -> int:
        r = self.rng.random()
        if r == 0:
            r = 1e-9
        return int(-math.log(r) * self.mL)

    def add(self, ids: List[str], vectors: np.ndarray) -> None:
        if len(ids) == 0:
            return

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dim {vectors.shape[1]} != index dim {self.dimension}")

        for i, doc_id in enumerate(ids):
            vec = vectors[i]
            if doc_id in self.id_to_int:
                # Update existing vector in-place
                node_idx = self.id_to_int[doc_id]
                self.vectors[node_idx] = vec
                # Note: For full graph restructuring on update, we keep topology intact
            else:
                self._insert_node(doc_id, vec)

    def _insert_node(self, doc_id: str, vector: np.ndarray) -> None:
        node_idx = len(self.int_to_id)
        self.id_to_int[doc_id] = node_idx
        self.int_to_id.append(doc_id)

        if self.vectors.shape[0] == 0:
            self.vectors = vector.reshape(1, -1)
        else:
            self.vectors = np.vstack([self.vectors, vector.reshape(1, -1)])

        level = self._sample_level()
        self.node_levels.append(level)

        # Extend layers if needed
        while len(self.graphs) <= level:
            self.graphs.append({})

        for l in range(level + 1):
            self.graphs[l][node_idx] = []

        curr_obj = self.enter_node

        if curr_obj is None:
            self.enter_node = node_idx
            self.max_level = level
            return

        # Phase 1: greedy search from top layer down to level + 1
        curr_dist = self._dist(node_idx, curr_obj)
        for l in range(self.max_level, level, -1):
            changed = True
            while changed:
                changed = False
                for neighbor in self.graphs[l].get(curr_obj, []):
                    d = self._dist(node_idx, neighbor)
                    if d < curr_dist:
                        curr_dist = d
                        curr_obj = neighbor
                        changed = True

        # Phase 2: search and connect from min(max_level, level) down to 0
        top_l = min(self.max_level, level)
        for l in range(top_l, -1, -1):
            w = self._search_layer_internal(vector, curr_obj, self.ef_construction, l)
            curr_obj = w[0][1] if w else curr_obj

            # Select neighbors
            m_max = self.M0 if l == 0 else self.M
            candidates = [node for _, node in w]
            neighbors = self._select_neighbors_heuristic(node_idx, candidates, m_max)

            # Bidirectional connections
            self.graphs[l][node_idx] = neighbors
            for neighbor in neighbors:
                if neighbor not in self.graphs[l]:
                    self.graphs[l][neighbor] = []
                self.graphs[l][neighbor].append(node_idx)
                # Prune neighbor connections if exceeding m_max
                if len(self.graphs[l][neighbor]) > m_max:
                    pruned = self._select_neighbors_heuristic(
                        neighbor, self.graphs[l][neighbor], m_max
                    )
                    self.graphs[l][neighbor] = pruned

        if level > self.max_level:
            self.max_level = level
            self.enter_node = node_idx

    def _select_neighbors_heuristic(
        self, base_node: int, candidate_nodes: List[int], m_max: int
    ) -> List[int]:
        """Simple diverse neighbor selection."""
        if len(candidate_nodes) <= m_max:
            return candidate_nodes

        # Sort candidates by distance to base_node
        scored = sorted([(self._dist(base_node, c), c) for c in candidate_nodes if c != base_node])
        selected = []
        for dist, c in scored:
            if len(selected) >= m_max:
                break
            selected.append(c)
        return selected

    def _search_layer_internal(
        self, query: np.ndarray, enter_point: int, ef: int, layer: int
    ) -> List[Tuple[float, int]]:
        """
        Internal search layer returning candidates as (distance, node_idx).
        """
        v_dist = self._dist_vec(query, enter_point)
        visited: Set[int] = {enter_point}
        # candidates: min-heap of (dist, node_idx)
        candidates: List[Tuple[float, int]] = [(v_dist, enter_point)]
        # best: max-heap of (-dist, node_idx)
        best: List[Tuple[float, int]] = [(-v_dist, enter_point)]

        while candidates:
            c_dist, c_node = heapq.heappop(candidates)
            furthest_best_dist = -best[0][0]

            if c_dist > furthest_best_dist:
                break

            neighbors = self.graphs[layer].get(c_node, [])
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    furthest_best_dist = -best[0][0]
                    n_dist = self._dist_vec(query, neighbor)

                    if n_dist < furthest_best_dist or len(best) < ef:
                        heapq.heappush(candidates, (n_dist, neighbor))
                        heapq.heappush(best, (-n_dist, neighbor))
                        if len(best) > ef:
                            heapq.heappop(best)

        # Return sorted list of (dist, node_idx)
        res = [(-d, node) for d, node in best]
        res.sort(key=lambda x: x[0])
        return res

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        if self.enter_node is None or len(self.int_to_id) == 0 or k <= 0:
            return []

        query = np.ascontiguousarray(query_vector, dtype=np.float32).flatten()
        curr_obj = self.enter_node

        # Phase 1: navigate from top layer down to 1
        curr_dist = self._dist_vec(query, curr_obj)
        for l in range(self.max_level, 0, -1):
            changed = True
            while changed:
                changed = False
                for neighbor in self.graphs[l].get(curr_obj, []):
                    d = self._dist_vec(query, neighbor)
                    if d < curr_dist:
                        curr_dist = d
                        curr_obj = neighbor
                        changed = True

        # Phase 2: search layer 0 with dynamic ef
        ef = max(self.ef_search, k)
        candidates = self._search_layer_internal(query, curr_obj, ef, 0)

        results: List[Tuple[str, float]] = []
        for dist, node_idx in candidates:
            doc_id = self.int_to_id[node_idx]
            if filter_ids is None or doc_id in filter_ids:
                results.append((doc_id, float(dist)))
                if len(results) >= k:
                    break

        return results

    def delete(self, ids: List[str]) -> int:
        """
        Delete nodes from the index.
        For high integrity, deleted nodes are unlinked from all graph layers.
        """
        ids_to_del = set(ids)
        deleted = 0

        for doc_id in ids_to_del:
            if doc_id not in self.id_to_int:
                continue

            node_idx = self.id_to_int[doc_id]
            # Remove from all graph layers
            for l in range(len(self.graphs)):
                if node_idx in self.graphs[l]:
                    neighbors = self.graphs[l].pop(node_idx)
                    for n in neighbors:
                        if n in self.graphs[l] and node_idx in self.graphs[l][n]:
                            self.graphs[l][n].remove(node_idx)

            deleted += 1

        # Re-index if entry node was deleted
        if self.enter_node is not None and self.int_to_id[self.enter_node] in ids_to_del:
            # Pick a surviving active node
            active_nodes = [i for i, uid in enumerate(self.int_to_id) if uid not in ids_to_del]
            if active_nodes:
                self.enter_node = active_nodes[0]
                self.max_level = self.node_levels[self.enter_node]
            else:
                self.enter_node = None
                self.max_level = -1

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
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "enter_node": self.enter_node,
            "max_level": self.max_level,
            "int_to_id": self.int_to_id,
            "node_levels": self.node_levels,
            "graphs": [
                {str(k): v for k, v in layer.items()} for layer in self.graphs
            ],
        }
        with open(f"{filepath_prefix}.hnsw.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)
        np.save(f"{filepath_prefix}.hnsw.npy", self.vectors)

    @classmethod
    def load(cls, filepath_prefix: str) -> "HNSWIndex":
        with open(f"{filepath_prefix}.hnsw.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        index = cls(
            dimension=meta["dimension"],
            metric=meta["metric"],
            M=meta["M"],
            ef_construction=meta["ef_construction"],
            ef_search=meta["ef_search"],
        )
        index.enter_node = meta["enter_node"]
        index.max_level = meta["max_level"]
        index.int_to_id = meta["int_to_id"]
        index.id_to_int = {doc_id: i for i, doc_id in enumerate(index.int_to_id)}
        index.node_levels = meta["node_levels"]
        index.graphs = [
            {int(k): v for k, v in layer.items()} for layer in meta["graphs"]
        ]
        index.vectors = np.load(f"{filepath_prefix}.hnsw.npy")
        return index
