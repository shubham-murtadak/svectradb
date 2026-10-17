"""
Vector math and distance metrics for SvectraDB.
Optimized using vectorized NumPy operations.
"""

from typing import Union
import numpy as np
from svectra.core.models import MetricType


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length (L2 norm)."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def normalize_matrix(m: np.ndarray) -> np.ndarray:
    """Row-wise normalize a 2D matrix of vectors."""
    if m.ndim == 1:
        return normalize_vector(m)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def compute_distance(
    u: np.ndarray,
    v: np.ndarray,
    metric: Union[MetricType, str] = MetricType.COSINE,
) -> float:
    """
    Compute scalar distance between two 1D vectors u and v.
    Lower distance always means closer / more similar.
    """
    metric = MetricType(metric)
    if metric == MetricType.COSINE:
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u == 0 or norm_v == 0:
            return 1.0
        similarity = float(np.dot(u, v) / (norm_u * norm_v))
        # Clamp to avoid numerical floating-point inaccuracies
        similarity = max(min(similarity, 1.0), -1.0)
        return float(1.0 - similarity)

    elif metric == MetricType.EUCLIDEAN:
        diff = u - v
        return float(np.linalg.norm(diff))

    elif metric == MetricType.DOT:
        # Negative dot product so lower is closer
        return float(-np.dot(u, v))

    elif metric == MetricType.MANHATTAN:
        return float(np.sum(np.abs(u - v)))

    else:
        raise ValueError(f"Unsupported metric: {metric}")


def compute_batch_distances(
    query: np.ndarray,
    matrix: np.ndarray,
    metric: Union[MetricType, str] = MetricType.COSINE,
) -> np.ndarray:
    """
    Compute pairwise distances between a 1D query vector and a 2D matrix (N, D).
    Returns a 1D numpy array of length N.
    """
    if matrix.shape[0] == 0:
        return np.array([], dtype=np.float32)

    metric = MetricType(metric)

    if metric == MetricType.COSINE:
        query_norm = np.linalg.norm(query)
        matrix_norms = np.linalg.norm(matrix, axis=1)

        if query_norm == 0:
            return np.ones(matrix.shape[0], dtype=np.float32)

        matrix_norms[matrix_norms == 0] = 1.0
        dots = np.dot(matrix, query)
        sims = dots / (matrix_norms * query_norm)
        np.clip(sims, -1.0, 1.0, out=sims)
        return 1.0 - sims

    elif metric == MetricType.EUCLIDEAN:
        diff = matrix - query
        return np.linalg.norm(diff, axis=1)

    elif metric == MetricType.DOT:
        # Negative dot product
        return -np.dot(matrix, query)

    elif metric == MetricType.MANHATTAN:
        return np.sum(np.abs(matrix - query), axis=1)

    else:
        raise ValueError(f"Unsupported metric: {metric}")


def distance_to_score(
    distance: float,
    metric: Union[MetricType, str] = MetricType.COSINE,
) -> float:
    """
    Transform geometric distance to an intuitive similarity score.
    Higher score always represents greater similarity.
    """
    metric = MetricType(metric)
    if metric == MetricType.COSINE:
        # Cosine distance = 1 - similarity => similarity = 1 - distance
        return float(1.0 - distance)
    elif metric in (MetricType.EUCLIDEAN, MetricType.MANHATTAN):
        # 1 / (1 + distance) maps [0, inf) to (0, 1]
        return float(1.0 / (1.0 + max(0.0, distance)))
    elif metric == MetricType.DOT:
        # Distance is -dot => score is dot
        return float(-distance)
    return float(-distance)
