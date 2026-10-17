"""
Unit tests for distance metrics and SIMD vector operations.
"""

import numpy as np
import pytest
from svectra.core.models import MetricType
from svectra.engine.distance import (
    compute_distance,
    compute_batch_distances,
    distance_to_score,
    normalize_vector,
    normalize_matrix,
)


def test_vector_normalization():
    v = np.array([3.0, 4.0], dtype=np.float32)
    normed = normalize_vector(v)
    assert np.isclose(np.linalg.norm(normed), 1.0)

    # Zero vector
    zero = np.zeros(4, dtype=np.float32)
    assert np.allclose(normalize_vector(zero), zero)


def test_cosine_distance():
    u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    # Identical vectors: cosine distance should be 0.0
    assert np.isclose(compute_distance(u, v, MetricType.COSINE), 0.0, atol=1e-6)

    # Orthogonal vectors: cosine distance should be 1.0
    w = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert np.isclose(compute_distance(u, w, MetricType.COSINE), 1.0, atol=1e-6)

    # Opposite vectors: cosine distance should be 2.0
    neg = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    assert np.isclose(compute_distance(u, neg, MetricType.COSINE), 2.0, atol=1e-6)


def test_euclidean_distance():
    u = np.array([0.0, 0.0], dtype=np.float32)
    v = np.array([3.0, 4.0], dtype=np.float32)
    assert np.isclose(compute_distance(u, v, MetricType.EUCLIDEAN), 5.0)


def test_batch_distances():
    query = np.array([1.0, 0.0], dtype=np.float32)
    matrix = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
    ], dtype=np.float32)

    # Cosine
    cos_dists = compute_batch_distances(query, matrix, MetricType.COSINE)
    assert len(cos_dists) == 3
    assert np.isclose(cos_dists[0], 0.0, atol=1e-6)
    assert np.isclose(cos_dists[1], 1.0, atol=1e-6)
    assert np.isclose(cos_dists[2], 2.0, atol=1e-6)

    # Euclidean
    euc_dists = compute_batch_distances(query, matrix, MetricType.EUCLIDEAN)
    assert np.isclose(euc_dists[0], 0.0)
    assert np.isclose(euc_dists[1], np.sqrt(2.0))
    assert np.isclose(euc_dists[2], 2.0)


def test_distance_to_score():
    assert np.isclose(distance_to_score(0.0, MetricType.COSINE), 1.0)
    assert np.isclose(distance_to_score(1.0, MetricType.COSINE), 0.0)
    assert np.isclose(distance_to_score(0.0, MetricType.EUCLIDEAN), 1.0)
