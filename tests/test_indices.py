"""
Unit tests for Flat, HNSW, and IVF vector indices.
"""

import os
import shutil
import tempfile
import numpy as np
import pytest

from svectra.core.models import MetricType
from svectra.engine.indices.flat import FlatIndex
from svectra.engine.indices.hnsw import HNSWIndex
from svectra.engine.indices.ivf import IVFFlatIndex


@pytest.fixture
def sample_data():
    np.random.seed(42)
    dim = 16
    n = 100
    vectors = np.random.randn(n, dim).astype(np.float32)
    # Normalize for cosine
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    ids = [f"doc_{i}" for i in range(n)]
    return ids, vectors, dim


def test_flat_index_operations(sample_data):
    ids, vectors, dim = sample_data
    index = FlatIndex(dimension=dim, metric=MetricType.COSINE)

    # 1. Add
    index.add(ids, vectors)
    assert index.count() == 100

    # 2. Search exact match
    query = vectors[10]
    results = index.search(query, k=5)
    assert len(results) == 5
    assert results[0][0] == ids[10]
    assert np.isclose(results[0][1], 0.0, atol=1e-5)

    # 3. Filtered search
    filtered_results = index.search(query, k=5, filter_ids={"doc_10", "doc_20"})
    assert len(filtered_results) == 2
    assert {r[0] for r in filtered_results} == {"doc_10", "doc_20"}

    # 4. Delete
    deleted = index.delete(["doc_10", "doc_11"])
    assert deleted == 2
    assert index.count() == 98
    assert index.get_vector("doc_10") is None


def test_hnsw_index_operations(sample_data):
    ids, vectors, dim = sample_data
    index = HNSWIndex(dimension=dim, metric=MetricType.COSINE, M=16, ef_construction=64, ef_search=32)

    # 1. Add
    index.add(ids, vectors)
    assert index.count() == 100

    # 2. Search
    query = vectors[5]
    results = index.search(query, k=5)
    assert len(results) == 5
    # Nearest neighbor should be doc_5
    assert results[0][0] == ids[5]
    assert np.isclose(results[0][1], 0.0, atol=1e-5)

    # 3. Persistence
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "hnsw_test")
        index.save(prefix)

        loaded_index = HNSWIndex.load(prefix)
        assert loaded_index.count() == 100
        loaded_res = loaded_index.search(query, k=5)
        assert loaded_res[0][0] == ids[5]


def test_ivf_index_operations(sample_data):
    ids, vectors, dim = sample_data
    index = IVFFlatIndex(dimension=dim, metric=MetricType.COSINE, nlist=10, nprobe=4)

    # 1. Add
    index.add(ids, vectors)
    assert index.count() == 100
    assert index.is_trained

    # 2. Search
    query = vectors[25]
    results = index.search(query, k=5)
    assert len(results) == 5
    assert results[0][0] == ids[25]
    assert np.isclose(results[0][1], 0.0, atol=1e-4)

    # 3. Persistence
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "ivf_test")
        index.save(prefix)

        loaded_index = IVFFlatIndex.load(prefix)
        assert loaded_index.count() == 100
        loaded_res = loaded_index.search(query, k=5)
        assert loaded_res[0][0] == ids[25]
