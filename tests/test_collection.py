"""
Integration tests for SvectraCollection.
"""

import os
import tempfile
import numpy as np
from svectra.core.collection import SvectraCollection
from svectra.core.models import CollectionConfig, IndexType, MetricType


def test_collection_in_memory_flow():
    config = CollectionConfig(
        name="test_coll",
        dimension=4,
        metric=MetricType.COSINE,
        index_type=IndexType.HNSW,
    )
    coll = SvectraCollection(config=config)

    # 1. Upsert
    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    ids = ["doc_x", "doc_y", "doc_z"]
    metas = [{"tag": "x"}, {"tag": "y"}, {"tag": "z"}]
    docs = ["Text X", "Text Y", "Text Z"]

    coll.upsert(vectors=vectors, ids=ids, metadatas=metas, documents=docs)
    assert coll.count() == 3

    # 2. Query
    results = coll.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].id == "doc_x"
    assert results[0].document == "Text X"
    assert results[0].metadata == {"tag": "x"}
    assert np.isclose(results[0].score, 1.0, atol=1e-5)

    # 3. Filtered query
    filtered = coll.query(
        vector=[1.0, 0.0, 0.0, 0.0],
        top_k=5,
        filter={"tag": {"$eq": "z"}},
    )
    assert len(filtered) == 1
    assert filtered[0].id == "doc_z"


def test_collection_persistence_and_recovery():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = CollectionConfig(
            name="persist_test",
            dimension=4,
            metric=MetricType.EUCLIDEAN,
            index_type=IndexType.FLAT,
        )

        # Session 1: Create, insert, flush
        coll1 = SvectraCollection(config=config, persist_dir=tmpdir)
        coll1.upsert(
            ids=["p1", "p2"],
            vectors=[[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
            documents=["First Doc", "Second Doc"],
            metadatas=[{"v": 1}, {"v": 2}],
        )
        coll1.flush()
        coll1.close()

        # Session 2: Reload from disk
        coll2 = SvectraCollection(config=config, persist_dir=tmpdir)
        assert coll2.count() == 2
        rec = coll2.get("p1")
        assert rec is not None
        assert rec.document == "First Doc"
        assert rec.metadata == {"v": 1}

        # Query
        res = coll2.query(vector=[1.0, 1.0, 1.0, 1.0], top_k=1)
        assert res[0].id == "p1"
        coll2.close()
