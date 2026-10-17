"""
Unit and integration tests for SvectraDB Client SDK.
"""

import tempfile
import svectradb


def test_embedded_client_workflow():
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Initialize embedded client
        client = svectradb.Client(path=tmpdir)

        # 2. Create collection
        col = client.create_collection(
            name="books",
            dimension=4,
            metric="cosine",
            index_type="hnsw",
        )
        assert col.config.name == "books"

        # 3. Upsert
        col.upsert(
            ids=["b1", "b2"],
            vectors=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            metadatas=[{"genre": "sci-fi"}, {"genre": "mystery"}],
            documents=["Dune", "Sherlock Holmes"],
        )
        assert col.count() == 2

        # 4. Query
        results = col.query(
            vector=[1.0, 0.0, 0.0, 0.0],
            top_k=1,
            filter={"genre": {"$eq": "sci-fi"}},
        )
        assert len(results) == 1
        assert results[0].id == "b1"
        assert results[0].document == "Dune"

        # 5. List collections
        colls = client.list_collections()
        assert "books" in colls

        client.close()
