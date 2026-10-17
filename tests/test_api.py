"""
API Integration tests for SvectraDB FastAPI endpoints.
"""

from fastapi.testclient import TestClient
from svectra.server.app import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "svectradb"


def test_metrics_endpoint():
    res = client.get("/api/v1/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "version" in data
    assert "total_collections" in data


def test_collection_lifecycle_api():
    coll_name = "test_api_coll"

    # 1. Create collection
    create_payload = {
        "name": coll_name,
        "dimension": 4,
        "metric": "cosine",
        "index_type": "flat",
    }
    res = client.post("/api/v1/collections", json=create_payload)
    assert res.status_code == 201
    assert res.json()["name"] == coll_name

    # 2. Upsert vectors
    upsert_payload = {
        "ids": ["v1", "v2"],
        "vectors": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        "metadatas": [{"cat": "a"}, {"cat": "b"}],
        "documents": ["Doc Alpha", "Doc Beta"],
    }
    res = client.post(f"/api/v1/collections/{coll_name}/upsert", json=upsert_payload)
    assert res.status_code == 200
    assert res.json()["count"] == 2

    # 3. Query vectors
    query_payload = {
        "vector": [1.0, 0.0, 0.0, 0.0],
        "top_k": 2,
    }
    res = client.post(f"/api/v1/collections/{coll_name}/query", json=query_payload)
    assert res.status_code == 200
    data = res.json()
    assert len(data["results"]) == 2
    assert data["results"][0]["id"] == "v1"

    # 4. Delete vectors
    del_payload = {"ids": ["v1"]}
    res = client.post(f"/api/v1/collections/{coll_name}/delete", json=del_payload)
    assert res.status_code == 200
    assert res.json()["deleted"] == 1

    # 5. Delete collection
    res = client.delete(f"/api/v1/collections/{coll_name}")
    assert res.status_code == 200
