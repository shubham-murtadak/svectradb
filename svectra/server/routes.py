"""
FastAPI REST API routes for SvectraDB.
"""

import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, status

from svectra.core.models import (
    CollectionConfig,
    CollectionStats,
    CreateCollectionRequest,
    DeleteRequest,
    QueryRequest,
    QueryResponse,
    SystemMetrics,
    UpsertRequest,
)

router = APIRouter(prefix="/api/v1")


@router.get("/health", tags=["System"])
def health_check():
    """Health check endpoint for Kubernetes liveness/readiness probes."""
    return {
        "status": "healthy",
        "service": "svectradb",
        "timestamp": time.time(),
    }


@router.get("/metrics", response_model=SystemMetrics, tags=["System"])
def get_metrics():
    """Retrieve runtime telemetry, collection counts, and latency statistics."""
    from svectra.server.app import get_db, app
    db = get_db()
    uptime = time.time() - getattr(app.state, "start_time", time.time())
    total_colls = len(db.collections)
    total_vecs = sum(c.count() for c in db.collections.values())
    total_mem = sum(c.stats().memory_bytes for c in db.collections.values()) / (1024 * 1024)

    queries = getattr(app.state, "total_queries", 0)
    total_lat = getattr(app.state, "total_query_latency_ms", 0.0)
    avg_lat = round(total_lat / queries, 3) if queries > 0 else 0.0

    return SystemMetrics(
        version="0.1.0",
        uptime_seconds=round(uptime, 1),
        total_collections=total_colls,
        total_vectors=total_vecs,
        total_memory_mb=round(total_mem, 2),
        queries_total=queries,
        average_query_latency_ms=avg_lat,
    )


@router.post("/collections", status_code=status.HTTP_201_CREATED, tags=["Collections"])
def create_collection(req: CreateCollectionRequest):
    from svectra.server.app import get_db
    db = get_db()

    try:
        coll = db.create_collection(
            name=req.name,
            dimension=req.dimension,
            metric=req.metric,
            index_type=req.index_type,
            index_params=req.index_params,
        )
        return coll.stats()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/collections", tags=["Collections"])
def list_collections():
    from svectra.server.app import get_db
    db = get_db()
    names = db.list_collections()
    colls_info = []
    for name in names:
        try:
            colls_info.append(db.get_collection(name).stats())
        except Exception:
            pass
    return {"collections": names, "details": colls_info}


@router.get("/collections/{name}", tags=["Collections"])
def get_collection(name: str):
    from svectra.server.app import get_db
    db = get_db()
    try:
        coll = db.get_collection(name)
        return coll.stats()
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")


@router.delete("/collections/{name}", tags=["Collections"])
def delete_collection(name: str):
    from svectra.server.app import get_db
    db = get_db()
    success = db.delete_collection(name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    return {"message": f"Collection '{name}' successfully deleted"}


@router.post("/collections/{name}/upsert", tags=["Vectors"])
def upsert_vectors(name: str, req: UpsertRequest):
    from svectra.server.app import get_db
    db = get_db()
    try:
        coll = db.get_collection(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    try:
        inserted_ids = coll.upsert(
            vectors=req.vectors,
            ids=req.ids,
            metadatas=req.metadatas,
            documents=req.documents,
        )
        return {"status": "success", "count": len(inserted_ids), "ids": inserted_ids}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/collections/{name}/query", response_model=QueryResponse, tags=["Vectors"])
def query_vectors(name: str, req: QueryRequest):
    from svectra.server.app import get_db, app
    db = get_db()
    try:
        coll = db.get_collection(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    start_t = time.perf_counter()
    try:
        results = coll.query(
            vector=req.vector,
            query_text=req.query_text,
            top_k=req.top_k,
            filter=req.filter,
            include_vector=req.include_vector,
            include_metadata=req.include_metadata,
            include_document=req.include_document,
        )
        latency_ms = (time.perf_counter() - start_t) * 1000.0

        # Update app query telemetry
        if hasattr(app.state, "total_queries"):
            app.state.total_queries += 1
            app.state.total_query_latency_ms += latency_ms

        return QueryResponse(
            collection=name,
            results=results,
            count=len(results),
            latency_ms=round(latency_ms, 3),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/collections/{name}/delete", tags=["Vectors"])
def delete_vectors(name: str, req: DeleteRequest):
    from svectra.server.app import get_db
    db = get_db()
    try:
        coll = db.get_collection(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    deleted_count = coll.delete(ids=req.ids, filter=req.filter)
    return {"status": "success", "deleted": deleted_count}


@router.post("/collections/{name}/flush", tags=["Collections"])
def flush_collection(name: str):
    from svectra.server.app import get_db
    db = get_db()
    try:
        coll = db.get_collection(name)
        coll.flush()
        return {"status": "success", "message": f"Collection '{name}' flushed to disk snapshot"}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")


@router.get("/collections/{name}/sample", tags=["Vectors"])
def get_sample_vectors(name: str, limit: int = Query(default=100, le=500)):
    """Fetch sample vectors and 2D projections for the dashboard visualization."""
    from svectra.server.app import get_db
    import numpy as np

    db = get_db()
    try:
        coll = db.get_collection(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    if coll.count() == 0:
        return {"points": [], "dimension": coll.config.dimension}

    # Extract sample IDs
    if hasattr(coll.index, "ids"):
        sample_ids = coll.index.ids[:limit]
    elif hasattr(coll.index, "int_to_id"):
        sample_ids = coll.index.int_to_id[:limit]
    else:
        sample_ids = list(coll.metadatas.keys())[:limit]

    vectors_list = []
    valid_ids = []
    for sid in sample_ids:
        vec = coll.index.get_vector(sid)
        if vec is not None:
            vectors_list.append(vec)
            valid_ids.append(sid)

    if not vectors_list:
        return {"points": [], "dimension": coll.config.dimension}

    matrix = np.array(vectors_list)
    # Perform fast 2D PCA projection for visualization
    if matrix.shape[1] > 2:
        mean = np.mean(matrix, axis=0)
        centered = matrix - mean
        if matrix.shape[0] >= 2:
            try:
                u, s, vt = np.linalg.svd(centered, full_matrices=False)
                proj_2d = np.dot(centered, vt[:2].T)
            except Exception:
                proj_2d = centered[:, :2]
        else:
            proj_2d = centered[:, :2]
    else:
        proj_2d = matrix[:, :2]

    points = []
    for i, sid in enumerate(valid_ids):
        points.append({
            "id": sid,
            "x": float(proj_2d[i, 0]),
            "y": float(proj_2d[i, 1]),
            "document": coll.documents.get(sid, "")[:100],
            "metadata": coll.metadatas.get(sid, {}),
        })

    return {"points": points, "dimension": coll.config.dimension, "count": len(points)}
