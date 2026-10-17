"""
Data models and schemas for SvectraDB.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class MetricType(str, Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT = "dot"
    MANHATTAN = "manhattan"


class IndexType(str, Enum):
    FLAT = "flat"
    HNSW = "hnsw"
    IVF = "ivf"


class VectorRecord(BaseModel):
    id: str
    vector: List[float]
    metadata: Optional[Dict[str, Any]] = None
    document: Optional[str] = None


class SearchResult(BaseModel):
    id: str
    score: float = Field(..., description="Similarity score (higher is more similar)")
    distance: float = Field(..., description="Raw geometric distance (lower is closer)")
    vector: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    document: Optional[str] = None


class CollectionConfig(BaseModel):
    name: str
    dimension: int
    metric: MetricType = MetricType.COSINE
    index_type: IndexType = IndexType.HNSW
    index_params: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=float)


class CollectionStats(BaseModel):
    name: str
    count: int
    dimension: int
    metric: MetricType
    index_type: IndexType
    memory_bytes: int
    segments_count: int
    created_at: float


class CreateCollectionRequest(BaseModel):
    name: str
    dimension: int
    metric: MetricType = MetricType.COSINE
    index_type: IndexType = IndexType.HNSW
    index_params: Optional[Dict[str, Any]] = None


class UpsertRequest(BaseModel):
    ids: Optional[List[str]] = None
    vectors: Optional[List[List[float]]] = None
    metadatas: Optional[List[Dict[str, Any]]] = None
    documents: Optional[List[str]] = None


class QueryRequest(BaseModel):
    vector: Optional[List[float]] = None
    query_text: Optional[str] = None
    top_k: int = 10
    filter: Optional[Dict[str, Any]] = None
    include_vector: bool = False
    include_metadata: bool = True
    include_document: bool = True


class QueryResponse(BaseModel):
    collection: str
    results: List[SearchResult]
    count: int
    latency_ms: float


class DeleteRequest(BaseModel):
    ids: Optional[List[str]] = None
    filter: Optional[Dict[str, Any]] = None


class SystemMetrics(BaseModel):
    version: str
    uptime_seconds: float
    total_collections: int
    total_vectors: int
    total_memory_mb: float
    queries_total: int
    average_query_latency_ms: float
