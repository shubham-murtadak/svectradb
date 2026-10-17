"""
HTTP REST API Client for SvectraDB.
Connects to a remote running SvectraDB server instance.
"""

from typing import Any, Dict, List, Optional, Union
import requests
import numpy as np

from svectra.core.models import (
    CollectionConfig,
    CollectionStats,
    IndexType,
    MetricType,
    QueryRequest,
    QueryResponse,
    SearchResult,
    VectorRecord,
)


class RemoteCollection:
    """Proxy representing a remote collection on a SvectraDB server."""

    def __init__(self, base_url: str, name: str, session: requests.Session):
        self.base_url = base_url.rstrip("/")
        self.name = name
        self.session = session
        self._url = f"{self.base_url}/api/v1/collections/{self.name}"

    def upsert(
        self,
        vectors: Optional[Union[List[List[float]], np.ndarray]] = None,
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ) -> List[str]:
        vec_list = None
        if vectors is not None:
            if isinstance(vectors, np.ndarray):
                vec_list = vectors.tolist()
            else:
                vec_list = vectors

        payload = {
            "ids": ids,
            "vectors": vec_list,
            "metadatas": metadatas,
            "documents": documents,
        }
        res = self.session.post(f"{self._url}/upsert", json=payload)
        res.raise_for_status()
        return res.json()["ids"]

    def query(
        self,
        vector: Optional[Union[List[float], np.ndarray]] = None,
        query_text: Optional[str] = None,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        include_vector: bool = False,
        include_metadata: bool = True,
        include_document: bool = True,
    ) -> List[SearchResult]:
        vec_list = None
        if vector is not None:
            if isinstance(vector, np.ndarray):
                vec_list = vector.tolist()
            else:
                vec_list = vector

        payload = {
            "vector": vec_list,
            "query_text": query_text,
            "top_k": top_k,
            "filter": filter,
            "include_vector": include_vector,
            "include_metadata": include_metadata,
            "include_document": include_document,
        }
        res = self.session.post(f"{self._url}/query", json=payload)
        res.raise_for_status()
        data = res.json()
        return [SearchResult(**r) for r in data["results"]]

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> int:
        payload = {"ids": ids, "filter": filter}
        res = self.session.post(f"{self._url}/delete", json=payload)
        res.raise_for_status()
        return res.json().get("deleted", 0)

    def count(self) -> int:
        return self.stats().count

    def stats(self) -> CollectionStats:
        res = self.session.get(self._url)
        res.raise_for_status()
        return CollectionStats(**res.json())

    def flush(self) -> None:
        res = self.session.post(f"{self._url}/flush")
        res.raise_for_status()


class RemoteClient:
    """Client for interacting with a remote SvectraDB cluster or single server."""

    def __init__(self, url: str = "http://localhost:8080", timeout: float = 30.0):
        self.base_url = url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self.timeout = timeout

    def create_collection(
        self,
        name: str,
        dimension: int,
        metric: MetricType = MetricType.COSINE,
        index_type: IndexType = IndexType.HNSW,
        index_params: Optional[Dict[str, Any]] = None,
    ) -> RemoteCollection:
        payload = {
            "name": name,
            "dimension": dimension,
            "metric": MetricType(metric).value,
            "index_type": IndexType(index_type).value,
            "index_params": index_params or {},
        }
        res = self.session.post(f"{self.base_url}/api/v1/collections", json=payload)
        res.raise_for_status()
        return RemoteCollection(self.base_url, name, self.session)

    def get_collection(self, name: str) -> RemoteCollection:
        res = self.session.get(f"{self.base_url}/api/v1/collections/{name}")
        res.raise_for_status()
        return RemoteCollection(self.base_url, name, self.session)

    def get_or_create_collection(
        self,
        name: str,
        dimension: int = 128,
        metric: MetricType = MetricType.COSINE,
        index_type: IndexType = IndexType.HNSW,
        index_params: Optional[Dict[str, Any]] = None,
    ) -> RemoteCollection:
        try:
            return self.get_collection(name)
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                return self.create_collection(
                    name=name,
                    dimension=dimension,
                    metric=metric,
                    index_type=index_type,
                    index_params=index_params,
                )
            raise

    def list_collections(self) -> List[str]:
        res = self.session.get(f"{self.base_url}/api/v1/collections")
        res.raise_for_status()
        return res.json().get("collections", [])

    def delete_collection(self, name: str) -> bool:
        res = self.session.delete(f"{self.base_url}/api/v1/collections/{name}")
        if res.status_code == 404:
            return False
        res.raise_for_status()
        return True

    def health(self) -> Dict[str, Any]:
        res = self.session.get(f"{self.base_url}/api/v1/health")
        res.raise_for_status()
        return res.json()

    def metrics(self) -> Dict[str, Any]:
        res = self.session.get(f"{self.base_url}/api/v1/metrics")
        res.raise_for_status()
        return res.json()
