"""
Collection management engine for SvectraDB.
Integrates vector indexing, metadata filtering, durability (WAL & snapshots),
and optional embedding transformation into a high-level, thread-safe interface.
"""

import os
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

from svectra.core.models import (
    CollectionConfig,
    CollectionStats,
    IndexType,
    MetricType,
    SearchResult,
    VectorRecord,
)
from svectra.embeddings.base import BaseEmbeddingFunction
from svectra.embeddings.fast_hash import FastHashEmbedding
from svectra.engine.distance import distance_to_score
from svectra.engine.filter import match_filter
from svectra.engine.indices.base import BaseIndex
from svectra.engine.indices.flat import FlatIndex
from svectra.engine.indices.hnsw import HNSWIndex
from svectra.engine.indices.ivf import IVFFlatIndex
from svectra.storage.snapshot import SnapshotManager
from svectra.storage.wal import SyncMode, WalOpCode, WriteAheadLog


class SvectraCollection:
    """
    Primary interface for vector operations within a named collection namespace.
    """

    def __init__(
        self,
        config: CollectionConfig,
        persist_dir: Optional[str] = None,
        embedding_function: Optional[BaseEmbeddingFunction] = None,
        wal_sync_mode: SyncMode = SyncMode.ALWAYS,
    ):
        self.config = config
        self.persist_dir = persist_dir
        self.embedding_function = embedding_function or FastHashEmbedding(dimension=config.dimension)
        self.wal_sync_mode = wal_sync_mode

        self.documents: Dict[str, str] = {}
        self.metadatas: Dict[str, Dict[str, Any]] = {}

        self.wal: Optional[WriteAheadLog] = None
        self.snapshot_manager: Optional[SnapshotManager] = None

        # Initialize storage managers if persist_dir is provided
        if self.persist_dir:
            coll_dir = os.path.join(self.persist_dir, "collections", self.config.name)
            os.makedirs(coll_dir, exist_ok=True)
            self.snapshot_manager = SnapshotManager(coll_dir)
            wal_path = os.path.join(coll_dir, "wal.bin")
            self.wal = WriteAheadLog(wal_path, sync_mode=self.wal_sync_mode)

        # Initialize index
        self.index: BaseIndex = self._create_index()

        # Recover from snapshot and replay WAL if available
        if self.persist_dir:
            self._recover()

    def _create_index(self) -> BaseIndex:
        """Instantiate index based on collection configuration."""
        params = self.config.index_params or {}
        dim = self.config.dimension
        metric = self.config.metric

        if self.config.index_type == IndexType.FLAT:
            return FlatIndex(dimension=dim, metric=metric)

        elif self.config.index_type == IndexType.HNSW:
            return HNSWIndex(
                dimension=dim,
                metric=metric,
                M=params.get("M", 16),
                ef_construction=params.get("ef_construction", 100),
                ef_search=params.get("ef_search", 64),
            )

        elif self.config.index_type == IndexType.IVF:
            return IVFFlatIndex(
                dimension=dim,
                metric=metric,
                nlist=params.get("nlist", 64),
                nprobe=params.get("nprobe", 8),
            )

        else:
            raise ValueError(f"Unsupported index type: {self.config.index_type}")

    def _recover(self) -> None:
        """Restore from snapshot and replay uncommitted WAL records."""
        if self.snapshot_manager and self.snapshot_manager.has_snapshot():
            res = self.snapshot_manager.load_checkpoint()
            if res:
                _, loaded_index, docs, metas = res
                self.index = loaded_index
                self.documents = docs
                self.metadatas = metas

        # Replay WAL
        if self.wal:
            replayed_count = 0
            for opcode, payload in self.wal.replay():
                if opcode == WalOpCode.INSERT:
                    ids = payload["ids"]
                    vectors = np.array(payload["vectors"], dtype=np.float32)
                    self.index.add(ids, vectors)

                    if payload.get("metadatas"):
                        for i, doc_id in enumerate(ids):
                            if payload["metadatas"][i] is not None:
                                self.metadatas[doc_id] = payload["metadatas"][i]

                    if payload.get("documents"):
                        for i, doc_id in enumerate(ids):
                            if payload["documents"][i] is not None:
                                self.documents[doc_id] = payload["documents"][i]

                    replayed_count += 1

                elif opcode == WalOpCode.DELETE:
                    del_ids = payload.get("ids", [])
                    self.index.delete(del_ids)
                    for did in del_ids:
                        self.documents.pop(did, None)
                        self.metadatas.pop(did, None)
                    replayed_count += 1

    def upsert(
        self,
        vectors: Optional[Union[List[List[float]], np.ndarray]] = None,
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Insert or update vectors into collection.
        If vectors are omitted but documents are provided, embeddings will be computed automatically.
        """
        # Determine batch count
        count = 0
        if vectors is not None:
            count = len(vectors)
        elif documents is not None:
            count = len(documents)
        elif ids is not None:
            count = len(ids)

        if count == 0:
            return []

        # Auto-generate IDs if not supplied
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(count)]
        elif len(ids) != count:
            raise ValueError(f"Length of ids ({len(ids)}) does not match batch count ({count})")

        # Auto-compute embeddings if vectors not given
        if vectors is None:
            if documents is None:
                raise ValueError("Must provide either 'vectors' or 'documents'")
            vectors_arr = self.embedding_function.embed_documents(documents)
        else:
            vectors_arr = np.ascontiguousarray(vectors, dtype=np.float32)

        if vectors_arr.ndim == 1:
            vectors_arr = vectors_arr.reshape(1, -1)

        if vectors_arr.shape[1] != self.config.dimension:
            raise ValueError(
                f"Vector dimension {vectors_arr.shape[1]} != collection dimension {self.config.dimension}"
            )

        # Append to WAL
        if self.wal:
            wal_payload = {
                "ids": ids,
                "vectors": vectors_arr.tolist(),
                "metadatas": metadatas,
                "documents": documents,
            }
            self.wal.append(WalOpCode.INSERT, wal_payload)

        # Update in-memory metadata and documents
        if metadatas is not None:
            for i, doc_id in enumerate(ids):
                if i < len(metadatas) and metadatas[i] is not None:
                    self.metadatas[doc_id] = metadatas[i]

        if documents is not None:
            for i, doc_id in enumerate(ids):
                if i < len(documents) and documents[i] is not None:
                    self.documents[doc_id] = documents[i]

        # Add to index
        self.index.add(ids, vectors_arr)
        return ids

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
        """
        Query the collection using a vector or query text with optional metadata filters.
        """
        if vector is None:
            if query_text is None:
                raise ValueError("Must provide either 'vector' or 'query_text'")
            query_vec = self.embedding_function.embed_query(query_text)
        else:
            query_vec = np.ascontiguousarray(vector, dtype=np.float32).flatten()

        if query_vec.shape[0] != self.config.dimension:
            raise ValueError(
                f"Query vector dimension {query_vec.shape[0]} != collection dimension {self.config.dimension}"
            )

        # Pre-filter candidate IDs if filter expression is provided
        filter_ids: Optional[Set[str]] = None
        if filter:
            # Check all existing IDs
            matched_ids = set()
            all_ids = list(self.metadatas.keys())
            # Also include IDs in index that may have no metadata
            if hasattr(self.index, "ids"):
                index_ids = self.index.ids
            elif hasattr(self.index, "int_to_id"):
                index_ids = self.index.int_to_id
            else:
                index_ids = []

            for doc_id in set(all_ids + index_ids):
                meta = self.metadatas.get(doc_id)
                if match_filter(meta, filter):
                    matched_ids.add(doc_id)

            filter_ids = matched_ids
            if not filter_ids:
                return []

        # Execute search on vector index
        raw_results = self.index.search(query_vec, k=top_k, filter_ids=filter_ids)

        # Format output SearchResults
        results: List[SearchResult] = []
        for doc_id, dist in raw_results:
            score = distance_to_score(dist, self.config.metric)
            vec = self.index.get_vector(doc_id).tolist() if include_vector else None
            meta = self.metadatas.get(doc_id) if include_metadata else None
            doc = self.documents.get(doc_id) if include_document else None

            results.append(
                SearchResult(
                    id=doc_id,
                    score=round(score, 6),
                    distance=round(dist, 6),
                    vector=vec,
                    metadata=meta,
                    document=doc,
                )
            )

        return results

    def get(self, id: str) -> Optional[VectorRecord]:
        """Retrieve record by ID."""
        vec = self.index.get_vector(id)
        if vec is None:
            return None
        return VectorRecord(
            id=id,
            vector=vec.tolist(),
            metadata=self.metadatas.get(id),
            document=self.documents.get(id),
        )

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Delete vectors matching IDs or filter expression.
        """
        targets = set()
        if ids:
            targets.update(ids)

        if filter:
            for doc_id, meta in list(self.metadatas.items()):
                if match_filter(meta, filter):
                    targets.add(doc_id)

        if not targets:
            return 0

        target_list = list(targets)

        # Log to WAL
        if self.wal:
            self.wal.append(WalOpCode.DELETE, {"ids": target_list})

        # Remove from index
        deleted_count = self.index.delete(target_list)

        # Remove payloads
        for doc_id in target_list:
            self.documents.pop(doc_id, None)
            self.metadatas.pop(doc_id, None)

        return deleted_count

    def flush(self) -> None:
        """Create an atomic snapshot checkpoint and truncate the WAL."""
        if self.snapshot_manager:
            self.snapshot_manager.save_checkpoint(
                config=self.config,
                index=self.index,
                documents=self.documents,
                metadatas=self.metadatas,
            )
            if self.wal:
                self.wal.truncate()

    def count(self) -> int:
        return self.index.count()

    def stats(self) -> CollectionStats:
        count = self.count()
        # Estimate memory usage
        vec_bytes = count * self.config.dimension * 4
        return CollectionStats(
            name=self.config.name,
            count=count,
            dimension=self.config.dimension,
            metric=self.config.metric,
            index_type=self.config.index_type,
            memory_bytes=vec_bytes,
            segments_count=1 if (self.snapshot_manager and self.snapshot_manager.has_snapshot()) else 0,
            created_at=self.config.created_at,
        )

    def close(self) -> None:
        self.flush()
        if self.wal:
            self.wal.close()
