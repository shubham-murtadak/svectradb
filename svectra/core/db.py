"""
SvectraDB Multi-Collection Database Orchestrator.
Manages collection lifecycles, configuration discovery, and persistence directories.
"""

import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional
from svectra.core.collection import SvectraCollection
from svectra.core.models import CollectionConfig, IndexType, MetricType


class SvectraDB:
    """
    Central database engine instance managing multiple isolated collections.
    Supports in-memory mode or persistent directory on disk.
    """

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir
        self.collections: Dict[str, SvectraCollection] = {}

        if self.persist_dir:
            os.makedirs(self.persist_dir, exist_ok=True)
            self._load_existing_collections()

    def _load_existing_collections(self) -> None:
        """Scan persistence directory and load saved collections."""
        colls_dir = os.path.join(self.persist_dir, "collections")
        if not os.path.exists(colls_dir):
            return

        for name in os.listdir(colls_dir):
            target_coll = os.path.join(colls_dir, name)
            if not os.path.isdir(target_coll):
                continue

            config_file = os.path.join(target_coll, "snapshots", "current", "config.json")
            if os.path.exists(config_file):
                try:
                    with open(config_file, "r", encoding="utf-8") as f:
                        config = CollectionConfig.model_validate_json(f.read())
                    self.collections[name] = SvectraCollection(
                        config=config,
                        persist_dir=self.persist_dir,
                    )
                except Exception as e:
                    print(f"Warning: Failed to load collection {name}: {e}")

    def create_collection(
        self,
        name: str,
        dimension: int,
        metric: MetricType = MetricType.COSINE,
        index_type: IndexType = IndexType.HNSW,
        index_params: Optional[Dict[str, Any]] = None,
    ) -> SvectraCollection:
        """Create and initialize a new vector collection."""
        if name in self.collections:
            raise ValueError(f"Collection '{name}' already exists.")

        config = CollectionConfig(
            name=name,
            dimension=dimension,
            metric=MetricType(metric),
            index_type=IndexType(index_type),
            index_params=index_params or {},
            created_at=time.time(),
        )

        collection = SvectraCollection(config=config, persist_dir=self.persist_dir)
        self.collections[name] = collection

        # Immediately write initial checkpoint if persistent
        if self.persist_dir:
            collection.flush()

        return collection

    def get_collection(self, name: str) -> SvectraCollection:
        """Retrieve existing collection by name."""
        if name not in self.collections:
            raise KeyError(f"Collection '{name}' does not exist.")
        return self.collections[name]

    def get_or_create_collection(
        self,
        name: str,
        dimension: int = 128,
        metric: MetricType = MetricType.COSINE,
        index_type: IndexType = IndexType.HNSW,
        index_params: Optional[Dict[str, Any]] = None,
    ) -> SvectraCollection:
        """Get collection if it exists, otherwise create it."""
        if name in self.collections:
            return self.collections[name]
        return self.create_collection(
            name=name,
            dimension=dimension,
            metric=metric,
            index_type=index_type,
            index_params=index_params,
        )

    def list_collections(self) -> List[str]:
        """List all collection names in database."""
        return sorted(list(self.collections.keys()))

    def delete_collection(self, name: str) -> bool:
        """Permanently drop a collection and remove its disk artifacts."""
        if name not in self.collections:
            return False

        coll = self.collections.pop(name)
        coll.close()

        if self.persist_dir:
            target_path = os.path.join(self.persist_dir, "collections", name)
            if os.path.exists(target_path):
                shutil.rmtree(target_path, ignore_errors=True)

        return True

    def flush_all(self) -> None:
        """Flush all active collections to disk."""
        for coll in self.collections.values():
            coll.flush()

    def close(self) -> None:
        """Cleanly close all collections."""
        for coll in self.collections.values():
            coll.close()
        self.collections.clear()
