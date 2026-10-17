"""
Snapshot and persistence manager for SvectraDB collections.
Handles atomic checkpoints, binary serialization, and crash recovery.
"""

import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from svectra.core.models import CollectionConfig, IndexType, MetricType
from svectra.engine.indices.base import BaseIndex
from svectra.engine.indices.flat import FlatIndex
from svectra.engine.indices.hnsw import HNSWIndex
from svectra.engine.indices.ivf import IVFFlatIndex


class SnapshotManager:
    """
    Manages atomic disk snapshots for a single collection.
    """

    def __init__(self, collection_dir: str):
        self.collection_dir = collection_dir
        self.snapshot_dir = os.path.join(collection_dir, "snapshots")
        self.current_snapshot_dir = os.path.join(self.snapshot_dir, "current")
        os.makedirs(self.snapshot_dir, exist_ok=True)

    def save_checkpoint(
        self,
        config: CollectionConfig,
        index: BaseIndex,
        documents: Dict[str, str],
        metadatas: Dict[str, Dict[str, Any]],
    ) -> str:
        """
        Atomically save index state, vectors, documents, and metadata.
        """
        tmp_dir = os.path.join(self.snapshot_dir, f"tmp_{int(time.time() * 1000)}")
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        # 1. Save config manifest
        config_path = os.path.join(tmp_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config.model_dump_json(indent=2))

        # 2. Save index structure & vectors
        index_prefix = os.path.join(tmp_dir, "index")
        index.save(index_prefix)

        # 3. Save payloads (documents and metadatas) in JSONL format
        payload_path = os.path.join(tmp_dir, "payloads.jsonl")
        with open(payload_path, "w", encoding="utf-8") as f:
            # Iterate through all IDs in index
            if hasattr(index, "ids"):
                all_ids = index.ids
            elif hasattr(index, "int_to_id"):
                all_ids = index.int_to_id
            else:
                all_ids = list(metadatas.keys())

            for doc_id in all_ids:
                record = {
                    "id": doc_id,
                    "metadata": metadatas.get(doc_id),
                    "document": documents.get(doc_id),
                }
                f.write(json.dumps(record) + "\n")

        # 4. Atomic swap: replace current_snapshot_dir
        backup_dir = os.path.join(self.snapshot_dir, "backup")
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)

        if os.path.exists(self.current_snapshot_dir):
            os.rename(self.current_snapshot_dir, backup_dir)

        os.rename(tmp_dir, self.current_snapshot_dir)

        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)

        return self.current_snapshot_dir

    def has_snapshot(self) -> bool:
        return os.path.exists(os.path.join(self.current_snapshot_dir, "config.json"))

    def load_checkpoint(
        self,
    ) -> Optional[Tuple[CollectionConfig, BaseIndex, Dict[str, str], Dict[str, Dict[str, Any]]]]:
        """
        Load the current checkpoint. Returns (config, index, documents, metadatas) or None.
        """
        if not self.has_snapshot():
            return None

        # 1. Load config
        config_path = os.path.join(self.current_snapshot_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = CollectionConfig.model_validate_json(f.read())

        # 2. Load index
        index_prefix = os.path.join(self.current_snapshot_dir, "index")
        if config.index_type == IndexType.FLAT:
            index = FlatIndex.load(index_prefix)
        elif config.index_type == IndexType.HNSW:
            index = HNSWIndex.load(index_prefix)
        elif config.index_type == IndexType.IVF:
            index = IVFFlatIndex.load(index_prefix)
        else:
            raise ValueError(f"Unknown index type: {config.index_type}")

        # 3. Load payloads
        documents: Dict[str, str] = {}
        metadatas: Dict[str, Dict[str, Any]] = {}
        payload_path = os.path.join(self.current_snapshot_dir, "payloads.jsonl")

        if os.path.exists(payload_path):
            with open(payload_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    doc_id = record["id"]
                    if record.get("document") is not None:
                        documents[doc_id] = record["document"]
                    if record.get("metadata") is not None:
                        metadatas[doc_id] = record["metadata"]

        return (config, index, documents, metadatas)
