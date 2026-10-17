"""
SvectraDB: High-Performance Vector Database for Modern AI Systems.
"""

from svectra.client.client import Client
from svectra.client.http_client import RemoteClient, RemoteCollection
from svectra.core.collection import SvectraCollection
from svectra.core.db import SvectraDB
from svectra.core.models import (
    CollectionConfig,
    CollectionStats,
    IndexType,
    MetricType,
    SearchResult,
    VectorRecord,
)

__version__ = "0.1.0"
__author__ = "Shubham Murtadak"

__all__ = [
    "Client",
    "RemoteClient",
    "RemoteCollection",
    "SvectraCollection",
    "SvectraDB",
    "CollectionConfig",
    "CollectionStats",
    "IndexType",
    "MetricType",
    "SearchResult",
    "VectorRecord",
    "__version__",
]
