"""
svectradb package alias.
Allows `import svectradb` to seamlessly expose all functionality from `svectra`.
"""

import sys
from svectra import (
    Client,
    RemoteClient,
    RemoteCollection,
    SvectraCollection,
    SvectraDB,
    CollectionConfig,
    CollectionStats,
    IndexType,
    MetricType,
    SearchResult,
    VectorRecord,
    __version__,
    __author__,
)

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
    "__version__,",
]
