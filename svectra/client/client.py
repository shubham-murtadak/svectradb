"""
Unified Client interface for SvectraDB.
Allows developers to seamlessly use SvectraDB as an in-memory/embedded engine
or connect to a remote production cluster with zero code changes.
"""

from typing import Optional, Union
from svectra.client.http_client import RemoteClient
from svectra.core.db import SvectraDB


def Client(
    path: Optional[str] = None,
    persist_dir: Optional[str] = None,
    url: Optional[str] = None,
) -> Union[SvectraDB, RemoteClient]:
    """
    Construct a SvectraDB client.

    Args:
        path: Local directory path for embedded persistence.
        persist_dir: Alias for `path`.
        url: URL of remote SvectraDB server (e.g. 'http://localhost:8080').

    Returns:
        `RemoteClient` if `url` is specified, otherwise `SvectraDB` embedded engine.

    Examples:
        >>> import svectradb
        >>> # 1. In-memory mode:
        >>> db = svectradb.Client()
        >>> # 2. Embedded on-disk mode:
        >>> db = svectradb.Client(path="./svectra_data")
        >>> # 3. Remote client-server mode:
        >>> db = svectradb.Client(url="http://localhost:8080")
    """
    if url is not None:
        return RemoteClient(url=url)

    target_dir = path or persist_dir
    return SvectraDB(persist_dir=target_dir)
