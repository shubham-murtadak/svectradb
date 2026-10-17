"""
Unit tests for binary Write-Ahead Log (WAL) and crash recovery.
"""

import os
import tempfile
from svectra.storage.wal import WriteAheadLog, WalOpCode, SyncMode


def test_wal_append_and_replay():
    with tempfile.TemporaryDirectory() as tmpdir:
        wal_path = os.path.join(tmpdir, "test.wal")
        wal = WriteAheadLog(wal_path, sync_mode=SyncMode.ALWAYS)

        # 1. Append records
        wal.append(WalOpCode.INSERT, {"ids": ["v1", "v2"], "vectors": [[0.1, 0.2], [0.3, 0.4]]})
        wal.append(WalOpCode.DELETE, {"ids": ["v1"]})
        wal.close()

        # 2. Replay
        wal2 = WriteAheadLog(wal_path)
        records = list(wal2.replay())
        assert len(records) == 2
        assert records[0][0] == WalOpCode.INSERT
        assert records[0][1]["ids"] == ["v1", "v2"]
        assert records[1][0] == WalOpCode.DELETE
        assert records[1][1]["ids"] == ["v1"]
        wal2.close()


def test_wal_crc32_corruption_handling():
    with tempfile.TemporaryDirectory() as tmpdir:
        wal_path = os.path.join(tmpdir, "corrupt.wal")
        wal = WriteAheadLog(wal_path, sync_mode=SyncMode.ALWAYS)

        wal.append(WalOpCode.INSERT, {"ids": ["valid_1"], "vectors": [[1.0, 2.0]]})
        wal.close()

        # Append corrupted bytes to tail
        with open(wal_path, "ab") as f:
            f.write(b"CORRUPTED_PARTIAL_BYTES_FROM_HARD_CRASH_9999")

        # Replay should safely stop at corrupted tail without raising exception
        wal_recover = WriteAheadLog(wal_path)
        records = list(wal_recover.replay())
        assert len(records) == 1
        assert records[0][1]["ids"] == ["valid_1"]
        wal_recover.close()
