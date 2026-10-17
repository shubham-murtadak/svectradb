"""
Binary Write-Ahead Log (WAL) with CRC32 checksums for SvectraDB.
Guarantees crash durability and ACID-like atomicity for mutating operations.
"""

from enum import Enum
import json
import os
import struct
import zlib
from typing import Any, Dict, Iterator, List, Optional, Tuple


WAL_MAGIC = b"SVW1"
HEADER_FORMAT = ">4sBII"  # Magic(4B) + OpCode(1B) + Length(4B) + CRC32(4B) = 13 bytes
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


class WalOpCode(int, Enum):
    INSERT = 1
    DELETE = 2
    CHECKPOINT = 3


class SyncMode(str, Enum):
    ALWAYS = "always"
    PERIODIC = "periodic"
    NONE = "none"


class WriteAheadLog:
    """
    Append-only binary log with CRC32 verification.
    """

    def __init__(self, wal_path: str, sync_mode: SyncMode = SyncMode.ALWAYS):
        self.wal_path = wal_path
        self.sync_mode = sync_mode
        self._file = None
        os.makedirs(os.path.dirname(wal_path), exist_ok=True)
        self._open()

    def _open(self) -> None:
        self._file = open(self.wal_path, "a+b")

    def append(self, op: WalOpCode, payload: Dict[str, Any]) -> int:
        """
        Append a binary frame to the WAL.
        Returns the byte offset where the frame was written.
        """
        payload_bytes = json.dumps(payload).encode("utf-8")
        payload_len = len(payload_bytes)
        crc = zlib.crc32(payload_bytes) & 0xFFFFFFFF

        header = struct.pack(HEADER_FORMAT, WAL_MAGIC, int(op), payload_len, crc)

        self._file.seek(0, os.SEEK_END)
        offset = self._file.tell()
        self._file.write(header + payload_bytes)

        if self.sync_mode == SyncMode.ALWAYS:
            self.sync()

        return offset

    def sync(self) -> None:
        """Flush and fsync the log file to physical storage."""
        if self._file and not self._file.closed:
            self._file.flush()
            try:
                os.fsync(self._file.fileno())
            except OSError:
                pass

    def replay(self) -> Iterator[Tuple[WalOpCode, Dict[str, Any]]]:
        """
        Iterate through and verify all recorded frames in the WAL.
        If a corrupted frame is encountered at the tail, it stops cleanly.
        """
        if not os.path.exists(self.wal_path):
            return

        with open(self.wal_path, "rb") as f:
            valid_pos = 0
            while True:
                header_bytes = f.read(HEADER_SIZE)
                if len(header_bytes) < HEADER_SIZE:
                    break

                magic, opcode_val, length, expected_crc = struct.unpack(
                    HEADER_FORMAT, header_bytes
                )

                if magic != WAL_MAGIC:
                    # Corrupted file header or unaligned read
                    break

                payload_bytes = f.read(length)
                if len(payload_bytes) < length:
                    # Incomplete tail write during sudden shutdown
                    break

                actual_crc = zlib.crc32(payload_bytes) & 0xFFFFFFFF
                if actual_crc != expected_crc:
                    # Checksum mismatch
                    break

                try:
                    payload = json.loads(payload_bytes.decode("utf-8"))
                    opcode = WalOpCode(opcode_val)
                    valid_pos = f.tell()
                    yield (opcode, payload)
                except Exception:
                    break

    def truncate(self) -> None:
        """Clear the WAL after a successful snapshot checkpoint."""
        if self._file and not self._file.closed:
            self._file.close()
        with open(self.wal_path, "wb") as f:
            f.truncate(0)
        self._open()

    def close(self) -> None:
        if self._file and not self._file.closed:
            self.sync()
            self._file.close()
