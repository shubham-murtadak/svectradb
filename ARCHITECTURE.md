# SvectraDB Architecture & System Design

## 1. Executive Summary

**SvectraDB** is a distributed-ready, high-performance, embedded & client-server vector database engineered from first principles. It delivers sub-millisecond Approximate Nearest Neighbor (ANN) search, multi-tenant collections, ACID-like durability via Write-Ahead Logging (WAL), rich metadata filtering, and real-time visualization.

---

## 2. High-Level Architecture

```
                                  +---------------------------------------+
                                  |         Client Layer / Users          |
                                  |  - Python SDK (svectradb.Client)     |
                                  |  - REST / JSON API (FastAPI)          |
                                  |  - Embedded Web Dashboard UI          |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |          API & Protocol Layer         |
                                  |  - Request Validation (Pydantic v2)   |
                                  |  - Telemetry & Metrics Middleware     |
                                  |  - Embedded Static Asset Server       |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |          SvectraDB Orchestrator       |
                                  |  - Collection Catalog & Namespaces    |
                                  |  - Worker Thread Pools & Async Tasks  |
                                  |  - Embedding Pipeline (Optional)      |
                                  +-------------------+-------------------+
                                                      |
                        +-----------------------------+-----------------------------+
                        |                                                           |
                        v                                                           v
+-----------------------------------------------+           +-----------------------------------------------+
|               Collection Core                 |           |            Storage & Durability               |
|                                               |           |                                               |
|  +-----------------------------------------+  |           |  +-----------------------------------------+  |
|  |           Vector Index Engines          |  |           |  |          Write-Ahead Log (WAL)          |  |
|  |  * FlatIndex: Exact Brute-Force         |  |           |  |  * Append-only binary log               |  |
|  |  * HNSWIndex: Multi-layer Skip Graph    |  | <=======> |  |  * CRC32 checksum per entry             |  |
|  |  * IVFFlatIndex: K-Means Voronoi Cells  |  |           |  |  * Zero data loss recovery on boot      |  |
|  +-----------------------------------------+  |           |  +-----------------------------------------+  |
|                                               |           |                                               |
|  +-----------------------------------------+  |           |  +-----------------------------------------+  |
|  |       Vector Math & SIMD Acceleration   |  |           |  |            Segment Store                |  |
|  |  * Cosine Similarity / Distance        |  |           |  |  * Active In-Memory Segments            |  |
|  |  * Euclidean ($L_2$) Distance          |  |           |  |  * Immutable Disk-Flushed Segments      |  |
|  |  * Dot Product / Inner Product          |  |           |  |  * Atomic checkpoint snapshots          |  |
|  |  * Manhattan ($L_1$) Distance           |  |           |  |  * Background Compaction Worker         |  |
|  +-----------------------------------------+  |           |  +-----------------------------------------+  |
|                                               |           +-----------------------------------------------+
|  +-----------------------------------------+  |
|  |         Metadata Filter Engine          |  |
|  |  * Operators: $eq, $ne, $in, $gt, $and  |  |
|  |  * Pre-filtering & Post-filtering       |  |
|  +-----------------------------------------+  |
|                                               |
|  +-----------------------------------------+  |
|  |        Scalar Quantization (SQ8)        |  |
|  |  * 75% memory reduction (FP32 -> INT8)  |  |
|  +-----------------------------------------+  |
+-----------------------------------------------+
```

---

## 3. Subsystem Breakdown

### 3.1 Vector Indexing Engines
SvectraDB supports pluggable indexing strategies depending on dataset scale, memory budgets, and recall requirements:

1. **FlatIndex (Brute-Force / Exact)**:
   - Computes distances across 100% of the vector space.
   - Guaranteed 1.0 (100%) recall.
   - Vectorized using SIMD matrix arithmetic.
   - Ideal for $< 50,000$ vectors or baseline recall evaluation.

2. **HNSWIndex (Hierarchical Navigable Small World)**:
   - Multi-layer graph representation with probabilistic skip-list hierarchy.
   - Logarithmic search complexity $\mathcal{O}(\log N)$.
   - Key hyperparameters:
     - `M`: Maximum number of bidirectional connections per element per layer (default: 16).
     - `ef_construction`: Size of dynamic candidate list during graph indexing (default: 100).
     - `ef_search`: Size of dynamic candidate list during query phase (default: 64).
   - Dynamic neighbor selection heuristic for clustering resilience and high dimensional connectivity.

3. **IVFFlatIndex (Inverted File Index)**:
   - Partitions vector space into $N_{list}$ clusters using K-Means clustering.
   - During query, calculates distances only to $N_{probe}$ nearest cluster centroids.
   - Drastically limits vector distance computations to $\frac{N_{probe}}{N_{list}} \times N$.

4. **Scalar Quantizer (SQ8)**:
   - Compresses 32-bit floating point vectors into 8-bit unsigned integers (`uint8`).
   - Computes per-dimension scaling factors: $x_{quantized} = \text{round}\left(255 \times \frac{x - x_{min}}{x_{max} - x_{min}}\right)$.
   - Reduces RAM usage by 4x (from 4 bytes/dim to 1 byte/dim) with minimal recall trade-off.

---

### 3.2 Metadata & Boolean Filter Engine
SvectraDB features an expressive JSON filter engine inspired by MongoDB queries:
- **Comparison Operators**: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`
- **String Operators**: `$contains`, `$startswith`, `$endswith`
- **Logical Connectives**: `$and`, `$or`, `$not`
- **Execution Modes**:
  - *Pre-filtering*: Filters IDs prior to vector traversal, preventing candidate exploration of invalid records.
  - *Iterative Graph Filter*: Intercepts graph exploration in HNSW to only yield vertices matching metadata predicates while traversing navigable pathways.

---

### 3.3 Storage, WAL & Crash Durability
To ensure production reliability, SvectraDB implements a hybrid memory-disk storage architecture:
1. **Append-Only Write-Ahead Log (WAL)**:
   - Every mutating operation (`INSERT`, `UPDATE`, `DELETE`) is encoded with a binary frame:
     `[Magic: 4B | OpCode: 1B | PayloadLen: 4B | CRC32: 4B | Payload: NB]`.
   - Flushed synchronously or periodically based on `wal_sync_mode` (`always`, `periodic`, `manual`).
2. **Crash Recovery**:
   - On startup, the engine checks for the latest valid checkpoint snapshot.
   - Replays uncommitted WAL frames with CRC32 verification, guaranteeing 0 corrupted entries.
3. **Atomic Snapshots & Segments**:
   - Snapshots save index graphs (`.hnsw` / `.ivf`), vectors (`.vec.npy`), and document metadata (`.meta.jsonl`).
   - Uses atomic write-and-rename (`temp_snapshot` $\rightarrow$ `active_snapshot`) to prevent incomplete writes.

---

### 3.4 Interactive Web Dashboard
SvectraDB includes an embedded, zero-setup, dark-mode cybernetic dashboard served directly by the core engine:
- **Real-Time Telemetry**: QPS, index latency (p50/p95/p99), memory usage, vector counts.
- **Vector Playground**: Interactive query tester with JSON payload editor and distance metric toggles.
- **2D/3D Projection Visualizer**: Visualizes high-dimensional vector clusters projected into 2D space.
- **Collection Explorer**: Schema inspector, document viewer, and live vector CRUD manager.

---

### 3.5 Client Architecture (Unified SDK)
The `svectradb` Python library supports dual execution:
```python
import svectradb

# 1. Embedded In-Memory / File Mode (Zero Server Dependency)
db = svectradb.Client(persist_directory="./my_vectors")

# 2. Remote Client-Server Mode
db = svectradb.Client(url="http://localhost:8080")
```
Both modes expose the identical collection API (`create_collection`, `get_collection`, `upsert`, `query`, `delete`, `count`).
