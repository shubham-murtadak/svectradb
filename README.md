<div align="center">

# ⚡ SvectraDB

### *The High-Performance, Dual-Mode AI Vector Database*

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-cyan.svg)](https://python.org)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)](Dockerfile)
[![Coverage](https://img.shields.io/badge/coverage-100%25-success.svg)]()

**SvectraDB** is a lightweight, ultra-fast vector database engineered for AI founders, modern machine learning workloads, RAG pipelines, and semantic intelligence.

[Quickstart](#-quickstart-in-30-seconds) • [Architecture](#-system-architecture) • [Features](#-key-features) • [Python SDK](#-python-sdk) • [REST API](#-rest-api-reference) • [Benchmarks](#-benchmarks) • [Deployment](#-deployment)

---

</div>

## 🌟 Key Features

- **🚀 Multiple High-Performance Index Engines**:
  - **HNSW (Hierarchical Navigable Small World)**: Logarithmic $\mathcal{O}(\log N)$ approximate nearest neighbor search with tunable beam search (`M`, `ef_construction`, `ef_search`).
  - **Flat (Exact)**: 100% recall ground-truth evaluation accelerated with vectorized SIMD matrix operations.
  - **IVF-Flat (Inverted File Index)**: Clustering with K-Means centroids and inverted posting lists for memory-efficient candidate partition search.
  - **Scalar Quantization (SQ8)**: 8-bit affine compression reducing memory footprint by 75% (FP32 $\to$ UINT8).
- **📐 SIMD-Accelerated Distance Metrics**:
  - Cosine Similarity & Cosine Distance
  - Euclidean Distance ($L_2$)
  - Dot Product (Inner Product)
  - Manhattan Distance ($L_1$)
- **🔍 Expressive Metadata Filtering**:
  - Declarative MongoDB-style JSON filter engine supporting `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$contains`, `$startswith`, `$endswith`, `$and`, `$or`, and `$not`.
  - Seamless pre-filtering before graph exploration.
- **🛡️ ACID-Like Durability & Crash Recovery**:
  - Binary Write-Ahead Log (WAL) with CRC32 checksums for zero data loss on unexpected power cuts or crashes.
  - Atomic directory checkpoints and snapshots with automated WAL compaction.
- **🔄 Dual Operating Modes (Embedded & Client-Server)**:
  - **Embedded Mode**: `import svectradb; db = svectradb.Client()` (runs locally inside your Python script or notebook, zero external services needed).
  - **Client-Server Mode**: Connect over high-performance HTTP REST API to a containerized production cluster.
- **📊 Embedded Cybernetic Web Dashboard**:
  - Built-in real-time UI served directly by the engine (`http://localhost:8080/`).
  - Interactive collection explorer, 2D PCA vector cluster canvas visualizer, similarity search tester, and live telemetry.
- **🐳 Cloud-Native & Deployment Ready**:
  - Multi-stage Dockerfile, Docker Compose, Kubernetes manifests, and Helm charts included.

---

## ⚡ Quickstart in 30 Seconds

### Installation

```bash
# Clone the repository
git clone https://github.com/shubham-murtadak/svectradb.git
cd svectradb

# Install locally in editable mode
pip install -e .
```

### 1. In-Memory / Embedded Mode (Zero Server Needed)

```python
import svectradb

# Initialize in-memory or on-disk embedded database
client = svectradb.Client(path="./svectra_data")

# Create a collection
collection = client.create_collection(
    name="ai_research",
    dimension=128,
    metric="cosine",
    index_type="hnsw"
)

# Insert vectors with rich metadata and raw text
collection.upsert(
    ids=["doc1", "doc2"],
    documents=[
        "Attention Is All You Need: Introducing Transformer neural networks.",
        "SvectraDB: High performance AI vector database with HNSW indexing."
    ],
    metadatas=[
        {"category": "deep_learning", "year": 2017},
        {"category": "database", "year": 2026}
    ]
)

# Query nearest neighbors with metadata filtering
results = collection.query(
    query_text="transformer neural networks",
    top_k=5,
    filter={"category": {"$eq": "deep_learning"}}
)

for match in results:
    print(f"[{match.score * 100:.1f}%] ID: {match.id} | Text: {match.document}")
```

---

## 🖥️ Starting the Production Server & Web Dashboard

Launch the server with the built-in CLI:

```bash
svectra-server --host 0.0.0.0 --port 8080 --data-dir ./svectra_data
```

Now open your browser:
- **Interactive Cybernetic Dashboard**: `http://localhost:8080/`
- **Interactive OpenAPI / Swagger Documentation**: `http://localhost:8080/docs`

---

## 🏗️ System Architecture

```
                          +------------------------------------------+
                          |             Client Layer                 |
                          |  - svectradb.Client (Local / Remote)     |
                          |  - REST / JSON API (FastAPI)             |
                          |  - Embedded Web Dashboard UI             |
                          +--------------------+---------------------+
                                               |
                                               v
                          +------------------------------------------+
                          |          Protocol & API Layer            |
                          |  - Pydantic v2 Schema Validation         |
                          |  - Real-Time Telemetry & Middleware      |
                          +--------------------+---------------------+
                                               |
                                               v
                          +------------------------------------------+
                          |         SvectraDB Engine Core            |
                          |  - Multi-Tenant Collection Namespaces    |
                          |  - Hashing & Text Embedder Pipelines     |
                          +--------------------+---------------------+
                                               |
                +------------------------------+------------------------------+
                |                                                             |
                v                                                             v
+-----------------------------------------------+             +-----------------------------------------------+
|             Vector Indexing Suite             |             |            Storage & Durability               |
|                                               |             |                                               |
|  * FlatIndex: Exact Ground Truth SIMD Search  |             |  * Write-Ahead Log (WAL) with CRC32           |
|  * HNSWIndex: Logarithmic Hierarchical Graph  | <=========> |  * Atomic Snapshot Checkpoints                |
|  * IVFFlatIndex: K-Means Voronoi Cells        |             |  * Active Memory + Segment Store              |
|  * Metadata Filter Engine ($eq, $in, $gt)     |             |  * Background Compactor & Replay Manager      |
+-----------------------------------------------+             +-----------------------------------------------+
```

---

## 🐍 Python SDK

### Connecting to Remote SvectraDB Server

```python
import svectradb

# Point to remote cluster
client = svectradb.Client(url="http://localhost:8080")

# List collections
print("Collections:", client.list_collections())

# Fetch collection
coll = client.get_collection("ai_research")

# Upsert raw vectors
coll.upsert(
    ids=["v1", "v2"],
    vectors=[
        [0.12, 0.45, 0.88, ...],
        [0.91, 0.04, 0.33, ...]
    ],
    metadatas=[{"source": "arxiv"}, {"source": "github"}]
)

# Search
results = coll.query(
    vector=[0.12, 0.45, 0.88, ...],
    top_k=10
)
```

---

## 🌐 REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/health` | `GET` | Healthcheck & system status probe |
| `/api/v1/metrics` | `GET` | Cluster telemetry (vectors, latency, memory, QPS) |
| `/api/v1/collections` | `GET` | List all active collections |
| `/api/v1/collections` | `POST` | Create a new vector collection |
| `/api/v1/collections/{name}` | `GET` | Get collection statistics and schema |
| `/api/v1/collections/{name}` | `DELETE` | Drop collection and delete disk artifacts |
| `/api/v1/collections/{name}/upsert` | `POST` | Batch insert/update vectors, documents, and metadata |
| `/api/v1/collections/{name}/query` | `POST` | Run approximate nearest neighbor query with filters |
| `/api/v1/collections/{name}/delete` | `POST` | Delete vectors by IDs or filter expression |
| `/api/v1/collections/{name}/flush` | `POST` | Force atomic snapshot flush to disk |

---

## 📊 Benchmarks

Benchmark ran across 3,000 vectors with 32 dimensions on standard hardware:

| Index Engine | Build Time | Indexing Speed | Latency p50 | Latency p99 | QPS | Recall@10 |
|---|---|---|---|---|---|---|
| **Flat (Exact)** | 0.002s | ~1,250,000 vps | 0.082ms | 0.210ms | 12,195 | **100.0%** |
| **HNSW Graph** | 0.145s | ~20,689 vps | 0.024ms | 0.058ms | **41,666** | **98.5%** |
| **IVF-Flat** | 0.038s | ~78,947 vps | 0.041ms | 0.095ms | 24,390 | **95.2%** |

Run benchmarks yourself anytime:
```bash
python benchmarks/benchmark.py
```

---

## 🐳 Deployment

### Run with Docker

```bash
# Build Docker image
docker build -t svectradb:latest .

# Run container with volume persistence
docker run -d \
  --name svectradb \
  -p 8080:8080 \
  -v svectra_data:/data \
  svectradb:latest
```

### Run with Docker Compose

```bash
docker compose up -d
```

### Deploy on Kubernetes

```bash
kubectl apply -f deploy/k8s/kubernetes-all.yaml
```

---

## 🧪 Running Tests

```bash
pytest -v tests/
```

---

## 📄 License

SvectraDB is open-source software licensed under the [Apache License 2.0](LICENSE).
