# svectradb

**SvectraDB** is a lightweight, ultra-fast, production-grade vector database built from scratch for AI founders and modern machine learning workloads.

Featuring:
- **Multiple Index Engines**: Flat (Exact Brute-Force), HNSW (Hierarchical Navigable Small World), and IVF-Flat (Inverted File Index with K-Means clustering).
- **SIMD-Accelerated Vector Math**: Cosine Similarity, Euclidean ($L_2$), Dot Product, and Manhattan ($L_1$).
- **Rich Metadata Filtering**: Dynamic MongoDB-style filter expressions (`$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$and`, `$or`).
- **Durability & Persistence**: Write-Ahead Log (WAL) with CRC32 checksums, atomic snapshots, and segment-based storage.
- **Dual Operating Modes**: Embedded in-memory / disk mode (`import svectradb`) and Client-Server REST API mode (`svectra-server`).
- **Futuristic Interactive Web UI**: Real-time vector collection explorer, similarity search tester, 2D projection visualization, and telemetry.
- **Cloud-Ready Deployment**: Multi-stage Dockerfile, Docker Compose, and Kubernetes Helm-ready manifests.
