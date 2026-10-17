"""
SvectraDB High-Performance Benchmark Suite.
Compares Flat (Exact Ground Truth) vs HNSW vs IVF-Flat on:
- Build / Indexing Speed (vectors/sec)
- Query Latency (p50, p95, p99 ms)
- Query Throughput (QPS)
- Recall@10 against Ground Truth
"""

import os
import sys
import time
from typing import Dict, List, Tuple
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from svectra.core.models import MetricType
from svectra.engine.indices.flat import FlatIndex
from svectra.engine.indices.hnsw import HNSWIndex
from svectra.engine.indices.ivf import IVFFlatIndex


def generate_benchmark_data(n_samples: int = 5000, dimension: int = 64, seed: int = 42):
    """Generate normalized random float32 vectors for cosine similarity testing."""
    rng = np.random.RandomState(seed)
    vectors = rng.randn(n_samples, dimension).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= norms
    ids = [f"vec_{i}" for i in range(n_samples)]
    return ids, vectors


def benchmark_index(
    name: str,
    index,
    ids: List[str],
    vectors: np.ndarray,
    queries: np.ndarray,
    ground_truth: List[List[str]],
    top_k: int = 10,
) -> Dict[str, float]:
    """Execute rigorous indexing and query benchmarks for a single index implementation."""
    # 1. Measure Build / Indexing Time
    t0 = time.perf_counter()
    index.add(ids, vectors)
    build_time = time.perf_counter() - t0
    index_speed = len(ids) / build_time if build_time > 0 else 0.0

    # 2. Measure Query Latency
    latencies = []
    hits = 0
    total_candidates = len(queries) * top_k

    for i, q in enumerate(queries):
        start_q = time.perf_counter()
        results = index.search(q, k=top_k)
        latencies.append((time.perf_counter() - start_q) * 1000.0)

        # Calculate Recall against ground truth
        result_ids = set(r[0] for r in results)
        gt_ids = set(ground_truth[i])
        hits += len(result_ids.intersection(gt_ids))

    latencies.sort()
    n_q = len(latencies)
    p50 = latencies[int(n_q * 0.50)]
    p95 = latencies[int(n_q * 0.95)]
    p99 = latencies[int(n_q * 0.99)]
    avg_latency = float(np.mean(latencies))
    qps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    recall_at_k = hits / total_candidates if total_candidates > 0 else 1.0

    return {
        "name": name,
        "build_time_s": round(build_time, 3),
        "index_speed_vps": round(index_speed, 1),
        "p50_ms": round(p50, 3),
        "p95_ms": round(p95, 3),
        "p99_ms": round(p99, 3),
        "qps": round(qps, 1),
        "recall_at_10": round(recall_at_k, 4),
    }


def run_benchmarks(n_vectors: int = 3000, n_queries: int = 100, dim: int = 64):
    print("=" * 78)
    print(f"  ⚡ SVECTRADB BENCHMARK SUITE: {n_vectors} Vectors | {dim} Dim | {n_queries} Queries")
    print("=" * 78)

    ids, vectors = generate_benchmark_data(n_samples=n_vectors, dimension=dim)
    _, query_vectors = generate_benchmark_data(n_samples=n_queries, dimension=dim, seed=999)

    # 1. Establish Ground Truth using Flat Index
    print("\n[1/3] Computing Ground Truth with Flat (Exact Brute-Force)...")
    flat = FlatIndex(dimension=dim, metric=MetricType.COSINE)
    flat.add(ids, vectors)
    ground_truth = []
    for q in query_vectors:
        res = flat.search(q, k=10)
        ground_truth.append([r[0] for r in res])

    flat_stats = benchmark_index(
        "Flat (Exact)",
        FlatIndex(dimension=dim, metric=MetricType.COSINE),
        ids,
        vectors,
        query_vectors,
        ground_truth,
    )

    # 2. Benchmark HNSW
    print("[2/3] Benchmarking HNSW Index (M=8, ef_c=32, ef_s=16)...")
    hnsw = HNSWIndex(dimension=dim, metric=MetricType.COSINE, M=8, ef_construction=32, ef_search=16)
    hnsw_stats = benchmark_index("HNSW Graph", hnsw, ids, vectors, query_vectors, ground_truth)

    # 3. Benchmark IVF-Flat
    print("[3/3] Benchmarking IVF-Flat Index (nlist=16, nprobe=4)...")
    ivf = IVFFlatIndex(dimension=dim, metric=MetricType.COSINE, nlist=16, nprobe=4)
    ivf_stats = benchmark_index("IVF-Flat", ivf, ids, vectors, query_vectors, ground_truth)

    # Display Comparative Table
    print("\n" + "=" * 78)
    print(f"{'Index Engine':<16} | {'Build Time':<10} | {'Indexing VPS':<12} | {'p50 (ms)':<9} | {'p99 (ms)':<9} | {'QPS':<8} | {'Recall@10':<9}")
    print("-" * 78)
    for s in [flat_stats, hnsw_stats, ivf_stats]:
        print(
            f"{s['name']:<16} | "
            f"{s['build_time_s']:>8}s | "
            f"{s['index_speed_vps']:>10.0f} | "
            f"{s['p50_ms']:>7.3f}ms | "
            f"{s['p99_ms']:>7.3f}ms | "
            f"{s['qps']:>8.0f} | "
            f"{s['recall_at_10'] * 100:>7.1f}%"
        )
    print("=" * 78)


if __name__ == "__main__":
    run_benchmarks(n_vectors=800, n_queries=30, dim=16)
