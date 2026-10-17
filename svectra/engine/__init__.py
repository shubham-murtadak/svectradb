"""
Vector mathematics, distance calculations, indexing, and filtering.
"""

from svectra.engine.distance import (
    compute_distance,
    compute_batch_distances,
    distance_to_score,
    normalize_vector,
    normalize_matrix,
)
from svectra.engine.filter import match_filter, compile_filter
from svectra.engine.quantization import ScalarQuantizer
from svectra.engine.indices import BaseIndex, FlatIndex, HNSWIndex, IVFFlatIndex

__all__ = [
    "compute_distance",
    "compute_batch_distances",
    "distance_to_score",
    "normalize_vector",
    "normalize_matrix",
    "match_filter",
    "compile_filter",
    "ScalarQuantizer",
    "BaseIndex",
    "FlatIndex",
    "HNSWIndex",
    "IVFFlatIndex",
]
