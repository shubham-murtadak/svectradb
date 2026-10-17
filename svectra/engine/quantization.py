"""
Scalar Quantization (SQ8) engine for SvectraDB.
Compresses 32-bit floating point vectors into 8-bit unsigned integers,
reducing memory footprint by 75% with minimal recall impact.
"""

from typing import Dict, Any, Tuple
import numpy as np


class ScalarQuantizer:
    """
    Affine 8-bit scalar quantizer mapping floats in [min_val, max_val] to [0, 255].
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.min_vals: np.ndarray = np.zeros(dimension, dtype=np.float32)
        self.max_vals: np.ndarray = np.ones(dimension, dtype=np.float32)
        self.scale: np.ndarray = np.ones(dimension, dtype=np.float32)
        self.is_fitted: bool = False

    def fit(self, vectors: np.ndarray) -> "ScalarQuantizer":
        """
        Compute per-dimension min and max bounds across the training vector matrix.
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[0] == 0:
            return self

        self.min_vals = np.min(vectors, axis=0).astype(np.float32)
        self.max_vals = np.max(vectors, axis=0).astype(np.float32)

        # Avoid division by zero
        diff = self.max_vals - self.min_vals
        diff[diff < 1e-7] = 1.0
        self.scale = 255.0 / diff
        self.is_fitted = True
        return self

    def quantize(self, vectors: np.ndarray) -> np.ndarray:
        """
        Convert float32 vectors to uint8.
        """
        if not self.is_fitted:
            self.fit(vectors)

        single = False
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
            single = True

        clipped = np.clip(vectors, self.min_vals, self.max_vals)
        q = np.round((clipped - self.min_vals) * self.scale).astype(np.uint8)

        if single:
            return q[0]
        return q

    def dequantize(self, q_vectors: np.ndarray) -> np.ndarray:
        """
        Convert uint8 vectors back to approximate float32 representation.
        """
        single = False
        if q_vectors.ndim == 1:
            q_vectors = q_vectors.reshape(1, -1)
            single = True

        approx = (q_vectors.astype(np.float32) / self.scale) + self.min_vals
        if single:
            return approx[0]
        return approx

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "min_vals": self.min_vals.tolist(),
            "max_vals": self.max_vals.tolist(),
            "scale": self.scale.tolist(),
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScalarQuantizer":
        sq = cls(dimension=data["dimension"])
        sq.min_vals = np.array(data["min_vals"], dtype=np.float32)
        sq.max_vals = np.array(data["max_vals"], dtype=np.float32)
        sq.scale = np.array(data["scale"], dtype=np.float32)
        sq.is_fitted = data["is_fitted"]
        return sq
