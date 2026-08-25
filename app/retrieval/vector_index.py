"""Exact local FAISS vector index behind a small abstraction."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class VectorIndexError(RuntimeError): pass
class VectorDimensionError(VectorIndexError): pass
class VectorFingerprintError(VectorIndexError): pass
class CorruptVectorIndexError(VectorIndexError): pass


class VectorIndex(Protocol):
    def add_or_update(self, ids: Sequence[int], vectors: NDArray[np.float32]) -> None: ...
    def delete(self, ids: Sequence[int]) -> None: ...
    def search(self, vector: NDArray[np.float32], limit: int) -> tuple[tuple[int, float], ...]: ...
    def save(self, path: Path) -> None: ...


class FaissVectorIndex:
    """Cosine search using normalized vectors and exact IndexIDMap2/FlatIP."""

    def __init__(self, dimension: int, model_fingerprint: str) -> None:
        if dimension <= 0:
            raise VectorDimensionError("Vector dimension must be positive")
        try:
            import faiss
        except ImportError as error:
            raise VectorIndexError("faiss-cpu is not installed") from error
        self._faiss = faiss
        self.dimension = dimension
        self.model_fingerprint = model_fingerprint
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))

    def _matrix(self, vectors: NDArray[np.float32]) -> NDArray[np.float32]:
        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.dimension:
            raise VectorDimensionError(f"Expected vectors with dimension {self.dimension}")
        return matrix

    def add_or_update(self, ids: Sequence[int], vectors: NDArray[np.float32]) -> None:
        matrix = self._matrix(vectors)
        int_ids = np.asarray(ids, dtype=np.int64)
        if len(int_ids) != len(matrix):
            raise ValueError("Vector and ID counts differ")
        self._index.remove_ids(int_ids)
        if len(int_ids):
            self._index.add_with_ids(matrix, int_ids)

    def delete(self, ids: Sequence[int]) -> None:
        self._index.remove_ids(np.asarray(ids, dtype=np.int64))

    def search(self, vector: NDArray[np.float32], limit: int) -> tuple[tuple[int, float], ...]:
        query = np.asarray(vector, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != self.dimension:
            raise VectorDimensionError(f"Expected query dimension {self.dimension}")
        scores, ids = self._index.search(np.ascontiguousarray(query[None, :]), limit)
        return tuple((int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(path))
        path.with_suffix(path.suffix + ".meta").write_text(
            f"1\n{self.dimension}\n{self.model_fingerprint}\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path, *, dimension: int, model_fingerprint: str) -> "FaissVectorIndex":
        meta = path.with_suffix(path.suffix + ".meta")
        try:
            version, saved_dim, saved_fp = meta.read_text(encoding="utf-8").splitlines()
            if version != "1": raise CorruptVectorIndexError("Unsupported vector index version")
            if int(saved_dim) != dimension: raise VectorDimensionError("Stored vector dimension mismatch")
            if saved_fp != model_fingerprint: raise VectorFingerprintError("Embedding model fingerprint mismatch")
            instance = cls(dimension, model_fingerprint)
            instance._index = instance._faiss.read_index(str(path))
            if instance._index.d != dimension: raise VectorDimensionError("FAISS dimension mismatch")
            return instance
        except (VectorIndexError, ValueError):
            raise
        except Exception as error:
            raise CorruptVectorIndexError(f"Could not load vector index: {error}") from error
