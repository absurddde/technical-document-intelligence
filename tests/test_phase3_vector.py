from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("faiss")
from app.retrieval.vector_index import (CorruptVectorIndexError, FaissVectorIndex,
    VectorDimensionError, VectorFingerprintError)


def test_vector_add_search_save_load_and_validation(tmp_path: Path) -> None:
    index = FaissVectorIndex(3, "model-a")
    index.add_or_update([10, 20], np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
    assert index.search(np.array([1, 0, 0], dtype=np.float32), 1)[0][0] == 10
    with pytest.raises(VectorDimensionError):
        index.add_or_update([1], np.ones((1, 2), dtype=np.float32))
    path = tmp_path / "index.faiss"
    index.save(path)
    assert FaissVectorIndex.load(path, dimension=3, model_fingerprint="model-a").search(
        np.array([0, 1, 0], dtype=np.float32), 1)[0][0] == 20
    with pytest.raises(VectorFingerprintError):
        FaissVectorIndex.load(path, dimension=3, model_fingerprint="model-b")

    path.write_bytes(b"broken")
    with pytest.raises(CorruptVectorIndexError):
        FaissVectorIndex.load(path, dimension=3, model_fingerprint="model-a")
