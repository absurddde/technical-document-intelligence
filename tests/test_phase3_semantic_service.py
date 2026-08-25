from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("faiss")

from app.domain.models import DocumentChunk
from app.persistence.connection import Database
from app.persistence.repositories import ContentRepository, DocumentRepository
from app.persistence.schema import initialize_schema
from app.services.semantic_index_service import SemanticIndexService
from tests.test_repository import make_fingerprint


class FakeBackend:
    dimension = 3
    model_fingerprint = "fake-v1"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        values = []
        for text in texts:
            base = [float(len(text)), float(text.count("a")), 1.0]
            values.append((base + [0.5] * self.dimension)[:self.dimension])
        matrix = np.asarray(values, dtype=np.float32)
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def _chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk("doc", "a.pdf", "a.pdf", 1, 1, None, 1, 1,
                         chunk_id, "en", text, False, None)


def test_incremental_embedding_skip_modify_reuse_and_remove(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    with database.transaction() as connection:
        initialize_schema(connection)
        repo = DocumentRepository(connection)
        first = repo.save_fingerprint(make_fingerprint("a.pdf", "a" * 64))
        second = repo.save_fingerprint(make_fingerprint("b.pdf", "b" * 64))
        for doc in (first, second): repo.ensure_index_state(doc.id, None, reset_pipeline=True)
        content = ContentRepository(connection)
        content.replace(first, (), (_chunk("a1", "same passage"),), "phase3")
        content.replace(second, (), (_chunk("b1", "same passage"),), "phase3")

    backend = FakeBackend()
    service = SemanticIndexService(database, backend, tmp_path / "data/indexes/semantic.faiss")
    result = service.synchronize()
    assert result.embedded == 1 and result.reused == 1
    assert len(backend.calls) == 1
    service.synchronize()
    assert len(backend.calls) == 1

    with database.transaction() as connection:
        first = DocumentRepository(connection).get_by_path("a.pdf")
        assert first is not None
        ContentRepository(connection).replace(first, (), (_chunk("a2", "changed passage"),), "phase3")
    assert service.synchronize().embedded == 1

    with database.transaction() as connection:
        second = DocumentRepository(connection).get_by_path("b.pdf")
        assert second is not None
        DocumentRepository(connection).mark_missing(second.id)
    result = service.synchronize()
    assert result.removed >= 1
    with database.transaction() as connection:
        ids = {row[0] for row in connection.execute("SELECT chunk_id FROM vector_index_metadata")}
    assert ids == {"a2"}


def test_dimension_and_fingerprint_change_reembeds_and_rebuilds(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    with database.transaction() as connection:
        initialize_schema(connection)
        repo = DocumentRepository(connection)
        document = repo.save_fingerprint(make_fingerprint("a.pdf", "c" * 64))
        repo.ensure_index_state(document.id, None, reset_pipeline=True)
        ContentRepository(connection).replace(
            document, (), (_chunk("a1", "dimension test"),), "phase3"
        )

    backend = FakeBackend()
    artifact = tmp_path / "data/indexes/semantic.faiss"
    SemanticIndexService(database, backend, artifact).synchronize()
    backend.dimension = 4
    backend.model_fingerprint = "fake-v2"

    result = SemanticIndexService(database, backend, artifact).synchronize()

    assert result.embedded == 1
    with database.transaction() as connection:
        metadata = connection.execute(
            "SELECT model_fingerprint, vector_dimension FROM vector_index_metadata"
        ).fetchone()
    assert tuple(metadata) == ("fake-v2", 4)
    from app.retrieval.vector_index import FaissVectorIndex
    loaded = FaissVectorIndex.load(
        artifact, dimension=4, model_fingerprint="fake-v2"
    )
    assert loaded.dimension == 4
