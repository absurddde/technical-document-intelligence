from pathlib import Path

import numpy as np

from app.domain.models import DocumentChunk
from app.persistence.repositories import ContentRepository, DocumentRepository
from app.retrieval.retrievers import FaissSemanticRetriever, SQLiteLexicalRetriever
from app.retrieval.vector_index import FaissVectorIndex
from tests.test_repository import make_fingerprint


def chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk("doc", "source.pdf", "source.pdf", 4, 4, "Navigation",
                         1, 1, chunk_id, "en", text, False, None)


def test_lexical_candidate_contains_provenance(connection) -> None:
    documents = DocumentRepository(connection)
    document = documents.save_fingerprint(make_fingerprint("source.pdf", "d" * 64))
    documents.ensure_index_state(document.id, None, reset_pipeline=True)
    ContentRepository(connection).replace(
        document, (), (chunk("lex", "GNSS navigation system"),), "phase4"
    )
    result = SQLiteLexicalRetriever(connection).search('"GNSS"', 30)[0]
    assert result.chunk_id == "lex"
    assert result.document_id == document.document_id
    assert result.page_start == 4 and result.section_title == "Navigation"
    assert result.score <= 0


class QueryBackend:
    dimension = 3
    model_fingerprint = "semantic-test"
    def embed_query(self, text): return np.array([1, 0, 0], dtype=np.float32)
    def embed_documents(self, texts): raise AssertionError


def test_semantic_candidate_maps_faiss_id_to_chunk(connection, tmp_path: Path) -> None:
    documents = DocumentRepository(connection)
    document = documents.save_fingerprint(make_fingerprint("source.pdf", "e" * 64))
    documents.ensure_index_state(document.id, None, reset_pipeline=True)
    ContentRepository(connection).replace(
        document, (), (chunk("semantic", "inertial navigation"),), "phase4"
    )
    connection.execute(
        """INSERT INTO vector_index_metadata(chunk_id,vector_id,content_hash,embedding,
           model_fingerprint,vector_dimension,index_artifact) VALUES(?,?,?,?,?,?,?)""",
        ("semantic", 7, "f" * 64, np.array([1, 0, 0], dtype=np.float32).tobytes(),
         "semantic-test", 3, "test.faiss"),
    )
    path = tmp_path / "test.faiss"
    index = FaissVectorIndex(3, "semantic-test")
    index.add_or_update([7], np.array([[1, 0, 0]], dtype=np.float32))
    index.save(path)

    result = FaissSemanticRetriever(connection, QueryBackend(), path).search("query", 30)[0]
    assert result.chunk_id == "semantic"
    assert result.score == 1.0
    assert result.model_fingerprint == "semantic-test"
