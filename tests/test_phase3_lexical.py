from pathlib import Path

from app.domain.models import DocumentStatus
from app.persistence.repositories import ContentRepository, DocumentRepository
from app.retrieval.lexical_index import SQLiteLexicalIndex
from tests.test_repository import make_fingerprint
from app.domain.models import DocumentChunk


def _chunk(chunk_id: str, text: str, section: str = "Güdüm") -> DocumentChunk:
    return DocumentChunk("doc", "teknik.pdf", "teknik.pdf", 1, 1, section, 1, 1,
                         chunk_id, "mixed", text, False, None)


def test_fts_unicode_acronym_phrase_bm25_and_incremental_update(connection) -> None:
    repo = DocumentRepository(connection)
    doc = repo.save_fingerprint(make_fingerprint("teknik.pdf", "a" * 64))
    repo.ensure_index_state(doc.id, None, reset_pipeline=True)
    content = ContentRepository(connection)
    content.replace(doc, (), (
        _chunk("one", "AESA radarı terminal güdüm sağlar. AESA AESA."),
        _chunk("two", "AESA sistemi ve farklı bir terminal mimarisi."),
    ), "phase3")
    lexical = SQLiteLexicalIndex(connection)

    assert lexical.tokenizer == "unicode61 remove_diacritics 0"
    assert lexical.search("güdüm")[0].text.startswith("AESA")
    assert lexical.search("AESA")[0].chunk_id == "one"
    assert [hit.chunk_id for hit in lexical.search('"terminal güdüm"')] == ["one"]
    assert lexical.search("AESA")[0].score <= lexical.search("AESA")[1].score

    content.replace(doc, (), (_chunk("three", "GNSS ile seyrüsefer."),), "phase3")
    assert not lexical.search("AESA")
    assert lexical.search("GNSS")[0].chunk_id == "three"
    lexical.rebuild()
    lexical.integrity_check()


def test_missing_document_is_filtered_from_lexical_results(connection) -> None:
    repo = DocumentRepository(connection)
    doc = repo.save_fingerprint(make_fingerprint("gone.pdf", "b" * 64))
    repo.ensure_index_state(doc.id, None, reset_pipeline=True)
    ContentRepository(connection).replace(doc, (), (_chunk("gone", "radar"),), "phase3")
    repo.mark_missing(doc.id)
    assert SQLiteLexicalIndex(connection).search("radar") == ()
