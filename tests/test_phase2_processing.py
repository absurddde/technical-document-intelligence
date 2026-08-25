from __future__ import annotations

from pathlib import Path

from app.domain.models import DocumentStatus, FileFingerprint, IndexAction, IndexPlanItem, ParsedBlock, ParsedDocument
from app.ingestion.errors import DocumentParseError
from app.persistence.connection import Database
from app.persistence.repositories import DocumentRepository
from app.persistence.schema import initialize_schema
from app.processing.cache import ParsedDocumentCache
from app.processing.chunker import StructureAwareChunker
from app.processing.cleaner import clean_text
from app.services.document_processing_service import DocumentProcessingService


def test_cleaning_preserves_turkish_and_acronyms() -> None:
    assert clean_text("  Güdüm   sistemi\t INS / EO/IR  ") == "Güdüm sistemi INS / EO/IR"


def test_chunk_metadata_and_structural_boundaries() -> None:
    blocks = (
        ParsedBlock("heading", "Navigation", page_number=1, section_title="Navigation", paragraph_index=0),
        ParsedBlock("paragraph", "The INS system provides navigation.", page_number=1, section_title="Navigation", paragraph_index=1),
        ParsedBlock("paragraph", "Güdüm sistemi ve GPS kullanılır.", page_number=2, section_title="Güdüm", paragraph_index=2, ocr_used=True, ocr_confidence=87.0),
    )
    chunks = StructureAwareChunker(200, 20).chunk(blocks, document_id="doc-1", file_name="a.pdf", file_path="C:/a.pdf")
    assert len(chunks) == 2
    assert chunks[0].page_start == chunks[0].page_end == 1
    assert chunks[1].page_start == chunks[1].page_end == 2
    assert chunks[1].ocr_used and chunks[1].ocr_confidence == 87.0
    assert chunks[1].language == "tr"
    assert chunks[1].document_id == "doc-1" and chunks[1].chunk_id


def test_cache_is_hash_and_pipeline_version_bound(tmp_path: Path) -> None:
    cache = ParsedDocumentCache(tmp_path)
    parsed = ParsedDocument((ParsedBlock("paragraph", "Gizli olmayan test"),), "a" * 64, "v1")
    cache.save(parsed)
    assert cache.load("a" * 64, "v1") == parsed
    assert cache.load("a" * 64, "v2") is None
    assert cache.load("b" * 64, "v1") is None


class _Parser:
    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def parse(self, path: Path, document_hash: str) -> ParsedDocument:
        self.calls += 1
        if self.fail:
            raise DocumentParseError("safe failure")
        return ParsedDocument((ParsedBlock("paragraph", "The technical system is ready.", page_number=1),), document_hash, "phase2-v1")


def _fingerprint(path: Path, digest: str) -> FileFingerprint:
    return FileFingerprint(path, str(path.resolve()).casefold(), path.name, path.suffix[1:], 1, 1, digest)


def test_broken_document_isolation_unchanged_skip_and_persistence(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    good_path, bad_path = tmp_path / "good.pdf", tmp_path / "bad.docx"
    good_path.write_bytes(b"x")
    bad_path.write_bytes(b"x")
    good, bad = _fingerprint(good_path, "a" * 64), _fingerprint(bad_path, "b" * 64)
    with database.transaction() as connection:
        initialize_schema(connection)
        repo = DocumentRepository(connection)
        for fingerprint in (good, bad):
            saved = repo.save_fingerprint(fingerprint)
            repo.ensure_index_state(saved.id, None, reset_pipeline=True)
    good_parser, bad_parser = _Parser(), _Parser(fail=True)
    service = DocumentProcessingService(database, good_parser, bad_parser, StructureAwareChunker(), ParsedDocumentCache(tmp_path / "cache"))
    plan = (
        IndexPlanItem(IndexAction.NEW, good.canonical_path, good),
        IndexPlanItem(IndexAction.NEW, bad.canonical_path, bad),
        IndexPlanItem(IndexAction.UNCHANGED, good.canonical_path, good),
    )
    result = service.process(plan)
    assert (result.processed, result.failed, result.skipped) == (1, 1, 1)
    assert good_parser.calls == 1
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1
        statuses = {row["file_name"]: row["status"] for row in connection.execute("SELECT file_name,status FROM documents")}
        assert statuses == {"good.pdf": DocumentStatus.INDEXED.value, "bad.docx": DocumentStatus.FAILED.value}
