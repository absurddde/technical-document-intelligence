"""Fault-isolated Phase 2 parsing and chunk persistence orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

from app.domain.models import IndexAction, IndexPlanItem
from app.ingestion.errors import DocumentParseError
from app.persistence.connection import Database
from app.persistence.repositories import ContentRepository, DocumentRepository
from app.persistence.schema import initialize_schema
from app.processing.cache import ParsedDocumentCache
from app.processing.chunker import StructureAwareChunker


@dataclass(frozen=True, slots=True)
class ProcessingResult:
    processed: int
    skipped: int
    failed: int


class DocumentProcessingService:
    """Process changed canonical documents while isolating per-file failures."""

    def __init__(self, database: Database, pdf_parser, docx_parser,
                 chunker: StructureAwareChunker, cache: ParsedDocumentCache,
                 pipeline_version: str = "phase2-v1") -> None:
        self._database = database
        self._parsers = {"pdf": pdf_parser, "docx": docx_parser}
        self._chunker = chunker
        self._cache = cache
        self._version = pipeline_version
        self._logger = logging.getLogger("document_intelligence.processing")

    def process(self, plan: tuple[IndexPlanItem, ...]) -> ProcessingResult:
        """Skip unchanged/duplicate paths and continue after malformed documents."""

        processed = skipped = failed = 0
        for item in plan:
            if item.action not in {IndexAction.NEW, IndexAction.MODIFIED} or item.fingerprint is None:
                skipped += 1
                continue
            fingerprint = item.fingerprint
            try:
                parsed = self._cache.load(fingerprint.sha256, self._version)
                if parsed is None:
                    parsed = self._parsers[fingerprint.file_type].parse(fingerprint.path, fingerprint.sha256)
                    self._cache.save(parsed)
                with self._database.transaction() as connection:
                    initialize_schema(connection)
                    document = DocumentRepository(connection).get_by_path(fingerprint.canonical_path)
                    if document is None:
                        raise RuntimeError("Inventory record is missing")
                    chunks = self._chunker.chunk(parsed.blocks, document_id=document.document_id,
                                                 file_name=document.file_name, file_path=document.canonical_path)
                    ContentRepository(connection).replace(document, parsed.blocks, chunks, self._version)
                processed += 1
            except (DocumentParseError, OSError, ValueError, KeyError, RuntimeError) as error:
                failed += 1
                code = type(error).__name__
                self._logger.warning("Document processing failed path=%s error_type=%s", fingerprint.canonical_path, code)
                with self._database.transaction() as connection:
                    initialize_schema(connection)
                    document = DocumentRepository(connection).get_by_path(fingerprint.canonical_path)
                    if document is not None:
                        ContentRepository(connection).mark_failed(document.id, code, self._version)
        return ProcessingResult(processed, skipped, failed)
