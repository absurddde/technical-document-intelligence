from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.generation.models import (
    ClaimSourceMapping, GenerationResult, SourceRecord,
)
from app.infrastructure.config import (
    AppConfig, ChunkingConfig, EmbeddingConfig, GenerationConfig, IndexingConfig,
    LlmConfig, LoggingConfig, OcrConfig, ParsingConfig, PathConfig,
    RetrievalConfig, VectorIndexConfig,
)
from app.retrieval.models import FusedCandidate, RetrievalDetails, SearchResult
from app.ingestion.scanner import FolderScanner
from app.persistence.connection import Database
from app.persistence.repositories import DocumentRepository
from app.services.inventory_service import InventoryService
import app.ui.backend as ui_backend
from app.ui.backend import LocalBackendFacade, MissingLocalResourceError
from app.ui.models import document_from_row, map_answer
from app.ui.workers import FunctionWorker


def config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        PathConfig(tmp_path / "app.db", tmp_path / "indexes", tmp_path / "cache", tmp_path / "logs", tmp_path / "models"),
        IndexingConfig(), LoggingConfig(), ParsingConfig(), OcrConfig(enabled=False),
        ChunkingConfig(), EmbeddingConfig(tmp_path / "missing-embedding"),
        VectorIndexConfig(), RetrievalConfig(), GenerationConfig(),
        LlmConfig(tmp_path / "missing.gguf"),
    )


def candidate() -> FusedCandidate:
    return FusedCandidate(
        "chunk-1", "doc-1", "guide.pdf", "Ataletsel sistem konumu kestirir.",
        4, 4, "Navigation", 0, "hash", fused_score=0.1, final_rank=1,
    )


def test_empty_query_is_rejected_before_models_or_index(tmp_path: Path) -> None:
    facade = LocalBackendFacade(config(tmp_path))
    with pytest.raises(ValueError, match="teknik bir soru"):
        facade.ask("   ", lambda _: None, lambda: False)


def test_missing_local_models_report_expected_paths(tmp_path: Path) -> None:
    facade = LocalBackendFacade(config(tmp_path))
    with pytest.raises(MissingLocalResourceError, match="missing-embedding"):
        facade.validate_embedding_model()
    with pytest.raises(MissingLocalResourceError, match="missing.gguf"):
        facade.validate_llm_model()


def test_insufficient_evidence_ui_flow_never_loads_llm(tmp_path: Path, monkeypatch) -> None:
    facade = LocalBackendFacade(config(tmp_path))
    facade.index_artifact.parent.mkdir(parents=True, exist_ok=True)
    facade.index_artifact.write_bytes(b"placeholder")
    monkeypatch.setattr(facade, "_embedding_backend", lambda: object())
    monkeypatch.setattr(ui_backend, "FaissSemanticRetriever", lambda *args: object())
    monkeypatch.setattr(ui_backend, "SQLiteLexicalRetriever", lambda *args: object())

    class FakeSearchService:
        def __init__(self, *args) -> None:
            pass

        def search(self, query: str) -> SearchResult:
            return SearchResult(query, query, None, True, (), ())

    monkeypatch.setattr(ui_backend, "SearchService", FakeSearchService)
    monkeypatch.setattr(
        facade, "_llm_backend",
        lambda: (_ for _ in ()).throw(AssertionError("LLM must not load")),
    )
    answer = facade.ask("kapsam dışı", lambda _: None, lambda: False)
    assert answer.insufficient_evidence is True
    assert answer.validation_status == "insufficient_evidence"


def test_explicit_inventory_does_not_mark_other_documents_missing(tmp_path: Path) -> None:
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.docx"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    database = Database(tmp_path / "inventory.db")
    service = InventoryService(database, FolderScanner())
    service.inventory_files((first,))
    service.inventory_files((second,))
    with database.connect() as connection:
        documents = DocumentRepository(connection).list_all()
    assert {item.file_name for item in documents} == {"first.pdf", "second.docx"}
    assert all(item.status.value == "discovered" for item in documents)


def test_citations_and_retrieval_details_map_to_ui_models() -> None:
    item = candidate()
    detail = RetrievalDetails(
        item.chunk_id, item.document_id, item.file_name, 4, 4, "Navigation",
        -2.0, 1, 0.88, 2, (), 0.0, 0.0, {"lexical": 0.01}, 0.01,
        1, item.text, item.chunk_id,
    )
    search = SearchResult("INS nedir?", "ins nedir", None, False, (item,), (detail,))
    source = SourceRecord("SOURCE_01", item.chunk_id, item.document_id, item.file_name, 4, 4, "Navigation", item.text)
    result = GenerationResult(
        "Ataletsel sistem konumu kestirir.", False, "valid", {"[1]": ("SOURCE_01",)},
        (source,), (item.chunk_id,),
        (ClaimSourceMapping("CLAIM_01", "Ataletsel sistem konumu kestirir.", "[1]", ("SOURCE_01",)),),
        (), 1,
    )
    mapped = map_answer(search, result)
    assert mapped.answer.endswith("[1]")
    assert mapped.sources[0].page_label == "Sayfa 4"
    assert mapped.sources[0].chunk_id == "chunk-1"
    assert mapped.retrieval[0].semantic_rank == 2


def test_document_status_requires_complete_vector_coverage() -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    row = connection.execute(
        "SELECT 'C:/a.pdf' canonical_path,'a.pdf' file_name,'pdf' file_type,"
        "'indexed' status,NULL last_error_code,2 chunk_count,1 vector_count,'tr' language"
    ).fetchone()
    assert document_from_row(row).status == "processed"


def test_worker_emits_success_and_failure_without_qt_event_loop() -> None:
    successes: list[object] = []
    failures: list[tuple[str, str]] = []
    success = FunctionWorker(lambda stage, cancelled: 42)
    success.signals.succeeded.connect(successes.append)
    success.run()
    failure = FunctionWorker(lambda stage, cancelled: (_ for _ in ()).throw(RuntimeError("boom")))
    failure.signals.failed.connect(lambda kind, message: failures.append((kind, message)))
    failure.run()
    assert successes == [42]
    assert failures == [("RuntimeError", "boom")]
