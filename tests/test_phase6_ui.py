from __future__ import annotations

import os
import sqlite3
import time
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


@pytest.mark.parametrize("dark", [True, False], ids=["dark-palette", "light-palette"])
def test_document_table_text_contrasts_with_application_palette(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dark: bool,
) -> None:
    """Production styling must preserve readable normal and selected table text."""
    from types import SimpleNamespace

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QColor, QPalette
    from PySide6.QtWidgets import QApplication
    from app.ui.__main__ import APPLICATION_STYLESHEET
    from app.ui.main_window import MainWindow
    from app.ui.models import DocumentItem

    def luminance(color: QColor) -> float:
        channels = [color.redF(), color.greenF(), color.blueF()]
        linear = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in channels]
        return sum(v * weight for v, weight in zip(linear, (0.2126, 0.7152, 0.0722)))

    def contrast(first: QColor, second: QColor) -> float:
        values = sorted((luminance(first), luminance(second)))
        return (values[1] + 0.05) / (values[0] + 0.05)

    application = QApplication.instance() or QApplication([])
    original_palette = application.palette()
    original_style = application.styleSheet()
    palette = QPalette(original_palette)
    roles = QPalette.ColorRole
    palette.setColor(roles.Window, QColor("#202020" if dark else "#f0f0f0"))
    palette.setColor(roles.WindowText, QColor("#ffffff" if dark else "#000000"))
    palette.setColor(roles.Base, QColor("#202020" if dark else "#ffffff"))
    palette.setColor(roles.Text, QColor("#ffffff" if dark else "#000000"))
    palette.setColor(roles.Highlight, QColor("#1769aa"))
    palette.setColor(roles.HighlightedText, QColor("#ffffff"))
    application.setPalette(palette)
    application.setStyleSheet(APPLICATION_STYLESHEET)
    document = DocumentItem(tmp_path / "Lİdar.pdf", "Lİdar.pdf", "PDF", "discovered", "tr")
    backend = SimpleNamespace(list_documents=lambda: (document,))
    window = MainWindow(backend)
    try:
        window.show()
        application.processEvents()
        table = window.document_table
        assert table.rowCount() == 1
        texts = [table.item(0, column).text() for column in range(4)]
        assert texts == ["Lİdar.pdf", "PDF", "Yeni", "tr"]
        colors = table.palette()
        assert contrast(colors.color(roles.Text), colors.color(roles.Base)) >= 4.5
        table.selectRow(0)
        application.processEvents()
        assert contrast(colors.color(roles.HighlightedText), colors.color(roles.Highlight)) >= 4.5
        assert colors.color(roles.Base) == palette.color(roles.Base)
        assert application.palette().color(roles.Window) == palette.color(roles.Window)
    finally:
        window.close()
        application.setStyleSheet(original_style)
        application.setPalette(original_palette)


@pytest.mark.skipif(os.name != "nt", reason="Windows filesystem casing regression")
def test_document_filter_uses_inventory_canonical_path(tmp_path: Path) -> None:
    """Mixed-case and Turkish paths must survive filtered registration results."""
    from app.infrastructure.paths import canonicalize_path

    path = tmp_path / "MixedCase" / "İnsansız Araç Lİdar 3B.pdf"
    path.parent.mkdir()
    path.write_bytes(b"local inventory fixture")
    facade = LocalBackendFacade(config(tmp_path))
    registered = facade.register_documents((path,))
    assert len(registered) == 1
    assert registered == facade.list_documents()
    assert registered == facade.list_documents((path,))
    assert registered == facade.list_documents((Path(canonicalize_path(path)),))
    assert registered == facade.register_documents((path,))
    assert facade.list_documents((tmp_path / "other.pdf",)) == ()


def test_manual_refresh_button_renders_persisted_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Click the connected refresh button and verify the visible table model."""
    from unittest.mock import patch

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from app.ui.main_window import MainWindow

    application = QApplication.instance() or QApplication([])
    facade = LocalBackendFacade(config(tmp_path))
    window = MainWindow(facade)
    window.show()
    try:
        assert window.document_table.rowCount() == 0
        path = tmp_path / "MixedCase.pdf"
        path.write_bytes(b"local inventory fixture")
        InventoryService(facade.database, FolderScanner()).inventory_files((path,))
        with patch.object(facade, "list_documents", wraps=facade.list_documents) as listing:
            window.refresh_button.click()
            application.processEvents()
            listing.assert_called_once_with()
        table = window.document_table
        assert table.rowCount() == 1
        assert table.item(0, 0).text() == path.name
        assert table.isVisible()
        assert not table.isRowHidden(0)
        assert table.viewport().rect().intersects(table.visualItemRect(table.item(0, 0)))
        window.refresh_button.click()
        assert table.rowCount() == 1
    finally:
        window.close()


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


@pytest.mark.parametrize("duplicate_copy", [False, True], ids=["same-path", "identical-copy"])
def test_add_documents_completes_in_qt_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, duplicate_copy: bool,
) -> None:
    """Real registration must refresh rows, clear busy text, and permit another add."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QFileDialog
    import pypdfium2 as pdfium
    from app.ui.main_window import MainWindow

    application = QApplication.instance() or QApplication([])
    path = tmp_path / "sample.pdf"
    pdf = pdfium.PdfDocument.new()
    page = pdf.new_page(200, 200)
    pdf.save(path)
    page.close()
    pdf.close()
    facade = LocalBackendFacade(config(tmp_path))
    window = MainWindow(facade)
    window.show()
    selected = path
    monkeypatch.setattr(
        QFileDialog, "getOpenFileNames", lambda *args: ([str(selected)], ""),
    )
    try:
        for attempt in range(2):
            if attempt and duplicate_copy:
                selected = tmp_path / "copy.pdf"
                selected.write_bytes(path.read_bytes())
            window.add_button.click()
            assert window._worker is not None
            assert not window.add_button.isEnabled()
            deadline = time.monotonic() + 5
            while window._worker is not None and time.monotonic() < deadline:
                application.processEvents()
                time.sleep(0.005)
            assert window._worker is None, "Registration worker did not finish within 5 seconds"
            assert window.thread_pool.waitForDone(1000)
            assert not window._elapsed_timer.isActive()
            assert not window.progress.isVisible()
            assert not window.cancel_button.isEnabled()
            assert window.add_button.isEnabled()
            assert window.index_button.isEnabled()
            documents = facade.list_documents()
            assert len(documents) == (2 if attempt and duplicate_copy else 1)
            assert window.document_table.rowCount() == len(documents)
            for row, document in enumerate(documents):
                assert window.document_table.item(row, 0).text() == document.file_name
                assert window.document_table.item(row, 2).text() == document.status_label
                assert window.document_table.item(row, 0).data(Qt.ItemDataRole.UserRole) == str(document.path)
                assert document.status == ("duplicate" if document.path != path else "discovered")
            assert path in window._paths_for_indexing()
            with facade.database.connect() as connection:
                state = connection.execute(
                    "SELECT inventory_status,parse_status,embedding_status "
                    "FROM document_index_state WHERE document_id="
                    "(SELECT id FROM documents WHERE status='discovered')"
                ).fetchone()
                assert tuple(state) == ("pending", "pending", "pending")
            assert window.operation_label.text() == "Belgeler kaydedildi. İndekslemeye hazır."
            assert window.status_label.text() == window.operation_label.text()
    finally:
        window.thread_pool.waitForDone(5000)
        application.processEvents()
        window.close()
