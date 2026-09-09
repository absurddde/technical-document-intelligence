"""Main PySide6 window for local document indexing and grounded questions."""

from __future__ import annotations

from functools import partial
import logging
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QThreadPool, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QSplitter, QStatusBar, QTableWidget, QTableWidgetItem,
    QTextBrowser, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from app.ui.backend import LocalBackendFacade
from app.ui.models import AnswerItem, DocumentItem, SourceItem
from app.ui.workers import FunctionWorker


class MainWindow(QMainWindow):
    """Responsive Phase 6 desktop shell over the production backend facade."""

    def __init__(self, backend: LocalBackendFacade) -> None:
        super().__init__()
        self.backend = backend
        self.thread_pool = QThreadPool(self)
        self.thread_pool.setMaxThreadCount(1)
        self._worker: FunctionWorker | None = None
        self._selected_paths: set[Path] = set()
        self._elapsed_seconds = 0
        self._logger = logging.getLogger("document_intelligence.ui")
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._tick_elapsed)
        self.setWindowTitle("Technical Document Intelligence")
        self.resize(1280, 820)
        self.setMinimumSize(980, 640)
        self._build_ui()
        self.refresh_documents()

    def _build_ui(self) -> None:
        root = QWidget(self)
        layout = QVBoxLayout(root)
        title = QLabel("Yerel Teknik Doküman Zekâsı")
        title.setObjectName("title")
        subtitle = QLabel("PDF ve DOCX belgeleriniz cihazdan çıkmadan işlenir.")
        subtitle.setObjectName("subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._document_panel())
        splitter.addWidget(self._query_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([410, 850])
        layout.addWidget(splitter, 1)
        self.setCentralWidget(root)

        status = QStatusBar(self)
        self.setStatusBar(status)
        self.status_label = QLabel("Hazır")
        status.addWidget(self.status_label, 1)
        status.addPermanentWidget(QLabel("Çevrimdışı · Yerel işlem"))

    def _document_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("Dokümanlar"))
        self.document_table = QTableWidget(0, 4)
        self.document_table.setHorizontalHeaderLabels(["Dosya", "Tür", "Durum", "Dil"])
        self.document_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.document_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.document_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.document_table.verticalHeader().setVisible(False)
        header = self.document_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.document_table, 1)

        first = QHBoxLayout()
        self.add_button = QPushButton("Belge Ekle")
        self.remove_button = QPushButton("Listeden Kaldır")
        self.refresh_button = QPushButton("Yenile")
        self.paths_button = QPushButton("Yerel Yollar")
        first.addWidget(self.add_button)
        first.addWidget(self.remove_button)
        first.addWidget(self.refresh_button)
        first.addWidget(self.paths_button)
        layout.addLayout(first)
        self.index_button = QPushButton("Seçili Belgeleri İşle / İndeksle")
        self.index_button.setObjectName("primary")
        layout.addWidget(self.index_button)
        self.document_hint = QLabel("Listeden kaldırma özgün dosyayı veya mevcut indeksi silmez.")
        self.document_hint.setWordWrap(True)
        self.document_hint.setObjectName("hint")
        layout.addWidget(self.document_hint)

        self.add_button.clicked.connect(self.add_documents)
        self.remove_button.clicked.connect(self.remove_from_view)
        self.refresh_button.clicked.connect(self.refresh_documents)
        self.paths_button.clicked.connect(self.show_local_paths)
        self.index_button.clicked.connect(self.start_indexing)
        return panel

    def _query_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("Teknik Soru"))
        self.question = QPlainTextEdit()
        self.question.setPlaceholderText("Teknik kavram veya soru girin...")
        self.question.setMaximumHeight(90)
        layout.addWidget(self.question)
        actions = QHBoxLayout()
        self.ask_button = QPushButton("Sor")
        self.ask_button.setObjectName("primary")
        self.cancel_button = QPushButton("İptal")
        self.cancel_button.setEnabled(False)
        actions.addWidget(self.ask_button)
        actions.addWidget(self.cancel_button)
        actions.addStretch(1)
        layout.addLayout(actions)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.operation_label = QLabel("Hazır")
        layout.addWidget(self.operation_label)
        layout.addWidget(self.progress)

        layout.addWidget(QLabel("Teknik Yanıt"))
        self.answer = QTextBrowser()
        self.answer.setOpenExternalLinks(False)
        self.answer.setPlaceholderText("Doğrulanmış Türkçe teknik yanıt burada gösterilir.")
        self.answer.setMinimumHeight(150)
        layout.addWidget(self.answer, 2)

        source_splitter = QSplitter(Qt.Orientation.Horizontal)
        source_box = QWidget()
        source_layout = QVBoxLayout(source_box)
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.addWidget(QLabel("Kullanılan Kaynaklar"))
        self.source_list = QListWidget()
        source_layout.addWidget(self.source_list)
        self.open_source_button = QPushButton("Özgün Dosyayı Aç")
        self.open_source_button.setEnabled(False)
        source_layout.addWidget(self.open_source_button)
        self.source_preview = QPlainTextEdit()
        self.source_preview.setReadOnly(True)
        self.source_preview.setPlaceholderText("Kaynak seçildiğinde kısa kanıt pasajı gösterilir.")
        source_splitter.addWidget(source_box)
        source_splitter.addWidget(self.source_preview)
        source_splitter.setSizes([330, 500])
        layout.addWidget(source_splitter, 2)

        self.retrieval_group = QGroupBox("Retrieval Details")
        self.retrieval_group.setCheckable(True)
        self.retrieval_group.setChecked(False)
        retrieval_layout = QVBoxLayout(self.retrieval_group)
        self.retrieval_tree = QTreeWidget()
        self.retrieval_tree.setHeaderLabels([
            "Sıra", "Dosya / Konum", "Lexical", "Semantic", "RRF", "Chunk ID"
        ])
        self.retrieval_tree.setRootIsDecorated(False)
        retrieval_layout.addWidget(self.retrieval_tree)
        self.retrieval_tree.setVisible(False)
        self.retrieval_group.toggled.connect(self.retrieval_tree.setVisible)
        layout.addWidget(self.retrieval_group)

        self.ask_button.clicked.connect(self.start_query)
        self.cancel_button.clicked.connect(self.cancel_operation)
        self.source_list.currentItemChanged.connect(self.show_source)
        self.open_source_button.clicked.connect(self.open_source)
        return panel

    def add_documents(self) -> None:
        names, _ = QFileDialog.getOpenFileNames(
            self, "PDF veya DOCX Belgeleri Seç", "", "Belgeler (*.pdf *.docx)"
        )
        paths = tuple(Path(name).resolve() for name in names)
        if not paths:
            return
        self._selected_paths.update(paths)
        self._start_worker(
            lambda stage, cancelled: self.backend.register_documents(paths),
            "Belgeler kaydediliyor...", self._documents_registered,
        )

    def _documents_registered(self, _: Any) -> None:
        self.refresh_documents()
        self._set_stage("Belgeler kaydedildi. İndekslemeye hazır.")

    def refresh_documents(self) -> None:
        try:
            documents = self.backend.list_documents()
        except Exception as error:
            self._show_error(type(error).__name__, str(error))
            return
        self._render_documents(documents)

    def _render_documents(self, documents: tuple[DocumentItem, ...]) -> None:
        self.document_table.setRowCount(len(documents))
        for row, document in enumerate(documents):
            values = (document.file_name, document.file_type, document.status_label, document.language)
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, str(document.path))
                if document.error_code:
                    item.setToolTip(f"Hata kodu: {document.error_code}")
                self.document_table.setItem(row, column, item)

    def remove_from_view(self) -> None:
        rows = sorted({index.row() for index in self.document_table.selectedIndexes()}, reverse=True)
        for row in rows:
            item = self.document_table.item(row, 0)
            if item:
                self._selected_paths.discard(Path(item.data(Qt.ItemDataRole.UserRole)))
            self.document_table.removeRow(row)
        self.status_label.setText("Seçim görünümden kaldırıldı; özgün dosya ve indeks korunuyor.")

    def _paths_for_indexing(self) -> tuple[Path, ...]:
        rows = sorted({index.row() for index in self.document_table.selectedIndexes()})
        if rows:
            return tuple(Path(self.document_table.item(row, 0).data(Qt.ItemDataRole.UserRole)) for row in rows)
        return tuple(sorted(self._selected_paths, key=str))

    def start_indexing(self) -> None:
        paths = self._paths_for_indexing()
        if not paths:
            QMessageBox.information(self, "Belge seçin", "İşlenecek en az bir PDF veya DOCX belge seçin.")
            return
        self._start_worker(
            partial(self.backend.index_documents, paths),
            "Belgeler hazırlanıyor...", self._index_complete,
        )

    def show_local_paths(self) -> None:
        config = self.backend.config
        QMessageBox.information(
            self, "Salt Okunur Yerel Yapılandırma",
            f"Veritabanı: {config.paths.database}\n"
            f"İndeks: {self.backend.index_artifact}\n"
            f"Embedding modeli: {config.embedding.model_path}\n"
            f"Dil modeli: {config.llm.model_path}\n\n"
            "Uygulama çalışma sırasında model indirmez.",
        )

    def _index_complete(self, result: dict[str, int]) -> None:
        self.refresh_documents()
        self.status_label.setText(
            f"İndeks tamamlandı · İşlenen {result['processed']} · Atlanan {result['skipped']} · Hatalı {result['failed']}"
        )

    def start_query(self) -> None:
        query = self.question.toPlainText().strip()
        if not query:
            QMessageBox.information(self, "Soru gerekli", "Lütfen teknik bir soru girin.")
            self.question.setFocus()
            return
        self.answer.clear()
        self.source_list.clear()
        self.source_preview.clear()
        self.retrieval_tree.clear()
        self._start_worker(partial(self.backend.ask, query), "Kaynaklar aranıyor...", self._query_complete)

    def _query_complete(self, result: AnswerItem) -> None:
        self.answer.setPlainText(result.answer)
        for source in result.sources:
            item = QListWidgetItem(
                f"{source.citation} {source.file_name}\n{source.page_label} · {source.section}"
            )
            item.setData(Qt.ItemDataRole.UserRole, source)
            self.source_list.addItem(item)
        for detail in result.retrieval:
            self.retrieval_tree.addTopLevelItem(QTreeWidgetItem([
                str(detail.rank), f"{detail.file_name} · {detail.locator}",
                self._rank_score(detail.lexical_rank, detail.lexical_score),
                self._rank_score(detail.semantic_rank, detail.semantic_score),
                f"{detail.rrf_score:.6f}", detail.chunk_id,
            ]))
            self.retrieval_tree.topLevelItem(self.retrieval_tree.topLevelItemCount() - 1).setToolTip(
                4, ", ".join(f"{name}: {value:.6f}" for name, value in detail.contributions.items())
            )
        self.retrieval_tree.resizeColumnToContents(0)
        self.retrieval_tree.resizeColumnToContents(1)
        self.status_label.setText(
            "Yetersiz kanıt: model çağrılmadı." if result.insufficient_evidence
            else f"Yanıt tamamlandı · Doğrulama: {result.validation_status}"
        )

    def show_source(self, current: QListWidgetItem | None, previous: QListWidgetItem | None = None) -> None:
        del previous
        source = current.data(Qt.ItemDataRole.UserRole) if current else None
        if not isinstance(source, SourceItem):
            self.source_preview.clear()
            self.open_source_button.setEnabled(False)
            return
        self.source_preview.setPlainText(
            f"{source.citation} · {source.source_id}\n"
            f"{source.file_name} · {source.page_label} · {source.section}\n"
            f"Chunk: {source.chunk_id}\n\n{source.preview}"
        )
        self.open_source_button.setEnabled(bool(source.file_path and source.file_path.is_file()))

    def open_source(self) -> None:
        item = self.source_list.currentItem()
        source = item.data(Qt.ItemDataRole.UserRole) if item else None
        if isinstance(source, SourceItem) and source.file_path and source.file_path.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(source.file_path)))

    def _start_worker(self, operation, initial_stage: str, on_success) -> None:
        if self._worker is not None:
            QMessageBox.information(self, "İşlem sürüyor", "Önce devam eden işlemin tamamlanmasını bekleyin.")
            return
        worker = FunctionWorker(operation)
        worker.signals.stage.connect(self._set_stage)
        worker.signals.succeeded.connect(on_success)
        worker.signals.failed.connect(self._show_error)
        worker.signals.finished.connect(self._operation_finished)
        self._worker = worker
        self._elapsed_seconds = 0
        self._set_stage(initial_stage)
        self._set_busy(True)
        self._elapsed_timer.start(1000)
        self.thread_pool.start(worker)

    def cancel_operation(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.cancel_button.setEnabled(False)
            self.operation_label.setText(
                "İptal istendi; güvenli aşama bekleniyor. Aktif model çağrısı kesilmeyecektir."
            )

    def closeEvent(self, event) -> None:
        """Request cooperative cancellation before allowing shutdown."""

        if self._worker is not None:
            self._worker.cancel()
            QMessageBox.information(
                self, "Operation still active",
                "Cancellation was requested. Close the application after the current "
                "safe operation boundary is reached.",
            )
            event.ignore()
            return
        event.accept()

    def _set_stage(self, text: str) -> None:
        self.operation_label.setText(text)
        self.status_label.setText(text)

    def _tick_elapsed(self) -> None:
        self._elapsed_seconds += 1
        base = self.operation_label.text().split(" · Geçen süre:", 1)[0]
        minutes, seconds = divmod(self._elapsed_seconds, 60)
        self.operation_label.setText(f"{base} · Geçen süre: {minutes:02d}:{seconds:02d}")

    def _set_busy(self, busy: bool) -> None:
        self.progress.setVisible(busy)
        self.cancel_button.setEnabled(busy)
        self.add_button.setEnabled(not busy)
        self.remove_button.setEnabled(not busy)
        self.refresh_button.setEnabled(not busy)
        self.paths_button.setEnabled(not busy)
        self.index_button.setEnabled(not busy)
        self.ask_button.setEnabled(not busy)

    @staticmethod
    def _rank_score(rank: int | None, score: float | None) -> str:
        if rank is None:
            return "—"
        return f"#{rank}" if score is None else f"#{rank} / {score:.4f}"

    def _operation_finished(self) -> None:
        self._elapsed_timer.stop()
        self._set_busy(False)
        self._worker = None

    def _show_error(self, error_type: str, message: str) -> None:
        safe = message or "İşlem tamamlanamadı. Ayrıntılar yerel günlükte saklandı."
        self._logger.error("UI operation failed error_type=%s", error_type)
        if error_type == "OperationCancelled":
            self.status_label.setText(safe)
            return
        QMessageBox.critical(self, "İşlem tamamlanamadı", safe)
        self.status_label.setText("İşlem başarısız oldu.")
