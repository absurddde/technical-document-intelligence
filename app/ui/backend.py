"""Thin Phase 6 coordination over the accepted local production services."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import threading

from app.generation.llm import LlamaCppGgufBackend
from app.infrastructure.config import AppConfig
from app.infrastructure.paths import canonicalize_path, ensure_local_directories
from app.ingestion.docx_parser import DocxParser
from app.ingestion.ocr import TesseractOcr
from app.ingestion.pdf_parser import PdfParser
from app.ingestion.scanner import FolderScanner
from app.persistence.connection import Database
from app.persistence.schema import initialize_schema
from app.processing.cache import ParsedDocumentCache
from app.processing.chunker import StructureAwareChunker
from app.retrieval.embeddings import SentenceTransformerEmbeddingBackend
from app.retrieval.retrievers import FaissSemanticRetriever, SQLiteLexicalRetriever
from app.services.document_processing_service import DocumentProcessingService
from app.services.generation_service import GenerationService
from app.services.inventory_service import InventoryService
from app.services.search_service import SearchService
from app.services.semantic_index_service import SemanticIndexService
from app.ui.models import AnswerItem, DocumentItem, document_from_row, map_answer


class OperationCancelled(RuntimeError):
    """Raised at safe cooperative cancellation boundaries."""


class MissingLocalResourceError(RuntimeError):
    """Raised when a configured local model or index is absent."""


class _ForbiddenBackend:
    def generate(self, *args, **kwargs) -> str:
        raise AssertionError("Insufficient-evidence flow invoked the LLM")


class LocalBackendFacade:
    """Own reusable local model instances and coordinate production services."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.database = Database(config.paths.database)
        self._embedding = None
        self._llm = None
        self._model_lock = threading.Lock()
        ensure_local_directories([
            config.paths.database.parent, config.paths.indexes,
            config.paths.cache, config.paths.logs,
        ])
        with self.database.transaction() as connection:
            initialize_schema(connection)

    @property
    def index_artifact(self) -> Path:
        return self.config.paths.indexes / self.config.vector_index.artifact

    def validate_embedding_model(self) -> None:
        """Fail clearly without attempting a download."""

        if not self.config.embedding.model_path.is_dir():
            raise MissingLocalResourceError(
                f"Yerel embedding modeli bulunamadı: {self.config.embedding.model_path}"
            )

    def validate_llm_model(self) -> None:
        """Fail clearly without attempting a download."""

        if not self.config.llm.model_path.is_file():
            raise MissingLocalResourceError(
                f"Yerel dil modeli bulunamadı: {self.config.llm.model_path}"
            )

    def _embedding_backend(self):
        self.validate_embedding_model()
        with self._model_lock:
            if self._embedding is None:
                cfg = self.config.embedding
                self._embedding = SentenceTransformerEmbeddingBackend(
                    cfg.model_path, device=cfg.device, batch_size=cfg.batch_size,
                    input_format=cfg.input_format,
                )
            return self._embedding

    def _llm_backend(self):
        self.validate_llm_model()
        with self._model_lock:
            if self._llm is None:
                self._llm = LlamaCppGgufBackend.from_config(self.config.llm)
            return self._llm

    def register_documents(self, paths: tuple[Path, ...]) -> tuple[DocumentItem, ...]:
        """Register explicit QFileDialog selections through inventory services."""

        if paths:
            InventoryService(
                self.database, FolderScanner(self.config.indexing.hash_block_size)
            ).inventory_files(paths)
        return self.list_documents(paths)

    def list_documents(self, paths: tuple[Path, ...] | None = None) -> tuple[DocumentItem, ...]:
        """Read persisted document and complete-vector status for display."""

        with self.database.connect() as connection:
            initialize_schema(connection)
            parameters: tuple[str, ...] = ()
            where = ""
            if paths:
                values = tuple(canonicalize_path(path) for path in paths)
                where = f"WHERE d.canonical_path IN ({','.join('?' for _ in values)})"
                parameters = values
            rows = connection.execute(
                f"""SELECT d.canonical_path,d.file_name,d.file_type,d.status,
                    s.last_error_code,COUNT(DISTINCT c.chunk_id) AS chunk_count,
                    COUNT(DISTINCT v.chunk_id) AS vector_count,
                    GROUP_CONCAT(DISTINCT c.language) AS language
                    FROM documents d
                    LEFT JOIN document_index_state s ON s.document_id=d.id
                    LEFT JOIN chunks c ON c.document_id=d.id
                    LEFT JOIN vector_index_metadata v ON v.chunk_id=c.chunk_id
                    {where}
                    GROUP BY d.id ORDER BY d.file_name COLLATE NOCASE""",
                parameters,
            ).fetchall()
        return tuple(document_from_row(row) for row in rows)

    def index_documents(
        self, paths: tuple[Path, ...], stage: Callable[[str], None],
        cancelled: Callable[[], bool],
    ) -> dict[str, int]:
        """Run incremental processing and indexing with safe stage boundaries."""

        stage("Belgeler hazırlanıyor...")
        plan = InventoryService(
            self.database, FolderScanner(self.config.indexing.hash_block_size)
        ).inventory_files(paths)
        if cancelled():
            raise OperationCancelled("İşlem iptal edildi.")

        parsing = self.config.parsing
        ocr = TesseractOcr(self.config.ocr.languages, self.config.ocr.tesseract_command) if self.config.ocr.enabled else None
        processor = DocumentProcessingService(
            self.database,
            PdfParser(
                ocr, min_text_characters=parsing.pdf_min_text_characters,
                min_alphanumeric_ratio=parsing.pdf_min_alphanumeric_ratio,
                dpi=self.config.ocr.dpi,
                remove_margins=parsing.remove_repeated_headers_footers,
                margin_lines=parsing.repeated_margin_lines,
                repeated_page_ratio=parsing.repeated_page_ratio,
                pipeline_version=parsing.pipeline_version,
            ),
            DocxParser(parsing.pipeline_version),
            StructureAwareChunker(self.config.chunking.chunk_size, self.config.chunking.overlap),
            ParsedDocumentCache(self.config.paths.cache), parsing.pipeline_version,
        )
        processed = skipped = failed = 0
        for item in plan:
            if cancelled():
                raise OperationCancelled("İşlem güvenli bir aşamada iptal edildi.")
            name = item.fingerprint.file_name if item.fingerprint else Path(item.canonical_path).name
            stage(f"Ayrıştırılıyor / gerekirse OCR: {name}")
            result = processor.process((item,))
            processed += result.processed
            skipped += result.skipped
            failed += result.failed
        if cancelled():
            raise OperationCancelled("İşlem güvenli bir aşamada iptal edildi.")

        stage("Yerel embedding modeli yükleniyor...")
        embedding = self._embedding_backend()
        stage("Embedding ve FAISS indeksi güncelleniyor...")
        semantic = SemanticIndexService(
            self.database, embedding, self.index_artifact,
            self.config.vector_index.version,
        ).synchronize()
        if cancelled():
            raise OperationCancelled("İndeks güvenle tamamlandı; sonuç gösterimi iptal edildi.")
        stage("Tamamlandı")
        return {
            "processed": processed, "skipped": skipped, "failed": failed,
            "embedded": semantic.embedded, "reused": semantic.reused,
            "total": semantic.total,
        }

    def ask(
        self, query: str, stage: Callable[[str], None],
        cancelled: Callable[[], bool],
    ) -> AnswerItem:
        """Run accepted retrieval and validated generation off the GUI thread."""

        if not query.strip():
            raise ValueError("Lütfen teknik bir soru girin.")
        if not self.index_artifact.is_file():
            raise MissingLocalResourceError(
                f"Arama indeksi bulunamadı: {self.index_artifact}"
            )
        stage("Kaynaklar aranıyor...")
        embedding = self._embedding_backend()
        with self.database.connect() as connection:
            search = SearchService(
                SQLiteLexicalRetriever(connection),
                FaissSemanticRetriever(connection, embedding, self.index_artifact),
                self.config.retrieval,
            ).search(query.strip())
            chunk_ids = tuple(item.chunk_id for item in search.selected)
            source_paths: dict[str, Path] = {}
            if chunk_ids:
                marks = ",".join("?" for _ in chunk_ids)
                source_paths = {
                    str(row[0]): Path(str(row[1]))
                    for row in connection.execute(
                        f"SELECT chunk_id,file_path FROM chunks WHERE chunk_id IN ({marks})",
                        chunk_ids,
                    )
                }
        if cancelled():
            raise OperationCancelled("Sorgu iptal edildi.")
        if search.insufficient_evidence:
            generation = GenerationService(
                _ForbiddenBackend(), self.config.generation
            ).generate(search)
            return map_answer(search, generation, source_paths)

        stage("Yerel dil modeli yükleniyor...")
        llm = self._llm_backend()
        if cancelled():
            raise OperationCancelled("Sorgu iptal edildi.")
        stage("Yanıt oluşturuluyor ve doğrulanıyor...")
        generation = GenerationService(llm, self.config.generation).generate(search)
        if cancelled():
            raise OperationCancelled(
                "Model çağrısı güvenle tamamlandı; iptal nedeniyle sonuç gösterilmedi."
            )
        stage("Tamamlandı")
        return map_answer(search, generation, source_paths)
