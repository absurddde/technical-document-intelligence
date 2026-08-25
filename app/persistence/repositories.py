"""Repositories for Phase 1 document inventory state."""

from __future__ import annotations

import sqlite3
import json
from uuid import uuid4

from app.domain.models import Document, DocumentChunk, DocumentStatus, FileFingerprint, ParsedBlock, RunStatus


def _document_from_row(row: sqlite3.Row) -> Document:
    return Document(
        id=row["id"],
        document_id=row["document_id"],
        canonical_path=row["canonical_path"],
        file_name=row["file_name"],
        file_type=row["file_type"],
        file_size=row["file_size"],
        modified_ns=row["modified_ns"],
        sha256=row["sha256"],
        status=DocumentStatus(row["status"]),
        duplicate_of_id=row["duplicate_of_id"],
    )


class DocumentRepository:
    """Persist one row per physical path, including duplicate paths."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def list_all(self) -> tuple[Document, ...]:
        """Return every known physical document path."""

        rows = self._connection.execute(
            "SELECT * FROM documents ORDER BY canonical_path"
        ).fetchall()
        return tuple(_document_from_row(row) for row in rows)

    def get_by_path(self, canonical_path: str) -> Document | None:
        """Find document metadata by its canonical path key."""

        row = self._connection.execute(
            "SELECT * FROM documents WHERE canonical_path = ?", (canonical_path,)
        ).fetchone()
        return _document_from_row(row) if row else None

    def save_fingerprint(
        self,
        fingerprint: FileFingerprint,
        status: DocumentStatus = DocumentStatus.DISCOVERED,
        duplicate_of_id: int | None = None,
    ) -> Document:
        """Insert or update metadata while preserving the path's document UUID."""

        existing = self.get_by_path(fingerprint.canonical_path)
        if existing is None:
            self._connection.execute(
                """
                INSERT INTO documents(
                    document_id, canonical_path, file_name, file_type, file_size,
                    modified_ns, sha256, status, duplicate_of_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    fingerprint.canonical_path,
                    fingerprint.file_name,
                    fingerprint.file_type,
                    fingerprint.file_size,
                    fingerprint.modified_ns,
                    fingerprint.sha256,
                    status.value,
                    duplicate_of_id,
                ),
            )
        else:
            self._connection.execute(
                """
                UPDATE documents SET
                    file_name = ?, file_type = ?, file_size = ?, modified_ns = ?,
                    sha256 = ?, status = ?, duplicate_of_id = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    fingerprint.file_name,
                    fingerprint.file_type,
                    fingerprint.file_size,
                    fingerprint.modified_ns,
                    fingerprint.sha256,
                    status.value,
                    duplicate_of_id,
                    existing.id,
                ),
            )
        saved = self.get_by_path(fingerprint.canonical_path)
        if saved is None:
            raise RuntimeError("Document save failed")
        return saved

    def mark_missing(self, document_id: int) -> None:
        """Mark a path absent without deleting its provenance metadata."""

        self._connection.execute(
            """
            UPDATE documents SET status = 'missing', updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (document_id,),
        )

    def ensure_index_state(
        self,
        document_id: int,
        run_id: int | None,
        *,
        reset_pipeline: bool = False,
        inventory_status: str = "pending",
    ) -> None:
        """Create pipeline state and reset it only for changed content."""

        self._connection.execute(
            """
            INSERT INTO document_index_state(document_id, run_id, inventory_status)
            VALUES (?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                run_id = excluded.run_id,
                inventory_status = CASE
                    WHEN ? THEN excluded.inventory_status
                    ELSE document_index_state.inventory_status
                END,
                parse_status = CASE WHEN ? THEN 'pending' ELSE parse_status END,
                chunk_status = CASE WHEN ? THEN 'pending' ELSE chunk_status END,
                embedding_status = CASE WHEN ? THEN 'pending' ELSE embedding_status END,
                lexical_status = CASE WHEN ? THEN 'pending' ELSE lexical_status END,
                vector_status = CASE WHEN ? THEN 'pending' ELSE vector_status END,
                last_error_code = CASE WHEN ? THEN NULL ELSE last_error_code END,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                document_id,
                run_id,
                inventory_status,
                reset_pipeline,
                reset_pipeline,
                reset_pipeline,
                reset_pipeline,
                reset_pipeline,
                reset_pipeline,
                reset_pipeline,
            ),
        )


class IndexingRunRepository:
    """Persist high-level inventory run progress without document content."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def start(self, root_path: str) -> int:
        """Start and return a new inventory run identifier."""

        cursor = self._connection.execute(
            "INSERT INTO indexing_runs(root_path, status) VALUES (?, 'running')",
            (root_path,),
        )
        return int(cursor.lastrowid)

    def finish(
        self,
        run_id: int,
        status: RunStatus,
        counts: dict[str, int],
    ) -> None:
        """Complete an inventory run with aggregate privacy-safe counts."""

        fields = (
            "discovered_count",
            "new_count",
            "unchanged_count",
            "modified_count",
            "duplicate_count",
            "missing_count",
            "failed_count",
        )
        values = [int(counts.get(field, 0)) for field in fields]
        self._connection.execute(
            """
            UPDATE indexing_runs SET
                finished_at = CURRENT_TIMESTAMP, status = ?,
                discovered_count = ?, new_count = ?, unchanged_count = ?,
                modified_count = ?, duplicate_count = ?, missing_count = ?,
                failed_count = ?
            WHERE id = ?
            """,
            (status.value, *values, run_id),
        )


class ContentRepository:
    """Atomically replace and retrieve Phase 2 parsed content and chunks."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def replace(self, document: Document, blocks: tuple[ParsedBlock, ...], chunks: tuple[DocumentChunk, ...], pipeline_version: str) -> None:
        """Replace derived content only after parsing and chunking succeeded."""

        self._connection.execute("DELETE FROM parsed_blocks WHERE document_id = ?", (document.id,))
        self._connection.execute("DELETE FROM chunks WHERE document_id = ?", (document.id,))
        self._connection.executemany(
            """INSERT INTO parsed_blocks(document_id, ordinal, kind, text, page_number,
               section_title, heading_path, paragraph_index, ocr_used, ocr_confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(document.id, i, b.kind, b.text, b.page_number, b.section_title,
              json.dumps(b.heading_path, ensure_ascii=False), b.paragraph_index,
              int(b.ocr_used), b.ocr_confidence) for i, b in enumerate(blocks)],
        )
        self._connection.executemany(
            """INSERT INTO chunks(document_id, chunk_id, ordinal, file_name, file_path,
               page_start, page_end, section_title, paragraph_start, paragraph_end,
               language, text, ocr_used, ocr_confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(document.id, c.chunk_id, i, c.file_name, c.file_path, c.page_start,
              c.page_end, c.section_title, c.paragraph_start, c.paragraph_end,
              c.language, c.text, int(c.ocr_used), c.ocr_confidence)
             for i, c in enumerate(chunks)],
        )
        self._connection.execute(
            """UPDATE document_index_state SET parse_status='complete', chunk_status='complete',
               pipeline_version=?, last_error_code=NULL, updated_at=CURRENT_TIMESTAMP
               WHERE document_id=?""", (pipeline_version, document.id)
        )
        self._connection.execute("UPDATE documents SET status='indexed', updated_at=CURRENT_TIMESTAMP WHERE id=?", (document.id,))

    def mark_failed(self, document_id: int, error_code: str, pipeline_version: str) -> None:
        """Record a privacy-safe per-document failure."""

        self._connection.execute("UPDATE documents SET status='failed', updated_at=CURRENT_TIMESTAMP WHERE id=?", (document_id,))
        self._connection.execute(
            """UPDATE document_index_state SET parse_status='failed', chunk_status='pending',
               pipeline_version=?, last_error_code=?, updated_at=CURRENT_TIMESTAMP WHERE document_id=?""",
            (pipeline_version, error_code, document_id),
        )

    def count_chunks(self, document_id: int) -> int:
        row = self._connection.execute("SELECT COUNT(*) FROM chunks WHERE document_id=?", (document_id,)).fetchone()
        return int(row[0])
