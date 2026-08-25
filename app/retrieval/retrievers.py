"""Adapters from SQLite FTS5 and FAISS to common retrieval hits."""

from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Protocol

from app.retrieval.embeddings import EmbeddingBackend
from app.retrieval.models import RetrievalHit
from app.retrieval.vector_index import FaissVectorIndex


class RankedRetriever(Protocol):
    def search(self, query: str, limit: int) -> tuple[RetrievalHit, ...]: ...


class SQLiteLexicalRetriever:
    """Return provenance-rich BM25 candidates from the existing FTS index."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def search(self, query: str, limit: int) -> tuple[RetrievalHit, ...]:
        if not query or limit <= 0:
            return ()
        rows = self._connection.execute(
            """SELECT c.chunk_id,d.document_id,c.file_name,c.text,
                      bm25(chunks_fts,0.0,2.0,1.5,1.0),c.page_start,c.page_end,
                      c.section_title,c.ordinal,v.content_hash
               FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.rowid
               JOIN documents d ON d.id=c.document_id
               LEFT JOIN vector_index_metadata v ON v.chunk_id=c.chunk_id
               WHERE chunks_fts MATCH ? AND d.status='indexed'
               ORDER BY 5 ASC,c.chunk_id ASC LIMIT ?""",
            (query, limit),
        ).fetchall()
        return tuple(RetrievalHit(
            str(row[0]), str(row[1]), str(row[2]), str(row[3]), float(row[4]),
            row[5], row[6], row[7], int(row[8]), str(row[9] or ""),
        ) for row in rows)


class FaissSemanticRetriever:
    """Embed a query, search local FAISS, and restore SQLite provenance."""

    def __init__(self, connection: sqlite3.Connection, backend: EmbeddingBackend,
                 artifact_path: Path) -> None:
        self._connection = connection
        self._backend = backend
        self._index = FaissVectorIndex.load(
            artifact_path, dimension=backend.dimension,
            model_fingerprint=backend.model_fingerprint,
        )

    def search(self, query: str, limit: int) -> tuple[RetrievalHit, ...]:
        if not query or limit <= 0:
            return ()
        matches = self._index.search(self._backend.embed_query(query), limit)
        if not matches:
            return ()
        score_by_id = {vector_id: score for vector_id, score in matches}
        marks = ",".join("?" for _ in matches)
        rows = self._connection.execute(
            f"""SELECT v.vector_id,c.chunk_id,d.document_id,c.file_name,c.text,
                       c.page_start,c.page_end,c.section_title,c.ordinal,v.content_hash
                FROM vector_index_metadata v JOIN chunks c ON c.chunk_id=v.chunk_id
                JOIN documents d ON d.id=c.document_id
                WHERE v.vector_id IN ({marks}) AND d.status='indexed'""",
            tuple(score_by_id),
        ).fetchall()
        by_id = {int(row[0]): row for row in rows}
        return tuple(RetrievalHit(
            str(by_id[vector_id][1]), str(by_id[vector_id][2]),
            str(by_id[vector_id][3]), str(by_id[vector_id][4]), score,
            by_id[vector_id][5], by_id[vector_id][6], by_id[vector_id][7],
            int(by_id[vector_id][8]), str(by_id[vector_id][9]),
            self._backend.model_fingerprint,
        ) for vector_id, score in matches if vector_id in by_id)
