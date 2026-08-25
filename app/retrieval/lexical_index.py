"""SQLite FTS5 lexical indexing and BM25 search."""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3

from app.persistence.schema import FTS5_TOKENIZER


@dataclass(frozen=True, slots=True)
class LexicalHit:
    chunk_id: str
    file_name: str
    section_title: str | None
    text: str
    score: float


class SQLiteLexicalIndex:
    """Search the external-content chunk FTS table without altering source text."""

    tokenizer = FTS5_TOKENIZER

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def search(self, query: str, limit: int = 20) -> tuple[LexicalHit, ...]:
        """Return best hits first; FTS5 BM25 uses lower (usually negative) scores."""

        if not query.strip() or limit <= 0:
            return ()
        rows = self._connection.execute(
            """SELECT c.chunk_id, c.file_name, c.section_title, c.text,
                      bm25(chunks_fts, 0.0, 2.0, 1.5, 1.0) AS score
               FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid
               JOIN documents d ON d.id = c.document_id
               WHERE chunks_fts MATCH ? AND d.status = 'indexed'
               ORDER BY score ASC, c.chunk_id ASC LIMIT ?""",
            (query, limit),
        ).fetchall()
        return tuple(LexicalHit(str(r[0]), str(r[1]), r[2], str(r[3]), float(r[4])) for r in rows)

    def rebuild(self) -> None:
        """Recover the external-content FTS index from authoritative chunk rows."""

        self._connection.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")

    def integrity_check(self) -> None:
        """Ask FTS5 to validate index/content consistency."""

        self._connection.execute("INSERT INTO chunks_fts(chunks_fts, rank) VALUES ('integrity-check', 1)")
