"""Phase 1 SQLite schema."""

from __future__ import annotations

import sqlite3


SCHEMA_VERSION = 1


def initialize_schema(connection: sqlite3.Connection) -> None:
    """Create the Phase 1 schema idempotently."""

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_info (
            version INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            document_id TEXT NOT NULL UNIQUE,
            canonical_path TEXT NOT NULL UNIQUE,
            file_name TEXT NOT NULL,
            file_type TEXT NOT NULL CHECK (file_type IN ('pdf', 'docx')),
            file_size INTEGER NOT NULL CHECK (file_size >= 0),
            modified_ns INTEGER NOT NULL,
            sha256 TEXT NOT NULL CHECK (length(sha256) = 64),
            status TEXT NOT NULL CHECK (
                status IN ('discovered', 'indexed', 'failed', 'missing', 'duplicate')
            ),
            duplicate_of_id INTEGER REFERENCES documents(id),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CHECK (duplicate_of_id IS NULL OR duplicate_of_id <> id)
        );

        CREATE INDEX IF NOT EXISTS idx_documents_sha256 ON documents(sha256);
        CREATE INDEX IF NOT EXISTS idx_documents_duplicate ON documents(duplicate_of_id);
        CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);

        CREATE TABLE IF NOT EXISTS indexing_runs (
            id INTEGER PRIMARY KEY,
            root_path TEXT NOT NULL,
            started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            finished_at TEXT,
            status TEXT NOT NULL CHECK (
                status IN ('running', 'completed', 'partial', 'failed')
            ),
            discovered_count INTEGER NOT NULL DEFAULT 0,
            new_count INTEGER NOT NULL DEFAULT 0,
            unchanged_count INTEGER NOT NULL DEFAULT 0,
            modified_count INTEGER NOT NULL DEFAULT 0,
            duplicate_count INTEGER NOT NULL DEFAULT 0,
            missing_count INTEGER NOT NULL DEFAULT 0,
            failed_count INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS document_index_state (
            document_id INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
            run_id INTEGER REFERENCES indexing_runs(id),
            inventory_status TEXT NOT NULL DEFAULT 'pending' CHECK (
                inventory_status IN ('pending', 'ready', 'failed', 'skipped')
            ),
            parse_status TEXT NOT NULL DEFAULT 'pending',
            chunk_status TEXT NOT NULL DEFAULT 'pending',
            embedding_status TEXT NOT NULL DEFAULT 'pending',
            lexical_status TEXT NOT NULL DEFAULT 'pending',
            vector_status TEXT NOT NULL DEFAULT 'pending',
            pipeline_version TEXT NOT NULL DEFAULT 'phase1',
            last_error_code TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    row = connection.execute("SELECT version FROM schema_info LIMIT 1").fetchone()
    if row is None:
        connection.execute(
            "INSERT INTO schema_info(version) VALUES (?)", (SCHEMA_VERSION,)
        )
    elif row[0] != SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported schema version: {row[0]}")

