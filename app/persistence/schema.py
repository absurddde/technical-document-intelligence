"""Phase 1 SQLite schema."""

from __future__ import annotations

import sqlite3


SCHEMA_VERSION = 3

# Kept explicit so tokenizer behaviour is reviewable and testable. remove_diacritics=0
# preserves Turkish characters in the indexed terms.
FTS5_TOKENIZER = "unicode61 remove_diacritics 0"


def initialize_schema(connection: sqlite3.Connection) -> None:
    """Create or safely migrate the local schema to the current version."""

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

        CREATE TABLE IF NOT EXISTS parsed_blocks (
            id INTEGER PRIMARY KEY,
            document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            kind TEXT NOT NULL CHECK (kind IN ('paragraph', 'heading', 'table')),
            text TEXT NOT NULL,
            page_number INTEGER,
            section_title TEXT,
            heading_path TEXT NOT NULL DEFAULT '[]',
            paragraph_index INTEGER,
            ocr_used INTEGER NOT NULL DEFAULT 0,
            ocr_confidence REAL,
            UNIQUE(document_id, ordinal)
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_id TEXT NOT NULL UNIQUE,
            ordinal INTEGER NOT NULL,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            page_start INTEGER,
            page_end INTEGER,
            section_title TEXT,
            paragraph_start INTEGER,
            paragraph_end INTEGER,
            language TEXT NOT NULL CHECK (language IN ('tr', 'en', 'mixed', 'unknown')),
            text TEXT NOT NULL,
            ocr_used INTEGER NOT NULL DEFAULT 0,
            ocr_confidence REAL,
            UNIQUE(document_id, ordinal)
        );
        CREATE INDEX IF NOT EXISTS idx_parsed_blocks_document ON parsed_blocks(document_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            chunk_id UNINDEXED, file_name, section_title, text,
            content='chunks', content_rowid='id',
            tokenize='unicode61 remove_diacritics 0'
        );
        CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, chunk_id, file_name, section_title, text)
            VALUES (new.id, new.chunk_id, new.file_name, new.section_title, new.text);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, file_name, section_title, text)
            VALUES ('delete', old.id, old.chunk_id, old.file_name, old.section_title, old.text);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, file_name, section_title, text)
            VALUES ('delete', old.id, old.chunk_id, old.file_name, old.section_title, old.text);
            INSERT INTO chunks_fts(rowid, chunk_id, file_name, section_title, text)
            VALUES (new.id, new.chunk_id, new.file_name, new.section_title, new.text);
        END;

        CREATE TABLE IF NOT EXISTS vector_index_metadata (
            chunk_id TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
            vector_id INTEGER NOT NULL UNIQUE,
            content_hash TEXT NOT NULL,
            embedding BLOB NOT NULL,
            model_fingerprint TEXT NOT NULL,
            vector_dimension INTEGER NOT NULL CHECK(vector_dimension > 0),
            embedded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            index_artifact TEXT NOT NULL,
            index_version INTEGER NOT NULL DEFAULT 1
        );
        CREATE INDEX IF NOT EXISTS idx_vector_content_model
            ON vector_index_metadata(content_hash, model_fingerprint, vector_dimension);
        CREATE TABLE IF NOT EXISTS embedding_cache (
            content_hash TEXT NOT NULL,
            model_fingerprint TEXT NOT NULL,
            vector_dimension INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            embedded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(content_hash, model_fingerprint, vector_dimension)
        );
        CREATE TABLE IF NOT EXISTS vector_index_state (
            artifact TEXT PRIMARY KEY,
            model_fingerprint TEXT NOT NULL,
            vector_dimension INTEGER NOT NULL,
            index_version INTEGER NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    row = connection.execute("SELECT version FROM schema_info LIMIT 1").fetchone()
    if row is None:
        connection.execute(
            "INSERT INTO schema_info(version) VALUES (?)", (SCHEMA_VERSION,)
        )
    elif row[0] in (1, 2, SCHEMA_VERSION):
        if row[0] < 3:
            connection.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
        connection.execute("UPDATE schema_info SET version = ?", (SCHEMA_VERSION,))
    else:
        raise RuntimeError(f"Unsupported schema version: {row[0]}")
