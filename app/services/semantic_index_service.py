"""Incremental embedding persistence and exact FAISS artifact rebuilding."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sqlite3

import numpy as np

from app.persistence.connection import Database
from app.persistence.schema import initialize_schema
from app.retrieval.embeddings import EmbeddingBackend
from app.retrieval.vector_index import FaissVectorIndex


@dataclass(frozen=True, slots=True)
class SemanticIndexResult:
    embedded: int
    reused: int
    removed: int
    total: int


class SemanticIndexService:
    """Embed only missing/changed active chunks, then atomically replace FAISS files."""

    def __init__(self, database: Database, backend: EmbeddingBackend,
                 artifact_path: Path, index_version: int = 1) -> None:
        self._database = database
        self._backend = backend
        self._artifact = artifact_path.resolve(strict=False)
        self._version = index_version

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def synchronize(self) -> SemanticIndexResult:
        """Synchronize metadata and rebuild the deletable exact index from cached vectors."""

        embedded = reused = 0
        with self._database.transaction() as connection:
            initialize_schema(connection)
            active = connection.execute(
                """SELECT c.chunk_id, c.text FROM chunks c JOIN documents d ON d.id=c.document_id
                   WHERE d.status='indexed' ORDER BY c.chunk_id"""
            ).fetchall()
            active_ids = {str(row[0]) for row in active}
            before = int(connection.execute("SELECT COUNT(*) FROM vector_index_metadata").fetchone()[0])
            if active_ids:
                marks = ",".join("?" for _ in active_ids)
                connection.execute(f"DELETE FROM vector_index_metadata WHERE chunk_id NOT IN ({marks})", tuple(active_ids))
            else:
                connection.execute("DELETE FROM vector_index_metadata")

            current = {str(r[0]): r for r in connection.execute(
                "SELECT chunk_id, content_hash, model_fingerprint, vector_dimension FROM vector_index_metadata"
            ).fetchall()}
            pending: list[tuple[str, str, str]] = []
            for row in active:
                chunk_id, text = str(row[0]), str(row[1])
                digest = self._hash(text)
                old = current.get(chunk_id)
                if old and old[1] == digest and old[2] == self._backend.model_fingerprint and old[3] == self._backend.dimension:
                    continue
                pending.append((chunk_id, text, digest))

            next_id = int(connection.execute("SELECT COALESCE(MAX(vector_id), 0) FROM vector_index_metadata").fetchone()[0]) + 1
            to_embed: list[tuple[str, str, str, int]] = []
            pending_vectors: dict[str, bytes] = {}
            deferred_duplicates: list[tuple[str, str, int]] = []
            for chunk_id, text, digest in pending:
                cached = connection.execute(
                    """SELECT embedding FROM embedding_cache WHERE content_hash=?
                       AND model_fingerprint=? AND vector_dimension=?""",
                    (digest, self._backend.model_fingerprint, self._backend.dimension),
                ).fetchone()
                vector_id_row = connection.execute(
                    "SELECT vector_id FROM vector_index_metadata WHERE chunk_id=?", (chunk_id,)
                ).fetchone()
                vector_id = int(vector_id_row[0]) if vector_id_row else next_id
                if not vector_id_row: next_id += 1
                if cached:
                    self._upsert(connection, chunk_id, vector_id, digest, bytes(cached[0]))
                    reused += 1
                elif digest in {item[2] for item in to_embed}:
                    deferred_duplicates.append((chunk_id, digest, vector_id))
                else:
                    to_embed.append((chunk_id, text, digest, vector_id))
            if to_embed:
                vectors = self._backend.embed_documents([item[1] for item in to_embed])
                if vectors.shape != (len(to_embed), self._backend.dimension):
                    raise ValueError("Embedding backend returned an invalid shape")
                for item, vector in zip(to_embed, vectors):
                    blob = np.asarray(vector, dtype=np.float32).tobytes()
                    pending_vectors[item[2]] = blob
                    connection.execute(
                        """INSERT OR REPLACE INTO embedding_cache(content_hash,model_fingerprint,
                           vector_dimension,embedding) VALUES(?,?,?,?)""",
                        (item[2], self._backend.model_fingerprint, self._backend.dimension, blob),
                    )
                    self._upsert(connection, item[0], item[3], item[2], blob)
                embedded = len(to_embed)
            for chunk_id, digest, vector_id in deferred_duplicates:
                self._upsert(connection, chunk_id, vector_id, digest, pending_vectors[digest])
                reused += 1

            rows = connection.execute(
                """SELECT vector_id, embedding FROM vector_index_metadata
                   WHERE model_fingerprint=? AND vector_dimension=? ORDER BY vector_id""",
                (self._backend.model_fingerprint, self._backend.dimension),
            ).fetchall()
            connection.execute(
                """INSERT INTO vector_index_state(artifact, model_fingerprint, vector_dimension, index_version)
                   VALUES(?,?,?,?) ON CONFLICT(artifact) DO UPDATE SET model_fingerprint=excluded.model_fingerprint,
                   vector_dimension=excluded.vector_dimension,index_version=excluded.index_version,updated_at=CURRENT_TIMESTAMP""",
                (self._artifact.name, self._backend.model_fingerprint, self._backend.dimension, self._version),
            )
            total = len(rows)
            removed = max(0, before - total)

        index = FaissVectorIndex(self._backend.dimension, self._backend.model_fingerprint)
        if rows:
            ids = [int(row[0]) for row in rows]
            vectors = np.vstack([np.frombuffer(row[1], dtype=np.float32) for row in rows])
            index.add_or_update(ids, vectors)
        temporary = self._artifact.with_suffix(self._artifact.suffix + ".tmp")
        index.save(temporary)
        temporary.replace(self._artifact)
        temporary.with_suffix(temporary.suffix + ".meta").replace(
            self._artifact.with_suffix(self._artifact.suffix + ".meta")
        )
        return SemanticIndexResult(embedded, reused, removed, total)

    def _upsert(self, connection: sqlite3.Connection, chunk_id: str, vector_id: int,
                digest: str, embedding: bytes) -> None:
        connection.execute(
            """INSERT INTO vector_index_metadata(chunk_id,vector_id,content_hash,embedding,
               model_fingerprint,vector_dimension,index_artifact,index_version)
               VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(chunk_id) DO UPDATE SET vector_id=excluded.vector_id,
               content_hash=excluded.content_hash,embedding=excluded.embedding,
               model_fingerprint=excluded.model_fingerprint,vector_dimension=excluded.vector_dimension,
               embedded_at=CURRENT_TIMESTAMP,index_artifact=excluded.index_artifact,index_version=excluded.index_version""",
            (chunk_id, vector_id, digest, embedding, self._backend.model_fingerprint,
             self._backend.dimension, self._artifact.name, self._version),
        )
