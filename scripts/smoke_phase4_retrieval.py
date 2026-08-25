"""Run a local BGE-M3 Phase 4 retrieval smoke test on synthetic chunks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from app.domain.models import DocumentChunk, FileFingerprint
from app.infrastructure.config import RetrievalConfig
from app.persistence.connection import Database
from app.persistence.repositories import ContentRepository, DocumentRepository
from app.persistence.schema import initialize_schema
from app.retrieval.embeddings import SentenceTransformerEmbeddingBackend
from app.retrieval.retrievers import FaissSemanticRetriever, SQLiteLexicalRetriever
from app.services.search_service import SearchService
from app.services.semantic_index_service import SemanticIndexService


MODEL_PATH = Path(r"D:\proje_Staj\models\embedding\bge-m3")


def fingerprint(file_name: str, digest: str) -> FileFingerprint:
    """Create deterministic synthetic inventory metadata."""

    return FileFingerprint(
        Path(file_name), file_name, file_name, "pdf", 1, 1, digest
    )


def main() -> None:
    """Index English fixtures and retrieve them with Turkish queries."""

    groups = {
        "inertial": [
            "The inertial navigation system estimates position, velocity and orientation using onboard inertial sensors.",
            "The radar antenna operates in the microwave frequency range and transmits electromagnetic signals.",
            "The propulsion system generates thrust required for vehicle acceleration during flight.",
        ],
        "guidance": [
            "The guidance system calculates steering commands to direct the vehicle toward the target.",
            "The structural frame carries mechanical loads during operation.",
            "The power supply distributes electrical energy to onboard subsystems.",
        ],
    }
    backend = SentenceTransformerEmbeddingBackend(
        MODEL_PATH, device="cpu", input_format="raw"
    )
    with TemporaryDirectory(prefix="phase4-smoke-") as temporary:
        root = Path(temporary)
        database = Database(root / "app.db")
        with database.transaction() as connection:
            initialize_schema(connection)
            documents = DocumentRepository(connection)
            content = ContentRepository(connection)
            for index, (group, passages) in enumerate(groups.items()):
                document = documents.save_fingerprint(
                    fingerprint(f"{group}.pdf", f"{index + 1:064x}")
                )
                documents.ensure_index_state(document.id, None, reset_pipeline=True)
                chunks = tuple(DocumentChunk(
                    document.document_id, document.file_name, document.canonical_path,
                    item + 1, item + 1, group.title(), item, item,
                    f"{group}-{chr(65 + item)}", "en", passage, False, None,
                ) for item, passage in enumerate(passages))
                content.replace(document, (), chunks, "phase4-smoke")

        artifact = root / "indexes" / "semantic.faiss"
        SemanticIndexService(database, backend, artifact).synchronize()
        connection = database.connect()
        try:
            service = SearchService(
                SQLiteLexicalRetriever(connection),
                FaissSemanticRetriever(connection, backend, artifact),
                RetrievalConfig(lexical_top_k=6, semantic_top_k=6,
                                fused_top_k=6, max_context_chunks=6),
            )
            output = {}
            for query in ("ataletsel seyrüsefer sistemi", "güdüm sistemi"):
                result = service.search(query)
                output[query] = [
                    {
                        "rank": item.final_rank,
                        "chunk_id": item.chunk_id,
                        "semantic_score": item.semantic_score,
                        "fused_score": item.fused_score,
                    }
                    for item in result.selected
                ]
            print(json.dumps(output, ensure_ascii=True, indent=2))
        finally:
            connection.close()


if __name__ == "__main__":
    main()
