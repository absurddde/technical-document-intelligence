"""Run the offline real-document lexical, semantic, and hybrid retrieval pilot."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from app.domain.models import ParsedBlock
from app.infrastructure.config import load_config
from app.persistence.connection import Database
from app.persistence.repositories import ContentRepository, DocumentRepository
from app.processing.chunker import StructureAwareChunker
from app.retrieval.embeddings import EmbeddingBackend, SentenceTransformerEmbeddingBackend
from app.retrieval.lexical_index import SQLiteLexicalIndex
from app.retrieval.query import QueryNormalizer
from app.retrieval.retrievers import FaissSemanticRetriever, SQLiteLexicalRetriever
from app.retrieval.vector_index import FaissVectorIndex
from app.services.search_service import SearchService
from app.services.semantic_index_service import SemanticIndexService


PILOT_FILES = (
    "BarısBilgiliAnlatım.pdf",
    "Türkçe_Autonomous Cooperative Levels of-Multıple_Hete.UVS.docx",
    "Suru_Zekasi_OTAG_Sonuc_Raporu.pdf",
    "İnsansız Kara Araçları için Lİdar Teknolojisi 3B Haritalama.pdf",
    "1608994972_stm-elektronik-harbin-yeniden-yukselisi.pdf",
    "20220126_army-approach-to-ras_final.pdf",
    "Future Warfare and Critical Technology.pdf",
    "ENABLING MUM-T WITHIN ARMY FORMATIONS.pdf",
    "Derin Öğrenme ile Resim ve Videolarda Nesnelerin tanınması ve Takibi.pdf",
    "WRS20 Modern Maritime Communications.pdf",
)

V4_OVERRIDES = (
    "20220126_army-approach-to-ras_final.pdf",
    "1608994972_stm-elektronik-harbin-yeniden-yukselisi.pdf",
)


@dataclass(frozen=True, slots=True)
class RetrievalCase:
    case_id: str
    query: str
    query_language: str
    category: str
    expected_files: tuple[str, ...]
    expected_locator: str | None
    rationale: str


CASES = (
    RetrievalCase("Q01", "sürü davranışında uzlaşma yöntemleri ve stratejileri", "tr", "tr_same",
                  ("Suru_Zekasi_OTAG_Sonuc_Raporu.pdf",), "page 100", "Turkish swarm-consensus table and discussion."),
    RetrievalCase("Q02", "LİDAR nokta bulutu ile 3B haritalama", "tr", "tr_same",
                  ("İnsansız Kara Araçları için Lİdar Teknolojisi 3B Haritalama.pdf",), "page 10", "Turkish LIDAR mapping and point-cloud evidence."),
    RetrievalCase("Q03", "elektronik taarruz karıştırma ve aldatma sistemleri", "tr", "tr_same",
                  ("1608994972_stm-elektronik-harbin-yeniden-yukselisi.pdf",), "pages 3-4", "Turkish electronic-warfare taxonomy."),
    RetrievalCase("Q04", "derin öğrenme ile videoda nesne tanıma ve takip", "tr", "tr_same",
                  ("Derin Öğrenme ile Resim ve Videolarda Nesnelerin tanınması ve Takibi.pdf",), "page 1", "Turkish title, abstract, and Faster R-CNN discussion."),
    RetrievalCase("Q05", "British Army robotics autonomous systems 2035 vision", "en", "en_same",
                  ("20220126_army-approach-to-ras_final.pdf",), "pages 4-5", "Army RAS context and 2035 vision."),
    RetrievalCase("Q06", "MUM-T synchronized employment of soldiers manned and unmanned systems", "en", "en_same",
                  ("ENABLING MUM-T WITHIN ARMY FORMATIONS.pdf",), "page 8", "Document's MUM-T definition."),
    RetrievalCase("Q07", "GMDSS VHF MF HF maritime safety communications", "en", "en_same",
                  ("WRS20 Modern Maritime Communications.pdf",), "page 4", "Maritime radio safety systems and bands."),
    RetrievalCase("Q08", "hypersonic technology strategic stability", "en", "en_same",
                  ("Future Warfare and Critical Technology.pdf",), "page 157", "Critical-technology strategic-stability discussion."),
    RetrievalCase("Q09", "object recognition and tracking Faster R-CNN", "en", "en_same",
                  ("Derin Öğrenme ile Resim ve Videolarda Nesnelerin tanınması ve Takibi.pdf",), "page 1", "English abstract in the Turkish computer-vision paper."),
    RetrievalCase("Q10", "Britanya Ordusu robotik otonom sistemler 2035 vizyonu", "tr", "tr_to_en",
                  ("20220126_army-approach-to-ras_final.pdf",), "pages 4-5", "Cross-language Army RAS concept."),
    RetrievalCase("Q11", "ordu formasyonlarında insanlı insansız takım oluşturma", "tr", "tr_to_en",
                  ("ENABLING MUM-T WITHIN ARMY FORMATIONS.pdf",), "page 8", "Cross-language MUM-T definition."),
    RetrievalCase("Q12", "deniz emniyet haberleşmesinde VHF MF HF ve uydu", "tr", "tr_to_en",
                  ("WRS20 Modern Maritime Communications.pdf",), "pages 4 and 8", "Cross-language maritime communications."),
    RetrievalCase("Q13", "geleceğin savaşlarında hipersonik kritik teknolojiler", "tr", "tr_to_en",
                  ("Future Warfare and Critical Technology.pdf",), "page 157", "Cross-language critical technology."),
    RetrievalCase("Q14", "swarm intelligence consensus methods and strategies", "en", "en_to_tr",
                  ("Suru_Zekasi_OTAG_Sonuc_Raporu.pdf",), "page 100", "Cross-language Turkish swarm report."),
    RetrievalCase("Q15", "3D mapping LIDAR point cloud autonomous ground vehicle", "en", "en_to_tr",
                  ("İnsansız Kara Araçları için Lİdar Teknolojisi 3B Haritalama.pdf",), "page 10", "Cross-language Turkish LIDAR paper."),
    RetrievalCase("Q16", "electronic warfare jamming deception systems", "en", "en_to_tr",
                  ("1608994972_stm-elektronik-harbin-yeniden-yukselisi.pdf",), "pages 3-4", "Cross-language Turkish EW report."),
    RetrievalCase("Q17", "\"Human Machine Teams\" RAS", "en", "acronym_exact",
                  ("20220126_army-approach-to-ras_final.pdf",), "page 4", "Exact phrase plus RAS acronym."),
    RetrievalCase("Q18", "\"map_update_distance_thresh\" 8000", "en", "acronym_exact",
                  ("İnsansız Kara Araçları için Lİdar Teknolojisi 3B Haritalama.pdf",), "page 10", "Exact parameter and numeric sampling evidence."),
    RetrievalCase("Q19", "chlorophyll photosynthesis in tropical orchids", "en", "weak", (), None,
                  "Deliberately out of scope for the defence-technology pilot."),
    RetrievalCase("Q20", "Osmanlı vergi defterlerinde on altıncı yüzyıl tımar sistemi", "tr", "weak", (), None,
                  "Deliberately out of scope for the collection."),
)


class TimedEmbeddingBackend:
    """Measure embedding calls without changing the production backend contract."""

    def __init__(self, delegate: EmbeddingBackend) -> None:
        self.delegate = delegate
        self.document_seconds = 0.0
        self.query_seconds = 0.0
        self.document_calls = 0
        self.document_texts = 0

    @property
    def dimension(self) -> int:
        return self.delegate.dimension

    @property
    def model_fingerprint(self) -> str:
        return self.delegate.model_fingerprint

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        started = time.perf_counter()
        result = self.delegate.embed_documents(texts)
        self.document_seconds += time.perf_counter() - started
        self.document_calls += 1
        self.document_texts += len(texts)
        return result

    def embed_query(self, text: str) -> NDArray[np.float32]:
        started = time.perf_counter()
        result = self.delegate.embed_query(text)
        self.query_seconds += time.perf_counter() - started
        return result


def copy_database(source: Path, destination: Path) -> None:
    """Copy a live SQLite database consistently using SQLite's backup API."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    with sqlite3.connect(source) as source_connection, sqlite3.connect(destination) as target:
        source_connection.backup(target)


def load_blocks(connection: sqlite3.Connection, document_id: int) -> tuple[ParsedBlock, ...]:
    rows = connection.execute(
        """SELECT kind,text,page_number,section_title,heading_path,paragraph_index,
                  ocr_used,ocr_confidence FROM parsed_blocks
           WHERE document_id=? ORDER BY ordinal""", (document_id,),
    ).fetchall()
    return tuple(ParsedBlock(
        str(row[0]), str(row[1]), row[2], row[3], tuple(json.loads(row[4] or "[]")),
        row[5], bool(row[6]), row[7],
    ) for row in rows)


def apply_v4_overrides(database_path: Path, override_path: Path, chunk_size: int,
                       overlap: int) -> None:
    """Replace only the two accepted multi-column documents using persisted v4 blocks."""

    chunker = StructureAwareChunker(chunk_size, overlap)
    database = Database(database_path)
    with database.transaction() as target, sqlite3.connect(override_path) as source:
        source.row_factory = sqlite3.Row
        documents = {document.file_name: document for document in DocumentRepository(target).list_all()}
        for file_name in V4_OVERRIDES:
            target_document = documents[file_name]
            source_document = source.execute(
                "SELECT id FROM documents WHERE file_name=?", (file_name,)
            ).fetchone()
            if source_document is None:
                raise RuntimeError(f"Accepted v4 override missing: {file_name}")
            blocks = load_blocks(source, int(source_document["id"]))
            chunks = chunker.chunk(
                blocks, document_id=target_document.document_id,
                file_name=target_document.file_name, file_path=target_document.canonical_path,
            )
            ContentRepository(target).replace(target_document, blocks, chunks, "phase2-v4")
        target.execute(
            "UPDATE document_index_state SET pipeline_version='phase2-v4' WHERE document_id IN "
            "(SELECT id FROM documents WHERE status='indexed')"
        )


def metric(ranks: list[int | None], cutoff: int) -> float:
    return sum(rank is not None and rank <= cutoff for rank in ranks) / len(ranks) if ranks else 0.0


def metrics(ranks: list[int | None]) -> dict[str, float]:
    return {
        "evaluated": len(ranks), "hit_at_1": metric(ranks, 1),
        "hit_at_3": metric(ranks, 3), "hit_at_5": metric(ranks, 5),
        "mrr": sum(1 / rank for rank in ranks if rank is not None) / len(ranks) if ranks else 0.0,
    }


def expected_rank(file_names: Sequence[str], expected: tuple[str, ...]) -> int | None:
    return next((rank for rank, file_name in enumerate(file_names, 1) if file_name in expected), None)


def short_hit(hit: Any, rank: int) -> dict[str, Any]:
    return {
        "rank": rank, "chunk_id": hit.chunk_id, "document": hit.file_name,
        "page_start": hit.page_start, "page_end": hit.page_end,
        "section": hit.section_title, "score": hit.score,
        "snippet": " ".join(hit.text.split())[:180],
    }


def classify_failure(case: RetrievalCase, lexical_rank: int | None,
                     semantic_rank: int | None, hybrid_rank: int | None) -> str | None:
    if hybrid_rank is not None and hybrid_rank <= 3:
        return None
    if case.category in {"tr_to_en", "en_to_tr"} and (semantic_rank is None or semantic_rank > 3):
        return "multilingual semantic weakness"
    if lexical_rank is None and semantic_rank is not None and semantic_rank <= 3:
        return "lexical mismatch"
    if any(rank is not None and rank <= 3 for rank in (lexical_rank, semantic_rank)):
        return "ranking/fusion issue"
    if any(rank is not None and rank <= 5 for rank in (lexical_rank, semantic_rank, hybrid_rank)):
        return "chunk boundary or ranking issue"
    return "insufficient retrieved evidence"


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    config = load_config(arguments.config)
    if config.embedding.input_format != "raw":
        raise RuntimeError("The real BGE-M3 pilot requires raw input mode")
    source_names = set(PILOT_FILES)
    with sqlite3.connect(arguments.parsing_database) as source:
        actual = {str(row[0]) for row in source.execute(
            "SELECT file_name FROM documents WHERE status='indexed'"
        )}
    if actual != source_names:
        raise RuntimeError(f"Parsing database document set differs: {sorted(actual ^ source_names)}")

    prior_report = None
    if arguments.reuse_index:
        if not arguments.database.is_file() or not arguments.report.is_file():
            raise RuntimeError("--reuse-index requires an existing pilot database and report")
        prior_report = json.loads(arguments.report.read_text(encoding="utf-8"))
    else:
        copy_database(arguments.parsing_database, arguments.database)
        apply_v4_overrides(
            arguments.database, arguments.v4_overrides,
            config.chunking.chunk_size, config.chunking.overlap,
        )
    database = Database(arguments.database)
    with database.transaction() as connection:
        lexical_index = SQLiteLexicalIndex(connection)
        lexical_index.rebuild()
        lexical_index.integrity_check()

    model_started = time.perf_counter()
    backend = TimedEmbeddingBackend(SentenceTransformerEmbeddingBackend(
        config.embedding.model_path, device=config.embedding.device,
        batch_size=config.embedding.batch_size, input_format=config.embedding.input_format,
    ))
    model_load_seconds = time.perf_counter() - model_started
    artifact = arguments.index_dir / config.vector_index.artifact
    service = SemanticIndexService(
        database, backend, artifact, config.vector_index.version
    )
    first_started = time.perf_counter()
    synchronized = service.synchronize()
    synchronized_seconds = time.perf_counter() - first_started
    if prior_report is None:
        first_result = synchronized
        first_seconds = synchronized_seconds
        first_embedding_seconds = backend.document_seconds
        first_document_calls = backend.document_calls
        second_started = time.perf_counter()
        second_result = service.synchronize()
        second_seconds = time.perf_counter() - second_started
        second_embedding_calls = backend.document_calls - first_document_calls
    else:
        if synchronized.embedded:
            raise RuntimeError("Existing pilot index unexpectedly required new embeddings")
        prior_index = prior_report["index"]
        first_result = type(synchronized)(**prior_index["first_result"])
        first_seconds = float(prior_index["first_index_seconds"])
        first_embedding_seconds = float(prior_index["embedding_seconds"])
        second_result = type(synchronized)(**prior_index["second_result"])
        second_seconds = float(prior_index["second_index_seconds"])
        second_embedding_calls = int(prior_index["second_embedding_calls"])

    faiss_started = time.perf_counter()
    faiss_index = FaissVectorIndex.load(
        artifact, dimension=backend.dimension, model_fingerprint=backend.model_fingerprint,
    )
    faiss_load_seconds = time.perf_counter() - faiss_started

    with database.connect() as connection:
        document_count = int(connection.execute(
            "SELECT COUNT(*) FROM documents WHERE status='indexed'"
        ).fetchone()[0])
        chunk_count = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        lexical_count = int(connection.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0])
        vector_count = int(connection.execute("SELECT COUNT(*) FROM vector_index_metadata").fetchone()[0])
        lexical = SQLiteLexicalRetriever(connection)
        semantic = FaissSemanticRetriever(connection, backend, artifact)
        hybrid = SearchService(lexical, semantic, config.retrieval)
        normalizer = QueryNormalizer()
        rows: list[dict[str, Any]] = []
        latency = {"lexical": [], "semantic": [], "hybrid": []}
        for case in CASES:
            normalized = normalizer.normalize(case.query)
            started = time.perf_counter()
            lexical_hits = lexical.search(normalized.lexical_query, 5)
            latency["lexical"].append(time.perf_counter() - started)
            started = time.perf_counter()
            semantic_hits = semantic.search(normalized.semantic_query, 5)
            latency["semantic"].append(time.perf_counter() - started)
            started = time.perf_counter()
            hybrid_result = hybrid.search(case.query)
            latency["hybrid"].append(time.perf_counter() - started)
            hybrid_hits = hybrid_result.selected[:5]
            lexical_rank = expected_rank([hit.file_name for hit in lexical_hits], case.expected_files)
            semantic_rank = expected_rank([hit.file_name for hit in semantic_hits], case.expected_files)
            hybrid_rank = expected_rank([hit.file_name for hit in hybrid_hits], case.expected_files)
            details = [{
                "fused_rank": detail.final_rank, "chunk_id": detail.chunk_id,
                "document": detail.file_name, "page_start": detail.page_start,
                "page_end": detail.page_end, "section": detail.section_title,
                "lexical_rank": detail.lexical_rank, "semantic_rank": detail.semantic_rank,
                "rrf_contributions": detail.rrf_contributions,
                "rrf_score": detail.rrf_score, "exact_phrase_boost": detail.exact_phrase_boost,
                "acronym_boost": detail.acronym_boost, "snippet": detail.text_preview,
            } for detail in hybrid_result.details[:5]]
            rows.append({
                **asdict(case), "lexical_expected_rank": lexical_rank,
                "semantic_expected_rank": semantic_rank, "hybrid_expected_rank": hybrid_rank,
                "hybrid_insufficient_evidence": hybrid_result.insufficient_evidence,
                "lexical_top5": [short_hit(hit, rank) for rank, hit in enumerate(lexical_hits, 1)],
                "semantic_top5": [short_hit(hit, rank) for rank, hit in enumerate(semantic_hits, 1)],
                "hybrid_top5": details,
                "failure_cause": classify_failure(case, lexical_rank, semantic_rank, hybrid_rank)
                if case.expected_files else None,
            })

    known = [row for row in rows if row["expected_files"]]
    summary_metrics = {
        mode: metrics([row[f"{mode}_expected_rank"] for row in known])
        for mode in ("lexical", "semantic", "hybrid")
    }
    category_metrics = {
        category: {
            mode: metrics([row[f"{mode}_expected_rank"] for row in known if row["category"] == category])
            for mode in ("lexical", "semantic", "hybrid")
        }
        for category in sorted({row["category"] for row in known})
    }
    weak = [{
        "case_id": row["case_id"], "query": row["query"],
        "hybrid_insufficient_evidence": row["hybrid_insufficient_evidence"],
        "semantic_top_score": row["semantic_top5"][0]["score"] if row["semantic_top5"] else None,
        "hybrid_top_document": row["hybrid_top5"][0]["document"] if row["hybrid_top5"] else None,
    } for row in rows if row["category"] == "weak"]
    report = {
        "policy": "offline local BGE-M3 retrieval only; no LLM, network, download, or Phase 6",
        "test_cases": rows,
        "index": {
            "documents": document_count, "chunks": chunk_count,
            "lexical_rows": lexical_count, "semantic_vectors": vector_count,
            "input_format": config.embedding.input_format,
            "model_path": str(config.embedding.model_path),
            "model_fingerprint": backend.model_fingerprint,
            "dimension": backend.dimension,
            "first_result": asdict(first_result), "second_result": asdict(second_result),
            "first_index_seconds": first_seconds,
            "embedding_seconds": first_embedding_seconds,
            "second_index_seconds": second_seconds,
            "second_embedding_calls": second_embedding_calls,
            "faiss_load_seconds": faiss_load_seconds,
            "faiss_vector_count": int(faiss_index._index.ntotal),
            "model_load_seconds": (
                float(prior_report["index"]["model_load_seconds"])
                if prior_report is not None else model_load_seconds
            ),
            "evaluation_reused_existing_index": arguments.reuse_index,
            "evaluation_reuse_sync_seconds": synchronized_seconds if arguments.reuse_index else None,
        },
        "metrics": summary_metrics, "category_metrics": category_metrics,
        "weak_queries": weak,
        "latency_ms": {
            mode: round(sum(values) / len(values) * 1000, 3)
            for mode, values in latency.items()
        },
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with arguments.report.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "case_id", "query", "query_language", "category", "expected_files",
            "expected_locator", "lexical_expected_rank", "semantic_expected_rank",
            "hybrid_expected_rank", "hybrid_insufficient_evidence", "failure_cause",
        ))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key], ensure_ascii=False) if key == "expected_files" else row[key]
                             for key in writer.fieldnames})
    print(json.dumps({"index": report["index"], "metrics": summary_metrics,
                      "category_metrics": category_metrics, "weak_queries": weak,
                      "latency_ms": report["latency_ms"]}, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "config/config.toml")
    parser.add_argument("--parsing-database", type=Path,
                        default=root / "data/pilot_reports/parsing_pilot.sqlite")
    parser.add_argument("--v4-overrides", type=Path,
                        default=root / "data/pilot_reports/multi_column_recheck/parsing_pilot.sqlite")
    parser.add_argument("--database", type=Path,
                        default=root / "data/pilot_reports/retrieval_pilot.sqlite")
    parser.add_argument("--index-dir", type=Path,
                        default=root / "data/pilot_reports/retrieval_index")
    parser.add_argument("--report", type=Path,
                        default=root / "data/pilot_reports/retrieval_pilot.json")
    parser.add_argument("--reuse-index", action="store_true",
                        help="Reuse the existing verified vectors and rerun retrieval evaluation only")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
