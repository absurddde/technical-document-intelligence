"""Run the production Phase 2 parsing pipeline on an explicit local pilot set."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import statistics
import time
from typing import Any

from app.domain.models import DocumentStatus, IndexAction, IndexPlanItem, ParsedDocument
from app.indexing.planner import IncrementalIndexPlanner
from app.ingestion.docx_parser import DocxParser
from app.ingestion.fingerprint import fingerprint_file
from app.ingestion.ocr import TesseractOcr
from app.ingestion.pdf_parser import PdfParser
from app.infrastructure.config import load_config
from app.persistence.connection import Database
from app.persistence.repositories import DocumentRepository
from app.persistence.schema import initialize_schema
from app.processing.cache import ParsedDocumentCache
from app.processing.chunker import StructureAwareChunker
from app.processing.cleaner import clean_text
from app.processing.language import detect_language
from app.services.document_processing_service import DocumentProcessingService


DEFAULT_FILES = (
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


class CountingCache:
    """Observe production parsed-cache hits and misses without changing semantics."""

    def __init__(self, delegate: ParsedDocumentCache) -> None:
        self.delegate = delegate
        self.hits = 0
        self.misses = 0

    def load(self, document_hash: str, pipeline_version: str) -> ParsedDocument | None:
        parsed = self.delegate.load(document_hash, pipeline_version)
        if parsed is None:
            self.misses += 1
        else:
            self.hits += 1
        return parsed

    def save(self, document: ParsedDocument) -> None:
        self.delegate.save(document)


class ObservedPdfParser(PdfParser):
    """Collect page-level telemetry while delegating all parsing to PdfParser."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.by_file: dict[str, dict[str, Any]] = {}
        self._current: dict[str, Any] | None = None

    def parse(self, path: Path, document_hash: str) -> ParsedDocument:
        telemetry: dict[str, Any] = {
            "native_character_counts": [], "ocr_required_pages": [],
            "ocr_attempted_pages": [], "ocr_succeeded_pages": [],
            "ocr_failed_pages": [], "ocr_character_counts": [], "ocr_errors": [],
        }
        self._current = telemetry
        started = time.perf_counter()
        try:
            parsed = super().parse(path, document_hash)
            telemetry["duration_seconds"] = time.perf_counter() - started
            return parsed
        finally:
            self.by_file[path.name] = telemetry
            self._current = None

    def needs_ocr(self, text: str) -> bool:
        required = super().needs_ocr(text)
        if self._current is not None:
            page_number = len(self._current["native_character_counts"]) + 1
            self._current["native_character_counts"].append(len(text or ""))
            if required:
                self._current["ocr_required_pages"].append(page_number)
        return required

    def _ocr_page(self, path: Path, page_index: int):
        if self._current is not None:
            self._current["ocr_attempted_pages"].append(page_index + 1)
        try:
            result = super()._ocr_page(path, page_index)
        except Exception as error:
            if self._current is not None:
                self._current["ocr_failed_pages"].append(page_index + 1)
                self._current["ocr_errors"].append(f"page {page_index + 1}: {type(error).__name__}: {error}")
            raise
        if self._current is not None:
            if result.text.strip():
                self._current["ocr_succeeded_pages"].append(page_index + 1)
            else:
                self._current["ocr_failed_pages"].append(page_index + 1)
                self._current["ocr_errors"].append(f"page {page_index + 1}: OCR returned empty text")
            self._current["ocr_character_counts"].append(len(result.text))
        return result


class ObservedParser:
    """Measure parser calls and retain metadata-only parsed results for inspection."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls: Counter[str] = Counter()
        self.durations: dict[str, float] = {}
        self.results: dict[str, ParsedDocument] = {}

    def parse(self, path: Path, document_hash: str) -> ParsedDocument:
        self.calls[path.name] += 1
        started = time.perf_counter()
        parsed = self.delegate.parse(path, document_hash)
        self.durations[path.name] = time.perf_counter() - started
        self.results[path.name] = parsed
        return parsed


def save_inventory(database: Database, plan: tuple[IndexPlanItem, ...]) -> None:
    """Persist explicit pilot fingerprints like the production inventory service."""

    with database.transaction() as connection:
        initialize_schema(connection)
        repository = DocumentRepository(connection)
        for item in plan:
            if item.fingerprint is None or item.action not in {IndexAction.NEW, IndexAction.MODIFIED}:
                continue
            document = repository.save_fingerprint(item.fingerprint, DocumentStatus.DISCOVERED)
            repository.ensure_index_state(document.id, None, reset_pipeline=True)


def build_plan(database: Database, fingerprints: tuple[Any, ...]) -> tuple[IndexPlanItem, ...]:
    """Use the production incremental planner for the explicit pilot files."""

    with database.connect() as connection:
        initialize_schema(connection)
        existing = DocumentRepository(connection).list_all()
    return IncrementalIndexPlanner().build_plan(fingerprints, existing)


def rss_mb() -> float | None:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
    except ImportError:
        return None


def pdf_page_count(path: Path) -> int:
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        return len(pdf.pages)


def quality_statistics(blocks: tuple[Any, ...], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate metadata-only quality warnings from parsed blocks and chunks."""

    chunk_sizes = [len(chunk["text"]) for chunk in chunks]
    chunk_hashes = Counter(hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest() for chunk in chunks)
    normalized_blocks: dict[str, set[int]] = defaultdict(set)
    for block in blocks:
        if block.page_number is not None and 3 <= len(block.text) <= 180:
            normalized = re.sub(r"\s+", " ", block.text).strip().casefold()
            normalized_blocks[normalized].add(block.page_number)
    repeated_short_blocks = sum(len(pages) >= 3 for pages in normalized_blocks.values())
    all_text = "\n".join(block.text for block in blocks)
    mojibake_count = sum(all_text.count(marker) for marker in ("Ã", "Ä", "Å", "�"))
    empty_chunks = sum(not chunk["text"].strip() for chunk in chunks)
    return {
        "chunk_count": len(chunks),
        "chunk_size_min": min(chunk_sizes) if chunk_sizes else 0,
        "chunk_size_median": round(statistics.median(chunk_sizes), 1) if chunk_sizes else 0,
        "chunk_size_max": max(chunk_sizes) if chunk_sizes else 0,
        "empty_chunks": empty_chunks,
        "small_chunks_under_80": sum(size < 80 for size in chunk_sizes),
        "oversized_chunks_over_1200": sum(size > 1200 for size in chunk_sizes),
        "duplicate_chunk_texts": sum(count - 1 for count in chunk_hashes.values() if count > 1),
        "repeated_short_blocks_across_3_pages": repeated_short_blocks,
        "mojibake_or_replacement_markers": mojibake_count,
        "hyphen_ended_blocks": sum(block.text.rstrip().endswith("-") for block in blocks),
    }


def database_document_stats(database: Database, file_name: str) -> tuple[tuple[Any, ...], list[dict[str, Any]]]:
    """Read persisted blocks and chunks for one document without logging content."""

    with database.connect() as connection:
        document = connection.execute("SELECT id FROM documents WHERE file_name=?", (file_name,)).fetchone()
        if document is None:
            return (), []
        block_rows = connection.execute(
            "SELECT kind,text,page_number,section_title,ocr_used,ocr_confidence FROM parsed_blocks WHERE document_id=? ORDER BY ordinal",
            (document["id"],),
        ).fetchall()
        chunk_rows = connection.execute(
            "SELECT text,language,ocr_used,ocr_confidence,page_start,page_end FROM chunks WHERE document_id=? ORDER BY ordinal",
            (document["id"],),
        ).fetchall()
    from app.domain.models import ParsedBlock

    blocks = tuple(ParsedBlock(
        row["kind"], row["text"], row["page_number"], row["section_title"],
        ocr_used=bool(row["ocr_used"]), ocr_confidence=row["ocr_confidence"],
    ) for row in block_rows)
    return blocks, [dict(row) for row in chunk_rows]


def build_document_report(
    path: Path, database: Database, pdf_observer: ObservedPdfParser,
    parser_durations: dict[str, float], parser_called: bool,
) -> dict[str, Any]:
    blocks, chunks = database_document_stats(database, path.name)
    text = "\n".join(block.text for block in blocks)
    cleaned_text = clean_text(text)
    kinds = Counter(block.kind for block in blocks)
    languages = Counter(chunk["language"] for chunk in chunks)
    telemetry = pdf_observer.by_file.get(path.name, {})
    pages = pdf_page_count(path) if path.suffix.casefold() == ".pdf" else None
    native_counts = telemetry.get("native_character_counts", [])
    ocr_counts = telemetry.get("ocr_character_counts", [])
    extracted_characters = (
        sum(count for index, count in enumerate(native_counts, 1)
            if index not in telemetry.get("ocr_required_pages", []))
        + sum(ocr_counts)
    ) if native_counts else len(text)
    report = {
        "file_name": path.name,
        "success": bool(blocks and chunks),
        "detected_language": detect_language(text),
        "chunk_language_distribution": dict(languages),
        "pages": pages,
        "docx_blocks": len(blocks) if path.suffix.casefold() == ".docx" else None,
        "extracted_character_count": extracted_characters,
        "cleaned_character_count": len(cleaned_text),
        "paragraph_blocks": kinds["paragraph"],
        "heading_blocks": kinds["heading"],
        "table_blocks": kinds["table"],
        "sections_detected": len({block.section_title for block in blocks if block.section_title}),
        "ocr_pages_attempted": telemetry.get("ocr_attempted_pages", []),
        "ocr_pages_succeeded": telemetry.get("ocr_succeeded_pages", []),
        "ocr_pages_failed": telemetry.get("ocr_failed_pages", []),
        "ocr_errors": telemetry.get("ocr_errors", []),
        "native_text_pages": pages - len(telemetry.get("ocr_required_pages", [])) if pages is not None else None,
        "processing_duration_seconds": round(parser_durations.get(path.name, 0.0), 3),
        "parser_called": parser_called,
        "warnings": [],
        **quality_statistics(blocks, chunks),
    }
    if report["mojibake_or_replacement_markers"]:
        report["warnings"].append("garbled_or_replacement_characters_observed")
    if report["repeated_short_blocks_across_3_pages"]:
        report["warnings"].append("repeated_short_blocks_remain_after_margin_cleaning")
    if report["duplicate_chunk_texts"]:
        report["warnings"].append("duplicate_chunk_text_observed")
    if report["oversized_chunks_over_1200"]:
        report["warnings"].append("chunk_exceeds_configured_size")
    if report["empty_chunks"]:
        report["warnings"].append("empty_chunk_observed")
    if path.name == "BarısBilgiliAnlatım.pdf" and chunks:
        ocr_text = "\n".join(block.text for block in blocks if block.ocr_used)
        tokens = re.findall(r"[^\W\d_]+", ocr_text, re.UNICODE)
        one_letter_ratio = sum(len(token) == 1 for token in tokens) / len(tokens) if tokens else 1.0
        report["ocr_sanity"] = {
            "nonempty": bool(ocr_text.strip()),
            "character_count": len(ocr_text),
            "alphanumeric_ratio": round(sum(char.isalnum() for char in ocr_text) / len(ocr_text), 3) if ocr_text else 0,
            "one_letter_token_ratio": round(one_letter_ratio, 3),
            "meaningful_text": len(tokens) >= 20 and one_letter_ratio < 0.35,
        }
    return report


def write_reports(report_dir: Path, report: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "parsing_pilot.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = report["documents"]
    if rows:
        fields = list(rows[0])
        with (report_dir / "parsing_pilot.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                flat = dict(row)
                for key, value in flat.items():
                    if isinstance(value, (dict, list)):
                        flat[key] = json.dumps(value, ensure_ascii=False)
                writer.writerow(flat)


def run(directory: Path, filenames: tuple[str, ...], report_dir: Path,
        force_cached_reprocess: bool = False) -> dict[str, Any]:
    """Run two production pipeline passes over an explicit local file set."""

    root = directory.resolve(strict=True)
    prior_report_path = report_dir / "parsing_pilot.json"
    prior_report = None
    if force_cached_reprocess and prior_report_path.is_file():
        prior_report = json.loads(prior_report_path.read_text(encoding="utf-8"))
    paths = tuple((root / name).resolve(strict=True) for name in filenames)
    if any(path.parent != root for path in paths):
        raise ValueError("Every pilot file must be directly inside the supplied directory")
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "config.toml")
    database = Database(report_dir / "parsing_pilot.sqlite")
    cache = CountingCache(ParsedDocumentCache(report_dir / "parsing_pilot_cache"))
    ocr = TesseractOcr(config.ocr.languages, config.ocr.tesseract_command)
    pdf = ObservedPdfParser(
        ocr, min_text_characters=config.parsing.pdf_min_text_characters,
        min_alphanumeric_ratio=config.parsing.pdf_min_alphanumeric_ratio,
        dpi=config.ocr.dpi,
        remove_margins=config.parsing.remove_repeated_headers_footers,
        margin_lines=config.parsing.repeated_margin_lines,
        repeated_page_ratio=config.parsing.repeated_page_ratio,
        pipeline_version=config.parsing.pipeline_version,
    )
    observed_pdf = ObservedParser(pdf)
    observed_docx = ObservedParser(DocxParser(config.parsing.pipeline_version))
    chunker = StructureAwareChunker(config.chunking.chunk_size, config.chunking.overlap)
    service = DocumentProcessingService(
        database, observed_pdf, observed_docx, chunker, cache,
        config.parsing.pipeline_version,
    )

    first_started = time.perf_counter()
    fingerprints = tuple(fingerprint_file(path) for path in paths)
    first_plan = build_plan(database, fingerprints)
    if force_cached_reprocess:
        first_plan = tuple(
            IndexPlanItem(
                IndexAction.MODIFIED, item.canonical_path, item.fingerprint,
                item.existing_document_id, item.duplicate_of_path,
            ) if item.action is IndexAction.UNCHANGED else item
            for item in first_plan
        )
    save_inventory(database, first_plan)
    rss_before = rss_mb()
    first_result = service.process(first_plan)
    rss_after = rss_mb()
    first_duration = time.perf_counter() - first_started
    documents = [build_document_report(
        path, database, pdf, {**observed_pdf.durations, **observed_docx.durations},
        bool(observed_pdf.calls[path.name] or observed_docx.calls[path.name]),
    ) for path in paths]
    if prior_report is not None:
        prior_documents = {item["file_name"]: item for item in prior_report["documents"]}
        preserved_fields = (
            "ocr_pages_attempted", "ocr_pages_succeeded", "ocr_pages_failed",
            "ocr_errors", "ocr_sanity", "native_text_pages",
            "extracted_character_count", "processing_duration_seconds",
        )
        for document in documents:
            prior = prior_documents.get(document["file_name"], {})
            if document["parser_called"]:
                continue
            for field in preserved_fields:
                if field in prior:
                    document[field] = prior[field]

    hits_before, misses_before = cache.hits, cache.misses
    second_started = time.perf_counter()
    second_fingerprints = tuple(fingerprint_file(path) for path in paths)
    second_plan = build_plan(database, second_fingerprints)
    second_result = service.process(second_plan)
    direct_cache_hits = sum(
        cache.load(fingerprint.sha256, config.parsing.pipeline_version) is not None
        for fingerprint in second_fingerprints
    )
    second_duration = time.perf_counter() - second_started

    report = {
        "policy": "production parsing/OCR/cleaning/language/chunk/cache only; no embedding, FAISS, retrieval, LLM, or network",
        "source_directory": str(root),
        "documents": documents,
        "summary": {
            "run_mode": "forced_cached_reprocess" if force_cached_reprocess else "full_parse",
            "document_count": len(paths),
            "successes": sum(item["success"] for item in documents),
            "failures": sum(not item["success"] for item in documents),
            "total_pdf_pages": sum(item["pages"] or 0 for item in documents),
            "total_chunks": sum(item["chunk_count"] for item in documents),
            "ocr_pages_attempted": sum(len(item["ocr_pages_attempted"]) for item in documents),
            "ocr_pages_succeeded": sum(len(item["ocr_pages_succeeded"]) for item in documents),
            "ocr_pages_failed": sum(len(item["ocr_pages_failed"]) for item in documents),
            "first_run_seconds": round(first_duration, 3),
            "first_run_result": asdict(first_result),
            "first_run_cache_hits": hits_before,
            "first_run_cache_misses": misses_before,
            "second_run_seconds": round(second_duration, 3),
            "second_run_result": asdict(second_result),
            "second_plan_actions": dict(Counter(item.action.value for item in second_plan)),
            "second_run_direct_cache_hits": direct_cache_hits,
            "second_run_new_cache_misses": cache.misses - misses_before,
            "parser_calls_after_second_run": sum(observed_pdf.calls.values()) + sum(observed_docx.calls.values()),
            "ocr_calls_after_second_run": sum(len(value["ocr_attempted_pages"]) for value in pdf.by_file.values()),
            "rss_before_processing_mb": rss_before,
            "rss_after_processing_mb": rss_after,
            "baseline_full_run": (
                prior_report["summary"].get("baseline_full_run") or prior_report["summary"]
                if prior_report else None
            ),
        },
    }
    write_reports(report_dir, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("\nDOCUMENTS")
    for item in documents:
        print(
            f"{item['file_name']} | success={item['success']} | language={item['detected_language']} "
            f"| chunks={item['chunk_count']} | OCR={len(item['ocr_pages_succeeded'])}/{len(item['ocr_pages_attempted'])} "
            f"| seconds={item['processing_duration_seconds']} | warnings={','.join(item['warnings']) or 'none'}"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--file", action="append", dest="files", help="Exact filename; repeat for each document")
    parser.add_argument(
        "--report-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "pilot_reports",
    )
    parser.add_argument(
        "--force-cached-reprocess", action="store_true",
        help="Rebuild persisted chunks from exact-version parsed cache without repeating OCR",
    )
    arguments = parser.parse_args()
    run(
        arguments.directory, tuple(arguments.files or DEFAULT_FILES),
        arguments.report_dir.resolve(), arguments.force_cached_reprocess,
    )


if __name__ == "__main__":
    main()
