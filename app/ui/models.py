"""Presentation models and provenance mapping for the desktop UI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.generation.models import GenerationResult
from app.retrieval.models import SearchResult


STATUS_LABELS = {
    "new": "Yeni",
    "discovered": "Yeni",
    "processed": "İşlendi",
    "indexed": "İndeksli",
    "unchanged": "Değişmedi",
    "failed": "Başarısız",
    "missing": "Dosya bulunamadı",
    "duplicate": "Kopya",
}


@dataclass(frozen=True, slots=True)
class DocumentItem:
    path: Path
    file_name: str
    file_type: str
    status: str
    language: str = "—"
    error_code: str | None = None

    @property
    def status_label(self) -> str:
        return STATUS_LABELS.get(self.status, self.status)


@dataclass(frozen=True, slots=True)
class SourceItem:
    citation: str
    source_id: str
    chunk_id: str
    file_name: str
    page_label: str
    section: str
    preview: str
    file_path: Path | None = None


@dataclass(frozen=True, slots=True)
class RetrievalItem:
    rank: int
    file_name: str
    locator: str
    chunk_id: str
    lexical_rank: int | None
    semantic_rank: int | None
    lexical_score: float | None
    semantic_score: float | None
    rrf_score: float
    contributions: dict[str, float]


@dataclass(frozen=True, slots=True)
class AnswerItem:
    answer: str
    sources: tuple[SourceItem, ...]
    retrieval: tuple[RetrievalItem, ...]
    insufficient_evidence: bool
    validation_status: str


def _page_label(start: int | None, end: int | None) -> str:
    if start is None:
        return "Sayfa bilgisi yok"
    return f"Sayfa {start}" if end in (None, start) else f"Sayfa {start}–{end}"


def _locator(page_start: int | None, page_end: int | None, section: str | None) -> str:
    values = [_page_label(page_start, page_end)]
    if section:
        values.append(section)
    return " · ".join(values)


def map_answer(
    search: SearchResult, generation: GenerationResult,
    source_paths: dict[str, Path] | None = None,
) -> AnswerItem:
    """Map validated backend provenance into compact UI records."""

    citations_by_source: dict[str, list[str]] = {}
    rendered = generation.final_paragraph
    for mapping in generation.claim_mappings:
        for source_id in mapping.source_ids:
            citations_by_source.setdefault(source_id, []).append(mapping.marker)
        if mapping.claim_text in rendered:
            rendered = rendered.replace(
                mapping.claim_text, f"{mapping.claim_text} {mapping.marker}", 1
            )

    selected_path = source_paths or {}
    sources = tuple(
        SourceItem(
            " ".join(dict.fromkeys(citations_by_source.get(source.source_id, ()))) or "—",
            source.source_id,
            source.chunk_id,
            source.file_name,
            _page_label(source.page_start, source.page_end),
            source.section_title or "—",
            " ".join(source.text.split())[:500],
            selected_path.get(source.chunk_id),
        )
        for source in generation.sources
        if source.source_id in citations_by_source
    )
    retrieval = tuple(
        RetrievalItem(
            detail.final_rank, detail.file_name,
            _locator(detail.page_start, detail.page_end, detail.section_title),
            detail.chunk_id, detail.lexical_rank, detail.semantic_rank,
            detail.lexical_score, detail.semantic_score, detail.rrf_score,
            dict(detail.rrf_contributions),
        )
        for detail in search.details
    )
    return AnswerItem(
        rendered, sources, retrieval, generation.insufficient_evidence,
        generation.validation_status,
    )


def document_from_row(row: Any) -> DocumentItem:
    """Translate a SQLite document/status row without exposing DB details to widgets."""

    status = str(row["status"])
    chunk_count = int(row["chunk_count"] or 0)
    vector_count = int(row["vector_count"] or 0)
    if status == "indexed" and chunk_count and vector_count < chunk_count:
        status = "processed"
    return DocumentItem(
        Path(str(row["canonical_path"])), str(row["file_name"]),
        str(row["file_type"]).upper(), status, str(row["language"] or "—"),
        str(row["last_error_code"]) if row["last_error_code"] else None,
    )
