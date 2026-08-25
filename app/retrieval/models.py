"""Structured Phase 4 retrieval and debug models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    chunk_id: str
    document_id: str
    file_name: str
    text: str
    score: float
    page_start: int | None = None
    page_end: int | None = None
    section_title: str | None = None
    ordinal: int = 0
    text_sha256: str = ""
    model_fingerprint: str | None = None


@dataclass(slots=True)
class FusedCandidate:
    chunk_id: str
    document_id: str
    file_name: str
    text: str
    page_start: int | None
    page_end: int | None
    section_title: str | None
    ordinal: int
    text_sha256: str
    lexical_score: float | None = None
    lexical_rank: int | None = None
    semantic_score: float | None = None
    semantic_rank: int | None = None
    model_fingerprint: str | None = None
    branch_ranks: dict[str, int] = field(default_factory=dict)
    rrf_contributions: dict[str, float] = field(default_factory=dict)
    exact_phrase_boost: float = 0.0
    acronym_boost: float = 0.0
    fused_score: float = 0.0
    final_rank: int | None = None


@dataclass(frozen=True, slots=True)
class RetrievalDetails:
    chunk_id: str
    document_id: str
    file_name: str
    page_start: int | None
    page_end: int | None
    section_title: str | None
    lexical_score: float | None
    lexical_rank: int | None
    semantic_score: float | None
    semantic_rank: int | None
    translation_branches: tuple[str, ...]
    exact_phrase_boost: float
    acronym_boost: float
    rrf_contributions: dict[str, float]
    rrf_score: float
    final_rank: int
    text_preview: str
    passage_reference: str


@dataclass(frozen=True, slots=True)
class SearchResult:
    original_query: str
    semantic_query: str
    translated_query: str | None
    insufficient_evidence: bool
    selected: tuple[FusedCandidate, ...]
    details: tuple[RetrievalDetails, ...]
