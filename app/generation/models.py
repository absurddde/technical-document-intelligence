"""Structured generation, provenance, and validation models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    temperature: float = 0.1
    max_output_tokens: int = 512
    seed: int = 42


@dataclass(frozen=True, slots=True)
class SourceRecord:
    source_id: str
    chunk_id: str
    document_id: str
    file_name: str
    page_start: int | None
    page_end: int | None
    section_title: str | None
    text: str


@dataclass(frozen=True, slots=True)
class BuiltContext:
    rendered: str
    sources: tuple[SourceRecord, ...]

    @property
    def allowlist(self) -> frozenset[str]:
        return frozenset(source.source_id for source in self.sources)


@dataclass(frozen=True, slots=True)
class GeneratedClaim:
    claim_id: str
    text: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StructuredGeneration:
    answer: str
    claims: tuple[GeneratedClaim, ...]


@dataclass(frozen=True, slots=True)
class ClaimSourceMapping:
    claim_id: str
    claim_text: str
    marker: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConflictRecord:
    concept: str
    unit: str
    values: tuple[str, ...]
    source_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GenerationResult:
    final_paragraph: str
    insufficient_evidence: bool
    validation_status: str
    citation_markers: dict[str, tuple[str, ...]]
    sources: tuple[SourceRecord, ...]
    referenced_chunk_ids: tuple[str, ...]
    claim_mappings: tuple[ClaimSourceMapping, ...]
    conflicts: tuple[ConflictRecord, ...]
    attempts: int
    validation_errors: tuple[str, ...] = ()
