"""Typed domain models used by the indexing inventory layer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

Language = Literal["tr", "en", "mixed", "unknown"]


class DocumentStatus(StrEnum):
    """Persistent lifecycle status of a document path."""

    DISCOVERED = "discovered"
    INDEXED = "indexed"
    FAILED = "failed"
    MISSING = "missing"
    DUPLICATE = "duplicate"


class IndexAction(StrEnum):
    """Action selected for a path during an inventory scan."""

    NEW = "new"
    UNCHANGED = "unchanged"
    MODIFIED = "modified"
    DUPLICATE = "duplicate"
    MISSING = "missing"


class RunStatus(StrEnum):
    """Status of an indexing inventory run."""

    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class FileFingerprint:
    """Content identity and filesystem metadata for a supported file."""

    path: Path
    canonical_path: str
    file_name: str
    file_type: str
    file_size: int
    modified_ns: int
    sha256: str


@dataclass(frozen=True, slots=True)
class Document:
    """Persistent metadata for one physical document path."""

    id: int
    document_id: str
    canonical_path: str
    file_name: str
    file_type: str
    file_size: int
    modified_ns: int
    sha256: str
    status: DocumentStatus
    duplicate_of_id: int | None


@dataclass(frozen=True, slots=True)
class IndexPlanItem:
    """A deterministic indexing decision for one document path."""

    action: IndexAction
    canonical_path: str
    fingerprint: FileFingerprint | None
    existing_document_id: int | None = None
    duplicate_of_path: str | None = None


@dataclass(frozen=True, slots=True)
class ParsedBlock:
    """One provenance-preserving structural unit extracted from a document."""

    kind: Literal["paragraph", "heading", "table"]
    text: str
    page_number: int | None = None
    section_title: str | None = None
    heading_path: tuple[str, ...] = ()
    paragraph_index: int | None = None
    ocr_used: bool = False
    ocr_confidence: float | None = None


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Parsed local document and its stable content identity."""

    blocks: tuple[ParsedBlock, ...]
    document_hash: str
    pipeline_version: str


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """Structure-aware text chunk with retrieval provenance metadata."""

    document_id: str
    file_name: str
    file_path: str
    page_start: int | None
    page_end: int | None
    section_title: str | None
    paragraph_start: int | None
    paragraph_end: int | None
    chunk_id: str
    language: Language
    text: str
    ocr_used: bool
    ocr_confidence: float | None
