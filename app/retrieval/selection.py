"""Retrieval deduplication and bounded context selection."""

from __future__ import annotations

from typing import Protocol

from app.retrieval.models import FusedCandidate


def _token_overlap(left: str, right: str) -> float:
    a, b = set(left.casefold().split()), set(right.casefold().split())
    return len(a & b) / max(1, min(len(a), len(b)))


class ContextSelector(Protocol):
    def select(self, candidates: tuple[FusedCandidate, ...],
               maximum: int) -> tuple[FusedCandidate, ...]: ...


class DiverseContextSelector:
    """Drop exact and adjacent near-duplicates while retaining distinct sources."""

    def __init__(self, adjacent_overlap_threshold: float = 0.80) -> None:
        self._threshold = adjacent_overlap_threshold

    def select(self, candidates: tuple[FusedCandidate, ...],
               maximum: int) -> tuple[FusedCandidate, ...]:
        selected: list[FusedCandidate] = []
        seen_chunks: set[str] = set()
        seen_hashes: set[str] = set()
        for candidate in candidates:
            if candidate.chunk_id in seen_chunks or candidate.text_sha256 in seen_hashes:
                continue
            adjacent_duplicate = any(
                prior.document_id == candidate.document_id
                and abs(prior.ordinal - candidate.ordinal) <= 1
                and _token_overlap(prior.text, candidate.text) >= self._threshold
                for prior in selected
            )
            if adjacent_duplicate:
                continue
            selected.append(candidate)
            seen_chunks.add(candidate.chunk_id)
            seen_hashes.add(candidate.text_sha256)
            if len(selected) >= maximum:
                break
        for rank, candidate in enumerate(selected, 1):
            candidate.final_rank = rank
        return tuple(selected)
