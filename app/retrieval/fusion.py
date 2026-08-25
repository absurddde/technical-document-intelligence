"""Branch-aware Reciprocal Rank Fusion and bounded lexical signals."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import re

from app.retrieval.models import FusedCandidate, RetrievalHit
from app.retrieval.query import NormalizedQuery


class ReciprocalRankFusion:
    """Fuse independent ranked lists without mixing incomparable raw scores."""

    def __init__(self, rrf_k: int = 60, exact_phrase_boost: float = 0.005,
                 acronym_boost: float = 0.005) -> None:
        if rrf_k <= 0 or exact_phrase_boost < 0 or acronym_boost < 0:
            raise ValueError("RRF and boost settings must be non-negative")
        self._k = rrf_k
        ceiling = 1.0 / (rrf_k + 1)
        self._phrase_boost = min(exact_phrase_boost, ceiling)
        self._acronym_boost = min(acronym_boost, ceiling)

    def fuse(self, branches: Sequence[tuple[str, Sequence[RetrievalHit]]],
             query: NormalizedQuery) -> tuple[FusedCandidate, ...]:
        candidates: dict[str, FusedCandidate] = {}
        for branch, hits in branches:
            for rank, hit in enumerate(hits, 1):
                digest = hit.text_sha256 or hashlib.sha256(hit.text.encode("utf-8")).hexdigest()
                candidate = candidates.setdefault(hit.chunk_id, FusedCandidate(
                    hit.chunk_id, hit.document_id, hit.file_name, hit.text,
                    hit.page_start, hit.page_end, hit.section_title, hit.ordinal, digest,
                    model_fingerprint=hit.model_fingerprint,
                ))
                contribution = 1.0 / (self._k + rank)
                candidate.branch_ranks[branch] = rank
                candidate.rrf_contributions[branch] = contribution
                candidate.fused_score += contribution
                if branch.endswith("lexical"):
                    if candidate.lexical_rank is None or rank < candidate.lexical_rank:
                        candidate.lexical_rank, candidate.lexical_score = rank, hit.score
                else:
                    if candidate.semantic_rank is None or rank < candidate.semantic_rank:
                        candidate.semantic_rank, candidate.semantic_score = rank, hit.score
                        candidate.model_fingerprint = hit.model_fingerprint

        for candidate in candidates.values():
            folded = candidate.text.casefold()
            if any(phrase.casefold() in folded for phrase in query.exact_phrases):
                candidate.exact_phrase_boost = self._phrase_boost
            if any(re.search(rf"(?<!\w){re.escape(acronym)}(?!\w)", candidate.text, re.IGNORECASE)
                   for acronym in query.detected_acronyms):
                candidate.acronym_boost = self._acronym_boost
            candidate.fused_score += candidate.exact_phrase_boost + candidate.acronym_boost
        return tuple(sorted(candidates.values(), key=lambda item: (-item.fused_score, item.chunk_id)))
