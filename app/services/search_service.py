"""Phase 4 hybrid retrieval orchestration without generation or reranking."""

from __future__ import annotations

from app.infrastructure.config import RetrievalConfig
from app.retrieval.fusion import ReciprocalRankFusion
from app.retrieval.models import RetrievalDetails, SearchResult
from app.retrieval.query import (DisabledQueryTranslator, QueryNormalizer,
                                 QueryTranslator, translate_preserving_acronyms)
from app.retrieval.retrievers import RankedRetriever
from app.retrieval.selection import ContextSelector, DiverseContextSelector


class SearchService:
    """Normalize, retrieve independent branches, fuse, deduplicate, and select."""

    def __init__(self, lexical: RankedRetriever, semantic: RankedRetriever,
                 config: RetrievalConfig, *, normalizer: QueryNormalizer | None = None,
                 translator: QueryTranslator | None = None,
                 selector: ContextSelector | None = None) -> None:
        self._lexical = lexical
        self._semantic = semantic
        self._config = config
        self._normalizer = normalizer or QueryNormalizer()
        self._translator = translator or DisabledQueryTranslator()
        self._selector = selector or DiverseContextSelector()
        self._fusion = ReciprocalRankFusion(
            config.rrf_k, config.exact_phrase_boost, config.acronym_boost
        )

    def search(self, query: str) -> SearchResult:
        """Return structured evidence only; this phase never invokes an LLM."""

        normalized = self._normalizer.normalize(query)
        branches = [
            ("original_lexical", self._lexical.search(
                normalized.lexical_query, self._config.lexical_top_k)),
            ("original_semantic", self._semantic.search(
                normalized.semantic_query, self._config.semantic_top_k)),
        ]
        translated_query: str | None = None
        if self._config.translation_enabled:
            translated_query = translate_preserving_acronyms(
                self._translator, normalized
            )
            translated = self._normalizer.normalize(translated_query)
            branches.extend([
                ("translated_lexical", self._lexical.search(
                    translated.lexical_query, self._config.lexical_top_k)),
                ("translated_semantic", self._semantic.search(
                    translated.semantic_query, self._config.semantic_top_k)),
            ])

        fused = self._fusion.fuse(branches, normalized)
        selected = self._selector.select(
            fused[:self._config.fused_top_k], self._config.max_context_chunks
        )
        semantic_scores = [
            hit.score for branch, hits in branches if branch.endswith("semantic")
            for hit in hits
        ]
        has_direct_lexical_signal = any(
            candidate.exact_phrase_boost > 0 or candidate.acronym_boost > 0
            for candidate in selected
        )
        insufficient = (
            not selected
            or selected[0].fused_score < self._config.minimum_evidence_threshold
            or (
                not has_direct_lexical_signal
                and (
                    not semantic_scores
                    or max(semantic_scores) < self._config.minimum_semantic_score
                )
            )
        )
        details = tuple(self._detail(candidate) for candidate in selected)
        return SearchResult(
            normalized.original_query, normalized.semantic_query, translated_query,
            insufficient, selected, details,
        )

    @staticmethod
    def _detail(candidate) -> RetrievalDetails:
        branches = tuple(
            branch for branch in candidate.branch_ranks if branch.startswith("translated_")
        )
        rrf_score = sum(candidate.rrf_contributions.values())
        return RetrievalDetails(
            candidate.chunk_id, candidate.document_id, candidate.file_name,
            candidate.page_start, candidate.page_end, candidate.section_title,
            candidate.lexical_score, candidate.lexical_rank,
            candidate.semantic_score, candidate.semantic_rank, branches,
            candidate.exact_phrase_boost, candidate.acronym_boost,
            dict(candidate.rrf_contributions), rrf_score,
            candidate.final_rank or 0, candidate.text[:240],
            candidate.chunk_id,
        )
