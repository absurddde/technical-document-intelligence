"""Validated grounded generation orchestration over Phase 4 search results."""

from __future__ import annotations

from app.generation.context import ContextBuilder
from app.generation.llm import LlmBackend
from app.generation.models import (ClaimSourceMapping, GenerationResult,
                                   GenerationSettings)
from app.generation.prompts import GROUNDED_SYSTEM_PROMPT, build_user_prompt
from app.generation.validation import (CitationValidator, ClaimCoverageValidator,
                                       ConflictDetector, ConflictValidator,
                                       GenerationValidationError,
                                       NumericClaimValidator,
                                       StructuredOutputParser,
                                       TurkishOutputValidator)
from app.infrastructure.config import GenerationConfig
from app.retrieval.models import SearchResult


INSUFFICIENT_EVIDENCE_MESSAGE = (
    "Seçilen dokümanlarda bu konu hakkında yeterli teknik bilgi bulunamadı."
)
VALIDATION_FAILURE_MESSAGE = (
    "Teknik açıklama kaynak doğrulamasından güvenli biçimde geçirilemedi."
)


class GenerationService:
    """Generate only from selected evidence and expose only validated output."""

    def __init__(self, backend: LlmBackend, config: GenerationConfig,
                 context_builder: ContextBuilder | None = None) -> None:
        self._backend = backend
        self._config = config
        self._context_builder = context_builder or ContextBuilder()
        self._parser = StructuredOutputParser()
        self._citations = CitationValidator()
        self._numbers = NumericClaimValidator()
        self._conflicts = ConflictDetector()
        self._conflict_validator = ConflictValidator()
        self._coverage = ClaimCoverageValidator()
        self._turkish = TurkishOutputValidator()

    def generate(self, search_result: SearchResult) -> GenerationResult:
        """Validate structured local-LLM output, with at most one configured retry."""

        context = self._context_builder.build(
            search_result.selected, self._config.max_context_chunks,
            self._config.max_context_characters,
        )
        if search_result.insufficient_evidence:
            return GenerationResult(
                INSUFFICIENT_EVIDENCE_MESSAGE, True, "insufficient_evidence", {},
                context.sources, (), (), (), 0,
            )

        conflicts = self._conflicts.detect(context)
        conflict_summary = "\n".join(
            f"{item.concept} ({item.unit}): "
            + ", ".join(f"{value} [{source}]" for value, source in zip(item.values, item.source_ids))
            for item in conflicts
        )
        settings = GenerationSettings(
            self._config.temperature, self._config.max_output_tokens,
            self._config.seed,
        )
        feedback = ""
        errors: list[str] = []
        maximum_attempts = 1 + self._config.max_regeneration_attempts
        for attempt in range(1, maximum_attempts + 1):
            raw = self._backend.generate(
                GROUNDED_SYSTEM_PROMPT,
                build_user_prompt(search_result.original_query, conflict_summary, feedback),
                context.rendered,
                settings,
            )
            try:
                generated = self._parser.parse(raw)
                self._citations.validate(generated, context)
                self._coverage.validate(generated)
                self._turkish.validate(generated)
                self._numbers.validate(generated, context)
                self._conflict_validator.validate(generated, conflicts)
                mappings = tuple(
                    ClaimSourceMapping(
                        claim.claim_id, claim.text, f"[{index}]", claim.source_ids
                    )
                    for index, claim in enumerate(generated.claims, 1)
                )
                markers = {mapping.marker: mapping.source_ids for mapping in mappings}
                referenced_ids = {
                    source_id for claim in generated.claims for source_id in claim.source_ids
                }
                referenced_chunks = tuple(
                    source.chunk_id for source in context.sources
                    if source.source_id in referenced_ids
                )
                return GenerationResult(
                    generated.answer, False, "valid", markers, context.sources,
                    referenced_chunks, mappings, conflicts, attempt, tuple(errors),
                )
            except GenerationValidationError as error:
                errors.append(str(error))
                feedback = str(error)

        return GenerationResult(
            VALIDATION_FAILURE_MESSAGE, False, "rejected", {}, context.sources,
            (), (), conflicts, maximum_attempts, tuple(errors),
        )
