"""Structured output, citation, numeric claim, and conflict validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import unicodedata

from app.generation.models import (BuiltContext, ConflictRecord, GeneratedClaim,
                                   StructuredGeneration)


class GenerationValidationError(ValueError):
    """Raised when generated content cannot safely be shown."""


class StructuredOutputParser:
    """Parse and strictly type-check the JSON generation contract."""

    def parse(self, raw: str) -> StructuredGeneration:
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as error:
            raise GenerationValidationError("Output is not valid JSON") from error
        if not isinstance(value, dict) or set(value) != {"answer", "claims"}:
            raise GenerationValidationError("Output must contain only answer and claims")
        answer, claims_value = value["answer"], value["claims"]
        if not isinstance(answer, str) or not answer.strip() or "\n" in answer.strip():
            raise GenerationValidationError("Answer must be one non-empty paragraph")
        if not isinstance(claims_value, list) or not claims_value:
            raise GenerationValidationError("At least one cited claim is required")
        claims: list[GeneratedClaim] = []
        seen: set[str] = set()
        for item in claims_value:
            if not isinstance(item, dict) or set(item) != {"claim_id", "text", "source_ids"}:
                raise GenerationValidationError("Malformed claim object")
            claim_id, text, source_ids = item["claim_id"], item["text"], item["source_ids"]
            if not isinstance(claim_id, str) or not re.fullmatch(r"CLAIM_[A-Za-z0-9_-]+", claim_id):
                raise GenerationValidationError("Malformed claim ID")
            if claim_id in seen or not isinstance(text, str) or not text.strip():
                raise GenerationValidationError("Duplicate claim ID or empty claim")
            if not isinstance(source_ids, list) or not source_ids or not all(isinstance(source, str) for source in source_ids):
                raise GenerationValidationError("Every factual claim requires citations")
            seen.add(claim_id)
            claims.append(GeneratedClaim(claim_id, text.strip(), tuple(source_ids)))
        return StructuredGeneration(answer.strip(), tuple(claims))


class CitationValidator:
    """Restrict all generated citations to request-local SOURCE IDs."""

    def validate(self, generation: StructuredGeneration, context: BuiltContext) -> None:
        for claim in generation.claims:
            for source_id in claim.source_ids:
                if not re.fullmatch(r"SOURCE_\d{2,}", source_id):
                    raise GenerationValidationError(f"Malformed source reference: {source_id}")
                if source_id not in context.allowlist:
                    raise GenerationValidationError(f"Invented source reference: {source_id}")


@dataclass(frozen=True, slots=True)
class NumericExpression:
    raw: str
    value: str
    unit: str
    start: int
    end: int


NUMERIC_PATTERN = re.compile(
    r"(?P<mach>\bMach\s*(?P<mach_value>\d+(?:[.,]\d+)?))"
    r"|(?P<regular>\b(?P<value>\d+(?:[.,]\d+)?(?:\s*[–-]\s*\d+(?:[.,]\d+)?)?)"
    r"\s*(?P<unit>GHz|MHz|kHz|Hz|km|kg|mm|cm|ms|m|%|°C|dB)(?!\w))",
    re.IGNORECASE,
)
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "for", "to", "and",
    "bir", "bu", "ve", "ile", "icin", "için", "olarak", "degeri", "değeri",
    "km", "kg", "ghz", "mhz", "khz", "hz", "mm", "cm", "ms", "m", "mach", "db",
}


def extract_numeric_expressions(text: str) -> tuple[NumericExpression, ...]:
    """Extract common technical values and their units conservatively."""

    found: list[NumericExpression] = []
    for match in NUMERIC_PATTERN.finditer(unicodedata.normalize("NFC", text)):
        if match.group("mach"):
            value, unit = match.group("mach_value"), "mach"
        else:
            value, unit = match.group("value"), match.group("unit").casefold()
        normalized_value = re.sub(r"\s+", "", value.replace(",", ".")).replace("-", "–")
        found.append(NumericExpression(match.group(0), normalized_value, unit,
                                       match.start(), match.end()))
    return tuple(found)


def _context_terms(text: str, expression: NumericExpression) -> set[str]:
    window = text[max(0, expression.start - 45):expression.end + 45].casefold()
    return {
        token for token in re.findall(r"[^\W\d_]{3,}", window, re.UNICODE)
        if token not in STOPWORDS
    }


class NumericClaimValidator:
    """Require value, unit, citation, and nearby-term support for every number."""

    def validate(self, generation: StructuredGeneration, context: BuiltContext) -> None:
        source_map = {source.source_id: source for source in context.sources}
        for claim in generation.claims:
            for expression in extract_numeric_expressions(claim.text):
                claim_terms = _context_terms(claim.text, expression)
                supported = False
                for source_id in claim.source_ids:
                    source = source_map[source_id]
                    for candidate in extract_numeric_expressions(source.text):
                        same_value = candidate.value == expression.value
                        same_unit = candidate.unit == expression.unit
                        term_overlap = claim_terms & _context_terms(source.text, candidate)
                        if same_value and same_unit and term_overlap:
                            supported = True
                            break
                    if supported:
                        break
                if not supported:
                    raise GenerationValidationError(
                        f"Unsupported numeric claim in {claim.claim_id}: {expression.raw}"
                    )


class ConflictDetector:
    """Detect obvious same-concept/unit numeric disagreements across sources."""

    def detect(self, context: BuiltContext) -> tuple[ConflictRecord, ...]:
        groups: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for source in context.sources:
            for expression in extract_numeric_expressions(source.text):
                before = source.text[max(0, expression.start - 35):expression.start]
                words = [word.casefold() for word in re.findall(r"[^\W\d_]{3,}", before, re.UNICODE)
                         if word.casefold() not in STOPWORDS]
                if not words:
                    continue
                concept = " ".join(words[-2:])
                groups.setdefault((concept, expression.unit), []).append(
                    (expression.value, source.source_id)
                )
        conflicts: list[ConflictRecord] = []
        for (concept, unit), entries in groups.items():
            evidence: dict[str, str] = {}
            for value, source_id in entries:
                evidence.setdefault(value, source_id)
            values = tuple(evidence)
            if len(values) > 1:
                source_ids = tuple(evidence.values())
                conflicts.append(ConflictRecord(concept, unit, values, source_ids))
        return tuple(conflicts)


class ConflictValidator:
    """Ensure every pre-detected conflicting value and source remains represented."""

    def validate(self, generation: StructuredGeneration,
                 conflicts: tuple[ConflictRecord, ...]) -> None:
        for conflict in conflicts:
            for value, source_id in zip(conflict.values, conflict.source_ids):
                represented = any(
                    source_id in claim.source_ids
                    and any(expr.value == value and expr.unit == conflict.unit
                            for expr in extract_numeric_expressions(claim.text))
                    for claim in generation.claims
                )
                if not represented:
                    raise GenerationValidationError(
                        f"Conflict value {value} {conflict.unit} from {source_id} was omitted"
                    )


class ClaimCoverageValidator:
    """Require structured claims to account for the complete answer text."""

    def validate(self, generation: StructuredGeneration) -> None:
        compact_answer = " ".join(generation.answer.casefold().split())
        uncovered = compact_answer
        for claim in generation.claims:
            compact_claim = " ".join(claim.text.casefold().split())
            if compact_claim not in compact_answer:
                raise GenerationValidationError(
                    f"Claim {claim.claim_id} is not represented in the answer"
                )
            uncovered = uncovered.replace(compact_claim, " ", 1)
        remaining_words = {
            word for word in re.findall(r"[^\W\d_]{3,}", uncovered, re.UNICODE)
            if word not in {"ve", "ile", "ancak", "ise", "diğer", "diger"}
        }
        if remaining_words:
            raise GenerationValidationError("Answer contains factual text not covered by claims")


class TurkishOutputValidator:
    """Reject clearly English prose while allowing embedded technical terminology."""

    _ENGLISH_MARKERS = frozenset({
        "the", "and", "using", "with", "from", "during", "system", "estimates",
        "position", "orientation", "documented", "range", "source", "uses",
    })
    _TURKISH_MARKERS = frozenset({
        "ve", "ile", "bir", "bu", "sistem", "sistemin", "kaynak", "kaynakta",
        "belgelenen", "belirtilmektedir", "kullanarak", "kullanır", "menzil",
        "konum", "yönelim", "seyrüsefer", "sırasında", "olarak", "değeri",
    })

    def validate(self, generation: StructuredGeneration) -> None:
        words = re.findall(r"[^\W\d_]+", generation.answer.casefold(), re.UNICODE)
        english = sum(word in self._ENGLISH_MARKERS for word in words)
        turkish = sum(word in self._TURKISH_MARKERS for word in words)
        if english >= 2 and english > turkish:
            raise GenerationValidationError("Answer must be written in Turkish")
