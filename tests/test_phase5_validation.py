import json

import pytest

from app.generation.context import ContextBuilder
from app.generation.models import GeneratedClaim, StructuredGeneration
from app.generation.validation import (CitationValidator, ConflictDetector,
    ConflictValidator, ClaimCoverageValidator, GenerationValidationError, NumericClaimValidator,
    StructuredOutputParser, extract_numeric_expressions)
from app.retrieval.models import FusedCandidate


def candidate(chunk_id: str, text: str, document="doc"):
    return FusedCandidate(chunk_id, document, f"{document}.pdf", text, 1, 1,
                          "Technical", 0, chunk_id, fused_score=.1)


def generation(text: str, sources=("SOURCE_01",)) -> StructuredGeneration:
    return StructuredGeneration(text, (GeneratedClaim("CLAIM_01", text, tuple(sources)),))


def test_citation_allowlist_success_and_rejections() -> None:
    context = ContextBuilder().build((candidate("one", "Radar sistemi."),), 1)
    CitationValidator().validate(generation("Radar sistemi açıklanmıştır."), context)
    with pytest.raises(GenerationValidationError, match="Invented"):
        CitationValidator().validate(generation("İddia.", ("SOURCE_99",)), context)
    with pytest.raises(GenerationValidationError, match="Malformed"):
        CitationValidator().validate(generation("İddia.", ("source-one",)), context)


def test_structured_parser_rejects_uncited_claim_and_arbitrary_prose() -> None:
    parser = StructuredOutputParser()
    with pytest.raises(GenerationValidationError, match="JSON"):
        parser.parse("ordinary prose")
    raw = json.dumps({"answer": "Bir iddia.", "claims": [
        {"claim_id": "CLAIM_01", "text": "Bir iddia.", "source_ids": []}
    ]})
    with pytest.raises(GenerationValidationError, match="requires citations"):
        parser.parse(raw)


def test_answer_text_outside_cited_claims_is_rejected() -> None:
    generated = StructuredGeneration(
        "Radar sistemi açıklanmıştır. Desteksiz performans yüksektir.",
        (GeneratedClaim("CLAIM_01", "Radar sistemi açıklanmıştır.", ("SOURCE_01",)),),
    )
    with pytest.raises(GenerationValidationError, match="not covered"):
        ClaimCoverageValidator().validate(generated)


def test_numeric_extraction_and_value_unit_context_validation() -> None:
    values = extract_numeric_expressions("Menzil 150 km, frekans 3.5 GHz, hız Mach 2, ağırlık 25 kg, başarı 95%, kalınlık 0.5 mm ve bant 10–12 GHz, CEP 5 m.")
    assert {(item.value, item.unit) for item in values} >= {
        ("150", "km"), ("3.5", "ghz"), ("2", "mach"), ("25", "kg"),
        ("95", "%"), ("0.5", "mm"), ("10–12", "ghz"), ("5", "m"),
    }
    context = ContextBuilder().build((candidate("one", "Sistemin menzili 150 km olarak verilir."),), 1)
    NumericClaimValidator().validate(generation("Sistemin menzili 150 km olarak belirtilmiştir."), context)


def test_unrelated_or_unsupported_number_is_rejected() -> None:
    context = ContextBuilder().build((candidate("one", "Farklı bileşenin ağırlığı 150 kg'dır."),), 1)
    with pytest.raises(GenerationValidationError, match="Unsupported numeric"):
        NumericClaimValidator().validate(generation("Sistemin menzili 150 km'dir."), context)
    with pytest.raises(GenerationValidationError, match="Unsupported numeric"):
        NumericClaimValidator().validate(generation("Sistemin ağırlığı 180 kg'dır."), context)


def test_obvious_numeric_conflict_detection_and_validation() -> None:
    context = ContextBuilder().build((
        candidate("a", "Sistem menzili 150 km olarak belirtilmiştir.", "a"),
        candidate("b", "Sistem menzili 180 km olarak belirtilmiştir.", "b"),
    ), 8)
    conflicts = ConflictDetector().detect(context)
    assert len(conflicts) == 1
    assert conflicts[0].values == ("150", "180")
    valid = StructuredGeneration(
        "Kaynaklarda sistem menzili için 150 km ve 180 km değerleri verilmektedir.",
        (
            GeneratedClaim("CLAIM_01", "Sistem menzili 150 km olarak verilmektedir.", ("SOURCE_01",)),
            GeneratedClaim("CLAIM_02", "Sistem menzili 180 km olarak verilmektedir.", ("SOURCE_02",)),
        ),
    )
    ConflictValidator().validate(valid, conflicts)
    with pytest.raises(GenerationValidationError, match="omitted"):
        ConflictValidator().validate(generation("Sistem menzili 180 km'dir.", ("SOURCE_02",)), conflicts)
