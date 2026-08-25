import json

from app.generation.llm import FakeLlmBackend
from app.infrastructure.config import GenerationConfig
from app.retrieval.models import FusedCandidate, SearchResult
from app.services.generation_service import (GenerationService,
    INSUFFICIENT_EVIDENCE_MESSAGE, VALIDATION_FAILURE_MESSAGE)


def candidate(chunk_id="one", text="Sistemin menzili 150 km olarak belirtilmiştir.",
              document="doc", page=4, section="Range"):
    return FusedCandidate(chunk_id, document, f"{document}.pdf", text, page, page,
                          section, 0, chunk_id, fused_score=.1, final_rank=1)


def search_result(chunks, insufficient=False):
    return SearchResult("menzil", "menzil", None, insufficient, tuple(chunks), ())


def output(answer, claims):
    return json.dumps({"answer": answer, "claims": claims}, ensure_ascii=False)


def test_fake_backend_is_deterministic_and_insufficient_evidence_bypasses_it() -> None:
    fake = FakeLlmBackend(("unused",))
    result = GenerationService(fake, GenerationConfig()).generate(
        search_result((candidate(),), insufficient=True)
    )
    assert result.final_paragraph == INSUFFICIENT_EVIDENCE_MESSAGE
    assert result.insufficient_evidence is True and result.attempts == 0
    assert fake.calls == []
    assert fake.backend_name == "fake-local" and fake.model_fingerprint == "fake-v1"


def test_generation_service_integration_and_future_ui_source_mapping() -> None:
    answer = "Sistemin menzili 150 km olarak belirtilmiştir."
    raw = output(answer, [{"claim_id": "CLAIM_01", "text": answer,
                           "source_ids": ["SOURCE_01"]}])
    fake = FakeLlmBackend((raw,))
    result = GenerationService(fake, GenerationConfig()).generate(search_result((candidate(),)))

    assert result.validation_status == "valid" and result.final_paragraph == answer
    assert "\n" not in result.final_paragraph
    assert result.citation_markers == {"[1]": ("SOURCE_01",)}
    assert result.referenced_chunk_ids == ("one",)
    assert result.sources[0].page_start == 4 and result.sources[0].section_title == "Range"
    assert result.claim_mappings[0].claim_id == "CLAIM_01"
    system_prompt, _, context, settings = fake.calls[0]
    assert "UNTRUSTED DATA" in system_prompt and "SOURCE_01" in context
    assert settings.temperature == .1


def test_unsupported_number_regenerates_once_then_succeeds() -> None:
    bad = output("Sistemin menzili 180 km'dir.", [{
        "claim_id": "CLAIM_01", "text": "Sistemin menzili 180 km'dir.",
        "source_ids": ["SOURCE_01"]}])
    good_text = "Sistemin menzili 150 km olarak belirtilmiştir."
    good = output(good_text, [{"claim_id": "CLAIM_01", "text": good_text,
                               "source_ids": ["SOURCE_01"]}])
    fake = FakeLlmBackend((bad, good))
    result = GenerationService(fake, GenerationConfig(max_regeneration_attempts=1)).generate(
        search_result((candidate(),))
    )
    assert result.validation_status == "valid" and result.attempts == 2
    assert len(fake.calls) == 2 and "Previous output was rejected" in fake.calls[1][1]


def test_maximum_regeneration_returns_safe_failure() -> None:
    bad = output("Menzil 999 km'dir.", [{"claim_id": "CLAIM_01",
        "text": "Menzil 999 km'dir.", "source_ids": ["SOURCE_01"]}])
    fake = FakeLlmBackend((bad, bad, bad))
    result = GenerationService(fake, GenerationConfig(max_regeneration_attempts=1)).generate(
        search_result((candidate(),))
    )
    assert result.validation_status == "rejected"
    assert result.final_paragraph == VALIDATION_FAILURE_MESSAGE
    assert result.attempts == 2 and len(fake.calls) == 2
    assert not result.citation_markers


def test_conflict_aware_generation_retains_both_sources() -> None:
    chunks = (
        candidate("a", "Sistem menzili 150 km olarak belirtilmiştir.", "a"),
        candidate("b", "Sistem menzili 180 km olarak belirtilmiştir.", "b"),
    )
    first_claim = "Bir kaynakta sistem menzili 150 km olarak verilmektedir"
    second_claim = "diğer kaynakta sistem menzili 180 km olarak verilmektedir"
    answer = f"{first_claim}; {second_claim}."
    raw = output(answer, [
        {"claim_id": "CLAIM_01", "text": first_claim, "source_ids": ["SOURCE_01"]},
        {"claim_id": "CLAIM_02", "text": second_claim, "source_ids": ["SOURCE_02"]},
    ])
    result = GenerationService(FakeLlmBackend((raw,)), GenerationConfig()).generate(
        search_result(chunks)
    )
    assert result.validation_status == "valid" and len(result.conflicts) == 1
    assert result.referenced_chunk_ids == ("a", "b")
