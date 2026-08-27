from app.infrastructure.config import RetrievalConfig
from app.retrieval.models import RetrievalHit
from app.services.search_service import SearchService


def hit(chunk: str, score: float, text: str = "technical passage") -> RetrievalHit:
    return RetrievalHit(chunk, chunk, f"{chunk}.pdf", text, score, 2, 2, "Systems", 1, chunk)


class RecordingRetriever:
    def __init__(self, results):
        self.results = results
        self.queries = []
    def search(self, query, limit):
        self.queries.append((query, limit))
        return tuple(self.results.get(query, ()))


class FakeTranslator:
    def translate(self, query, source_language, target_language):
        return query.replace("g\u00fcd\u00fcm sistemi", "guidance system")


def config(**overrides) -> RetrievalConfig:
    values = RetrievalConfig().__dict__ if hasattr(RetrievalConfig(), "__dict__") else {
        name: getattr(RetrievalConfig(), name) for name in RetrievalConfig.__dataclass_fields__
    }
    values.update(overrides)
    return RetrievalConfig(**values)


def test_translation_disabled_and_insufficient_evidence() -> None:
    lexical, semantic = RecordingRetriever({}), RecordingRetriever({})
    result = SearchService(lexical, semantic, config()).search("güdüm sistemi")
    assert result.translated_query is None
    assert result.insufficient_evidence is True
    assert len(lexical.queries) == len(semantic.queries) == 1


def test_four_branch_fusion_keeps_original_and_translated_queries() -> None:
    lexical = RecordingRetriever({
        '"g\u00fcd\u00fcm" OR "sistemi"': (hit("original-only", -1),),
        '"guidance" OR "system"': (hit("translated", -2),),
    })
    semantic = RecordingRetriever({
        "g\u00fcd\u00fcm sistemi": (hit("original-semantic", .7),),
        "guidance system": (hit("translated", .9),),
    })
    result = SearchService(
        lexical, semantic, config(translation_enabled=True), translator=FakeTranslator()
    ).search("g\u00fcd\u00fcm sistemi")

    assert result.semantic_query == "g\u00fcd\u00fcm sistemi"
    assert result.translated_query == "guidance system"
    assert len(lexical.queries) == len(semantic.queries) == 2
    translated = next(item for item in result.details if item.chunk_id == "translated")
    assert translated.translation_branches == ("translated_lexical", "translated_semantic")
    assert set(translated.rrf_contributions) == {"translated_lexical", "translated_semantic"}
    original = next(item for item in result.details if item.chunk_id == "original-only")
    assert "original_lexical" in original.rrf_contributions


def test_retrieval_details_and_evidence_threshold() -> None:
    lexical = RecordingRetriever({'"GNSS"': (hit("one", -3, "GNSS passage"),)})
    semantic = RecordingRetriever({"GNSS": (hit("one", .88, "GNSS passage"),)})
    result = SearchService(lexical, semantic, config(minimum_evidence_threshold=.01)).search("GNSS")
    detail = result.details[0]
    assert result.insufficient_evidence is False
    assert detail.lexical_rank == detail.semantic_rank == detail.final_rank == 1
    assert detail.lexical_score == -3 and detail.semantic_score == .88
    assert detail.acronym_boost > 0 and detail.rrf_score > 0
    assert detail.page_start == 2 and detail.section_title == "Systems"


def test_low_semantic_similarity_is_insufficient_despite_rrf_rank() -> None:
    lexical = RecordingRetriever({'"orchids"': (hit("lexical", -1, "unrelated orchids"),)})
    semantic = RecordingRetriever({"orchids": (hit("semantic", .40, "weak neighbor"),)})
    result = SearchService(lexical, semantic, config()).search("orchids")
    assert result.selected
    assert result.insufficient_evidence is True
