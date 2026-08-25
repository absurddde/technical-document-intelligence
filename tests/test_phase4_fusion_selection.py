import hashlib

from app.retrieval.fusion import ReciprocalRankFusion
from app.retrieval.models import RetrievalHit
from app.retrieval.query import QueryNormalizer
from app.retrieval.selection import DiverseContextSelector


def hit(chunk: str, score: float, text: str | None = None, *, document: str = "doc",
        ordinal: int = 0, digest: str = "") -> RetrievalHit:
    value = text or f"text {chunk}"
    return RetrievalHit(chunk, document, f"{document}.pdf", value, score,
                        1, 1, "Section", ordinal, digest)


def test_rrf_orders_and_keeps_single_branch_candidates() -> None:
    fused = ReciprocalRankFusion(rrf_k=10).fuse((
        ("original_lexical", (hit("both", -2), hit("lexical", -1))),
        ("original_semantic", (hit("both", .9), hit("semantic", .8))),
    ), QueryNormalizer().normalize("query"))

    assert [item.chunk_id for item in fused] == ["both", "lexical", "semantic"]
    assert fused[1].semantic_rank is None
    assert fused[2].lexical_rank is None
    assert set(fused[0].rrf_contributions) == {"original_lexical", "original_semantic"}


def test_exact_and_acronym_boosts_are_separate_and_bounded() -> None:
    fusion = ReciprocalRankFusion(rrf_k=60, exact_phrase_boost=99, acronym_boost=99)
    fused = fusion.fuse((("original_lexical", (
        hit("boosted", -1, "GNSS terminal güdüm architecture"),
    )),), QueryNormalizer().normalize('GNSS "terminal güdüm"'))
    ceiling = 1 / 61
    assert fused[0].exact_phrase_boost == ceiling
    assert fused[0].acronym_boost == ceiling
    assert fused[0].fused_score == sum(fused[0].rrf_contributions.values()) + 2 * ceiling


def test_deduplicates_chunk_hash_and_adjacent_overlap_and_limits_context() -> None:
    common_hash = hashlib.sha256(b"same").hexdigest()
    candidates = ReciprocalRankFusion(60).fuse((("original_semantic", (
        hit("one", .9, "alpha beta gamma delta", ordinal=1, digest=common_hash),
        hit("same-hash", .8, "different display", document="other", digest=common_hash),
        hit("adjacent", .7, "alpha beta gamma delta extra", ordinal=2),
        hit("distinct", .6, "radar antenna frequency", ordinal=3),
        hit("overflow", .5, "power supply voltage", ordinal=4),
    )),), QueryNormalizer().normalize("alpha"))

    selected = DiverseContextSelector(.8).select(candidates, 2)
    assert [item.chunk_id for item in selected] == ["one", "distinct"]
    assert [item.final_rank for item in selected] == [1, 2]
