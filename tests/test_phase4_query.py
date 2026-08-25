from app.retrieval.query import QueryNormalizer, translate_preserving_acronyms


def test_turkish_unicode_acronyms_eoir_and_exact_phrases_are_preserved() -> None:
    original = '  GNSS   destekli  "ataletsel seyrüsefer" ve EO/IR  '
    normalized = QueryNormalizer().normalize(original)

    assert normalized.original_query == original
    assert "seyrüsefer" in normalized.semantic_query
    assert normalized.detected_acronyms == ("GNSS", "EO/IR")
    assert normalized.exact_phrases == ("ataletsel seyrüsefer",)
    assert '"ataletsel seyrüsefer"' in normalized.lexical_query
    assert '"EO IR"' in normalized.lexical_query


class MarkerTranslator:
    def translate(self, query: str, source_language: str, target_language: str) -> str:
        assert "GNSS" not in query
        assert "ZXQACRONYM0QXZ" in query
        return query.replace("destekli ataletsel seyrüsefer sistemi", "assisted inertial navigation system")


def test_translation_protects_acronyms() -> None:
    query = QueryNormalizer().normalize("GNSS destekli ataletsel seyrüsefer sistemi")
    translated = translate_preserving_acronyms(MarkerTranslator(), query)
    assert translated == "GNSS assisted inertial navigation system"
