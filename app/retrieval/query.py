"""Unicode-safe query normalization and optional local translation contracts."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Protocol


ACRONYM_PATTERN = re.compile(
    r"(?<!\w)(?:EO/IR|INS|GPS|GNSS|SAR|AESA|RF|IR|EW|ECM|ECCM|CEP|BVR|LOS|NLOS)(?!\w)",
    re.IGNORECASE,
)
QUOTED_PATTERN = re.compile(r'"([^"\r\n]+)"')


@dataclass(frozen=True, slots=True)
class NormalizedQuery:
    original_query: str
    lexical_query: str
    semantic_query: str
    detected_acronyms: tuple[str, ...]
    exact_phrases: tuple[str, ...]


class QueryNormalizer:
    """Create lexical and semantic forms without transliterating source input."""

    def normalize(self, query: str) -> NormalizedQuery:
        """Normalize Unicode/spacing while retaining the exact original string."""

        normalized = unicodedata.normalize("NFC", query)
        compact = " ".join(normalized.split())
        phrases = tuple(
            " ".join(match.group(1).split())
            for match in QUOTED_PATTERN.finditer(compact)
            if match.group(1).strip()
        )
        acronyms = tuple(dict.fromkeys(match.group(0).upper() for match in ACRONYM_PATTERN.finditer(compact)))
        semantic = QUOTED_PATTERN.sub(lambda match: match.group(1), compact)
        semantic = re.sub(r"[^\w\s/\-]+", " ", semantic, flags=re.UNICODE)
        semantic = " ".join(semantic.split())

        without_phrases = QUOTED_PATTERN.sub(" ", compact)
        tokens = re.findall(r"[\w]+(?:/[\w]+)?", without_phrases, flags=re.UNICODE)
        lexical_parts = [f'"{phrase.replace(chr(34), "")}"' for phrase in phrases]
        lexical_parts.extend(f'"{token.replace("/", " ")}"' for token in tokens)
        lexical = " OR ".join(dict.fromkeys(lexical_parts))
        return NormalizedQuery(query, lexical, semantic, acronyms, phrases)


class QueryTranslator(Protocol):
    """Contract for a completely local optional query translator."""

    def translate(self, query: str, source_language: str,
                  target_language: str) -> str: ...


class DisabledQueryTranslator:
    """Production-safe translator used until a local model is configured."""

    def translate(self, query: str, source_language: str,
                  target_language: str) -> str:
        return query


def translate_preserving_acronyms(translator: QueryTranslator, query: NormalizedQuery,
                                   source_language: str = "tr",
                                   target_language: str = "en") -> str:
    """Protect detected technical acronyms around a local translation call."""

    protected = query.semantic_query
    replacements: dict[str, str] = {}
    for index, acronym in enumerate(query.detected_acronyms):
        marker = f"ZXQACRONYM{index}QXZ"
        protected = re.sub(re.escape(acronym), marker, protected, flags=re.IGNORECASE)
        replacements[marker] = acronym
    translated = translator.translate(protected, source_language, target_language)
    for marker, acronym in replacements.items():
        translated = re.sub(re.escape(marker), acronym, translated, flags=re.IGNORECASE)
    return translated
