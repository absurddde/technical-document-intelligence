"""Small deterministic Turkish/English language heuristic with no model dependency."""

from __future__ import annotations

import re
from app.domain.models import Language

_TR = {"ve", "bir", "bu", "için", "ile", "olarak", "sistem", "olan", "de", "da"}
_EN = {"the", "and", "of", "for", "with", "system", "is", "are", "to", "in"}


def detect_language(text: str) -> Language:
    """Return broad metadata only; it never filters retrieval."""

    words = re.findall(r"[^\W\d_]+", text.casefold(), flags=re.UNICODE)
    if not words:
        return "unknown"
    tr = sum(word in _TR for word in words) + sum(any(char in word for char in "çğıöşü") for word in words)
    en = sum(word in _EN for word in words)
    if tr and en:
        return "mixed"
    if tr:
        return "tr"
    if en:
        return "en"
    return "unknown"
