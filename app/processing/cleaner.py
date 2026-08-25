"""Conservative text cleaning that preserves technical meaning."""

from __future__ import annotations

from collections import Counter
import re


def clean_text(text: str) -> str:
    """Normalize excessive whitespace without case-folding or ASCII conversion."""

    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def remove_repeated_margins(pages: list[str], margin_lines: int = 2, page_ratio: float = 0.7) -> list[str]:
    """Remove only exact, short lines repeated in the same page-margin position."""

    if len(pages) < 3 or margin_lines <= 0:
        return pages
    threshold = max(3, int(len(pages) * page_ratio + 0.999))
    candidates: Counter[tuple[str, int]] = Counter()
    page_lines = [[line for line in page.splitlines() if line.strip()] for page in pages]
    for lines in page_lines:
        for index, line in enumerate(lines[:margin_lines]):
            if len(line) <= 160:
                candidates[(line.strip(), index)] += 1
        for offset, line in enumerate(reversed(lines[-margin_lines:])):
            if len(line) <= 160:
                candidates[(line.strip(), -(offset + 1))] += 1
    repeated = {key for key, count in candidates.items() if count >= threshold}
    cleaned: list[str] = []
    for lines in page_lines:
        kept = [line for i, line in enumerate(lines) if (line.strip(), i) not in repeated and (line.strip(), i - len(lines)) not in repeated]
        cleaned.append("\n".join(kept))
    return cleaned
