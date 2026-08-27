"""Text-first PDF parsing with page-level local OCR fallback."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Sequence

from app.domain.models import ParsedBlock, ParsedDocument
from app.ingestion.errors import DocumentParseError
from app.ingestion.ocr import OcrEngine
from app.processing.cleaner import (clean_text, join_hyphenated_line_breaks,
                                    remove_repeated_margins)


def render_table_text(table: list[list[object | None]]) -> str:
    """Render meaningful table rows while ignoring empty decorative grids."""

    rows: list[str] = []
    for row in table:
        cells = [clean_text(str(cell or "")) for cell in row]
        if any(cells):
            rows.append(" | ".join(cells))
    return clean_text("\n".join(rows))


def _word_value(word: dict[str, Any], key: str) -> float:
    return float(word[key])


def _persistent_gutters(words: Sequence[dict[str, Any]], page_width: float) -> list[tuple[float, float]]:
    """Find well-supported vertical whitespace bands between text columns."""

    if len(words) < 12 or page_width <= 0:
        return []
    heights = sorted(_word_value(word, "bottom") - _word_value(word, "top") for word in words)
    median_height = heights[len(heights) // 2]
    body_words = [word for word in words
                  if _word_value(word, "bottom") - _word_value(word, "top") < median_height * 1.35]
    if len(body_words) < 12:
        body_words = list(words)
    steps = 1000
    occupied = [False] * (steps + 1)
    for word in body_words:
        start = max(0, min(steps, int(_word_value(word, "x0") / page_width * steps)))
        end = max(start, min(steps, int(_word_value(word, "x1") / page_width * steps) + 1))
        for index in range(start, end + 1):
            occupied[index] = True

    empty_runs: list[tuple[float, float]] = []
    start_index: int | None = None
    for index, is_occupied in enumerate(occupied + [True]):
        if not is_occupied and start_index is None:
            start_index = index
        elif is_occupied and start_index is not None:
            left, right = start_index / steps * page_width, index / steps * page_width
            if right - left >= page_width * 0.015:
                empty_runs.append((left, right))
            start_index = None

    supported: list[tuple[float, float]] = []
    for left, right in empty_runs:
        midpoint = (left + right) / 2
        if not page_width * 0.12 <= midpoint <= page_width * 0.88:
            continue
        left_words = [word for word in body_words if _word_value(word, "x1") <= left]
        right_words = [word for word in body_words if _word_value(word, "x0") >= right - page_width / steps]
        if min(len(left_words), len(right_words)) < max(6, int(len(body_words) * 0.08)):
            continue
        left_top = min(_word_value(word, "top") for word in left_words)
        left_bottom = max(_word_value(word, "bottom") for word in left_words)
        right_top = min(_word_value(word, "top") for word in right_words)
        right_bottom = max(_word_value(word, "bottom") for word in right_words)
        overlap = max(0.0, min(left_bottom, right_bottom) - max(left_top, right_top))
        shorter_span = min(left_bottom - left_top, right_bottom - right_top)
        if shorter_span > 0 and overlap / shorter_span >= 0.35:
            supported.append((left, right))

    # At most two strongest separated gutters: two columns normally, three when
    # the source page provides equally strong evidence (as in the Army pilot).
    strongest = sorted(supported, key=lambda gap: gap[1] - gap[0], reverse=True)
    chosen: list[tuple[float, float]] = []
    for gap in strongest:
        if all(abs((gap[0] + gap[1]) - (old[0] + old[1])) > page_width * 0.12 for old in chosen):
            chosen.append(gap)
        if len(chosen) == 2:
            break
    return sorted(chosen)


def _words_to_lines(words: Sequence[dict[str, Any]], y_tolerance: float = 3.0) -> list[str]:
    lines: list[list[dict[str, Any]]] = []
    for word in sorted(words, key=lambda item: (_word_value(item, "top"), _word_value(item, "x0"))):
        if not lines or abs(_word_value(word, "top") - _word_value(lines[-1][0], "top")) > y_tolerance:
            lines.append([word])
        else:
            lines[-1].append(word)
    return [" ".join(str(word["text"]) for word in sorted(line, key=lambda item: _word_value(item, "x0")))
            for line in lines]


def order_words_by_columns(words: Sequence[dict[str, Any]], page_width: float) -> str | None:
    """Return column-major text only when persistent gutter evidence is strong."""

    gutters = _persistent_gutters(words, page_width)
    if not gutters:
        return None
    boundaries = [0.0, *[(left + right) / 2 for left, right in gutters], page_width]
    regions: list[list[dict[str, Any]]] = [[] for _ in range(len(boundaries) - 1)]
    median_height = sorted(_word_value(word, "bottom") - _word_value(word, "top") for word in words)[len(words) // 2]
    spanning: list[dict[str, Any]] = []
    for word in words:
        center = (_word_value(word, "x0") + _word_value(word, "x1")) / 2
        region_index = next(index for index in range(len(regions)) if boundaries[index] <= center <= boundaries[index + 1])
        regions[region_index].append(word)

    # Large lines above the column bodies are treated as full-width headings.
    grouped = []
    for word in sorted(words, key=lambda item: (_word_value(item, "top"), _word_value(item, "x0"))):
        if not grouped or abs(_word_value(word, "top") - _word_value(grouped[-1][0], "top")) > 3.0:
            grouped.append([word])
        else:
            grouped[-1].append(word)
    for line in grouped:
        height = max(_word_value(word, "bottom") - _word_value(word, "top") for word in line)
        crosses_gutter = any(
            _word_value(word, "x0") < midpoint < _word_value(word, "x1")
            for word in line for midpoint in boundaries[1:-1]
        )
        if crosses_gutter and height >= median_height * 1.35:
            spanning.extend(line)
            for region in regions:
                region[:] = [word for word in region if word not in line]

    ordered_lines = _words_to_lines(spanning)
    for region in regions:
        ordered_lines.extend(_words_to_lines(region))
    return "\n".join(line for line in ordered_lines if line.strip())


def extract_native_page_text(page: Any) -> str:
    """Extract native text, correcting only strongly evidenced column layouts."""

    words = page.extract_words() or []
    ordered = order_words_by_columns(words, float(page.width))
    return ordered if ordered is not None else (page.extract_text() or "")


class PdfParser:
    """Extract page provenance and OCR only pages with insufficient native text."""

    def __init__(self, ocr: OcrEngine | None, *, min_text_characters: int = 40,
                 min_alphanumeric_ratio: float = 0.20, dpi: int = 300,
                 remove_margins: bool = True, margin_lines: int = 2,
                 repeated_page_ratio: float = 0.70, pipeline_version: str = "phase2-v1") -> None:
        self._ocr = ocr
        self._min_chars = min_text_characters
        self._min_ratio = min_alphanumeric_ratio
        self._dpi = dpi
        self._remove_margins = remove_margins
        self._margin_lines = margin_lines
        self._repeated_ratio = repeated_page_ratio
        self._version = pipeline_version

    def needs_ocr(self, text: str) -> bool:
        """Apply configurable page-level native-text sufficiency thresholds."""

        compact = re.sub(r"\s+", "", text or "")
        if len(compact) < self._min_chars:
            return True
        return sum(char.isalnum() for char in compact) / len(compact) < self._min_ratio

    def _ocr_page(self, path: Path, page_index: int):
        if self._ocr is None:
            raise DocumentParseError("OCR is required for this page but is disabled")
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(str(path))
            try:
                image = pdf[page_index].render(scale=self._dpi / 72).to_pil()
                return self._ocr.recognize(image)
            finally:
                pdf.close()
        except DocumentParseError:
            raise
        except Exception as error:
            raise DocumentParseError("PDF page rendering for OCR failed") from error

    def parse(self, path: Path, document_hash: str) -> ParsedDocument:
        """Parse a PDF while converting library errors to safe domain errors."""

        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                native_pages = [extract_native_page_text(page) for page in pdf.pages]
                tables = [page.extract_tables() or [] for page in pdf.pages]
        except Exception as error:
            message = "Encrypted PDF cannot be parsed" if "password" in str(error).lower() or "encrypt" in str(error).lower() else "Invalid or unreadable PDF"
            raise DocumentParseError(message) from error

        page_texts: list[str] = []
        ocr_metadata: list[tuple[bool, float | None]] = []
        for index, native in enumerate(native_pages):
            if self.needs_ocr(native):
                result = self._ocr_page(path, index)
                page_texts.append(clean_text(join_hyphenated_line_breaks(result.text)))
                ocr_metadata.append((True, result.confidence))
            else:
                page_texts.append(clean_text(join_hyphenated_line_breaks(native)))
                ocr_metadata.append((False, None))
        if self._remove_margins:
            page_texts = remove_repeated_margins(page_texts, self._margin_lines, self._repeated_ratio)

        blocks: list[ParsedBlock] = []
        paragraph_index = 0
        for page_index, text in enumerate(page_texts):
            used, confidence = ocr_metadata[page_index]
            for paragraph in re.split(r"\n\s*\n|\n", text):
                cleaned = clean_text(paragraph)
                if cleaned:
                    blocks.append(ParsedBlock("paragraph", cleaned, page_index + 1, paragraph_index=paragraph_index, ocr_used=used, ocr_confidence=confidence))
                    paragraph_index += 1
            for table in tables[page_index]:
                table_text = render_table_text(table)
                if table_text:
                    blocks.append(ParsedBlock("table", table_text, page_index + 1, paragraph_index=paragraph_index, ocr_used=False))
                    paragraph_index += 1
        return ParsedDocument(tuple(blocks), document_hash, self._version)
