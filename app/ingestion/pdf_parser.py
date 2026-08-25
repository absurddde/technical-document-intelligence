"""Text-first PDF parsing with page-level local OCR fallback."""

from __future__ import annotations

from pathlib import Path
import re

from app.domain.models import ParsedBlock, ParsedDocument
from app.ingestion.errors import DocumentParseError
from app.ingestion.ocr import OcrEngine
from app.processing.cleaner import clean_text, remove_repeated_margins


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
                native_pages = [page.extract_text() or "" for page in pdf.pages]
                tables = [page.extract_tables() or [] for page in pdf.pages]
        except Exception as error:
            message = "Encrypted PDF cannot be parsed" if "password" in str(error).lower() or "encrypt" in str(error).lower() else "Invalid or unreadable PDF"
            raise DocumentParseError(message) from error

        page_texts: list[str] = []
        ocr_metadata: list[tuple[bool, float | None]] = []
        for index, native in enumerate(native_pages):
            if self.needs_ocr(native):
                result = self._ocr_page(path, index)
                page_texts.append(clean_text(result.text))
                ocr_metadata.append((True, result.confidence))
            else:
                page_texts.append(clean_text(native))
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
                rows = [" | ".join(clean_text(str(cell or "")) for cell in row) for row in table]
                table_text = clean_text("\n".join(rows))
                if table_text:
                    blocks.append(ParsedBlock("table", table_text, page_index + 1, paragraph_index=paragraph_index, ocr_used=False))
                    paragraph_index += 1
        return ParsedDocument(tuple(blocks), document_hash, self._version)
