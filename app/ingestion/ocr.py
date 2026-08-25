"""Completely local Tesseract OCR abstraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Any

from app.ingestion.errors import OcrUnavailableError


@dataclass(frozen=True, slots=True)
class OcrResult:
    text: str
    confidence: float | None


class OcrEngine(Protocol):
    def recognize(self, image: Any) -> OcrResult: ...


class TesseractOcr:
    """Run preinstalled Tesseract without downloading runtime data."""

    def __init__(self, languages: tuple[str, ...] = ("tur", "eng"), command: str | None = None) -> None:
        self._languages = languages
        self._command = command

    def _module(self):
        try:
            import pytesseract
        except ImportError as error:
            raise OcrUnavailableError("pytesseract is not installed") from error
        if self._command:
            command = Path(self._command)
            if not command.is_file():
                raise OcrUnavailableError("Configured Tesseract executable was not found")
            pytesseract.pytesseract.tesseract_cmd = str(command)
        try:
            available = set(pytesseract.get_languages(config=""))
        except Exception as error:
            raise OcrUnavailableError("Tesseract executable is unavailable") from error
        missing = set(self._languages) - available
        if missing:
            raise OcrUnavailableError("Tesseract language data unavailable: " + ", ".join(sorted(missing)))
        return pytesseract

    def recognize(self, image: Any) -> OcrResult:
        """Extract text and mean confidence from one in-memory page image."""

        module = self._module()
        try:
            data = module.image_to_data(image, lang="+".join(self._languages), output_type=module.Output.DICT)
        except Exception as error:
            raise OcrUnavailableError("Local Tesseract OCR failed") from error
        words: list[str] = []
        confidences: list[float] = []
        for text, raw_confidence in zip(data.get("text", []), data.get("conf", []), strict=False):
            if str(text).strip():
                words.append(str(text).strip())
                try:
                    value = float(raw_confidence)
                    if value >= 0:
                        confidences.append(value)
                except (TypeError, ValueError):
                    pass
        confidence = sum(confidences) / len(confidences) if confidences else None
        return OcrResult(" ".join(words), confidence)
