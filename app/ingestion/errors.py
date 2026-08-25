"""Safe document-processing errors that never include document contents."""


class DocumentParseError(RuntimeError):
    """A document could not be parsed safely."""


class OcrUnavailableError(DocumentParseError):
    """Local Tesseract or requested language data is unavailable."""
