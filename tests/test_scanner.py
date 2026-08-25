from pathlib import Path

from app.ingestion.fingerprint import calculate_sha256
from app.ingestion.scanner import FolderScanner
from tests.conftest import write_document


def test_scanner_discovers_only_pdf_and_docx_case_insensitive(tmp_path: Path) -> None:
    write_document(tmp_path / "a.PDF", b"pdf-a")
    write_document(tmp_path / "nested" / "b.docx", b"docx-b")
    write_document(tmp_path / "ignored.txt", b"ignored")

    result = FolderScanner().scan(tmp_path)

    assert {item.file_name for item in result.fingerprints} == {"a.PDF", "b.docx"}
    assert not result.failures


def test_sha256_is_exact_content_identity(tmp_path: Path) -> None:
    first = write_document(tmp_path / "first.pdf", b"same-content")
    second = write_document(tmp_path / "second.docx", b"same-content")

    assert calculate_sha256(first) == calculate_sha256(second)
    assert len(calculate_sha256(first)) == 64

