from pathlib import Path
import sqlite3

from app.domain.models import DocumentStatus, FileFingerprint
from app.persistence.repositories import DocumentRepository


def make_fingerprint(path: str, digest: str) -> FileFingerprint:
    return FileFingerprint(
        path=Path(path),
        canonical_path=path,
        file_name=Path(path).name,
        file_type=Path(path).suffix.lstrip("."),
        file_size=12,
        modified_ns=123,
        sha256=digest,
    )


def test_duplicate_keeps_own_path_metadata_and_links_canonical(
    connection: sqlite3.Connection,
) -> None:
    repository = DocumentRepository(connection)
    digest = "a" * 64
    canonical = repository.save_fingerprint(make_fingerprint("one.pdf", digest))
    duplicate = repository.save_fingerprint(
        make_fingerprint("copies/two.pdf", digest),
        status=DocumentStatus.DUPLICATE,
        duplicate_of_id=canonical.id,
    )

    assert duplicate.id != canonical.id
    assert duplicate.canonical_path == "copies/two.pdf"
    assert duplicate.duplicate_of_id == canonical.id
    assert len(repository.list_all()) == 2


def test_upsert_preserves_document_uuid(connection: sqlite3.Connection) -> None:
    repository = DocumentRepository(connection)
    original = repository.save_fingerprint(make_fingerprint("one.pdf", "a" * 64))
    updated = repository.save_fingerprint(make_fingerprint("one.pdf", "b" * 64))

    assert updated.id == original.id
    assert updated.document_id == original.document_id
    assert updated.sha256 == "b" * 64


def test_index_state_is_reset_only_when_content_changes(
    connection: sqlite3.Connection,
) -> None:
    repository = DocumentRepository(connection)
    saved = repository.save_fingerprint(make_fingerprint("one.pdf", "a" * 64))
    repository.ensure_index_state(saved.id, None, reset_pipeline=True)
    connection.execute(
        """
        UPDATE document_index_state SET parse_status = 'complete',
            chunk_status = 'complete' WHERE document_id = ?
        """,
        (saved.id,),
    )

    repository.ensure_index_state(saved.id, None, reset_pipeline=False)
    unchanged = connection.execute(
        "SELECT * FROM document_index_state WHERE document_id = ?", (saved.id,)
    ).fetchone()
    assert unchanged["parse_status"] == "complete"
    assert unchanged["chunk_status"] == "complete"

    repository.ensure_index_state(saved.id, None, reset_pipeline=True)
    changed = connection.execute(
        "SELECT * FROM document_index_state WHERE document_id = ?", (saved.id,)
    ).fetchone()
    assert changed["parse_status"] == "pending"
    assert changed["chunk_status"] == "pending"
