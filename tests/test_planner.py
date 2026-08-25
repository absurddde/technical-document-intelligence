from pathlib import Path

from app.domain.models import Document, DocumentStatus, FileFingerprint, IndexAction
from app.indexing.planner import IncrementalIndexPlanner


def fingerprint(path: str, digest: str, size: int = 10) -> FileFingerprint:
    return FileFingerprint(
        path=Path(path),
        canonical_path=path,
        file_name=Path(path).name,
        file_type=Path(path).suffix.lstrip("."),
        file_size=size,
        modified_ns=1,
        sha256=digest,
    )


def document(identifier: int, path: str, digest: str) -> Document:
    return Document(
        id=identifier,
        document_id=f"doc-{identifier}",
        canonical_path=path,
        file_name=Path(path).name,
        file_type=Path(path).suffix.lstrip("."),
        file_size=10,
        modified_ns=1,
        sha256=digest,
        status=DocumentStatus.DISCOVERED,
        duplicate_of_id=None,
    )


def test_planner_classifies_all_incremental_actions() -> None:
    old = (
        document(1, "a.pdf", "a" * 64),
        document(2, "b.pdf", "b" * 64),
        document(3, "missing.docx", "c" * 64),
    )
    current = (
        fingerprint("a.pdf", "a" * 64),
        fingerprint("b.pdf", "d" * 64),
        fingerprint("new.docx", "e" * 64),
        fingerprint("new-copy.docx", "e" * 64),
    )

    plan = IncrementalIndexPlanner().build_plan(current, old)
    actions = {item.canonical_path: item.action for item in plan}

    assert actions == {
        "a.pdf": IndexAction.UNCHANGED,
        "b.pdf": IndexAction.MODIFIED,
        "missing.docx": IndexAction.MISSING,
        "new-copy.docx": IndexAction.NEW,
        "new.docx": IndexAction.DUPLICATE,
    }


def test_size_or_timestamp_never_overrides_strict_hash_identity() -> None:
    old = document(1, "a.pdf", "a" * 64)
    changed_metadata = fingerprint("a.pdf", "a" * 64, size=999)

    plan = IncrementalIndexPlanner().build_plan((changed_metadata,), (old,))

    assert plan[0].action is IndexAction.UNCHANGED

