from pathlib import Path
import sqlite3

from app.domain.models import IndexAction
from app.ingestion.scanner import FolderScanner
from app.persistence.connection import Database
from app.services.inventory_service import InventoryService
from tests.conftest import write_document


def test_inventory_service_persists_duplicates_and_incremental_state(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    write_document(archive / "a.pdf", b"shared")
    write_document(archive / "b.pdf", b"shared")
    database = Database(tmp_path / "data" / "app.db")
    service = InventoryService(database, FolderScanner())

    first_plan = service.inventory(archive)
    second_plan = service.inventory(archive)

    assert {item.action for item in first_plan} == {
        IndexAction.NEW,
        IndexAction.DUPLICATE,
    }
    assert {item.action for item in second_plan} == {
        IndexAction.UNCHANGED,
        IndexAction.DUPLICATE,
    }
    with sqlite3.connect(database.path) as connection:
        rows = connection.execute(
            "SELECT canonical_path, duplicate_of_id FROM documents ORDER BY canonical_path"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0][1] is None
        assert rows[1][1] is not None


def test_inventory_marks_removed_path_missing(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    document = write_document(archive / "a.pdf", b"content")
    service = InventoryService(
        Database(tmp_path / "app.db"), FolderScanner()
    )
    service.inventory(archive)
    document.unlink()

    plan = service.inventory(archive)

    assert len(plan) == 1
    assert plan[0].action is IndexAction.MISSING

