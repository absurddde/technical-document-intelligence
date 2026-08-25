"""Shared Phase 1 test fixtures."""

from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from app.persistence.schema import initialize_schema


@pytest.fixture
def connection() -> sqlite3.Connection:
    database = sqlite3.connect(":memory:")
    database.row_factory = sqlite3.Row
    database.execute("PRAGMA foreign_keys = ON")
    initialize_schema(database)
    yield database
    database.close()


def write_document(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path

