"""SQLite connection management."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
import sqlite3


class Database:
    """Create configured SQLite connections for a local database."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve(strict=False)

    def connect(self) -> sqlite3.Connection:
        """Open a connection with integrity-oriented defaults."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Commit a unit of work or roll it back on failure."""

        connection = self.connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

