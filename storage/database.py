# Opens SQLite connections for the storage layer.
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Iterator


class SqliteDatabase:
    """Small database connection wrapper for SQLite.

    storage.repository uses this class whenever it needs a database
    connection. The database file path can be passed directly or read from
    SQLITE_DATABASE_PATH.
    """

    def __init__(self, db_path: str | None = None) -> None:
        """Store the SQLite database file path, defaulting to workflows.db."""

        self._db_path = db_path or os.getenv("SQLITE_DATABASE_PATH", "workflows.db")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Open one SQLite connection for a repository operation."""

        try:
            connection = sqlite3.connect(self._db_path)
            # Enable WAL mode for better concurrency and foreign keys for safety
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            yield connection
            connection.commit()
        except sqlite3.Error as exc:
            raise RuntimeError(f"SQLite operation failed: {exc}") from exc
        finally:
            connection.close()
