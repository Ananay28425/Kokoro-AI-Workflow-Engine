# Opens PostgreSQL connections for the storage layer.
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator


class PostgresDatabase:
    """Small database connection wrapper for PostgreSQL.

    storage.repository uses this class whenever it needs a database
    connection. The DSN can be passed directly or read from DATABASE_URL.
    """

    def __init__(self, dsn: str | None = None) -> None:
        """Store the PostgreSQL connection string and fail early if missing."""

        self._dsn = dsn or os.getenv("DATABASE_URL")
        if not self._dsn:
            raise RuntimeError("DATABASE_URL must be set for PostgreSQL persistence")

    @contextmanager
    def connect(self) -> Iterator[Any]:
        """Open one PostgreSQL connection for a repository operation."""

        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError(
                "psycopg is required for PostgreSQL persistence. "
                "Install the project dependencies or install psycopg[binary]."
            ) from exc

        try:
            with psycopg.connect(self._dsn) as connection:
                yield connection
        except psycopg.Error as exc:
            raise RuntimeError(f"PostgreSQL operation failed: {exc}") from exc
