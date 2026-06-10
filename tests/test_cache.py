# Tests deterministic hash helpers used by storage and cache-related code.
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest

from storage.database import PostgresDatabase
from storage.hash_utils import stable_sha256
from storage.repository import PostgresWorkflowRepository


class FakeConnection:
    """Fake PostgreSQL connection used to test repository SQL safely."""

    def __init__(self) -> None:
        """Create storage for captured SQL statements and parameters."""

        self.calls: list[tuple[str, tuple[Any, ...] | None]] = []
        self.row: tuple[int] | None = (123,)

    def execute(self, query: str, params: tuple[Any, ...] | None = None) -> "FakeConnection":
        """Capture SQL and return self like psycopg cursor helpers do."""

        self.calls.append((query, params))
        return self

    def fetchone(self) -> tuple[int] | None:
        """Return the configured fake database row."""

        return self.row


class FakeDatabase:
    """Fake database wrapper that yields one fake connection."""

    def __init__(self) -> None:
        """Create a fake connection for repository tests."""

        self.connection = FakeConnection()

    @contextmanager
    def connect(self) -> Iterator[FakeConnection]:
        """Yield the fake connection like PostgresDatabase.connect."""

        yield self.connection


def test_stable_sha256_is_deterministic() -> None:
    """Check that the same input hashes the same way and different input differs."""

    assert stable_sha256("abc") == stable_sha256("abc")
    assert stable_sha256("abc") != stable_sha256("abcd")


def test_stable_sha256_handles_large_input_quickly() -> None:
    """Check hashing a larger prompt-sized input still returns a fixed digest."""

    digest = stable_sha256("abc" * 10_000)

    assert len(digest) == 64


def test_postgres_database_requires_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check that missing database configuration fails before any DB call."""

    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        PostgresDatabase()


def test_repository_initialize_creates_table_and_index() -> None:
    """Check database initialization SQL includes table safety and an index."""

    database = FakeDatabase()
    PostgresWorkflowRepository(database).initialize()
    sql = "\n".join(query for query, _ in database.connection.calls)

    assert "CREATE TABLE IF NOT EXISTS workflow_runs" in sql
    assert "CHECK (status IN ('succeeded', 'failed'))" in sql
    assert "CREATE INDEX IF NOT EXISTS workflow_runs_created_at_idx" in sql


def test_repository_save_run_serializes_json_and_returns_id() -> None:
    """Check save_run writes JSON safely and returns the inserted id."""

    database = FakeDatabase()
    run_id = PostgresWorkflowRepository(database).save_run(
        workflow_name="workflow",
        status="succeeded",
        inputs={"path": Path("input.txt")},
        outputs={"answer": "done"},
    )

    assert run_id == 123
    query, params = database.connection.calls[0]
    assert "INSERT INTO workflow_runs" in query
    assert params is not None
    assert params[0] == "workflow"
    assert params[1] == "succeeded"
    assert '"path": "input.txt"' in params[2]


def test_repository_rejects_invalid_status() -> None:
    """Check invalid run statuses never reach the database."""

    database = FakeDatabase()

    with pytest.raises(ValueError, match="invalid workflow run status"):
        PostgresWorkflowRepository(database).save_run(
            workflow_name="workflow",
            status="unknown",
            inputs={},
            outputs={},
        )

    assert database.connection.calls == []


def test_repository_raises_when_insert_returns_no_row() -> None:
    """Check save_run fails loudly if PostgreSQL does not return an id."""

    database = FakeDatabase()
    database.connection.row = None

    with pytest.raises(RuntimeError, match="failed to persist"):
        PostgresWorkflowRepository(database).save_run(
            workflow_name="workflow",
            status="failed",
            inputs={},
            outputs={},
            error="boom",
        )
