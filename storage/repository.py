# Persists workflow run records through SQLite.
from __future__ import annotations

import json
from typing import Any

from storage.database import SqliteDatabase

VALID_RUN_STATUSES = {"succeeded", "failed"}


class SqliteWorkflowRepository:
    """Repository that saves workflow execution metadata.

    core.executor calls this class through the WorkflowRepository contract when
    CLI persistence is enabled.
    """

    def __init__(self, database: SqliteDatabase) -> None:
        """Receive the database connection wrapper used for all queries."""

        self._database = database

    def initialize(self) -> None:
        """Create the workflow_runs table if it does not already exist."""

        with self._database.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_name TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('succeeded', 'failed')),
                    inputs TEXT NOT NULL,
                    outputs TEXT NOT NULL,
                    error TEXT,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS workflow_runs_created_at_idx
                ON workflow_runs (created_at DESC)
                """
            )

    def save_run(
        self,
        *,
        workflow_name: str,
        status: str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        error: str | None = None,
    ) -> int:
        """Insert one workflow run and return its new database id."""

        if status not in VALID_RUN_STATUSES:
            allowed = ", ".join(sorted(VALID_RUN_STATUSES))
            raise ValueError(
                f"invalid workflow run status '{status}'. Expected one of: {allowed}"
            )
        with self._database.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO workflow_runs (workflow_name, status, inputs, outputs, error)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    workflow_name,
                    status,
                    json.dumps(inputs, default=str),
                    json.dumps(outputs, default=str),
                    error,
                ),
            )
            row_id = cursor.lastrowid
        if row_id is None:
            raise RuntimeError("failed to persist workflow run")
        return int(row_id)
