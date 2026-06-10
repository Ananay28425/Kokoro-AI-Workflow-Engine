# Defines simple storage data models for workflow run records.
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class WorkflowRun:
    """In-memory representation of one row from workflow_runs.

    This model belongs to the storage layer and can be returned by future read
    queries without exposing raw database rows to the rest of the app.
    """

    id: int
    workflow_name: str
    status: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    error: str | None
    created_at: datetime
