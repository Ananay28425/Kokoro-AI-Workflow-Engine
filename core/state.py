# Stores values shared between workflow steps while the executor is running.
from __future__ import annotations

from typing import Any


class WorkflowState(dict[str, Any]):
    """Dictionary used by steps to pass data to each other.

    A read_file step may write "text", a summarize step may read that text and
    write "summary", and a speak step may read the summary.
    """

    def require(self, key: str) -> Any:
        """Return a required state value or raise a clear error."""

        if key not in self:
            raise KeyError(f"workflow state is missing required key: {key}")
        return self[key]
