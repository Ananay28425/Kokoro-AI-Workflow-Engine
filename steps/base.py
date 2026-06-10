# Provides the base class shared by concrete workflow steps.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BaseStep:
    """Common parent for concrete step classes in steps/.

    It stores the public step name used for debugging and keeps every step on
    the same simple run(state, config) interface used by core.executor.
    """

    name: str

    def run(self, state: dict[str, Any], config: dict[str, Any]) -> Any:
        """Define the method every real step must implement."""

        raise NotImplementedError
