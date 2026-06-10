# Maps workflow step type names to concrete step classes and factories.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from core.interfaces import Step


StepFactory = Callable[[], Step]


@dataclass
class StepRegistry:
    """Factory registry for workflow steps.

    The executor passes a step type from YAML, such as "summarize", and this
    registry returns the concrete step object from the steps/ package.
    """

    _factories: dict[str, StepFactory] = field(default_factory=dict)

    def register(self, step_type: str, factory: StepFactory) -> None:
        """Register a factory function for one YAML step type."""

        step_type = step_type.strip()
        if not step_type:
            raise ValueError("step_type must not be empty")
        if step_type in self._factories:
            raise ValueError(f"step_type is already registered: {step_type}")
        self._factories[step_type] = factory

    def create(self, step_type: str) -> Step:
        """Build a step object for the requested YAML step type."""

        try:
            return self._factories[step_type]()
        except KeyError as exc:
            available = ", ".join(sorted(self._factories)) or "<none>"
            raise KeyError(f"unknown step type '{step_type}'. Available: {available}") from exc

    def registered_types(self) -> tuple[str, ...]:
        """Return all known step type names in a stable order."""

        return tuple(sorted(self._factories))


def build_default_registry() -> StepRegistry:
    """Create the production registry used by the CLI.

    This connects core.registry to ai/ adapters and steps/ implementations.
    Imports stay inside the function so tests can import the registry without
    immediately loading local model or audio dependencies.
    """

    from ai.kokoro import KokoroTTSAdapter
    from ai.llama_cpp import LlamaCppAdapter
    from steps.read_file import ReadFileStep
    from steps.speak import SpeakStep
    from steps.summarize import SummarizeStep

    registry = StepRegistry()
    registry.register("read_file", ReadFileStep)
    registry.register("summarize", lambda: SummarizeStep(LlamaCppAdapter()))
    registry.register("speak", lambda: SpeakStep(KokoroTTSAdapter()))
    return registry
