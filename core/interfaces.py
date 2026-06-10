# Defines shared contracts between core, step, AI, and storage modules.
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class Step(Protocol):
    """Contract for any executable workflow step.

    Classes in steps/ follow this shape so core.executor can call them without
    knowing their concrete type.
    """

    name: str

    def run(self, state: dict[str, Any], config: dict[str, Any]) -> Any:
        """Run one step using shared workflow state and step config."""


class LLMClient(Protocol):
    """Contract for local text generation clients used by AI steps."""

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.2) -> str:
        """Generate text from a prompt through a local model adapter."""


class TTSClient(Protocol):
    """Contract for local text-to-speech clients used by speech steps."""

    def synthesize(self, text: str, output_path: Path, *, voice: str | None = None) -> Path:
        """Turn text into an audio file and return the saved file path."""


class WorkflowRepository(Protocol):
    """Contract for saving workflow run metadata in storage modules."""

    def save_run(
        self,
        *,
        workflow_name: str,
        status: str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        error: str | None = None,
    ) -> int:
        """Persist one workflow run and return the database id."""
