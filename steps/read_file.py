# Implements the workflow step that reads text from a local file.
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from steps.base import BaseStep


class ReadFileConfig(BaseModel):
    """Validated settings for ReadFileStep.

    The YAML config provides these fields, and Pydantic checks them before the
    step touches the filesystem.
    """

    model_config = ConfigDict(extra="forbid")

    path: Path
    output_key: str = Field(default="text", min_length=1)


class ReadFileStep(BaseStep):
    """Read a UTF-8 text file and store its content in workflow state.

    This step is called by core.executor and usually feeds text into later AI
    steps such as steps.summarize.SummarizeStep.
    """

    def __init__(self) -> None:
        """Set the step name used by the registry and debugging output."""

        super().__init__("read_file")

    def run(self, state: dict[str, Any], config: dict[str, Any]) -> str:
        """Read the configured file and write the text to state."""

        parsed = ReadFileConfig.model_validate(config)
        if not parsed.path.is_file():
            raise FileNotFoundError(f"input file must be a readable file: {parsed.path}")
        try:
            text = parsed.path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"input file must be valid UTF-8 text: {parsed.path}") from exc
        state[parsed.output_key] = text
        return text
