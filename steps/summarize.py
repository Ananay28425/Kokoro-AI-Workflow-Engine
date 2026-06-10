# Implements the workflow step that summarizes text with a local LLM adapter.
from __future__ import annotations

from string import Formatter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.interfaces import LLMClient
from steps.base import BaseStep


class SummarizeConfig(BaseModel):
    """Validated settings for SummarizeStep.

    YAML controls which state key to read, which key to write, and how the
    prompt is sent to the LLM client from ai/.
    """

    model_config = ConfigDict(extra="forbid")

    input_key: str = Field(default="text", min_length=1)
    output_key: str = Field(default="summary", min_length=1)
    prompt: str = Field(default="Summarize the following text clearly and concisely:\n\n{text}", min_length=1)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    @field_validator("prompt")
    @classmethod
    def _validate_prompt_contract(cls, value: str) -> str:
        """Require prompts to use only the supported {text} placeholder."""

        fields = [field_name for _, field_name, _, _ in Formatter().parse(value) if field_name is not None]
        unsupported = sorted(set(fields) - {"text"})
        if unsupported:
            raise ValueError(f"unsupported prompt placeholder(s): {', '.join(unsupported)}")
        if "text" not in fields:
            raise ValueError("prompt must include the {text} placeholder")
        return value


class SummarizeStep(BaseStep):
    """Create a summary by calling an LLM client.

    The step depends on core.interfaces.LLMClient, so tests can pass a fake
    client while production code passes ai.llama_cpp.LlamaCppAdapter.
    """

    def __init__(self, llm: LLMClient) -> None:
        """Receive the LLM adapter used to generate summaries."""

        super().__init__("summarize")
        self._llm = llm

    def run(self, state: dict[str, Any], config: dict[str, Any]) -> str:
        """Read text from state, call the LLM, and store the summary."""

        parsed = SummarizeConfig.model_validate(config)
        text = state.get(parsed.input_key)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"state key '{parsed.input_key}' must contain non-empty text")
        # The prompt is a simple contract: the input text is inserted where
        # "{text}" appears, then ai.llama_cpp handles local inference.
        prompt = parsed.prompt.format(text=text)
        summary = self._llm.complete(
            prompt,
            max_tokens=parsed.max_tokens,
            temperature=parsed.temperature,
        )
        if not summary.strip():
            raise RuntimeError("LLM returned an empty summary")
        state[parsed.output_key] = summary
        return summary
