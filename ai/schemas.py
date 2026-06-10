# Defines small validated request objects for AI adapter inputs.
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class CompletionRequest(BaseModel):
    """Validated data needed to ask the local LLM for text."""

    prompt: str
    max_tokens: int = 512
    temperature: float = 0.2


class SpeechRequest(BaseModel):
    """Validated data needed to ask Kokoro to create an audio file."""

    text: str
    output_path: Path
    voice: str | None = None
