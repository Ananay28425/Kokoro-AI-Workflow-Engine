# Implements the workflow step that turns text into speech with Kokoro.
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core.interfaces import TTSClient
from steps.base import BaseStep


class SpeakConfig(BaseModel):
    """Validated settings for SpeakStep.

    YAML chooses which text to read from state, where to save the audio file,
    and which optional Kokoro voice to use.
    """

    model_config = ConfigDict(extra="forbid")

    input_key: str = Field(default="summary", min_length=1)
    output_key: str = Field(default="audio_path", min_length=1)
    output_path: Path = Path("assets/audio/output.wav")
    voice: str | None = None


class SpeakStep(BaseStep):
    """Generate a local audio file from text in workflow state.

    The step depends on core.interfaces.TTSClient, so production can use
    ai.kokoro.KokoroTTSAdapter and tests can use a fake adapter.
    """

    def __init__(self, tts: TTSClient) -> None:
        """Receive the text-to-speech adapter used by this step."""

        super().__init__("speak")
        self._tts = tts

    def run(self, state: dict[str, Any], config: dict[str, Any]) -> str:
        """Read text from state, synthesize speech, and store the audio path."""

        parsed = SpeakConfig.model_validate(config)
        text = state.get(parsed.input_key)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"state key '{parsed.input_key}' must contain non-empty text")
        output_path = self._tts.synthesize(text, parsed.output_path, voice=parsed.voice)
        if output_path.is_dir():
            raise RuntimeError(f"TTS output path is a directory, not a file: {output_path}")
        state[parsed.output_key] = str(output_path)
        return str(output_path)
