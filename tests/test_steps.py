# Tests concrete workflow steps with fake AI and speech adapters.
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from steps.read_file import ReadFileStep
from steps.speak import SpeakStep
from steps.summarize import SummarizeStep


class FakeLLM:
    """Fake LLM adapter used by SummarizeStep tests."""

    def __init__(self, response: str = "short summary") -> None:
        """Store the fake response returned to the summarize step."""

        self.response = response

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.2) -> str:
        """Return a predictable summary without loading llama.cpp."""

        assert "source text" in prompt
        return self.response


class FakeTTS:
    """Fake TTS adapter used by SpeakStep tests."""

    def synthesize(self, text: str, output_path: Path, *, voice: str | None = None) -> Path:
        """Write small fake audio bytes without loading Kokoro."""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"audio")
        return output_path


class DirectoryTTS:
    """Fake TTS adapter that incorrectly returns a directory path."""

    def synthesize(self, text: str, output_path: Path, *, voice: str | None = None) -> Path:
        """Return a directory so SpeakStep can reject it."""

        output_path.mkdir(parents=True, exist_ok=True)
        return output_path


def test_read_file_step_stores_text(tmp_path: Path) -> None:
    """Check that ReadFileStep reads a file and stores its text in state."""

    source = tmp_path / "input.txt"
    source.write_text("hello", encoding="utf-8")
    state: dict[str, object] = {}

    result = ReadFileStep().run(state, {"path": source, "output_key": "text"})

    assert result == "hello"
    assert state["text"] == "hello"


def test_read_file_step_rejects_directory_path(tmp_path: Path) -> None:
    """Check that ReadFileStep rejects directories before reading."""

    with pytest.raises(FileNotFoundError, match="readable file"):
        ReadFileStep().run({}, {"path": tmp_path})


def test_read_file_step_rejects_invalid_utf8(tmp_path: Path) -> None:
    """Check that binary or invalid text files produce a clear error."""

    source = tmp_path / "binary.txt"
    source.write_bytes(b"\xff\xfe")

    with pytest.raises(ValueError, match="UTF-8"):
        ReadFileStep().run({}, {"path": source})


def test_read_file_step_rejects_unknown_config_fields(tmp_path: Path) -> None:
    """Check that typoed YAML config fields are not silently ignored."""

    source = tmp_path / "input.txt"
    source.write_text("hello", encoding="utf-8")

    with pytest.raises(ValidationError):
        ReadFileStep().run({}, {"path": source, "unknown": True})


def test_summarize_step_uses_llm_and_stores_summary() -> None:
    """Check that SummarizeStep calls the LLM and stores the summary."""

    state: dict[str, object] = {"text": "source text"}

    result = SummarizeStep(FakeLLM()).run(state, {"input_key": "text", "output_key": "summary"})

    assert result == "short summary"
    assert state["summary"] == "short summary"


def test_summarize_step_rejects_missing_text() -> None:
    """Check that SummarizeStep fails clearly when prior steps did not produce text."""

    with pytest.raises(ValueError, match="non-empty text"):
        SummarizeStep(FakeLLM()).run({}, {})


def test_summarize_step_rejects_prompt_without_text_placeholder() -> None:
    """Check that prompts must include the workflow text."""

    with pytest.raises(ValidationError):
        SummarizeStep(FakeLLM()).run({"text": "source text"}, {"prompt": "Summarize this"})


def test_summarize_step_rejects_unsupported_prompt_placeholder() -> None:
    """Check that unsupported prompt variables fail during validation."""

    with pytest.raises(ValidationError):
        SummarizeStep(FakeLLM()).run({"text": "source text"}, {"prompt": "{text} {other}"})


def test_summarize_step_rejects_empty_llm_response() -> None:
    """Check that empty model responses are treated as failures."""

    with pytest.raises(RuntimeError, match="empty summary"):
        SummarizeStep(FakeLLM("  ")).run({"text": "source text"}, {})


def test_summarize_step_rejects_invalid_token_boundary() -> None:
    """Check that max_tokens has a useful lower boundary."""

    with pytest.raises(ValidationError):
        SummarizeStep(FakeLLM()).run({"text": "source text"}, {"max_tokens": 0})


def test_speak_step_uses_tts_and_stores_audio_path(tmp_path: Path) -> None:
    """Check that SpeakStep calls TTS and stores the generated audio path."""

    output = tmp_path / "out.wav"
    state: dict[str, object] = {"summary": "say this"}

    result = SpeakStep(FakeTTS()).run(state, {"output_path": output})

    assert result == str(output)
    assert output.read_bytes() == b"audio"
    assert state["audio_path"] == str(output)


def test_speak_step_rejects_missing_text() -> None:
    """Check that SpeakStep fails clearly when no summary exists."""

    with pytest.raises(ValueError, match="non-empty text"):
        SpeakStep(FakeTTS()).run({}, {})


def test_speak_step_rejects_directory_returned_by_tts(tmp_path: Path) -> None:
    """Check that SpeakStep catches a broken TTS adapter result."""

    with pytest.raises(RuntimeError, match="directory"):
        SpeakStep(DirectoryTTS()).run({"summary": "say this"}, {"output_path": tmp_path / "audio_dir"})
