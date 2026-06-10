# Tests the Kokoro adapter behavior without importing the real Kokoro package.
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ai.llama_cpp import LlamaCppAdapter
from ai.kokoro import KokoroTTSAdapter


class CallablePipeline:
    """Fake callable pipeline shaped like a simple Kokoro audio generator."""

    def __call__(self, text: str, voice: str | None = None) -> bytes:
        """Return predictable bytes so the adapter can write them to disk."""

        return f"{voice}:{text}".encode("utf-8")


class GeneratorPipeline:
    """Fake Kokoro pipeline that returns generator-style audio chunks."""

    def __call__(self, text: str, voice: str | None = None) -> list[tuple[str, str, bytes]]:
        """Return chunk tuples shaped like common Kokoro output."""

        return [("graphemes", "phonemes", b"chunk-1"), ("graphemes", "phonemes", b"chunk-2")]


class FakeCompletionClient:
    """Fake llama.cpp client that behaves like the callable API."""

    def __call__(self, prompt: str, *, max_tokens: int, temperature: float) -> dict[str, Any]:
        """Return a completion response shaped like llama-cpp-python."""

        return {"choices": [{"text": f"answer:{prompt}:{max_tokens}:{temperature}"}]}


class FakeCreateCompletionClient:
    """Fake llama.cpp client that behaves like the create_completion API."""

    def create_completion(self, *, prompt: str, max_tokens: int, temperature: float) -> dict[str, Any]:
        """Return a completion response from a method instead of __call__."""

        return {"choices": [{"message": {"content": f"chat:{prompt}"}}]}


def test_kokoro_adapter_writes_callable_pipeline_output(tmp_path: Path) -> None:
    """Check that KokoroTTSAdapter writes bytes from a callable pipeline."""

    adapter = KokoroTTSAdapter()
    adapter._pipeline = CallablePipeline()

    output = adapter.synthesize("hello", tmp_path / "hello.wav", voice="af")

    assert output.read_bytes() == b"af:hello"


def test_kokoro_adapter_writes_generator_pipeline_chunks(tmp_path: Path) -> None:
    """Check that generator-style Kokoro chunks are normalized and written."""

    adapter = KokoroTTSAdapter(pipeline=GeneratorPipeline())

    output = adapter.synthesize("hello", tmp_path / "hello.wav")

    assert output.read_bytes() == b"chunk-1chunk-2"


def test_kokoro_adapter_rejects_directory_output_path(tmp_path: Path) -> None:
    """Check that audio output must point to a file, not a directory."""

    adapter = KokoroTTSAdapter(pipeline=CallablePipeline())

    with pytest.raises(ValueError, match="file path"):
        adapter.synthesize("hello", tmp_path)


def test_llama_adapter_uses_injected_callable_client() -> None:
    """Check llama.cpp adapter behavior without importing llama-cpp-python."""

    adapter = LlamaCppAdapter(client=FakeCompletionClient())

    assert adapter.complete("prompt", max_tokens=3, temperature=0.5) == "answer:prompt:3:0.5"


def test_llama_adapter_uses_create_completion_client() -> None:
    """Check compatibility with clients that expose create_completion."""

    adapter = LlamaCppAdapter(client=FakeCreateCompletionClient())

    assert adapter.complete("prompt") == "chat:prompt"


def test_llama_adapter_rejects_missing_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check that missing LLAMA_CPP_MODEL_PATH gives an actionable error."""

    monkeypatch.delenv("LLAMA_CPP_MODEL_PATH", raising=False)

    with pytest.raises(RuntimeError, match="LLAMA_CPP_MODEL_PATH"):
        LlamaCppAdapter().complete("prompt")


def test_llama_adapter_rejects_missing_model_file(tmp_path: Path) -> None:
    """Check that a bad model path is caught before importing llama.cpp."""

    missing_model = tmp_path / "missing.gguf"

    with pytest.raises(RuntimeError, match="missing model file"):
        LlamaCppAdapter(model_path=str(missing_model)).complete("prompt")


def test_llama_adapter_rejects_unexpected_response_shape() -> None:
    """Check that malformed model responses fail loudly."""

    adapter = LlamaCppAdapter(client=lambda prompt, max_tokens, temperature: {"choices": [{}]})

    with pytest.raises(RuntimeError, match="unexpected response shape"):
        adapter.complete("prompt")
