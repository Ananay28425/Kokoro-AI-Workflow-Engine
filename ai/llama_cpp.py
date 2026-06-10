# Adapts llama-cpp-python so workflow steps can call a local LLM.
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class LlamaCppAdapter:
    """Local LLM adapter used by steps.summarize.

    This class hides the llama-cpp-python API behind the LLMClient contract from
    core.interfaces. The model is loaded lazily so importing the app does not
    immediately require a model file or llama.cpp runtime.
    """

    def __init__(self, model_path: str | None = None, client: Any | None = None, **kwargs: Any) -> None:
        """Store model settings without loading the model yet."""

        self._model_path = model_path or os.getenv("LLAMA_CPP_MODEL_PATH")
        self._kwargs = kwargs
        self._client: Any | None = client

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.2) -> str:
        """Send a prompt to llama.cpp and return the generated text."""

        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        client = self._get_client()
        if callable(client):
            response = client(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        elif hasattr(client, "create_completion"):
            response = client.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            raise RuntimeError("llama.cpp client does not expose a completion API")

        text = self._extract_text(response)
        if not text:
            raise RuntimeError("llama.cpp returned an empty completion")
        return text

    def _extract_text(self, response: Any) -> str:
        """Extract generated text from supported llama.cpp response shapes."""

        try:
            choice = response["choices"][0]
            if "text" in choice:
                return str(choice["text"]).strip()
            if "message" in choice and "content" in choice["message"]:
                return str(choice["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("llama.cpp returned an unexpected response shape") from exc
        raise RuntimeError("llama.cpp returned an unexpected response shape")

    def _get_client(self) -> Any:
        """Load and cache the llama.cpp client the first time it is needed."""

        if self._client is not None:
            return self._client
        if not self._model_path:
            raise RuntimeError(
                "LLAMA_CPP_MODEL_PATH must be set for local inference. "
                "Point it at a local GGUF model file."
            )
        model_path = Path(self._model_path)
        if not model_path.is_file():
            raise RuntimeError(f"LLAMA_CPP_MODEL_PATH points to a missing model file: {model_path}")
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is required for local inference. "
                "Install the local LLM extra or install llama-cpp-python directly."
            ) from exc
        self._client = Llama(model_path=str(model_path), **self._kwargs)
        return self._client
