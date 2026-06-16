# Adapts Kokoro so workflow steps can synthesize local speech audio.
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any


class KokoroTTSAdapter:
    """Local text-to-speech adapter used by steps.speak.

    This class hides the Kokoro package behind the TTSClient contract from
    core.interfaces. The Kokoro pipeline is loaded lazily so tests and CLI
    validation do not need audio dependencies.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        lang_code: str = "a",
        sample_rate: int = 24000,
        pipeline: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self._model = model
        self._lang_code = lang_code
        self._sample_rate = sample_rate
        self._kwargs = kwargs
        self._pipeline: Any | None = pipeline

    def synthesize(
        self, text: str, output_path: Path, *, voice: str | None = None
    ) -> Path:
        if not text.strip():
            raise ValueError("text must not be empty")
        if output_path.exists() and output_path.is_dir():
            raise ValueError(
                f"audio output path must be a file path, not a directory: {output_path}"
            )

        voice = voice or "af_heart"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pipeline = self._get_pipeline()

        if hasattr(pipeline, "save"):
            pipeline.save(text, str(output_path), voice=voice)
        elif callable(pipeline):
            audio = pipeline(text, voice=voice)
            self._write_audio_output(audio, output_path)
        else:
            raise RuntimeError("unsupported Kokoro pipeline interface")

        if not output_path.exists():
            raise RuntimeError(
                f"Kokoro did not create the expected audio file: {output_path}"
            )
        return output_path

    def _write_audio_output(self, audio: Any, output_path: Path) -> None:
        """Write bytes or Kokoro audio chunks to the requested file path."""

        chunks = self._extract_audio_chunks(audio)
        if not chunks:
            raise RuntimeError("Kokoro returned no audio data")

        if all(isinstance(chunk, (bytes, bytearray)) for chunk in chunks):
            output_path.write_bytes(b"".join(bytes(chunk) for chunk in chunks))
            return

        try:
            import numpy as np
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError(
                "Kokoro produced numeric audio data, so numpy and soundfile are required "
                "to write the WAV file."
            ) from exc

        normalized: list[Any] = []
        for chunk in chunks:
            if isinstance(chunk, (bytes, bytearray)):
                raise RuntimeError(
                    "Kokoro returned mixed binary and numeric audio data"
                )

            try:
                arr = np.asarray(chunk, dtype=np.float32)
            except Exception as exc:
                raise RuntimeError(
                    f"Kokoro returned an unsupported audio chunk type: {type(chunk)!r}"
                ) from exc

            if arr.size == 0:
                continue

            normalized.append(arr.reshape(-1))

        if not normalized:
            raise RuntimeError("Kokoro returned no usable numeric audio data")

        audio_data = (
            normalized[0]
            if len(normalized) == 1
            else np.concatenate(normalized, axis=0)
        )
        sf.write(str(output_path), audio_data, self._sample_rate)

    def _extract_audio_chunks(self, audio: Any) -> list[Any]:
        """Normalize Kokoro output into a list of audio chunks."""

        if isinstance(audio, (bytes, bytearray)):
            return [audio]

        try:
            import numpy as np

            if isinstance(audio, np.ndarray):
                return [audio]
        except Exception:
            pass

        direct_audio = getattr(audio, "audio", None)
        if direct_audio is not None:
            return [direct_audio]

        if isinstance(audio, tuple):
            if len(audio) >= 3:
                return [audio[2]]
            return [audio]

        if isinstance(audio, Iterable) and not isinstance(
            audio, (str, bytes, bytearray, dict)
        ):
            chunks: list[Any] = []
            for item in audio:
                if isinstance(item, (bytes, bytearray)):
                    chunks.append(item)
                    continue

                try:
                    import numpy as np

                    if isinstance(item, np.ndarray):
                        chunks.append(item)
                        continue
                except Exception:
                    pass

                item_audio = getattr(item, "audio", None)
                if item_audio is not None:
                    chunks.append(item_audio)
                    continue

                if isinstance(item, tuple) and len(item) >= 3:
                    chunks.append(item[2])
                    continue

                chunks.append(item)

            if chunks:
                return chunks

        return [audio]

    def _get_pipeline(self) -> Any:
        """Load and cache the Kokoro pipeline the first time speech is needed."""

        if self._pipeline is not None:
            return self._pipeline
        try:
            from kokoro import KPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Kokoro is required for local text-to-speech. "
                "Install the local TTS extra or install Kokoro directly."
            ) from exc

        pipeline_kwargs = {"lang_code": self._lang_code, **self._kwargs}
        if self._model is not None:
            pipeline_kwargs["model"] = self._model

        try:
            self._pipeline = KPipeline(**pipeline_kwargs)
        except TypeError as exc:
            if "model" not in pipeline_kwargs:
                raise RuntimeError(
                    f"could not initialize Kokoro pipeline: {exc}"
                ) from exc
            pipeline_kwargs.pop("model")
            self._pipeline = KPipeline(**pipeline_kwargs)

        return self._pipeline
