"""Local-only LLM backend contracts and llama.cpp-compatible adapter."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
from pathlib import Path
from typing import Protocol

from app.generation.models import GenerationSettings


class LlmBackend(Protocol):
    @property
    def backend_name(self) -> str: ...
    @property
    def model_fingerprint(self) -> str: ...
    @property
    def context_size(self) -> int | None: ...
    @property
    def local_model_path(self) -> Path | None: ...
    @property
    def runtime_info(self) -> str: ...
    def generate(self, system_prompt: str, user_prompt: str, context: str,
                 generation_settings: GenerationSettings) -> str: ...


class FakeLlmBackend:
    """Deterministically return queued JSON outputs for fast validation tests."""

    backend_name = "fake-local"
    model_fingerprint = "fake-v1"
    context_size = 4096
    local_model_path = None
    runtime_info = "deterministic-test"

    def __init__(self, outputs: Sequence[str]) -> None:
        self._outputs = tuple(outputs)
        self.calls: list[tuple[str, str, str, GenerationSettings]] = []

    def generate(self, system_prompt: str, user_prompt: str, context: str,
                 generation_settings: GenerationSettings) -> str:
        self.calls.append((system_prompt, user_prompt, context, generation_settings))
        index = min(len(self.calls) - 1, len(self._outputs) - 1)
        if index < 0:
            raise RuntimeError("FakeLlmBackend requires at least one output")
        return self._outputs[index]


class LlamaCppGgufBackend:
    """Optional local GGUF adapter; imports llama-cpp-python only when selected."""

    backend_name = "llama.cpp"

    def __init__(self, model_path: Path, *, context_size: int = 4096,
                 n_gpu_layers: int = 0) -> None:
        path = model_path.expanduser().resolve(strict=False)
        if not path.is_file() or path.suffix.casefold() != ".gguf":
            raise RuntimeError(f"Local GGUF model file does not exist: {path}")
        try:
            from llama_cpp import Llama
        except ImportError as error:
            raise RuntimeError("llama-cpp-python is not installed") from error
        stat = path.stat()
        self._path = path
        self._context_size = context_size
        self._fingerprint = hashlib.sha256(
            f"{path}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
        ).hexdigest()
        self._runtime = f"llama.cpp n_gpu_layers={n_gpu_layers}"
        self._model = Llama(model_path=str(path), n_ctx=context_size,
                            n_gpu_layers=n_gpu_layers, verbose=False)

    @property
    def model_fingerprint(self) -> str:
        return self._fingerprint

    @property
    def context_size(self) -> int:
        return self._context_size

    @property
    def local_model_path(self) -> Path:
        return self._path

    @property
    def runtime_info(self) -> str:
        return self._runtime

    def generate(self, system_prompt: str, user_prompt: str, context: str,
                 generation_settings: GenerationSettings) -> str:
        response = self._model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{user_prompt}\n\n{context}"},
            ],
            temperature=generation_settings.temperature,
            max_tokens=generation_settings.max_output_tokens,
            seed=generation_settings.seed,
            response_format={"type": "json_object"},
        )
        return str(response["choices"][0]["message"]["content"])
