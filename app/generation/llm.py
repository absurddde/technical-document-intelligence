"""Local-only LLM backend contracts and llama.cpp-compatible adapter."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import os
from pathlib import Path
import time
from typing import Any, Protocol

from app.generation.models import GenerationSettings
from app.infrastructure.config import LlmConfig


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

    @classmethod
    def from_config(cls, config: LlmConfig) -> LlamaCppGgufBackend:
        """Create the backend solely from validated local configuration."""

        return cls(
            config.model_path, context_size=config.context_size,
            n_gpu_layers=config.n_gpu_layers, n_batch=config.n_batch,
            n_threads=config.n_threads, use_mmap=config.use_mmap,
            use_mlock=config.use_mlock,
        )

    _OUTPUT_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "claims": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string", "pattern": "^CLAIM_[A-Za-z0-9_-]+$"},
                        "text": {"type": "string"},
                        "source_ids": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string", "pattern": "^SOURCE_[0-9]{2,}$"},
                        },
                    },
                    "required": ["claim_id", "text", "source_ids"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["answer", "claims"],
        "additionalProperties": False,
    }

    def __init__(self, model_path: Path, *, context_size: int = 4096,
                 n_gpu_layers: int = 0, n_batch: int = 128,
                 n_threads: int | None = None, use_mmap: bool = True,
                 use_mlock: bool = False, verbose: bool = False) -> None:
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
        self._n_threads = n_threads or sensible_cpu_thread_count()
        load_started = time.perf_counter()
        self._model = Llama(
            model_path=str(path), n_ctx=context_size, n_batch=n_batch,
            n_threads=self._n_threads, n_threads_batch=self._n_threads,
            n_gpu_layers=n_gpu_layers, use_mmap=use_mmap,
            use_mlock=use_mlock, verbose=verbose,
        )
        self.load_time_seconds = time.perf_counter() - load_started
        self.last_generation_seconds: float | None = None
        self.last_completion_tokens: int | None = None
        self.last_tokens_per_second: float | None = None
        self.metadata: dict[str, str] = dict(self._model.metadata)
        self.chat_format: str = self._model.chat_format
        self._runtime = (
            f"llama.cpp cpu-only n_gpu_layers={n_gpu_layers} n_ctx={context_size} "
            f"n_batch={n_batch} n_threads={self._n_threads} "
            f"use_mmap={use_mmap} use_mlock={use_mlock} chat_format={self.chat_format}"
        )

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
        started = time.perf_counter()
        response = self._model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{user_prompt}\n\n{context}\n\n/no_think"},
            ],
            temperature=generation_settings.temperature,
            max_tokens=generation_settings.max_output_tokens,
            seed=generation_settings.seed,
            response_format={"type": "json_object", "schema": self._OUTPUT_SCHEMA},
        )
        self.last_generation_seconds = time.perf_counter() - started
        usage = response.get("usage", {})
        tokens = usage.get("completion_tokens")
        self.last_completion_tokens = int(tokens) if tokens is not None else None
        self.last_tokens_per_second = (
            self.last_completion_tokens / self.last_generation_seconds
            if self.last_completion_tokens is not None and self.last_generation_seconds > 0
            else None
        )
        return str(response["choices"][0]["message"]["content"])


def sensible_cpu_thread_count() -> int:
    """Choose a conservative physical-core estimate, capped at eight threads."""

    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
    except ImportError:
        physical = None
    if not physical:
        logical = os.cpu_count() or 1
        physical = max(1, logical // 2)
    return min(int(physical), 8)
