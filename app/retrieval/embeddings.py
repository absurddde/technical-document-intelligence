"""Offline embedding abstractions and a local Sentence Transformers backend."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


class LocalModelError(RuntimeError):
    """Raised when a configured local embedding model cannot be used."""


@runtime_checkable
class EmbeddingBackend(Protocol):
    @property
    def dimension(self) -> int: ...
    @property
    def model_fingerprint(self) -> str: ...
    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]: ...
    def embed_query(self, text: str) -> NDArray[np.float32]: ...


class SentenceTransformerEmbeddingBackend:
    """Normalized embeddings loaded exclusively from a local directory."""

    def __init__(self, model_path: Path, *, device: str = "cpu", batch_size: int = 32,
                 input_format: Literal["raw", "e5"] = "raw") -> None:
        path = model_path.expanduser().resolve(strict=False)
        if not path.is_dir():
            raise LocalModelError(f"Local embedding model directory does not exist: {path}")
        if input_format not in {"raw", "e5"}:
            raise ValueError(f"Unsupported embedding input format: {input_format}")
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                str(path), device=device, local_files_only=True, trust_remote_code=False
            )
        except ImportError as error:
            raise LocalModelError("sentence-transformers is not installed") from error
        except Exception as error:
            raise LocalModelError(f"Could not load local embedding model at {path}: {error}") from error
        self._batch_size = batch_size
        self._input_format = input_format
        dimension_getter = getattr(
            self._model,
            "get_embedding_dimension",
            self._model.get_sentence_embedding_dimension,
        )
        self._dimension = int(dimension_getter())
        fingerprint = hashlib.sha256(
            f"{path}|{self._dimension}|input_format={input_format}".encode("utf-8")
        )
        for item in sorted((entry for entry in path.rglob("*") if entry.is_file()), key=lambda p: str(p.relative_to(path))):
            stat = item.stat()
            fingerprint.update(f"{item.relative_to(path)}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8"))
        self._fingerprint = fingerprint.hexdigest()

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_fingerprint(self) -> str:
        return self._fingerprint

    def _encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        values = self._model.encode(
            list(texts), batch_size=self._batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        return np.asarray(values, dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        formatted = (
            [f"passage: {text}" for text in texts]
            if self._input_format == "e5"
            else list(texts)
        )
        return self._encode(formatted)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        formatted = f"query: {text}" if self._input_format == "e5" else text
        return self._encode([formatted])[0]
