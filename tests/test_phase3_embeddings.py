from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from app.retrieval.embeddings import (EmbeddingBackend, LocalModelError,
                                      SentenceTransformerEmbeddingBackend)


class FakeModel:
    def get_sentence_embedding_dimension(self): return 3
    def encode(self, texts, **kwargs):
        self.texts = texts
        self.kwargs = kwargs
        return np.ones((len(texts), 3), dtype=np.float32)


def test_embedding_protocol_missing_model_and_e5_offline_flags(tmp_path: Path) -> None:
    with pytest.raises(LocalModelError, match="does not exist"):
        SentenceTransformerEmbeddingBackend(tmp_path / "missing")

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    fake = FakeModel()
    with patch("sentence_transformers.SentenceTransformer", return_value=fake) as constructor:
        backend = SentenceTransformerEmbeddingBackend(model_dir, input_format="e5")
        backend.embed_documents(["English passage"])
        assert fake.texts == ["passage: English passage"]
        backend.embed_query("Türkçe sorgu")
        assert fake.texts == ["query: Türkçe sorgu"]
        assert fake.kwargs["normalize_embeddings"] is True
        assert isinstance(backend, EmbeddingBackend)
    kwargs = constructor.call_args.kwargs
    assert kwargs["local_files_only"] is True
    assert kwargs["trust_remote_code"] is False
    assert Path(constructor.call_args.args[0]).is_absolute()


def test_raw_input_format_does_not_add_e5_prefixes(tmp_path: Path) -> None:
    model_dir = tmp_path / "bge"
    model_dir.mkdir()
    fake = FakeModel()
    with patch("sentence_transformers.SentenceTransformer", return_value=fake):
        backend = SentenceTransformerEmbeddingBackend(model_dir, input_format="raw")
        backend.embed_documents(["Raw passage"])
        assert fake.texts == ["Raw passage"]
        backend.embed_query("Raw query")
        assert fake.texts == ["Raw query"]


def test_input_format_changes_model_fingerprint(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with patch("sentence_transformers.SentenceTransformer", return_value=FakeModel()):
        raw = SentenceTransformerEmbeddingBackend(model_dir, input_format="raw")
        e5 = SentenceTransformerEmbeddingBackend(model_dir, input_format="e5")
    assert raw.model_fingerprint != e5.model_fingerprint
