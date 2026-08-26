from pathlib import Path

import pytest

from app.infrastructure.config import load_config


def test_load_config_resolves_paths_from_project_root(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "config.toml"
    config_path.write_text(
        """
[paths]
database = "data/database/app.db"
indexes = "data/indexes"
cache = "data/cache"
logs = "data/logs"
models = "models"
[indexing]
strict_hash_verification = true
hash_block_size = 4096
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.paths.database == (tmp_path / "data/database/app.db").resolve()
    assert config.indexing.strict_hash_verification is True
    assert config.indexing.hash_block_size == 4096
    assert config.embedding.model_path == (tmp_path / "models/embedding/bge-m3").resolve()
    assert config.embedding.input_format == "raw"
    assert config.llm.model_path == (
        tmp_path / "models/llm/qwen3-8b/Qwen3-8B-Q4_K_M.gguf"
    ).resolve()
    assert config.llm.n_gpu_layers == 0 and config.llm.n_batch == 128


def test_load_config_rejects_missing_local_paths(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[paths]\ndatabase='db.sqlite'", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing path configuration"):
        load_config(path)
