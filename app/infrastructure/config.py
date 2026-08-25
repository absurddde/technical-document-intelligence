"""Typed application configuration loaded from a local TOML file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

from app.infrastructure.paths import resolve_local_path


@dataclass(frozen=True, slots=True)
class PathConfig:
    """Resolved local storage locations."""

    database: Path
    indexes: Path
    cache: Path
    logs: Path
    models: Path


@dataclass(frozen=True, slots=True)
class IndexingConfig:
    """Phase 1 indexing behavior."""

    strict_hash_verification: bool = True
    hash_block_size: int = 1024 * 1024


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Privacy-safe rotating logging settings."""

    level: str = "INFO"
    max_bytes: int = 2_000_000
    backup_count: int = 3


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Complete typed Phase 1 configuration."""

    paths: PathConfig
    indexing: IndexingConfig
    logging: LoggingConfig


def load_config(config_path: Path) -> AppConfig:
    """Load and validate configuration from a local TOML file."""

    resolved_config = config_path.expanduser().resolve(strict=True)
    with resolved_config.open("rb") as stream:
        raw = tomllib.load(stream)

    base_dir = resolved_config.parent.parent
    path_values = raw.get("paths", {})
    required = ("database", "indexes", "cache", "logs", "models")
    missing = [key for key in required if not path_values.get(key)]
    if missing:
        raise ValueError(f"Missing path configuration: {', '.join(missing)}")

    indexing = raw.get("indexing", {})
    logging = raw.get("logging", {})
    hash_block_size = int(indexing.get("hash_block_size", 1024 * 1024))
    if hash_block_size <= 0:
        raise ValueError("indexing.hash_block_size must be positive")

    return AppConfig(
        paths=PathConfig(
            **{
                key: resolve_local_path(base_dir, str(path_values[key]))
                for key in required
            }
        ),
        indexing=IndexingConfig(
            strict_hash_verification=bool(
                indexing.get("strict_hash_verification", True)
            ),
            hash_block_size=hash_block_size,
        ),
        logging=LoggingConfig(
            level=str(logging.get("level", "INFO")).upper(),
            max_bytes=int(logging.get("max_bytes", 2_000_000)),
            backup_count=int(logging.get("backup_count", 3)),
        ),
    )

