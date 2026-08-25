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
class ParsingConfig:
    pipeline_version: str = "phase2-v1"
    pdf_min_text_characters: int = 40
    pdf_min_alphanumeric_ratio: float = 0.20
    remove_repeated_headers_footers: bool = True
    repeated_margin_lines: int = 2
    repeated_page_ratio: float = 0.70


@dataclass(frozen=True, slots=True)
class OcrConfig:
    enabled: bool = True
    languages: tuple[str, ...] = ("tur", "eng")
    tesseract_command: str | None = None
    dpi: int = 300


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    chunk_size: int = 1200
    overlap: int = 150


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Complete typed Phase 1 configuration."""

    paths: PathConfig
    indexing: IndexingConfig
    logging: LoggingConfig
    parsing: ParsingConfig
    ocr: OcrConfig
    chunking: ChunkingConfig


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
    parsing = raw.get("parsing", {})
    ocr = raw.get("ocr", {})
    chunking = raw.get("chunking", {})
    hash_block_size = int(indexing.get("hash_block_size", 1024 * 1024))
    if hash_block_size <= 0:
        raise ValueError("indexing.hash_block_size must be positive")

    chunk_size = int(chunking.get("chunk_size", 1200))
    overlap = int(chunking.get("overlap", 150))
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunking requires chunk_size > overlap >= 0")
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
        parsing=ParsingConfig(
            pipeline_version=str(parsing.get("pipeline_version", "phase2-v1")),
            pdf_min_text_characters=int(parsing.get("pdf_min_text_characters", 40)),
            pdf_min_alphanumeric_ratio=float(parsing.get("pdf_min_alphanumeric_ratio", 0.20)),
            remove_repeated_headers_footers=bool(parsing.get("remove_repeated_headers_footers", True)),
            repeated_margin_lines=int(parsing.get("repeated_margin_lines", 2)),
            repeated_page_ratio=float(parsing.get("repeated_page_ratio", 0.70)),
        ),
        ocr=OcrConfig(
            enabled=bool(ocr.get("enabled", True)),
            languages=tuple(str(value) for value in ocr.get("languages", ["tur", "eng"])),
            tesseract_command=str(ocr.get("tesseract_command", "")) or None,
            dpi=int(ocr.get("dpi", 300)),
        ),
        chunking=ChunkingConfig(chunk_size=chunk_size, overlap=overlap),
    )
