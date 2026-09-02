"""Local path normalization and validation."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from dataclasses import dataclass


PRODUCT_DIRECTORY = "TechnicalDocumentIntelligence"


def enforce_offline_environment() -> None:
    """Disable download and telemetry paths in Hugging Face-compatible libraries."""

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    """Locations used by source and frozen application runtimes."""

    packaged: bool
    application_dir: Path
    resource_dir: Path
    config_file: Path
    user_data_dir: Path
    database: Path
    indexes: Path
    cache: Path
    logs: Path
    models: Path


def discover_runtime_paths(
    *, frozen: bool | None = None, executable: Path | None = None,
    module_file: Path | None = None, local_app_data: Path | None = None,
    bundle_dir: Path | None = None,
) -> RuntimePaths:
    """Resolve paths without embedding a checkout or user-specific location."""

    is_packaged = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    source_file = module_file or Path(__file__)
    if is_packaged:
        app_dir = (executable or Path(sys.executable)).resolve(strict=False).parent
        resources = (bundle_dir or Path(getattr(sys, "_MEIPASS", app_dir))).resolve(strict=False)
        config_file = app_dir / "config" / "config.toml"
        if not config_file.is_file():
            config_file = resources / "config" / "config.toml"
        local_root = local_app_data or Path(os.environ.get("LOCALAPPDATA", app_dir))
        data_root = local_root / PRODUCT_DIRECTORY
        models = app_dir / "models"
    else:
        app_dir = source_file.resolve(strict=False).parents[2]
        resources = app_dir
        config_file = app_dir / "config" / "config.toml"
        data_root = app_dir / "data"
        models = app_dir / "models"
    return RuntimePaths(
        packaged=is_packaged, application_dir=app_dir, resource_dir=resources,
        config_file=config_file, user_data_dir=data_root,
        database=data_root / "database" / "app.db", indexes=data_root / "indexes",
        cache=data_root / "cache", logs=data_root / "logs", models=models,
    )


def canonicalize_path(path: Path) -> str:
    """Return a stable absolute path key with Windows-aware case handling."""

    resolved = path.expanduser().resolve(strict=False)
    return os.path.normcase(os.path.normpath(str(resolved)))


def resolve_local_path(base_dir: Path, configured_path: str) -> Path:
    """Resolve a configured local path without requiring it to exist."""

    candidate = Path(configured_path).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve(strict=False)


def ensure_local_directories(paths: list[Path]) -> None:
    """Create only the explicitly configured local application directories."""

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
