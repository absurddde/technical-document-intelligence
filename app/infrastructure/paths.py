"""Local path normalization and validation."""

from __future__ import annotations

import os
from pathlib import Path


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

