"""Strict content fingerprint generation."""

from __future__ import annotations

import hashlib
from pathlib import Path

from app.domain.models import FileFingerprint
from app.infrastructure.paths import canonicalize_path


def calculate_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    """Calculate the exact SHA-256 content identity of a local file."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_file(path: Path, block_size: int = 1024 * 1024) -> FileFingerprint:
    """Read metadata and the strict hash for one supported document."""

    resolved = path.expanduser().resolve(strict=True)
    stat = resolved.stat()
    suffix = resolved.suffix.lower().lstrip(".")
    if suffix not in {"pdf", "docx"}:
        raise ValueError(f"Unsupported document type: {resolved.suffix}")
    return FileFingerprint(
        path=resolved,
        canonical_path=canonicalize_path(resolved),
        file_name=resolved.name,
        file_type=suffix,
        file_size=stat.st_size,
        modified_ns=stat.st_mtime_ns,
        sha256=calculate_sha256(resolved, block_size),
    )

