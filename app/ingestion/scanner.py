"""Recursive discovery of supported local documents."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

from app.domain.models import FileFingerprint
from app.ingestion.fingerprint import fingerprint_file
from app.infrastructure.paths import canonicalize_path


@dataclass(frozen=True, slots=True)
class ScanFailure:
    """Privacy-safe description of a file inventory failure."""

    canonical_path: str
    error_type: str


@dataclass(frozen=True, slots=True)
class ScanResult:
    """Successful fingerprints and isolated failures from one scan."""

    fingerprints: tuple[FileFingerprint, ...]
    failures: tuple[ScanFailure, ...]


class FolderScanner:
    """Discover PDF/DOCX files and produce strict fingerprints."""

    def __init__(self, hash_block_size: int = 1024 * 1024) -> None:
        self._hash_block_size = hash_block_size
        self._logger = logging.getLogger("document_intelligence.scanner")

    def scan(self, root: Path) -> ScanResult:
        """Recursively scan a local directory without parsing document content."""

        resolved_root = root.expanduser().resolve(strict=True)
        if not resolved_root.is_dir():
            raise NotADirectoryError(str(resolved_root))

        paths = sorted(
            (
                path
                for path in resolved_root.rglob("*")
                if path.is_file() and path.suffix.lower() in {".pdf", ".docx"}
            ),
            key=canonicalize_path,
        )
        return self.scan_paths(tuple(paths))

    def scan_paths(self, paths: tuple[Path, ...]) -> ScanResult:
        """Fingerprint an explicit set of supported local document paths."""

        unique_paths = sorted(
            {path.expanduser().resolve(strict=True) for path in paths},
            key=canonicalize_path,
        )
        unsupported = [
            path for path in unique_paths
            if path.suffix.lower() not in {".pdf", ".docx"}
        ]
        if unsupported:
            raise ValueError(f"Unsupported document type: {unsupported[0].suffix or unsupported[0].name}")

        fingerprints: list[FileFingerprint] = []
        failures: list[ScanFailure] = []
        for path in unique_paths:
            if not path.is_file():
                failures.append(ScanFailure(canonicalize_path(path), "NotAFileError"))
                continue
            try:
                fingerprints.append(
                    fingerprint_file(path, block_size=self._hash_block_size)
                )
            except (OSError, ValueError) as error:
                safe_path = canonicalize_path(path)
                failures.append(ScanFailure(safe_path, type(error).__name__))
                self._logger.warning(
                    "Document fingerprint failed path=%s error_type=%s",
                    safe_path,
                    type(error).__name__,
                )
        return ScanResult(tuple(fingerprints), tuple(failures))
