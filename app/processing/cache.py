"""Private local parsed/OCR result cache keyed by hash and pipeline version."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from app.domain.models import ParsedBlock, ParsedDocument


class ParsedDocumentCache:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve(strict=False)

    def _path(self, document_hash: str, pipeline_version: str) -> Path:
        version_key = hashlib.sha256(pipeline_version.encode("utf-8")).hexdigest()[:12]
        return self._root / "parsed" / f"{document_hash}-{version_key}.json"

    def load(self, document_hash: str, pipeline_version: str) -> ParsedDocument | None:
        """Return a valid exact-version cache entry or safely ignore it."""

        path = self._path(document_hash, pipeline_version)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["document_hash"] != document_hash or data["pipeline_version"] != pipeline_version:
                return None
            blocks = tuple(ParsedBlock(**{**item, "heading_path": tuple(item.get("heading_path", ()))}) for item in data["blocks"])
            return ParsedDocument(blocks, document_hash, pipeline_version)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def save(self, document: ParsedDocument) -> None:
        """Persist parsed text locally using an atomic replace."""

        path = self._path(document.document_hash, document.pipeline_version)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        payload = {"document_hash": document.document_hash, "pipeline_version": document.pipeline_version,
                   "blocks": [asdict(block) for block in document.blocks]}
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
