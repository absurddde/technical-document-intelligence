"""Structure-aware document chunking."""

from __future__ import annotations

import hashlib
import re

from app.domain.models import DocumentChunk, ParsedBlock
from app.processing.language import detect_language


class StructureAwareChunker:
    def __init__(self, chunk_size: int = 1200, overlap: int = 150) -> None:
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("chunk_size must be greater than overlap")
        self._size = chunk_size
        self._overlap = overlap

    def _split_block(self, block: ParsedBlock) -> list[ParsedBlock]:
        if len(block.text) <= self._size:
            return [block]
        sentences = re.split(r"(?<=[.!?])\s+", block.text)
        parts: list[str] = []
        current = ""
        for sentence in sentences:
            tokens = [sentence[i:i + self._size] for i in range(0, len(sentence), self._size)] if len(sentence) > self._size else [sentence]
            for token in tokens:
                candidate = f"{current} {token}".strip()
                if current and len(candidate) > self._size:
                    parts.append(current)
                    current = token
                else:
                    current = candidate
        if current:
            parts.append(current)
        return [ParsedBlock(block.kind, part, block.page_number, block.section_title,
                            block.heading_path, block.paragraph_index, block.ocr_used,
                            block.ocr_confidence) for part in parts]

    def chunk(self, blocks: tuple[ParsedBlock, ...], *, document_id: str,
              file_name: str, file_path: str) -> tuple[DocumentChunk, ...]:
        """Group whole structural units, breaking only oversized individual units."""

        units = [unit for block in blocks for unit in self._split_block(block)]
        groups: list[list[ParsedBlock]] = []
        current: list[ParsedBlock] = []
        current_length = 0
        for unit in units:
            boundary = current and (unit.page_number != current[-1].page_number or unit.section_title != current[-1].section_title)
            if current and (boundary or current_length + len(unit.text) + 2 > self._size):
                groups.append(current)
                overlap_units: list[ParsedBlock] = []
                overlap_length = 0
                if not boundary:
                    for prior in reversed(current):
                        proposed = overlap_length + len(prior.text) + (2 if overlap_units else 0)
                        if proposed > self._overlap:
                            break
                        if proposed + len(unit.text) + 2 > self._size:
                            break
                        overlap_units.insert(0, prior)
                        overlap_length = proposed
                current = overlap_units
                current_length = overlap_length
            current.append(unit)
            current_length += len(unit.text) + 2
        if current:
            groups.append(current)

        chunks: list[DocumentChunk] = []
        for ordinal, group in enumerate(groups):
            text = "\n\n".join(block.text for block in group)
            pages = [block.page_number for block in group if block.page_number is not None]
            paragraphs = [block.paragraph_index for block in group if block.paragraph_index is not None]
            confidences = [block.ocr_confidence for block in group if block.ocr_confidence is not None]
            identity = hashlib.sha256(f"{document_id}:{ordinal}:{text}".encode("utf-8")).hexdigest()[:24]
            chunks.append(DocumentChunk(
                document_id, file_name, file_path, min(pages) if pages else None,
                max(pages) if pages else None, group[-1].section_title,
                min(paragraphs) if paragraphs else None, max(paragraphs) if paragraphs else None,
                identity, detect_language(text), text, any(block.ocr_used for block in group),
                sum(confidences) / len(confidences) if confidences else None,
            ))
        return tuple(chunks)
