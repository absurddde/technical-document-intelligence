"""Build deterministic, provenance-preserving untrusted source blocks."""

from __future__ import annotations

from html import escape

from app.generation.models import BuiltContext, SourceRecord
from app.retrieval.models import FusedCandidate


class ContextBuilder:
    """Render only selected chunks and map deterministic request-local IDs."""

    def build(self, chunks: tuple[FusedCandidate, ...], maximum: int) -> BuiltContext:
        sources: list[SourceRecord] = []
        blocks: list[str] = []
        for index, chunk in enumerate(chunks[:maximum], 1):
            source_id = f"SOURCE_{index:02d}"
            source = SourceRecord(
                source_id, chunk.chunk_id, chunk.document_id, chunk.file_name,
                chunk.page_start, chunk.page_end, chunk.section_title, chunk.text,
            )
            sources.append(source)
            page = self._locator(chunk.page_start, chunk.page_end)
            blocks.append(
                f'<SOURCE id="{source_id}">\n'
                f'document_id: {escape(chunk.document_id)}\n'
                f'chunk_id: {escape(chunk.chunk_id)}\n'
                f'file_name: {escape(chunk.file_name)}\n'
                f'page: {page}\n'
                f'section: {escape(chunk.section_title or "")}\n'
                f'text:\n{escape(chunk.text, quote=False)}\n</SOURCE>'
            )
        return BuiltContext("\n\n".join(blocks), tuple(sources))

    @staticmethod
    def _locator(start: int | None, end: int | None) -> str:
        if start is None:
            return ""
        return str(start) if end in (None, start) else f"{start}-{end}"
