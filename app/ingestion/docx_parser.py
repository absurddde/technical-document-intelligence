"""DOCX parser preserving headings, paragraphs, and tables."""

from __future__ import annotations

from pathlib import Path

from app.domain.models import ParsedBlock, ParsedDocument
from app.ingestion.errors import DocumentParseError
from app.processing.cleaner import clean_text


class DocxParser:
    def __init__(self, pipeline_version: str = "phase2-v1") -> None:
        self._version = pipeline_version

    def parse(self, path: Path, document_hash: str) -> ParsedDocument:
        """Parse body elements in document order without inventing page numbers."""

        try:
            from docx import Document as OpenDocument
            from docx.table import Table
            from docx.text.paragraph import Paragraph
            document = OpenDocument(path)
        except Exception as error:
            raise DocumentParseError("Invalid or unreadable DOCX") from error
        headings: list[str] = []
        blocks: list[ParsedBlock] = []
        paragraph_index = 0
        for child in document.element.body.iterchildren():
            if child.tag.endswith("}p"):
                paragraph = Paragraph(child, document)
                text = clean_text(paragraph.text)
                if not text:
                    continue
                style = paragraph.style.name if paragraph.style is not None else ""
                if style.lower().startswith("heading"):
                    try:
                        level = max(1, int(style.split()[-1]))
                    except ValueError:
                        level = 1
                    headings[:] = headings[:level - 1]
                    headings.append(text)
                    blocks.append(ParsedBlock("heading", text, section_title=text, heading_path=tuple(headings), paragraph_index=paragraph_index))
                else:
                    section = headings[-1] if headings else None
                    blocks.append(ParsedBlock("paragraph", text, section_title=section, heading_path=tuple(headings), paragraph_index=paragraph_index))
                paragraph_index += 1
            elif child.tag.endswith("}tbl"):
                table = Table(child, document)
                rows = [" | ".join(clean_text(cell.text) for cell in row.cells) for row in table.rows]
                text = clean_text("\n".join(rows))
                if text:
                    blocks.append(ParsedBlock("table", text, section_title=headings[-1] if headings else None, heading_path=tuple(headings), paragraph_index=paragraph_index))
                    paragraph_index += 1
        return ParsedDocument(tuple(blocks), document_hash, self._version)
