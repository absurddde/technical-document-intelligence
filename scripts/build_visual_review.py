"""Build a local HTML visual-QA package from parsed Phase 2 pilot results."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from html import escape
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any


@dataclass(frozen=True, slots=True)
class ReviewPage:
    file_name: str
    page_number: int
    rationale: str
    assessment: str


DEFAULT_PAGES = (
    ReviewPage("BarısBilgiliAnlatım.pdf", 1, "OCR-produced Turkish text", "OCR is visibly related to the handwritten source and retains several acronyms, but handwriting recognition is noisy and incomplete."),
    ReviewPage("BarısBilgiliAnlatım.pdf", 2, "Second OCR page", "OCR is recognizably related to the visible notes, with substantial character and word corruption typical of this handwriting."),
    ReviewPage("20220126_army-approach-to-ras_final.pdf", 4, "Army RAS duplicate investigation, first page", "The source differs from page 5. Empty decorative grid output is gone, but three-column text is interleaved and is not reliable linear reading order."),
    ReviewPage("20220126_army-approach-to-ras_final.pdf", 5, "Army RAS duplicate investigation, second page", "The source differs from page 4. The former duplicate was an empty-grid artifact; remaining multi-column text is interleaved."),
    ReviewPage("Suru_Zekasi_OTAG_Sonuc_Raporu.pdf", 18, "Turkish long-report diagram page", "Narrative text is usable in top-down order; diagram labels are retained but flattened into the surrounding block stream."),
    ReviewPage("Suru_Zekasi_OTAG_Sonuc_Raporu.pdf", 100, "Turkish long-report table page", "The native paragraph stream interleaves table cells, but a structured table block retains the rows and columns in usable pipe-delimited form."),
    ReviewPage("İnsansız Kara Araçları için Lİdar Teknolojisi 3B Haritalama.pdf", 10, "Turkish numerical/technical table page", "Technical prose reads correctly; numerical point-cloud values are retained in a structured table block."),
    ReviewPage("1608994972_stm-elektronik-harbin-yeniden-yukselisi.pdf", 4, "Dense Turkish multi-column page", "The visual page has two columns, while extracted blocks alternate between columns and sometimes join their lines; linear reading order is not usable."),
    ReviewPage("Future Warfare and Critical Technology.pdf", 4, "English long-report control page", "English text and publication metadata follow the visible order; a meaningful table-like layout is also retained."),
    ReviewPage("Future Warfare and Critical Technology.pdf", 108, "English long-report OCR-triggered divider", "OCR correctly preserves the visible section-divider title."),
    ReviewPage("ENABLING MUM-T WITHIN ARMY FORMATIONS.pdf", 10, "Table-heavy English MUM-T page", "Slide text, acronyms, and callouts are retained; visual grouping is flattened, while meaningful table/callout blocks remain available."),
    ReviewPage("WRS20 Modern Maritime Communications.pdf", 1, "OCR-triggered maritime communications page", "OCR preserves the visible title, date, event name, URL, and acronym with minor punctuation noise."),
    ReviewPage("WRS20 Modern Maritime Communications.pdf", 8, "Normal English technical control", "Headings and bullet text are retained in a usable top-down order."),
)


def connect(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    return connection


def render_page(pdf_path: Path, page_number: int, output_path: Path, scale: float = 2.25) -> None:
    """Render one PDF page locally to a readable PNG."""

    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(pdf_path))
    try:
        if not 1 <= page_number <= len(document):
            raise ValueError(f"Page {page_number} is outside {pdf_path.name}")
        image = document[page_number - 1].render(scale=scale).to_pil()
        image.save(output_path, format="PNG", optimize=True)
    finally:
        document.close()


def multi_column_signal(pdf_path: Path, page_number: int) -> dict[str, Any]:
    """Calculate a diagnostic page-geometry signal without reparsing the document."""

    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number - 1]
        words = page.extract_words() or []
        lines: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for word in words:
            lines[round(float(word["top"]) / 4)].append(word)
        split_lines = 0
        for line_words in lines.values():
            ordered = sorted(line_words, key=lambda item: float(item["x0"]))
            gaps = [float(right["x0"]) - float(left["x1"]) for left, right in zip(ordered, ordered[1:])]
            if gaps and max(gaps) > float(page.width) * 0.18:
                split_lines += 1
        return {"possible_multi_column": split_lines >= 4, "wide_gap_line_count": split_lines}


def document_repetition_metadata(connection: sqlite3.Connection, document_id: int) -> tuple[set[str], Counter[str]]:
    rows = connection.execute(
        "SELECT text,page_number FROM parsed_blocks WHERE document_id=?", (document_id,)
    ).fetchall()
    pages_by_text: dict[str, set[int]] = defaultdict(set)
    counts: Counter[str] = Counter()
    for row in rows:
        normalized = " ".join(row["text"].casefold().split())
        counts[normalized] += 1
        if row["page_number"] is not None and 3 <= len(normalized) <= 180:
            pages_by_text[normalized].add(int(row["page_number"]))
    repeated_margins = {text for text, pages in pages_by_text.items() if len(pages) >= 3}
    return repeated_margins, counts


def page_data(connection: sqlite3.Connection, review: ReviewPage) -> dict[str, Any]:
    document = connection.execute(
        "SELECT id,document_id FROM documents WHERE file_name=?", (review.file_name,)
    ).fetchone()
    if document is None:
        raise RuntimeError(f"Parsed document is missing: {review.file_name}")
    blocks = connection.execute(
        """SELECT ordinal,kind,text,page_number,section_title,ocr_used,ocr_confidence
           FROM parsed_blocks WHERE document_id=? AND page_number=? ORDER BY ordinal""",
        (document["id"], review.page_number),
    ).fetchall()
    chunks = connection.execute(
        """SELECT ordinal,chunk_id,page_start,page_end,language,ocr_used,ocr_confidence
           FROM chunks WHERE document_id=? AND page_start<=? AND page_end>=? ORDER BY ordinal""",
        (document["id"], review.page_number, review.page_number),
    ).fetchall()
    repeated_margins, counts = document_repetition_metadata(connection, int(document["id"]))
    block_items: list[dict[str, Any]] = []
    flags: set[str] = set()
    for row in blocks:
        normalized = " ".join(row["text"].casefold().split())
        block_flags: list[str] = []
        if len(row["text"]) < 40:
            block_flags.append("very_short_block")
        if len(row["text"]) > 1200:
            block_flags.append("very_long_block")
        if normalized in repeated_margins:
            block_flags.append("repeated_header_footer_candidate")
        if counts[normalized] > 1:
            block_flags.append("duplicate_block_candidate")
        if row["ocr_used"]:
            block_flags.append("ocr_used")
        if row["kind"] == "table":
            block_flags.append("table_block_present")
        flags.update(block_flags)
        block_items.append({
            "ordinal": row["ordinal"], "kind": row["kind"], "text": row["text"],
            "section_title": row["section_title"], "ocr_used": bool(row["ocr_used"]),
            "ocr_confidence": row["ocr_confidence"], "flags": block_flags,
        })
    chunk_items = []
    for row in chunks:
        crosses = row["page_start"] != row["page_end"]
        if crosses:
            flags.add("chunk_crosses_page_boundary")
        chunk_items.append({
            "ordinal": row["ordinal"], "chunk_id": row["chunk_id"],
            "page_start": row["page_start"], "page_end": row["page_end"],
            "language": row["language"], "ocr_used": bool(row["ocr_used"]),
            "ocr_confidence": row["ocr_confidence"], "crosses_page": crosses,
        })
    return {
        "file_name": review.file_name, "page_number": review.page_number,
        "rationale": review.rationale, "assessment": review.assessment,
        "document_id": document["document_id"],
        "blocks": block_items, "chunks": chunk_items, "flags": sorted(flags),
    }


def duplicate_chunk_evidence(connection: sqlite3.Connection) -> dict[str, Any]:
    """Record metadata-only evidence for the Army RAS pages 4–5 duplicate."""

    row = connection.execute(
        "SELECT id FROM documents WHERE file_name=?", ("20220126_army-approach-to-ras_final.pdf",)
    ).fetchone()
    chunks = connection.execute(
        "SELECT chunk_id,text,page_start,page_end FROM chunks WHERE document_id=? AND page_start IN (4,5)",
        (row["id"],),
    ).fetchall()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        digest = hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()
        groups[digest].append({
            "chunk_id": chunk["chunk_id"], "page_start": chunk["page_start"],
            "page_end": chunk["page_end"], "character_count": len(chunk["text"]),
        })
    duplicates = [values for values in groups.values() if len(values) > 1]
    return {"duplicate_groups": duplicates, "interpretation": "manual visual confirmation required"}


def block_html(block: dict[str, Any]) -> str:
    badges = "".join(f'<span class="badge warning">{escape(flag)}</span>' for flag in block["flags"])
    meta = f'Block {block["ordinal"]} · {escape(block["kind"])}'
    if block["ocr_used"]:
        meta += f' · OCR confidence {block["ocr_confidence"]}'
    return (
        '<article class="block">'
        f'<div class="block-meta">{meta}{badges}</div>'
        f'<pre>{escape(block["text"])}</pre>'
        '</article>'
    )


def page_html(item: dict[str, Any]) -> str:
    flags = "".join(f'<span class="badge">{escape(flag)}</span>' for flag in item["flags"])
    chunks = "".join(
        '<li>'
        f'#{chunk["ordinal"]} <code>{escape(chunk["chunk_id"])}</code> · '
        f'pages {chunk["page_start"]}–{chunk["page_end"]} · {escape(chunk["language"])}'
        f'{" · OCR" if chunk["ocr_used"] else ""}'
        f'{" · crosses page" if chunk["crosses_page"] else ""}'
        '</li>' for chunk in item["chunks"]
    ) or "<li>No associated chunk</li>"
    blocks = "".join(block_html(block) for block in item["blocks"])
    return f"""
    <section class="review" id="review-{item['review_number']}">
      <header>
        <div><span class="index">{item['review_number']:02d}</span>
        <h2>{escape(item['file_name'])} — page {item['page_number']}</h2></div>
        <p>{escape(item['rationale'])}</p>
        <p class="assessment"><strong>Human visual assessment:</strong> {escape(item['assessment'])}</p>
        <div class="badges">{flags or '<span class="badge ok">no automated warning</span>'}</div>
      </header>
      <div class="columns">
        <div class="original"><h3>Original rendered page</h3><a href="{item['image_name']}"><img src="{item['image_name']}" alt="Rendered original page"></a></div>
        <div class="parsed"><h3>Parsed blocks in persisted order</h3>{blocks or '<p>No persisted text blocks.</p>'}</div>
      </div>
      <details><summary>Associated chunk metadata ({len(item['chunks'])})</summary><ul>{chunks}</ul></details>
    </section>"""


def build_html(items: list[dict[str, Any]], duplicate_evidence: dict[str, Any]) -> str:
    navigation = "".join(
        f'<a href="#review-{item["review_number"]}">{item["review_number"]:02d}. {escape(item["file_name"])} p.{item["page_number"]}</a>'
        for item in items
    )
    evidence = escape(json.dumps(duplicate_evidence, ensure_ascii=False, indent=2))
    sections = "".join(page_html(item) for item in items)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phase 2 Content-Level Visual QA</title>
<style>
:root{{--ink:#17202a;--muted:#667085;--line:#d0d5dd;--paper:#fff;--bg:#eef1f5;--accent:#315b7d;--warn:#9a3412}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 "Segoe UI",Arial,sans-serif}}
.top{{background:#13293d;color:white;padding:28px 34px}} .top h1{{margin:0 0 6px;font-size:26px}} .top p{{margin:0;color:#d6e2ec}}
nav{{display:flex;gap:7px;overflow:auto;padding:12px 20px;background:white;border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5}}
nav a{{white-space:nowrap;color:var(--accent);text-decoration:none;border:1px solid var(--line);border-radius:6px;padding:6px 9px}}
main{{max-width:1800px;margin:auto;padding:20px}} .review{{background:var(--paper);border:1px solid var(--line);border-radius:10px;margin:0 0 24px;overflow:hidden}}
.review>header{{padding:18px 22px;border-bottom:1px solid var(--line)}} .review h2{{font-size:19px;margin:0;display:inline}} .review header p{{color:var(--muted);margin:6px 0}}
.review header .assessment{{color:var(--ink);background:#f8fafc;border-left:4px solid var(--accent);padding:8px 10px}}
.index{{display:inline-block;background:var(--accent);color:white;border-radius:5px;padding:2px 7px;margin-right:8px;font-weight:700}}
.columns{{display:grid;grid-template-columns:minmax(420px,48%) 1fr;gap:18px;padding:18px}} h3{{font-size:15px;margin:0 0 10px}}
.original img{{width:100%;height:auto;border:1px solid var(--line);box-shadow:0 3px 12px #0002}} .parsed{{max-height:1050px;overflow:auto;padding-right:5px}}
.block{{border:1px solid var(--line);border-radius:6px;margin:0 0 8px}} .block-meta{{background:#f8fafc;padding:6px 9px;font-size:12px;color:var(--muted)}}
pre{{font:13px/1.45 Consolas,"Segoe UI",sans-serif;white-space:pre-wrap;word-break:break-word;margin:0;padding:9px}}
.badge{{display:inline-block;background:#fff4e5;color:var(--warn);border:1px solid #fed7aa;border-radius:999px;padding:2px 7px;margin:2px 4px 2px 0;font-size:11px}}
.badge.ok{{background:#ecfdf3;color:#027a48;border-color:#abefc6}} .block-meta .badge{{margin-left:6px}} details{{padding:12px 20px;border-top:1px solid var(--line)}} code{{font-size:11px}}
.evidence{{background:#fff;border:1px solid var(--line);border-radius:10px;padding:18px;margin-bottom:20px}} .evidence pre{{background:#f8fafc}}
@media(max-width:950px){{.columns{{grid-template-columns:1fr}}.parsed{{max-height:none}}}}
</style></head><body>
<header class="top"><h1>Phase 2 Content-Level Visual QA</h1><p>Local diagnostic package · persisted parsed blocks versus original rendered pages · no reparse/OCR/models</p></header>
<nav>{navigation}</nav><main>
<section class="evidence"><h2>Army RAS duplicate metadata</h2><pre>{evidence}</pre></section>
{sections}</main></body></html>"""


def build(source_dir: Path, database_path: Path, output_dir: Path,
          pages: tuple[ReviewPage, ...] = DEFAULT_PAGES) -> dict[str, Any]:
    """Render selected originals and combine them with existing persisted text."""

    source_root = source_dir.resolve(strict=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with connect(database_path.resolve(strict=True)) as connection:
        items: list[dict[str, Any]] = []
        for number, review in enumerate(pages, 1):
            pdf_path = (source_root / review.file_name).resolve(strict=True)
            if pdf_path.parent != source_root:
                raise ValueError("Review files must be directly inside the supplied directory")
            image_name = f"review_{number:02d}_page_{review.page_number:03d}.png"
            render_page(pdf_path, review.page_number, output_dir / image_name)
            item = page_data(connection, review)
            geometry = multi_column_signal(pdf_path, review.page_number)
            item.update(geometry)
            if geometry["possible_multi_column"]:
                item["flags"] = sorted(set(item["flags"]) | {"possible_multi_column_layout"})
            item["review_number"] = number
            item["image_name"] = image_name
            items.append(item)
        duplicate_evidence = duplicate_chunk_evidence(connection)
    manifest = {
        "policy": "local visual QA only; images rendered from originals; text read from existing Phase 2 SQLite result",
        "database": str(database_path.resolve()), "pages": items,
        "army_ras_duplicate": duplicate_evidence,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "index.html").write_text(
        build_html(items, duplicate_evidence), encoding="utf-8"
    )
    print(f"Created {output_dir / 'index.html'} with {len(items)} reviewed pages")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    build(arguments.source_dir, arguments.database, arguments.output_dir)


if __name__ == "__main__":
    main()
