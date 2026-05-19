"""
PDF to JSON parser using pymupdf4llm.

Converts a PDF file into the Science-Parse-compatible JSON format used by
the rest of this project (data_preprocessing.py / inspect_pdf.py).

Output JSON schema
------------------
{
    "name": "<filename>",
    "metadata": {
        "source": "pymupdf4llm",
        "title": "<string>",
        "authors": [],
        "abstractText": "<string>",
        "sections": [
            {"heading": "<string>", "text": "<string>"},
            ...
        ]
    }
}

Usage
-----
    # From Python
    from Trainer.pdf_parser import parse_pdf_to_json, parse_pdf_to_json_file

    data = parse_pdf_to_json("path/to/paper.pdf")

    # Or save directly to a JSON file
    parse_pdf_to_json_file("path/to/paper.pdf", "path/to/output.json")

    # CLI
    python -m Trainer.pdf_parser path/to/paper.pdf [path/to/output.json]
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pymupdf4llm


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Heuristic: a line is a section heading when it looks like
#   "1 Introduction", "2.3 Related Work", "Abstract", "Conclusion", etc.
_HEADING_RE = re.compile(
    r"^(?:"
    r"(?:\d+(?:\.\d+)*\s+[A-Z])"   # numbered: "1 Intro", "2.3 Method"
    r"|(?:[A-Z][A-Z\s]{2,}$)"       # ALL-CAPS short line
    r"|(?:[A-Z][a-z]+(?: [A-Z][a-z]+){0,5}$)"  # Title Case, ≤6 words
    r")",
    re.MULTILINE,
)

# Markdown heading patterns produced by pymupdf4llm
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)


def _split_markdown_into_sections(
    md_text: str,
) -> List[Dict[str, str]]:
    """
    Split pymupdf4llm Markdown output into a list of
    ``{"heading": ..., "text": ...}`` dicts that mirror the Science-Parse
    section format consumed by ``_extract_paper_text`` in
    ``data_preprocessing.py``.
    """
    sections: List[Dict[str, str]] = []
    current_heading = ""
    current_lines: List[str] = []

    for line in md_text.splitlines():
        m = _MD_HEADING_RE.match(line)
        if m:
            # Flush the previous section
            body = "\n".join(current_lines).strip()
            if body or current_heading:
                sections.append({"heading": current_heading, "text": body})
            current_heading = m.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Flush the last section
    body = "\n".join(current_lines).strip()
    if body or current_heading:
        sections.append({"heading": current_heading, "text": body})

    return sections


def _strip_md_formatting(text: str) -> str:
    """Remove Markdown bold/italic markers from a string."""
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.*?)_{1,3}", r"\1", text)
    return text.strip()


# Keywords that strongly indicate a heading/line is an author affiliation,
# page header, journal name, or other non-title content. Lowercase comparison.
_AFFILIATION_KEYWORDS = (
    "university", "universit", "department", "departament", "institute",
    "laboratory", " lab ", "academy", "faculty", "school of",
    "google", "microsoft", "facebook", "openai", "deepmind", "amazon",
    "@", "email", "e-mail", ".edu", ".com", ".org", "http", "www.",
    "street", "avenue", "boulevard", "road",
    "proceedings of", "in proc.", "preprint", "submitted", "arxiv:",
    "doi:", "isbn:", "issn:",
    "copyright", "©",
)


def _looks_like_author_or_affiliation(text: str) -> bool:
    """Heuristic: is this short heading-or-line actually an author/affiliation/
    journal header rather than a paper title?"""
    if not text:
        return True
    t = text.lower().strip()
    if len(t) < 4 or len(t) > 220:   # titles are rarely shorter than 4 chars or > 220
        return True
    # Affiliation/contact keywords
    if any(kw in t for kw in _AFFILIATION_KEYWORDS):
        return True
    # Pure-numbers / page numbers / DOI fragments
    if re.fullmatch(r"[\d\.\-:/\s]+", t):
        return True
    # All lowercase short heading → likely a section sub-heading, not a title
    if t == t.lower() and len(t) < 40 and " " not in t.strip():
        return True
    return False


def _extract_title_by_font_size(pdf_path: str) -> str:
    """Find the most likely paper title using font-size analysis of page 1.

    Academic papers almost always typeset the title in the LARGEST font on
    the first page. We collect all text spans by font size, sort
    descending, and walk down the list until we find spans that don't look
    like authors/affiliations/page headers.
    """
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz   # older pymupdf
        except ImportError:
            return ""

    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return ""

    if len(doc) == 0:
        doc.close()
        return ""

    page = doc[0]
    try:
        blocks = page.get_text("dict").get("blocks", [])
    except Exception:
        doc.close()
        return ""

    # Group consecutive same-size spans, sort by size desc, then by y-position.
    # entries: list[(size, y_top, text)]
    entries: List[Tuple[float, float, str]] = []
    for block in blocks:
        if block.get("type", 0) != 0:   # 0 = text block
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                txt = (span.get("text") or "").strip()
                if not txt:
                    continue
                size = float(span.get("size", 0.0))
                y_top = float(span.get("bbox", [0, 0, 0, 0])[1])
                entries.append((round(size, 1), y_top, txt))

    doc.close()
    if not entries:
        return ""

    # Bucket text by font size; iterate sizes from largest to smallest
    sizes = sorted({e[0] for e in entries}, reverse=True)
    for size in sizes:
        same_size = [(y, t) for s, y, t in entries if s == size]
        same_size.sort(key=lambda yt: yt[0])  # top-of-page first
        # Join adjacent spans into one candidate title string
        candidate = " ".join(t for _, t in same_size).strip()
        candidate = re.sub(r"\s+", " ", candidate)
        candidate = _strip_md_formatting(candidate)
        if _looks_like_author_or_affiliation(candidate):
            continue
        # Real-title sanity: should have at least one space (multi-word) AND
        # at least one alphabetic character.
        if " " not in candidate or not re.search(r"[A-Za-z]", candidate):
            continue
        # Looks plausible
        return candidate

    return ""


def _extract_title_and_abstract(
    sections: List[Dict[str, str]],
    doc_metadata: Dict[str, Any],
    title_hint: str = "",
) -> Tuple[str, str, List[Dict[str, str]]]:
    """
    Try to pull the paper title and abstract from the section list, the PDF
    document metadata, or a pre-computed font-size title hint.

    Returns ``(title, abstract, remaining_sections)``.
    """
    # 1. Title precedence: font-size-detected hint > PDF metadata > headings
    title: str = ""
    if title_hint and not _looks_like_author_or_affiliation(title_hint):
        title = title_hint
    elif doc_metadata.get("title"):
        meta_title = _strip_md_formatting(str(doc_metadata["title"])).strip()
        if not _looks_like_author_or_affiliation(meta_title):
            title = meta_title

    abstract: str = ""
    remaining: List[Dict[str, str]] = []

    for sec in sections:
        heading = sec["heading"]
        heading_lower = heading.lower()
        text = sec["text"]
        text_stripped = text.strip()

        # Section heading that IS the title (bold Markdown heading like **Paper Title**)
        bold_heading = re.match(r"^\*{1,3}(.+?)\*{1,3}$", heading.strip())
        if bold_heading and not title:
            candidate = bold_heading.group(1).strip()
            if not _looks_like_author_or_affiliation(candidate):
                title = candidate
                continue   # consume the section

        # Headingless first block: may start with a bold title line
        if not title and not heading_lower:
            lines = text.split("\n")
            first_line = lines[0].strip() if lines else ""
            bold_match = re.match(r"^\*{1,3}(.+?)\*{1,3}$", first_line)
            if bold_match:
                cand = bold_match.group(1).strip()
                if not _looks_like_author_or_affiliation(cand):
                    title = cand
                    rest = "\n".join(lines[1:]).strip()
                    if rest:
                        remaining.append({"heading": heading, "text": rest})
                    continue
            elif first_line and not _looks_like_author_or_affiliation(_strip_md_formatting(first_line)):
                title = _strip_md_formatting(first_line)
                continue

        # Abstract: only accept a section whose heading literally references
        # "abstract" AND whose content is substantial (≥ 80 chars). This skips
        # bogus initial sections that contain just page-header text.
        if re.search(r"\babstract\b", heading_lower) and len(text_stripped) >= 80 and not abstract:
            abstract = text_stripped
            continue

        # Otherwise keep as a body section. Drop sections that are noise:
        #   - heading matches the detected title exactly (it's the title block)
        #   - heading OR short content looks like author/affiliation/journal header
        clean_heading = _strip_md_formatting(heading)
        if title and clean_heading.strip().lower() == title.strip().lower():
            continue
        text_preview = text_stripped[:300]
        heading_bad = _looks_like_author_or_affiliation(clean_heading)
        content_bad = _looks_like_author_or_affiliation(text_preview)
        if (heading_bad or content_bad) and len(text_stripped) < 300:
            continue
        remaining.append({"heading": clean_heading, "text": text})

    return title.strip(), abstract.strip(), remaining


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_pdf_to_json(
    pdf_path: str | Path,
    *,
    page_chunks: bool = False,
    include_images: bool = False,
) -> Dict[str, Any]:
    """
    Parse a PDF file with pymupdf4llm and return a Science-Parse-compatible
    JSON dict.

    Parameters
    ----------
    pdf_path:
        Path to the source PDF.
    page_chunks:
        If *True*, pymupdf4llm splits the document into per-page Markdown
        chunks (useful for very long PDFs).  The chunks are concatenated
        before section splitting.
    include_images:
        Passed to ``pymupdf4llm.to_markdown``; set to *True* to embed base64
        images in the Markdown (increases output size significantly).

    Returns
    -------
    dict
        A dict matching the Science-Parse JSON schema used by the project.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # ------------------------------------------------------------------
    # 1. Convert PDF → Markdown via pymupdf4llm
    # ------------------------------------------------------------------
    if page_chunks:
        chunks = pymupdf4llm.to_markdown(
            str(pdf_path),
            page_chunks=True,
            show_progress=False,
            write_images=include_images,
        )
        md_text: str = "\n\n".join(
            chunk["text"] if isinstance(chunk, dict) else str(chunk)
            for chunk in chunks
        )
    else:
        md_text = pymupdf4llm.to_markdown(
            str(pdf_path),
            show_progress=False,
            write_images=include_images,
        )

    # ------------------------------------------------------------------
    # 2. Extract PDF-level metadata (author list, embedded title, etc.)
    # ------------------------------------------------------------------
    import pymupdf  # bundled with pymupdf4llm

    doc = pymupdf.open(str(pdf_path))
    raw_meta: Dict[str, Any] = doc.metadata or {}
    doc.close()

    # Authors: the PDF metadata rarely has a proper list, but we try.
    authors: List[str] = []
    if raw_meta.get("author"):
        # May be comma- or semicolon-separated
        raw_authors = re.split(r"[;,]", raw_meta["author"])
        authors = [a.strip() for a in raw_authors if a.strip()]

    # ------------------------------------------------------------------
    # 3. Font-size title detection (most reliable for academic PDFs)
    # ------------------------------------------------------------------
    font_size_title = _extract_title_by_font_size(str(pdf_path))

    # ------------------------------------------------------------------
    # 4. Split Markdown into sections
    # ------------------------------------------------------------------
    all_sections = _split_markdown_into_sections(md_text)

    title, abstract, body_sections = _extract_title_and_abstract(
        all_sections, raw_meta, title_hint=font_size_title
    )

    # Final fallback: raw PDF metadata title even if it contains affiliation
    # keywords — better than nothing for downstream display.
    if not title and raw_meta.get("title"):
        title = raw_meta["title"].strip()

    # ------------------------------------------------------------------
    # 4. Build the output dict
    # ------------------------------------------------------------------
    result: Dict[str, Any] = {
        "name": pdf_path.name,
        "metadata": {
            "source": "pymupdf4llm",
            "title": title,
            "authors": authors,
            "abstractText": abstract,
            "sections": body_sections,
        },
    }

    return result


def parse_pdf_to_json_file(
    pdf_path: str | Path,
    output_path: Optional[str | Path] = None,
    *,
    page_chunks: bool = False,
    include_images: bool = False,
    indent: int = 2,
) -> Path:
    """
    Parse a PDF and write the result as a JSON file.

    Parameters
    ----------
    pdf_path:
        Path to the source PDF.
    output_path:
        Destination JSON path.  Defaults to ``<pdf_stem>.pdf.json`` next to
        the PDF (matching the Science-Parse naming convention).
    page_chunks, include_images:
        Forwarded to :func:`parse_pdf_to_json`.
    indent:
        JSON indentation width.

    Returns
    -------
    Path
        The path to the written JSON file.
    """
    pdf_path = Path(pdf_path)

    if output_path is None:
        output_path = pdf_path.with_name(pdf_path.name + ".json")
    output_path = Path(output_path)

    data = parse_pdf_to_json(
        pdf_path,
        page_chunks=page_chunks,
        include_images=include_images,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=indent)

    print(f"[pdf_parser] Written → {output_path}")
    return output_path


def parse_directory(
    pdf_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    *,
    recursive: bool = False,
    skip_existing: bool = True,
    page_chunks: bool = False,
) -> List[Path]:
    """
    Batch-parse all PDFs in *pdf_dir*.

    Parameters
    ----------
    pdf_dir:
        Directory containing ``*.pdf`` files.
    output_dir:
        Directory for the output JSON files.  Defaults to the same directory
        as each PDF.
    recursive:
        If *True*, descend into sub-directories.
    skip_existing:
        Skip a PDF if the expected ``.pdf.json`` file already exists.
    page_chunks:
        Forwarded to :func:`parse_pdf_to_json`.

    Returns
    -------
    list[Path]
        Paths to all written JSON files.
    """
    pdf_dir = Path(pdf_dir)
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = list(pdf_dir.glob(pattern))

    if not pdf_files:
        print(f"[pdf_parser] No PDFs found in {pdf_dir}")
        return []

    written: List[Path] = []
    for pdf_file in pdf_files:
        if output_dir is not None:
            out_path = Path(output_dir) / (pdf_file.name + ".json")
        else:
            out_path = pdf_file.with_name(pdf_file.name + ".json")

        if skip_existing and out_path.exists():
            print(f"[pdf_parser] Skipping (exists): {out_path.name}")
            continue

        try:
            written.append(
                parse_pdf_to_json_file(pdf_file, out_path, page_chunks=page_chunks)
            )
        except Exception as exc:
            print(f"[pdf_parser] ERROR processing {pdf_file.name}: {exc}", file=sys.stderr)

    return written


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _cli() -> None:
    """
    Command-line interface::

        python -m Trainer.pdf_parser <pdf_path> [output_json]

    If *output_json* is omitted the result is printed to stdout.
    """
    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <pdf_path> [output_json]")
        sys.exit(1)

    pdf_arg = sys.argv[1]
    output_arg = sys.argv[2] if len(sys.argv) >= 3 else None

    if output_arg:
        parse_pdf_to_json_file(pdf_arg, output_arg)
    else:
        data = parse_pdf_to_json(pdf_arg)
        print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()


