"""
Heuristic parser that extracts structured sections from raw ICLR review text.

Typical ICLR review patterns we handle:

  Summary:
  This paper proposes ...

  Strengths:
  - Novel approach
  - Strong experiments

  Weaknesses:
  - Missing baseline X
  - Notation is unclear

  Questions:
  - Did you compare against Y?

Also recognizes Pros/Cons, Positives/Negatives, Issues/Concerns, etc.

Public API
----------
  parse_review(text: str) -> Optional[ParsedReview]
      Returns a dict with summary, strengths, weaknesses, questions.
      Returns None if the review can't be parsed (missing required sections).

  format_structured_target(parsed: ParsedReview) -> str
      Renders the structured dict back into the canonical
      "SUMMARY: ... STRENGTHS: ... WEAKNESSES: ... QUESTIONS: ..." string
      used as the seq2seq training target.

  parse_structured_output(text: str) -> ParsedReview
      Inverse of format_structured_target — used at inference to split
      the model's generated text back into sections for display.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, TypedDict


# ---------------------------------------------------------------------------
# Section header recognition
# ---------------------------------------------------------------------------

# Each (canonical_name, [regex patterns]).  All matched case-insensitively at
# the start of a line, with optional colon/dash/period/whitespace after.
_SECTION_PATTERNS: List[tuple] = [
    ("summary", [
        r"summary",
        r"paper\s+summary",
        r"overview",
        r"brief\s+summary",
    ]),
    ("strengths", [
        r"strengths?",
        r"pros",
        r"positives?",
        r"strong\s+points",
        r"things\s+i\s+like[d]?",
        r"what\s+i\s+like[d]?",
        r"merits",
        r"advantages?",
    ]),
    ("weaknesses", [
        r"weakness(?:es)?",
        r"cons",
        r"negatives?",
        r"weak\s+points",
        r"things\s+i\s+(?:dis)?like[d]?",
        r"issues",
        r"concerns",
        r"limitations?",
        r"shortcomings",
        r"flaws",
        r"drawbacks?",
    ]),
    ("questions", [
        r"questions?",
        r"queries",
        r"questions?\s+for\s+(?:the\s+)?authors?",
    ]),
    # Sections we recognize but discard:
    ("_ignore", [
        r"clarity",
        r"originality",
        r"significance",
        r"quality",
        r"recommendation",
        r"confidence",
        r"reproducibility",
        r"detailed\s+comments",
        r"general\s+comments",
        r"minor\s+comments?",
        r"typos?",
        r"references?",
        r"rating",
        r"score",
    ]),
]

# Pre-compile a single regex per section that matches "header :" lines.
_HEADER_RE = re.compile(
    r"^\s*(" +
    "|".join(
        f"(?P<{name}>" + "|".join(pats) + ")"
        for name, pats in _SECTION_PATTERNS
    ) +
    r")\s*[:\-\.\)]*\s*$",
    re.IGNORECASE,
)

# Bullet detection (start-of-line markers commonly used in ICLR reviews)
_BULLET_RE = re.compile(r"^\s*(?:[-*•+]|\d+[.)\]])\s+(.*)$")


class ParsedReview(TypedDict, total=False):
    summary:    str
    strengths:  List[str]
    weaknesses: List[str]
    questions:  List[str]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _which_section(line: str) -> Optional[str]:
    """If `line` looks like a section header (whole line), return its canonical name."""
    m = _HEADER_RE.match(line.strip())
    if not m:
        return None
    for name, _ in _SECTION_PATTERNS:
        if m.groupdict().get(name):
            return name
    return None


def _split_bullets(lines: List[str]) -> List[str]:
    """Convert a block of lines to a bullet list. If no explicit bullets,
    treat each non-empty paragraph as one bullet (paragraphs split by blank lines).
    Returns deduplicated, cleaned bullets."""
    bullets: List[str] = []
    current: List[str] = []

    for line in lines:
        m = _BULLET_RE.match(line)
        if m:
            # flush previous bullet if any
            if current:
                bullets.append(" ".join(s.strip() for s in current).strip())
                current = []
            current.append(m.group(1))
        elif line.strip() == "":
            # paragraph break flushes the current bullet
            if current:
                bullets.append(" ".join(s.strip() for s in current).strip())
                current = []
        else:
            current.append(line)

    if current:
        bullets.append(" ".join(s.strip() for s in current).strip())

    # cleanup: drop empty / very short fragments, collapse whitespace
    cleaned: List[str] = []
    seen = set()
    for b in bullets:
        b = re.sub(r"\s+", " ", b).strip()
        if len(b) < 8:
            continue
        if b.lower() in seen:
            continue
        seen.add(b.lower())
        cleaned.append(b)
    return cleaned


def parse_review(text: str, allow_freeform: bool = False) -> Optional[ParsedReview]:
    """Parse raw review text into a structured dict.

    Lenient: accepts a review if ANY recognized section (summary header
    explicitly named, OR a strengths bucket, OR a weaknesses bucket, OR
    a questions bucket) was detected.

    If `allow_freeform=True`, also accepts reviews with no recognized
    sections at all and returns them as summary-only (the first ~1200 chars
    of the review go into `summary`, all bullet lists stay empty).

    Returns None if the text is shorter than 50 characters OR if it has
    no structure and `allow_freeform` is False.
    """
    if not text or len(text.strip()) < 50:
        return None

    lines = text.replace("\r", "").split("\n")
    sections: Dict[str, List[str]] = {
        "summary":    [],
        "strengths":  [],
        "weaknesses": [],
        "questions":  [],
    }
    current = "summary"
    saw_explicit_header = False

    for line in lines:
        sec = _which_section(line)
        if sec == "_ignore":
            current = "_dropped"
            saw_explicit_header = True
            continue
        if sec is not None:
            current = sec
            saw_explicit_header = True
            continue
        if current == "_dropped":
            continue
        sections[current].append(line)

    summary_text = " ".join(
        re.sub(r"\s+", " ", l.strip()) for l in sections["summary"] if l.strip()
    ).strip()
    strengths  = _split_bullets(sections["strengths"])
    weaknesses = _split_bullets(sections["weaknesses"])
    questions  = _split_bullets(sections["questions"])

    # Trim runaway summary
    if len(summary_text) > 600:
        summary_text = summary_text[:600].rsplit(" ", 1)[0] + "..."

    has_any_structure = saw_explicit_header or bool(strengths) or bool(weaknesses) or bool(questions)

    if has_any_structure:
        return {
            "summary":    summary_text or "(no summary provided)",
            "strengths":  strengths,
            "weaknesses": weaknesses,
            "questions":  questions,
        }

    if not allow_freeform:
        return None

    # Freeform fallback: dump the whole text as summary. The model will
    # still learn the structured output format from the parseable samples,
    # while these unstructured ones at least teach it how to summarize.
    freeform_summary = re.sub(r"\s+", " ", text.strip())
    if len(freeform_summary) > 1200:
        freeform_summary = freeform_summary[:1200].rsplit(" ", 1)[0] + "..."
    return {
        "summary":    freeform_summary,
        "strengths":  [],
        "weaknesses": [],
        "questions":  [],
    }


# ---------------------------------------------------------------------------
# Target / output formatting
# ---------------------------------------------------------------------------

def format_structured_target(parsed: ParsedReview) -> str:
    """Render a parsed review back into the canonical training-target format.

    The model learns to reproduce exactly this format, so keep it stable.
    """
    parts = [f"SUMMARY: {parsed.get('summary', '').strip()}"]

    parts.append("STRENGTHS:")
    for b in parsed.get("strengths", []):
        parts.append(f"- {b}")

    parts.append("WEAKNESSES:")
    for b in parsed.get("weaknesses", []):
        parts.append(f"- {b}")

    if parsed.get("questions"):
        parts.append("QUESTIONS:")
        for b in parsed["questions"]:
            parts.append(f"- {b}")

    return "\n".join(parts)


# Inverse parser — splits model output back into a structured dict.

_OUTPUT_HEADER_RE = re.compile(
    r"^\s*(SUMMARY|STRENGTHS|WEAKNESSES|QUESTIONS)\s*:\s*(.*)$",
    re.IGNORECASE,
)


def parse_structured_output(text: str) -> ParsedReview:
    """Parse the model's generated output back into a structured dict.

    Lenient: if the model misses a section, it just stays empty.
    """
    sections: Dict[str, List[str]] = {
        "summary":    [],
        "strengths":  [],
        "weaknesses": [],
        "questions":  [],
    }
    current = "summary"

    for raw_line in text.replace("\r", "").split("\n"):
        m = _OUTPUT_HEADER_RE.match(raw_line)
        if m:
            current = m.group(1).lower()
            rest = m.group(2).strip()
            if rest:
                sections[current].append(rest)
            continue
        if raw_line.strip():
            sections[current].append(raw_line)

    summary = " ".join(re.sub(r"\s+", " ", l.strip()) for l in sections["summary"] if l.strip()).strip()
    return {
        "summary":    summary,
        "strengths":  _split_bullets(sections["strengths"]),
        "weaknesses": _split_bullets(sections["weaknesses"]),
        "questions":  _split_bullets(sections["questions"]),
    }
