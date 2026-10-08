"""Split a lease into clauses so each one can be checked on its own."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

MAX_CLAUSE_CHARS = 1500

# Lines that start a new clause: "1.", "12)", "Section 4", "ARTICLE II", "4.2", or an ALL-CAPS heading.
HEADING = re.compile(
    r"""^\s*(
        (section|article|clause)\s+[\dIVXLC]+[.:)]?   # Section 4 / ARTICLE II
      | \d{1,2}(\.\d{1,2})*[.)]\s                     # 1.  4.2.  12)
      | [A-Z][A-Z &/'-]{3,60}:?\s*$                   # ALL CAPS HEADING
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# Ontario's Standard Form of Lease (2229E). Its appendix ("General Information") and the page of
# translations are the government's explanations, not terms anyone agreed to, so they're removed.
# Sections 1-3 and 17 are names, addresses, contacts and signatures: form details, not terms.
STANDARD_FORM = re.compile(r"2229E|Residential Tenancy Agreement \(Standard Form of Lease\)", re.I)
APPENDIX_START = re.compile(r"^\s*Appendix:\s*General Information", re.I | re.M)
APPENDIX_END = re.compile(r"ontario\s*\.\s*ca\s*/\s*standardlease[^\n]*|Page\s*15\s*of\s*14[^\n]*", re.I)
PAGE_FURNITURE = re.compile(r"^\s*(?:2229E\s*\(\d{4}/\d{2}\)\s*(?:Page\s*\d+\s*of\s*\d+)?|Page\s*\d+(?:\s*of\s*\d+)?|"
                            r"©\s*(?:Queen|King)'?s Printer for Ontario,? \d{4}.*)\s*$", re.I | re.M)
FORM_DETAIL_SECTIONS = {1, 2, 3, 17}


@dataclass
class Clause:
    index: int
    heading: str
    text: str
    form_detail: bool = False  # standard-form names/addresses/signatures: shown, not checked


def load_lease(path: str | Path) -> str:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8")


def _strip_standard_form_appendix(text: str) -> str:
    start = APPENDIX_START.search(text)
    if not start:
        return text
    ends = list(APPENDIX_END.finditer(text, start.end()))
    end = ends[-1].end() if ends else len(text)
    return text[:start.start()] + "\n" + text[end:]


def split_clauses(lease_text: str) -> list[Clause]:
    standard = bool(STANDARD_FORM.search(lease_text))
    if standard:
        lease_text = _strip_standard_form_appendix(lease_text)
    lease_text = PAGE_FURNITURE.sub("", lease_text)

    blocks: list[list[str]] = []
    for line in lease_text.splitlines():
        if not line.strip():
            if blocks and blocks[-1]:
                blocks[-1].append("")
            continue
        # The all-caps branch of HEADING is case-insensitive too, so check casing explicitly.
        is_heading = bool(HEADING.match(line)) and (
            not re.fullmatch(r"\s*[A-Za-z &/'-]+:?\s*", line) or line.strip().isupper()
        )
        if is_heading or not blocks:
            blocks.append([])
        blocks[-1].append(line.rstrip())

    # Two-column PDFs split headings over several lines ("COVENANTS" / "ANNUAL" / "INCREASE"), each
    # looking like its own heading. A block that is only a short all-caps line is a heading
    # fragment: carry it forward and join it to the next block with real text.
    merged: list[tuple[str, str]] = []
    carried: list[str] = []
    for block in blocks:
        text = "\n".join(block).strip()
        if not text:
            continue
        if len(text) < 40 and text.upper() == text and len(text.split()) <= 4:
            carried.append(text)
            continue
        heading = block[0].strip()[:80]
        if carried:
            # "COVENANTS" / "ANNUAL" + a block starting "INCREASE": one heading over three lines.
            joiner = " " if heading.upper() == heading else " · "
            heading = (" ".join(carried) + joiner + heading)[:80]
            carried = []
        merged.append((heading, text))

    clauses: list[Clause] = []
    for heading, text in merged:
        section = re.match(r"^\s*(\d{1,2})\.\s+[A-Z]", text) if standard else None
        form_detail = (bool(section) and int(section.group(1)) in FORM_DETAIL_SECTIONS) or \
                      (standard and not merged.index((heading, text)) and not section)  # the cover page
        for piece in _split_long(text):
            clauses.append(Clause(index=len(clauses), heading=heading, text=piece, form_detail=form_detail))
    return clauses


# Item letters that a two-column PDF pulls out of the text into one run ("A. B. C. D. The Tenant…",
# "i. ii. iii."). They can't mark where terms start, so they're removed.
LETTER_RUN = re.compile(r"(?<![\w.])(?:(?:[A-Z]|[ivxl]{1,5})\.\s+){2,}")
# How a new term usually begins, in additional-terms sections.
TERM_OPENER = re.compile(r"^(?:The (?:Tenant|Landlord|Resident|Guarantor)s?\b|It is (?:agreed|understood)|"
                         r"In (?:the event|case)\b|If\b|Under this\b|Notwithstanding\b|No\s|Any\s|All\s|"
                         r"Where\b|Upon\b|Tenants?\b|Residents?\b|Refunds?\b|By (?:signing|agreeing)\b)")
MIN_SPLIT_CHARS = 300  # shorter blocks are one term


def _split_long(text: str) -> list[str]:
    """Break long blocks into separate terms: at sentences that start a new term, else by size."""
    text = LETTER_RUN.sub("", text)
    if len(text) <= MIN_SPLIT_CHARS:
        return [text.strip()]
    sentences = [x for x in re.split(r"\n\s*\n|(?<=[.;])\s+", text) if x.strip()]
    out, current = [], ""
    for sentence in sentences:
        new_term = bool(TERM_OPENER.match(sentence.strip())) and len(current) > 150
        if current and (new_term or len(current) + len(sentence) > MAX_CLAUSE_CHARS):
            out.append(current.strip())
            current = ""
        current += sentence + " "
    if current.strip():
        out.append(current.strip())
    return out
