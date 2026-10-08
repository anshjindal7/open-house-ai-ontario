"""Parse the Residential Tenancies Act, 2006 (Ontario) into sections.

Source: e-Laws, https://www.ontario.ca/laws/statute/06r17 (Word download), consolidated
from September 21, 2026. © King's Printer for Ontario, 2026. This is not an official version.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

SOURCES = Path(__file__).parents[1] / "sources"
OUT = Path(__file__).parents[1] / "corpus" / "rta_sections.jsonl"

SECTION_START = re.compile(r"^(\d+(?:\.\d+)?) +(?=\(1\)|[A-Z“\"])")
PART = re.compile(r"^PART [IVXLC.\d]+ (.+)$")
AMENDMENT_NOTE = re.compile(r"\s*(\d{4}, c\. \d+(, Sched\. [\w.]+)?, s\. [\w. ()]+;?\s*)+$")


@dataclass
class Section:
    number: str
    title: str
    part: str
    text: str

    @property
    def cite(self) -> str:
        return f"RTA s. {self.number}"


def parse(raw: str) -> list[Section]:
    # The Word export uses U+2028 inside headings; Python's splitlines() would break on it.
    raw = raw.replace("\u2028", " ").replace("\u00a0", " ").replace("\r", "")
    # Drop the table of contents: the body starts at the second "PART I INTRODUCTION".
    starts = [m.start() for m in re.finditer(r"^PART I INTRODUCTION", raw, re.M)]
    lines = raw[starts[-1]:].splitlines()

    lines = [l.strip() for l in lines if l.strip() and "HYPERLINK" not in l]
    sections: list[Section] = []
    part, pending_title, skipping = "", "", False
    for i, line in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if line.startswith("Section Amendments with date in force"):
            skipping = True
            continue
        if m := PART.match(line):
            part, skipping = m.group(1).title(), False
            continue
        if m := SECTION_START.match(line):
            sections.append(Section(m.group(1), pending_title, part, ""))
            skipping, pending_title = False, ""
            line = line[m.end():]
        elif SECTION_START.match(nxt) or (nxt.startswith("(") and len(line) < 100 and not line.startswith("(")):
            # The line right before a section or subsection is its marginal note (heading).
            pending_title = line
            if sections and not SECTION_START.match(nxt) and not skipping:
                sections[-1].text += f"\n[{line}]"
            skipping = False
            continue
        elif skipping or not sections:
            continue
        sections[-1].text += ("\n" if sections[-1].text else "") + AMENDMENT_NOTE.sub("", line)

    for s in sections:
        s.text = re.sub(r"(\n\[[^\]]+\])+$", "", s.text).strip()
    return [s for s in sections if s.text]


def main() -> None:
    sections = parse((SOURCES / "rta.txt").read_text(encoding="utf-8", errors="replace"))
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as f:
        for s in sections:
            f.write(json.dumps({**asdict(s), "cite": s.cite}) + "\n")
    print(f"{len(sections)} sections -> {OUT}")


if __name__ == "__main__":
    main()
