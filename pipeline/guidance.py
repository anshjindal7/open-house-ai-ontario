"""Split the LTB's Interpretation Guidelines and brochures into searchable sections.

    python -m pipeline.guidance

Source: Tribunals Ontario (Landlord and Tenant Board). Interpretation Guidelines explain "the
Board's usual interpretation of the law"; brochures explain the rules for landlords and tenants.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

SRC = Path(__file__).parents[1] / "sources" / "ltb_guidance"
OUT = Path(__file__).parents[1] / "corpus" / "ltb_guidance.jsonl"
MAX_SECTION = 1600
SKIP_HEADINGS = re.compile(r"table of contents|disponible|for more information|contact the|french", re.I)


def _text(el) -> str:
    for sup in el.find_all("sup"):
        sup.decompose()  # footnote markers
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()


def _blocks(start):
    """Paragraphs and list items after a heading, up to the next heading of the same or higher level."""
    level = int(start.name[1])
    for el in start.find_all_next():
        if el.name in {"h1", "h2", "h3", "h4"} and int(el.name[1]) <= level:
            return
        if el.name in {"h3", "h4"}:
            yield f"[{_text(el)}]"
        elif el.name in {"p", "li"} and not el.find_parent("li"):
            t = _text(el)
            if t:
                yield ("• " if el.name == "li" else "") + t


def parse(path: Path) -> list[dict]:
    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    kind, name = path.stem.split("--", 1)
    num = re.match(r"(\d+) - ", name)
    doc = (f"LTB Interpretation Guideline {int(num.group(1))}: {name[num.end():]}" if num
           else f"LTB brochure: {name.replace('_', ' ')}")
    url = ("https://tribunalsontario.ca/documents/ltb/"
           + ("Interpretation Guidelines/" if kind == "guideline" else "Brochures/") + name + ".html")

    sections = []
    for h2 in soup.find_all("h2"):
        heading = _text(h2)
        if SKIP_HEADINGS.search(heading):
            continue
        body = "\n".join(_blocks(h2)).strip()
        if len(body) < 120:
            continue
        # Long sections split at sub-headings so each piece stays on one topic.
        chunks, current = [], ""
        for line in body.split("\n"):
            # Prefer splitting at a sub-heading; split anywhere if a section has none.
            too_long = len(current) + len(line) > MAX_SECTION
            if current and too_long and (line.startswith("[") or len(current) > MAX_SECTION * 1.5):
                chunks.append(current.strip())
                current = ""
            current += line + "\n"
        chunks.append(current.strip())
        for i, chunk in enumerate(chunks):
            sub = re.match(r"\[([^\]]+)\]", chunk)
            title = heading + (f": {sub.group(1)}" if sub and i else "")
            sections.append({"doc": doc, "kind": kind, "heading": title, "text": chunk, "url": url})
    return sections


def main() -> None:
    rows = [s for f in sorted(SRC.glob("*.html")) for s in parse(f)]
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"id": f"guide:{i}", **r}, ensure_ascii=False) + "\n")
    docs = {r["doc"] for r in rows}
    print(f"{len(rows)} sections from {len(docs)} documents -> {OUT}")


if __name__ == "__main__":
    main()
