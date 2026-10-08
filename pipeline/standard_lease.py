"""Parse Ontario's Residential Tenancy Agreement (Standard Form of Lease), form 2229E.

Source: https://forms.mgcs.gov.on.ca/dataset/047-2229 (English, dated 2020/12).
Sections 1-17 are the lease itself; Parts A-U of the appendix are the Government of
Ontario's own plain-language explanations. Each section points at its Parts ("See Part H").
"""

from __future__ import annotations

import json
import re
from pathlib import Path

SOURCES = Path(__file__).parents[1] / "sources"
OUT = Path(__file__).parents[1] / "corpus" / "standard_lease.jsonl"

SECTION = re.compile(r"^\s*(\d{1,2})\. ([A-Z][^\n]{2,40}?)\s*$", re.M)
PART = re.compile(r"^\s*([A-U])\.\s+([A-Z][^\n]+?)\s*$", re.M)


def _clean(text: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\n(?!\s*[•\d])", " ", text)).strip()


def parse(raw: str) -> tuple[list[dict], dict[str, dict]]:
    body, appendix = raw.split("Appendix: General Information", 1)

    heads = [m for m in SECTION.finditer(body) if 1 <= int(m.group(1)) <= 17]
    sections = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(body)
        text = body[m.end():end]
        refs = set()
        for ref in re.findall(r"Parts? ([A-U](?:(?:,| and) [A-U])*) (?:in|of) General Information", text):
            refs.update(re.findall(r"[A-U]", ref))
        sections.append({"number": int(m.group(1)), "title": m.group(2).strip(),
                         "text": _clean(text), "parts": sorted(refs)})

    part_heads = list(PART.finditer(appendix))
    parts = {}
    for i, m in enumerate(part_heads):
        end = part_heads[i + 1].start() if i + 1 < len(part_heads) else len(appendix)
        parts[m.group(1)] = {"letter": m.group(1), "title": m.group(2).strip(),
                             "text": _clean(appendix[m.end():end])}
    return sections, parts


def main() -> None:
    sections, parts = parse((SOURCES / "2229e_standard_lease.txt").read_text())
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as f:
        for s in sections:
            f.write(json.dumps({"kind": "section", **s}) + "\n")
        for p in parts.values():
            f.write(json.dumps({"kind": "part", **p}) + "\n")
    print(f"{len(sections)} lease sections, {len(parts)} explanation parts -> {OUT}")
    for s in sections:
        print(f"  {s['number']:2}. {s['title']:32} -> Parts {', '.join(s['parts']) or '-'}")


if __name__ == "__main__":
    main()
