"""Pointers like {2.3} or {2.3-5}: "source 2, sentence 3 (to 5)".

The model writes pointers instead of retyping quotes; the app replaces them with the exact
sentences. So a quote can never be misremembered, and the text always comes from the source
the pointer names. Sentences are numbered by `Source.sentences` (see sources.segments).
"""

from __future__ import annotations

import re

from .sources import FIELD, Source

POINTER_RE = re.compile(r"\{(\d)\.(\d+)(?:-(?:\d\.)?(\d+))?\}")
QUOTE_RE = re.compile(r'(:[ \n])"([\s\S]+?)"(?=[ \t]*(?:\n|$))')
CITE_RE = re.compile(r"\[(\d)\]")


def _norm(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", s).strip().lower()


def resolve(n: int, a: int, b: int | None, sources: list[Source]) -> str | None:
    """The exact text a pointer stands for, or None if it points at nothing."""
    if not 1 <= n <= len(sources):
        return None
    sents = sources[n - 1].sentences
    b = b or a
    if not 1 <= a <= b <= len(sents):
        return None
    return " ".join(sents[a - 1:b])


def expand(answer: str, sources: list[Source]) -> str:
    """Replace pointers with the quoted text (for the command line and the fact-checker)."""
    def sub(m):
        text = resolve(int(m.group(1)), int(m.group(2)), int(m.group(3)) if m.group(3) else None, sources)
        return f'"{text}"' if text else m.group(0)
    return POINTER_RE.sub(sub, answer)


def _covering(quote: str, src: Source) -> tuple[int, int] | None:
    """The sentence range (1-based, inclusive) that contains a verbatim quote."""
    spans, pos, joined = [], 0, []
    for s in src.sentences:
        t = _norm(s)
        spans.append((pos, pos + len(t)))
        joined.append(t)
        pos += len(t) + 1
    text = " ".join(joined)
    # Quotes of a whole section include marginal headings ("[Same]") and window marks ("…"),
    # which aren't numbered sentences; drop them before matching.
    lines = []
    for line in quote.split("\n"):
        line = FIELD.sub("", line.strip()).removeprefix("… ").removesuffix(" …").strip()
        if line and not re.fullmatch(r"\[[^\]]+\]", line) and line != "…":
            lines.append(line)
    q = _norm(" ".join(lines).removesuffix("…"))
    start = text.find(q)
    if start < 0 or not q:
        return None
    end = start + len(q)
    hit = [k for k, (a, b) in enumerate(spans, 1) if a < end and start < b]
    return (hit[0], hit[-1]) if hit else None


def pointerize(answer: str, sources: list[Source]) -> str:
    """Turn every verbatim quote in a training answer into a pointer. Raises if one isn't found."""
    text = answer.replace("“", '"').replace("”", '"')
    out, last = [], 0
    for m in QUOTE_RE.finditer(text):
        lead, quote = m.group(1), m.group(2)
        line_start = text.rfind("\n", 0, m.start()) + 1
        cites = [int(c) for c in CITE_RE.findall(text[line_start:m.start()])]
        order = ([cites[-1]] if cites else []) + [n for n in range(1, len(sources) + 1) if not cites or n != cites[-1]]
        for n in order:
            if n <= len(sources) and (rng := _covering(quote, sources[n - 1])):
                a, b = rng
                pointer = f"{{{n}.{a}}}" if a == b else f"{{{n}.{a}-{b}}}"
                break
        else:
            raise ValueError(f"quote not found in any source: {quote[:80]}")
        out.append(text[last:m.start()] + lead.rstrip() + " " + pointer)
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def check(answer: str, sources: list[Source]) -> list[str]:
    """Pointers that don't point at anything."""
    bad = []
    for m in POINTER_RE.finditer(answer):
        if resolve(int(m.group(1)), int(m.group(2)), int(m.group(3)) if m.group(3) else None, sources) is None:
            bad.append(m.group(0))
    return bad


POINTER_LINE = re.compile(r"^(?P<bullet>- )?\*\*.+?\*\*(?P<lead>[^\n{]*?)(?P<ptr>\{\d\.\d+(?:-(?:\d\.)?\d+)?\})\s*$")


def source_title(src: Source) -> str:
    """A heading taken from the source itself, so it can't disagree with the quote under it."""
    label = src.label
    if src.kind == "rta":
        m = re.match(r"Residential Tenancies Act, 2006, s\. ([\d.]+): (.*)", label)
        return f"RTA s. {m.group(1)} · {m.group(2)}" if m else label
    if src.kind == "lease":
        m = re.match(r"Ontario Standard Form of Lease, General Information Part ([A-U]): (.*?)( \(Part.*)?$", label)
        if m:
            return f"Standard lease, Part {m.group(1)} · {m.group(2)}"
        m = re.match(r"Ontario Standard Form of Lease, Section (\d+): (.*)", label)
        return f"Standard lease, Section {m.group(1)} · {m.group(2)}" if m else label
    if src.kind == "guide":
        doc, _, heading = label.partition(', "')
        doc = re.sub(r"LTB Interpretation Guideline (\d+):.*", r"LTB Guideline \1", doc).replace("LTB brochure: ", "LTB brochure, ")
        return f"{doc} · {heading.rstrip('\"').split(':')[0]}"
    return label


CASE_LINE = re.compile(r"^(?P<bullet>- )?In \[(?P<n>\d)\][^\n]*?(?:(?P<lab>Why|What was ordered):\s*(?P<ptr>\{[^}]+\}))?\s*$")
OUTCOME_TEXT = {"tenant": "decided the case in the tenant's favour", "landlord": "decided the case in the landlord's favour"}


def restructure(answer: str, sources: list[Source]) -> str:
    """Make every line that refers to a source say exactly what that source is:

    - quote lines get the quoted source's own title as their heading
    - decision lines are rebuilt from the decision itself (citation and outcome), since a model
      can copy a citation from an earlier answer that doesn't match the passage it points at
    - lines with a pointer to nothing are dropped, as are repeated lines (a small model can loop)
    """
    out, seen = [], set()
    for line in answer.split("\n"):
        key = line.strip()
        if key and key in seen:
            continue
        seen.add(key)
        if check(line, sources):
            continue
        c = CASE_LINE.match(key)
        if c and 1 <= int(c.group("n")) <= len(sources) and sources[int(c.group("n")) - 1].kind == "case":
            n = int(c.group("n"))
            src = sources[n - 1]
            outcome = OUTCOME_TEXT.get(src.side, "made the order below")
            ptr = c.group("ptr") if c.group("ptr") and c.group("ptr").startswith(f"{{{n}.") else None
            out.append(f"{c.group('bullet') or ''}In [{n}] {src.label}, the Board {outcome}." + (f" {c.group('lab')}: {ptr}" if ptr else ""))
            continue
        m = POINTER_LINE.match(line.strip())
        if m:
            n = int(m.group("ptr")[1])
            if 1 <= n <= len(sources):
                line = f"{m.group('bullet') or ''}**{source_title(sources[n - 1])}.** {m.group('ptr')}"
        out.append(line)
    return "\n".join(out)
