"""Check a model answer against the sources it was given, before anyone sees it.

Deterministic checks (always on):
  quotes      every "quoted" passage appears word for word in the source it cites
  citations   every [n] points at a source that was actually provided
  facts       numbers, amounts, deadlines and section numbers written outside quotes appear in
              the sources the same sentence cites (or, if it cites none, in any source)

Optional Jev check (TYPESAFE_API_KEY set): each plain-language sentence that cites a source is
judged against that source's text with a yes/no question.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from .pointers import POINTER_RE, check as check_pointers, expand
from .sources import Source

QUOTE_RE = re.compile(r'(:[ \n])"([\s\S]+?)"(?=[ \t]*(?:\n|$))')
CITE_RE = re.compile(r"\[(\d)\]")
# Things a small model is most likely to get subtly wrong: amounts, durations, section numbers.
FACT_RE = re.compile(
    r"\$\s?\d[\d,]*(?:\.\d+)?"                                   # $500, $1,850.00
    r"|\b\d+(?:\.\d+)?\s?%"                                      # 5%
    r"|\b\d+\s*(?:days?|weeks?|months?|years?|hours?)\b"         # 90 days, 24 hours
    r"|\b(?:s\.|section|subsection)\s*\d+(?:\.\d+)?(?:\s*\(\d+\))?",  # s. 20, section 48.1
    re.I)
NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "seven": "7", "twelve": "12", "sixty": "60", "ninety": "90"}


def _norm(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", s).strip().lower()


def _fact_key(fact: str) -> str:
    """'90 days' -> '90 day', 's. 20' -> '20', '$1,850' -> '1850' so wording differences don't matter."""
    f = fact.lower().replace(",", "").replace("$", "").strip()
    f = re.sub(r"^(?:s\.|section|subsection)\s*", "", f)
    f = re.sub(r"\s+", " ", f)
    return re.sub(r"(day|week|month|year|hour)s\b", r"\1", f)


def _facts_in(text: str) -> set[str]:
    t = text.lower()
    for word, digit in NUMBER_WORDS.items():
        t = re.sub(rf"\b{word}\b", digit, t)
    return {_fact_key(m.group(0)) for m in FACT_RE.finditer(t)}


@dataclass
class Issue:
    kind: str     # "pointer" | "quote" | "citation" | "fact" | "outcome" | "jev"
    detail: str
    text: str     # the offending quote, citation or sentence


@dataclass
class Report:
    checked_quotes: int = 0
    checked_facts: int = 0
    issues: list[Issue] = field(default_factory=list)
    jev_used: bool = False

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict:
        return {"ok": self.ok, "quotes": self.checked_quotes, "facts": self.checked_facts, "jev": self.jev_used,
                "issues": [vars(i) for i in self.issues]}


def _sentences_outside_quotes(answer: str) -> list[str]:
    plain = QUOTE_RE.sub(r'\1"…"', answer.replace("“", '"').replace("”", '"'))
    parts = re.split(r"(?<=[.!?])\s+|\n+", plain)
    return [p.strip() for p in parts if len(p.strip()) > 3]


def verify(answer: str, sources: list[Source], question: str = "") -> Report:
    """`question`: facts the person stated themselves (an amount, a lease section) count as known."""
    report = Report()
    for bad in check_pointers(answer, sources):
        report.issues.append(Issue("pointer", f"{bad} doesn't point at a sentence in the sources", bad))
    report.checked_quotes += len(POINTER_RE.findall(answer))  # pointers are exact by construction
    answer = POINTER_RE.sub("", answer)  # what's left is the model's own words, checked below
    by_n = {i + 1: s for i, s in enumerate(sources)}
    shown = {n: _norm(s.shown) for n, s in by_n.items()}
    text = answer.replace("“", '"').replace("”", '"')

    outside_quotes = QUOTE_RE.sub(r'\1"…"', text)  # "[9]" inside a quoted decision is a paragraph number
    for n in {int(m.group(1)) for m in CITE_RE.finditer(outside_quotes)}:
        if n not in by_n:
            report.issues.append(Issue("citation", f"[{n}] doesn't match any source", f"[{n}]"))

    for m in QUOTE_RE.finditer(text):
        quote = m.group(2).removesuffix("…").removesuffix(" …").strip()
        if len(quote) < 15:
            continue
        report.checked_quotes += 1
        # The source a quote belongs to is the last citation before it on the same line.
        line_start = text.rfind("\n", 0, m.start()) + 1
        cites = [int(c) for c in CITE_RE.findall(text[line_start:m.start()])]
        targets = [cites[-1]] if cites and cites[-1] in shown else list(shown)
        if not any(_norm(quote) in shown[n] for n in targets):
            where = f"source [{targets[0]}]" if len(targets) == 1 else "any source"
            report.issues.append(Issue("quote", f"not found word for word in {where}", quote[:200]))

    for m in re.finditer(r"\[(\d)\][^\n]*?decided the case in the (tenant|landlord)'s favour", text):
        n, side = int(m.group(1)), m.group(2)
        if n in by_n and by_n[n].kind == "case" and f"decided in the {side}'s favour" not in by_n[n].shown:
            report.issues.append(Issue("outcome", f"source [{n}] wasn't decided in the {side}'s favour", m.group(0)[:200]))

    asked = _facts_in(question)
    all_facts = set().union(asked, *(_facts_in(s.shown + " " + s.label) for s in sources))
    for sentence in _sentences_outside_quotes(text):
        facts = _facts_in(sentence)
        if not facts:
            continue
        cited = [int(c) for c in CITE_RE.findall(sentence) if int(c) in by_n]
        pool = set().union(asked, *(_facts_in(by_n[n].shown + " " + by_n[n].label) for n in cited)) if cited else all_facts
        for fact in facts:
            report.checked_facts += 1
            if fact not in pool:
                report.issues.append(Issue("fact", f"\"{fact}\" isn't in the cited sources", sentence[:200]))
    return report


async def jev_check(answer: str, sources: list[Source], report: Report, question: str = "") -> Report:
    """Jev's two checks, sent together as one request (Jev answers every question in parallel):

    supported   each plain-language sentence that cites a source is backed by that source
    relevant    each statute/guidance passage the answer quotes actually helps answer the question
                ("verified" isn't "relevant": a real passage can still be off-topic)
    """
    if not os.environ.get("TYPESAFE_API_KEY"):
        return report
    from typesafe_sdk import AsyncTypeSafeClient, Noul

    from . import jev

    by_n = {i + 1: s for i, s in enumerate(sources)}
    pairs = []
    for sentence in _sentences_outside_quotes(answer):
        cited = [int(c) for c in CITE_RE.findall(sentence) if int(c) in by_n]
        claim = POINTER_RE.sub("", CITE_RE.sub("", sentence)).replace('"…"', "").strip(" .*-")
        if cited and len(claim.split()) >= 6:
            pairs.append((sentence, claim, by_n[cited[-1]]))
    passages = jev.pointed_passages(answer, sources) if question else []
    if not pairs and not passages:
        return report

    state: dict[str, str] = {}
    questions: dict = {}
    for i, (_, claim, src) in enumerate(pairs, 1):
        state[f"statement_{i}"], state[f"source_{i}"] = claim, src.shown
        questions[f"s{i}"] = Noul(instructions=f"Is statement_{i} fully supported by source_{i}, without adding "
                                               f"facts source_{i} doesn't contain?")
    if passages:
        state["question"] = question
        for i, (_, text) in enumerate(passages, 1):
            state[f"passage_{i}"] = text
            questions[f"p{i}"] = Noul(instructions=f"Does passage_{i} contain information that helps answer the "
                                                   "question (a rule, right, duty or outcome that applies to it)?")

    async with AsyncTypeSafeClient() as client:
        r = await client.system_one(state=state, questions=questions)
    for i, (sentence, _, _) in enumerate(pairs, 1):
        if r.nouls[f"s{i}"].noul < 0.5:
            report.issues.append(Issue("jev", "Jev judged this not supported by its source", sentence[:200]))
    for i, (pointer, text) in enumerate(passages, 1):
        p = r.nouls[f"p{i}"].noul
        if p < jev.RELEVANT_MIN:
            report.issues.append(Issue("relevance", f"{pointer} may not address the question (Jev: {p:.0%})", text[:200]))
    report.jev_used = True
    return report
