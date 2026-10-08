"""Decide whether a question is one of the known scenarios, and answer those from the playbook.

Matching by embedding similarity alone is too blunt (unrelated questions score as high as real
paraphrases), but choosing one option from a short list is something even a small model does
reliably. So: shortlist the 5 nearest scenarios by similarity, ask the model to pick one or
"none", then build the answer from the playbook. Its headings were written and reviewed in
advance, its quotes are exact, and it always shows a decision each side won.
"""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np

from .playbooks import _question_index, pick, scenarios
from .sources import Source, _embedder, balanced_cases, case_why, retrieve

ROUTER_PROMPT = (
    "You sort questions about renting in Ontario. Pick an option only if the question is about "
    "that exact situation; a question that merely shares words with an option (eviction, rent, "
    "repairs) is not a match. If unsure, pick N. Reply with the option's letter only."
)


def shortlist(question: str, k: int = 5) -> list[dict]:
    vectors, owner = _question_index()
    q = next(iter(_embedder().embed([question])))
    best: dict[int, float] = {}
    for sim, i in zip(vectors @ q, owner):
        best[i] = max(best.get(i, -1.0), float(sim))
    order = sorted(best, key=lambda i: -best[i])[:k]
    return [scenarios()[i] for i in order]


@contextmanager
def adapters_off(model):
    """Route with the plain base model: it routes more carefully than our answer-trained adapter
    (1 wrong route vs 2 on evals/run_router_eval.py). Turning the LoRA layers' scale to 0 gives
    base behaviour without loading the 8B model twice."""
    from mlx_lm.tuner.lora import LoRALinear

    layers = [m for _, m in model.named_modules() if isinstance(m, LoRALinear)]
    saved = [m.scale for m in layers]
    for m in layers:
        m.scale = 0.0
    try:
        yield
    finally:
        for m, sc in zip(layers, saved):
            m.scale = sc


def route(question: str, model, tok, chat_args: dict) -> dict | None:
    """The scenario the model picks for this question, or None."""
    from mlx_lm import generate

    options = shortlist(question)
    letters = "ABCDE"[: len(options)]
    lines = [f"{l}. {sc['topic']}. Examples: " + " / ".join(f'"{q}"' for q in sc["questions"][:3])
             for l, sc in zip(letters, options)]
    lines.append("N. none of these")
    messages = [{"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": f"Question: {question}\n\nOptions:\n" + "\n".join(lines)}]
    prompt = tok.apply_chat_template(messages, **chat_args)
    with adapters_off(model):
        reply = generate(model, tok, prompt=prompt, max_tokens=6)
    m = re.search(r"\b([A-EN])\b", re.sub(r"<think>[\s\S]*?</think>", "", reply).strip().upper())
    if not m or m.group(1) == "N" or m.group(1) not in letters:
        return None
    return options[letters.index(m.group(1))]


def _title(src: Source) -> str:
    from .pointers import source_title
    return source_title(src)


def playbook_answer(sc: dict, question: str, keep: set[int] | None = None) -> tuple[str, list[Source]]:
    """A complete answer from the playbook: reviewed headings, exact quotes, both sides' decisions.

    `keep`: indexes of the branches to include (see tailor()); None keeps all of them.
    Quotes are pointers into the returned sources, so the app renders them like any other answer.
    """
    resolved, sources = [], []
    for i, b in enumerate(sc["branches"]):
        if keep is not None and i not in keep:
            continue
        src, quote = pick(b["src"], b["about"], b.get("extend", 0))
        if src.id not in {s.id for s in sources}:
            sources.append(src)
        resolved.append((b, src, quote))
    from .pointers import _covering

    lines = [sc["opener"], ""]
    for b, src, quote in resolved:
        n = [s.id for s in sources].index(src.id) + 1
        view = sources[n - 1]
        rng = _covering(quote, view)
        if not rng:  # two branches share a source whose window can't show both quotes
            continue
        a, z = rng
        pointer = f"{{{n}.{a}}}" if a == z else f"{{{n}.{a}-{z}}}"
        lines.append(f"**{b['when']}.** {pointer}")

    cases = balanced_cases(question, exclude={s.id for s in sources})
    if cases:
        start = len(sources)
        sources += cases
        both = len({c.side for c in cases} & {"tenant", "landlord"}) == 2
        lines += ["", "**At the Landlord and Tenant Board.** " +
                  ("Two real decisions on similar issues, one each side won:" if both else "A real decision on a similar issue:")]
        for i, c in enumerate(cases, start + 1):
            side = {"tenant": "in the tenant's favour", "landlord": "in the landlord's favour"}.get(c.side)
            why = case_why(c)
            outcome = f"the Board decided the case {side}" if side else "the Board made the order below"
            lines.append(f"In [{i}] {c.label}, {outcome}." + (f" Why: {{{i}.{why}}}" if why else ""))
    lines += ["", "Every situation is a little different. For advice on yours, the LTB (1-888-332-3234) "
                  "or a community legal clinic can help."]
    return "\n".join(lines), sources


NEGOTIATION = re.compile(r"negotiat|bring up|my side|suggestions? for me|approach (?:a|the|this) conversation|what should i say", re.I)
OTHER = {"tenant": "landlord", "landlord": "tenant"}


def wants_negotiation_help(question: str, role: str | None) -> bool:
    return bool(role) and bool(NEGOTIATION.search(question))


def _resolve(sc: dict) -> tuple[list[tuple[dict, Source, str]], list[Source]]:
    resolved, sources = [], []
    for b in sc["branches"]:
        src, quote = pick(b["src"], b["about"], b.get("extend", 0))
        if src.id not in {s.id for s in sources}:
            sources.append(src)
        resolved.append((b, src, quote))
    return resolved, sources


def _pointer(quote: str, sources: list[Source], src: Source) -> str | None:
    from .pointers import _covering

    n = [s.id for s in sources].index(src.id) + 1
    rng = _covering(quote, sources[n - 1])
    if not rng:
        return None
    a, z = rng
    return f"{{{n}.{a}}}" if a == z else f"{{{n}.{a}-{z}}}"


def playbook_negotiation(sc: dict, role: str, question: str) -> tuple[str, list[Source]]:
    """Suggestions for the asker's side, labelled as such, plus the other side's strongest point."""
    resolved, sources = _resolve(sc)
    mine = [r for r in resolved if r[0]["side"] in (role, "both")]
    theirs = [r for r in resolved if r[0]["side"] == OTHER[role]]
    lines = [f"Here are some suggestions for your side as the {role}. These are suggestions only, not legal "
             "advice, and the other side may see it differently.", "", "**Points in your favour**"]
    for b, src, q in mine:
        if p := _pointer(q, sources, src):
            lines.append(f"- **{b['when']}.** {p}")
    won = [c for c in balanced_cases(question, exclude={s.id for s in sources}) if c.side == role][:1]
    for c in won:
        sources.append(c)
        lines.append(f"- In [{len(sources)}] {c.label}, the Board decided the case in the {role}'s favour.")
    if theirs:
        b, src, q = theirs[0]
        if p := _pointer(q, sources, src):
            lines += ["", "**Be ready for this point from the other side**", f"- **{b['when']}.** {p}"]
    lines += ["", "**For the conversation itself.** Keep it in writing where you can, so there's a record of what "
                  "was agreed. If you can't agree, either of you can apply to the Landlord and Tenant Board."]
    return "\n".join(lines), sources


async def tailor(sc: dict, question: str) -> set[int]:
    """With Jev: keep only the playbook branches whose passage helps answer this question.

    A guide covers a whole situation; a particular question may need only part of it (an eviction
    question doesn't need "before a tenancy starts"). Always keeps at least the best branch.
    """
    from . import jev

    quotes = [pick(b["src"], b["about"], b.get("extend", 0))[1] for b in sc["branches"]]
    scores = await jev.relevance(question, quotes)
    keep = {i for i, p in enumerate(scores) if p >= jev.RELEVANT_MIN}
    keep |= {i for i, b in enumerate(sc["branches"]) if b.get("always")} if keep else set()
    return keep or {max(range(len(scores)), key=scores.__getitem__)}


async def selected_answer(question: str, sources: list[Source], previous: str = "", top: int = 4) -> tuple[str, dict]:
    """An answer built from the sentences Jev rates most relevant. The model writes nothing.

    Up to `top` statute/guidance passages (at most 2 per source), then any decision whose
    reasoning fits, all as pointers. If nothing clears the threshold, says so plainly.
    """
    from . import jev
    from .pointers import source_title

    scores = await jev.score_sentences(question, sources, previous)
    good = sorted(((p, n, k) for (n, k), p in scores.items() if p >= jev.RELEVANT_MIN), reverse=True)
    rules, per_source = [], {}
    for p, n, k in good:
        if sources[n - 1].kind == "case" or per_source.get(n, 0) >= 2 or len(rules) >= top:
            continue
        per_source[n] = per_source.get(n, 0) + 1
        rules.append((n, k))
    cases = []
    for p, n, k in good:
        if sources[n - 1].kind == "case" and n not in {c[0] for c in cases} and len(cases) < 2:
            cases.append((n, k))

    if not rules:
        return ("I couldn't find the part of Ontario law that answers this directly. The official sources I "
                "searched don't cover it closely enough to quote. Try rewording the question, or check with the "
                "LTB (1-888-332-3234) or a community legal clinic.", {"selected": 0})

    lines = ["Here's what Ontario's official sources say about this.", ""]
    # Keep sentences from the same source together and in the source's own order.
    for n in dict.fromkeys(n for n, _ in rules):
        ks = sorted(k for m, k in rules if m == n)
        span = f"{{{n}.{ks[0]}}}" if len(ks) == 1 else (f"{{{n}.{ks[0]}-{ks[-1]}}}" if ks[-1] - ks[0] == len(ks) - 1 else None)
        if span:
            lines.append(f"**{source_title(sources[n - 1])}.** {span}")
        else:  # separate sentences from one source: one heading, each quote under it
            lines.append(f"**{source_title(sources[n - 1])}.** {{{n}.{ks[0]}}}")
            lines += [f"{{{n}.{k}}}" for k in ks[1:]]
    if cases:
        lines += ["", "**At the Landlord and Tenant Board.** A real decision on a similar issue:" if len(cases) == 1
                  else "**At the Landlord and Tenant Board.** Real decisions on similar issues:"]
        for n, k in cases:
            src = sources[n - 1]
            k = case_why(src, k)
            outcome = {"tenant": "decided the case in the tenant's favour", "landlord": "decided the case in the landlord's favour"}.get(src.side, "made the order below")
            lines.append(f"In [{n}] {src.label}, the Board {outcome}." + (f" Why: {{{n}.{k}}}" if k else ""))
    lines += ["", "Every situation is a little different. For advice on yours, the LTB (1-888-332-3234) or a community legal clinic can help."]
    return "\n".join(lines), {"selected": len(rules) + len(cases), "scored": len(scores)}


CLOSING = "Every situation is a little different."
LEASE_SUMMARY = {
    "conflicts": "A term in your lease may conflict with this (quoted below). Where a lease term conflicts with the Act, the Act wins.",
    "matches": "Your lease says the same (quoted below).",
}
# Fixed wording for how a lease line relates to the law quoted above it (Jev picks which applies).
RELATION_NOTE = {
    "matches": "_This lines up with the law quoted above._",
    "conflicts": "_This may conflict with the law quoted above. Where a lease term conflicts with the Act, the Act "
                 "wins (RTA s. 4, quoted below)._",
}
VOID_NOTE = "a provision in a tenancy agreement that is inconsistent with this Act or the regulations is void"


async def lease_section(question: str, body: str, sources: list[Source], terms: list[Source],
                        previous: str = "") -> tuple[str, list[Source], str]:
    """Add what the user's own lease says about the question, quoted from the lease itself.

    Jev picks the lease sentences that bear on the question and runs the lease checks on those
    terms; a flagged term is marked, with the Act's rule that conflicting terms are void.
    Pointers go up to source 9, so the lease gets whatever room is left. Also returns a one-line
    summary of how the lease relates to the law, for the top of the answer ("" if none).
    """
    from . import jev
    from .checks import CHECKS, LABELS
    from .clauses import Clause
    from .pointers import source_title
    from .scanner import scan_lease
    from .sources import is_form_text, load_sources

    room = 9 - len(sources)
    if room < 1 or not terms:
        return body, sources, ""
    terms = terms[:room]
    scores = await jev.score_sentences(question, terms, previous)
    good = sorted(((p, n, k) for (n, k), p in scores.items()
                   if p >= jev.RELEVANT_MIN and not is_form_text(terms[n - 1].sentences[k - 1])), reverse=True)[:4]
    if not good:
        return body, sources, ""
    used = list(dict.fromkeys(n for _, n, _ in good))
    picked = {n: sorted(k for _, nn, k in good if nn == n) for n in used}
    law = [t for _, t in jev.pointed_passages(body, sources)]
    quoted = [" ".join(terms[n - 1].sentences[k - 1] for k in picked[n]) for n in used]
    # The lease checks run on exactly the lines quoted, so a flag is always about what the reader sees.
    findings, relations = await asyncio.gather(
        scan_lease([Clause(i, terms[n - 1].label, q) for i, (n, q) in enumerate(zip(used, quoted))]),
        jev.compare_terms(law, quoted))
    laws = {c.key: c for c in CHECKS}
    out = list(sources)
    lines = ["", "**What your lease says.**"]
    flagged, kinds = False, []
    for f, rel, n in zip(findings, relations, used):
        out.append(terms[n - 1])
        m = len(out)
        ks = picked[n]
        lines.append(f"**{source_title(terms[n - 1])}.** {{{m}.{ks[0]}}}")
        lines += [f"{{{m}.{k}}}" for k in ks[1:]]
        sure = rel
        if f.flags or sure == "conflicts":  # either check is enough to say it may conflict
            flagged = True
            kinds.append("conflicts")
            lines.append(RELATION_NOTE["conflicts"])
            if f.flags:
                names = ", ".join(f"{LABELS.get(k, k)} ({laws[k].law})" for k in f.flags)
                lines.append(f"_Open House's lease check flags this: {names}._")
        elif sure == "matches":
            kinds.append(sure)
            lines.append(RELATION_NOTE[sure])
    rta4 = load_sources().get("rta:4")
    if flagged and rta4 and len(out) < 9:
        out.append(rta4)
        k = next((k for k, t in enumerate(rta4.sentences, 1) if "void" in t), 1)
        lines.append(f"**{source_title(rta4)}.** {{{len(out)}.{k}}}")
    text = "\n".join(lines)
    i = body.rfind(CLOSING)
    body = (body[:i].rstrip() + "\n" + text + "\n\n" + body[i:]) if i >= 0 else body + "\n" + text
    summary = next((LEASE_SUMMARY[r] for r in ("conflicts", "matches") if r in kinds), "")
    return body, out, summary


async def drop_unrelated_cases(question: str, body: str, sources: list[Source]) -> tuple[str, list[Source]]:
    """With Jev: remove example decisions that don't fit the question, and fix reasons that don't
    support the decision's outcome (shown instead: what was ordered).

    Decisions are found by search, and a near miss (a fire for a broken door) teaches nothing.
    If none fit, the whole Board section goes.
    """
    from . import jev
    from .pointers import POINTER_RE, resolve
    from .sources import case_ordered

    lines = body.split("\n")
    at = [i for i, l in enumerate(lines) if re.match(r"(- )?In \[\d\]", l)]
    if not at:
        return body, sources
    items = []
    for i in at:
        n = int(re.search(r"\[(\d)\]", lines[i]).group(1))
        src = sources[n - 1]
        app = re.search(r"^Application: (.*)$", src.shown, re.M)
        why = " ".join(w for w in (resolve(int(m.group(1)), int(m.group(2)), int(m.group(3)) if m.group(3) else None, sources)
                                   for m in POINTER_RE.finditer(lines[i])) if w)
        outcome = {"tenant": "The Board decided in the tenant's favour.", "landlord": "The Board decided in the landlord's favour."}.get(src.side, "")
        items.append((((app.group(1) if app else "") + " " + why).strip(), why if outcome else "", outcome))
    scores = await jev.case_checks(question, items)
    gone = {i for i, (p, _) in zip(at, scores) if p < jev.RELEVANT_MIN}
    for i, (_, k) in zip(at, scores):
        if i in gone or k >= 0.5:
            continue
        n = int(re.search(r"\[(\d)\]", lines[i]).group(1))
        order = case_ordered(sources[n - 1])
        lines[i] = re.sub(r"\s*Why:\s*\{[^}]+\}", f" What was ordered: {{{n}.{order}}}" if order else "", lines[i])
    if not gone:
        return "\n".join(lines), sources
    dropped = {int(re.search(r"\[(\d)\]", lines[i]).group(1)) for i in gone}
    lines = [l for i, l in enumerate(lines) if i not in gone]
    if len(gone) == len(at):  # no decision fits: drop the heading too
        lines = [l for l in lines if not l.startswith("**At the Landlord and Tenant Board.**")]
    else:
        lines = [("**At the Landlord and Tenant Board.** A real decision on a similar issue:"
                  if l.startswith("**At the Landlord and Tenant Board.**") else l) for l in lines]
    return renumber(re.sub(r"\n{3,}", "\n\n", "\n".join(lines)), sources, dropped)


def renumber(body: str, sources: list[Source], drop: set[int]) -> tuple[str, list[Source]]:
    """Remove sources nobody points at any more, and renumber [n] and {n.k} to match."""
    new = {}
    for n in range(1, len(sources) + 1):
        if n not in drop:
            new[n] = len(new) + 1
    body = re.sub(r"\[(\d)\]", lambda m: f"[{new.get(int(m.group(1)), m.group(1))}]", body)
    body = re.sub(r"\{(\d)\.", lambda m: f"{{{new.get(int(m.group(1)), m.group(1))}.", body)
    body = re.sub(r"-(\d)\.(\d+)\}", lambda m: f"-{new.get(int(m.group(1)), m.group(1))}.{m.group(2)}}}", body)
    return body, [s for n, s in enumerate(sources, 1) if n not in drop]


SHORT_ANSWERS = Path(__file__).parents[2] / "train" / "short_answers.json"
BRANCH_CITE = re.compile(r"\[b(\d)\]")


@lru_cache(maxsize=1)
def _short_answers() -> dict:
    return json.loads(SHORT_ANSWERS.read_text())["short"] if SHORT_ANSWERS.exists() else {}


def short_answer(sc: dict, option: str | None = None) -> tuple[str, set[int]]:
    """A guide's short answer (the version for `option` if given) and the branches it cites."""
    entry = _short_answers().get(sc["id"])
    if not entry:
        return "", set()
    text = entry.get("options", {}).get(option, {}).get("say") or entry["say"]
    return text, {int(b) for b in BRANCH_CITE.findall(text)}


def short_options(sc: dict) -> dict:
    return _short_answers().get(sc["id"], {}).get("options", {})


def cite_short(text: str, sc: dict, sources: list[Source]) -> str:
    """Turn [bN] into the source number of branch N, placed before the sentence's full stop."""
    ids = [s.id for s in sources]

    def num(m):
        b = sc["branches"][int(m.group(1))]
        src, _ = pick(b["src"], b["about"], b.get("extend", 0))
        return f"[{ids.index(src.id) + 1}]" if src.id in ids else ""
    text = BRANCH_CITE.sub(num, text)
    text = re.sub(r"(?:\[\d\]\s*)+", lambda m: " ".join(f"[{n}]" for n in sorted(set(re.findall(r"\d", m.group(0))))) + " ", text)
    text = re.sub(r"([.!?])((?:\s*\[\d\])+)(?=\s|$)", lambda m: m.group(2) + m.group(1), text)  # "rule. [1]" -> "rule [1]."
    return re.sub(r"\s+([.!?])", r"\1", re.sub(r"\s{2,}", " ", text)).replace(" [", " [").strip()
