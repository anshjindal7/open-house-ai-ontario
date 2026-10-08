"""Jev (TypeSafe AI) for the typed decisions in Open House.

Jev returns probabilities over options we define, never free text, so it can't invent an
answer. It's used where a yes/no or pick-one judgment is exactly what's needed:

  route       which known scenario a question is about (or none), with a confidence
  relevance   for each passage an answer quotes: does it actually help answer the question?

Everything here is optional. Without TYPESAFE_API_KEY the app falls back to the local router
and skips the relevance check. Note: with Jev on, the question and the quoted passages are sent
to TypeSafe's API.
"""

from __future__ import annotations

import os

from .playbooks import scenarios
from .sources import Source

ROUTE_MIN_CONFIDENCE = 0.6  # below this, treat as "none" and let the model answer
RELEVANT_MIN = 0.65         # below this a passage is flagged as possibly off-topic (best on evals/run_relevance_eval.py: 92%)


def enabled() -> bool:
    return bool(os.environ.get("TYPESAFE_API_KEY"))


def _route_criteria() -> dict[str, str]:
    criteria = {sc["id"]: f"{sc['topic']}. For example: " + " / ".join(sc["questions"][:3]) for sc in scenarios()}
    criteria["none"] = "None of these: the question is about a different situation, even if it shares words with one."
    return criteria


async def route(question: str) -> tuple[dict | None, float]:
    """The scenario Jev picks, and its confidence. (None, confidence) if it picks none or isn't sure."""
    from typesafe_sdk import AsyncTypeSafeClient, Choice

    async with AsyncTypeSafeClient() as client:
        r = await client.system_one(
            state={"question": question},
            questions={"situation": Choice(
                instructions="Which situation is this question about? Pick none unless it is clearly about that situation.",
                criteria=_route_criteria())},
        )
    ans = r.choices["situation"]
    if ans.choice == "none" or ans.confidence < ROUTE_MIN_CONFIDENCE:
        return None, ans.confidence
    return next(sc for sc in scenarios() if sc["id"] == ans.choice), ans.confidence


async def relevance(question: str, passages: list[str]) -> list[float]:
    """For each passage, the probability that it helps answer the question. One parallel request."""
    from typesafe_sdk import AsyncTypeSafeClient, Noul

    if not passages:
        return []
    state = {"question": question, **{f"passage_{i}": p for i, p in enumerate(passages, 1)}}
    questions = {f"p{i}": Noul(instructions=f"Does passage_{i} contain information that helps answer the question "
                                            "(a rule, right, duty or outcome that applies to it)?")
                 for i in range(1, len(passages) + 1)}
    async with AsyncTypeSafeClient() as client:
        r = await client.system_one(state=state, questions=questions)
    return [r.nouls[f"p{i}"].noul for i in range(1, len(passages) + 1)]


def pointed_passages(answer: str, sources: list[Source]) -> list[tuple[str, str]]:
    """(pointer, exact text) for every statute/guidance passage the answer quotes. Decisions are
    excluded: a case is an example, not a rule, and is matched by search, not by the model."""
    from .pointers import POINTER_RE, resolve

    out, seen = [], set()
    for m in POINTER_RE.finditer(answer):
        n = int(m.group(1))
        if m.group(0) in seen or not 1 <= n <= len(sources) or sources[n - 1].kind == "case":
            continue
        seen.add(m.group(0))
        text = resolve(n, int(m.group(2)), int(m.group(3)) if m.group(3) else None, sources)
        if text:
            out.append((m.group(0), text))
    return out


async def is_followup(previous: str, question: str) -> float:
    """Probability that `question` continues the previous one rather than starting a new topic."""
    from typesafe_sdk import AsyncTypeSafeClient, Noul

    async with AsyncTypeSafeClient() as client:
        r = await client.system_one(
            state={"previous_question": previous, "new_question": question},
            questions={"followup": Noul(instructions="Is the new question a follow-up that depends on the previous "
                                                     "question (it only makes sense together with it), rather than a "
                                                     "new, separate topic?")},
        )
    return r.nouls["followup"].noul


MAX_PER_SOURCE = 12  # sentences scored per source (one request scores them all in parallel)


async def score_sentences(question: str, sources: list[Source], previous: str = "") -> dict[tuple[int, int], float]:
    """Relevance of every sentence the sources show, keyed by (source number, sentence number).

    For decisions only the "Key reasons" sentences count: a decision is an example of the rule,
    worth showing when its reasoning fits. `previous` gives context for follow-up questions.
    """
    from typesafe_sdk import AsyncTypeSafeClient, Noul

    state: dict[str, str] = {"question": question}
    if previous:
        state["previous_question"] = previous
    questions, keys = {}, {}
    for n, src in enumerate(sources, 1):
        sents = src.sentences
        if src.kind == "case":
            reasons = src.shown.split("Key reasons:", 1)[-1]
            picks = [k for k, t in enumerate(sents, 1) if t and t in reasons][:2]
        else:
            picks = list(range(1, len(sents) + 1))[:MAX_PER_SOURCE]
        for k in picks:
            key = f"n{n}_{k}"
            state[f"sentence_{n}_{k}"] = sents[k - 1]
            questions[key] = Noul(instructions=f"Does sentence_{n}_{k} directly help answer the question"
                                               + (" (read together with previous_question)" if previous else "")
                                               + " with a rule, right, duty or outcome that applies to it?")
            keys[key] = (n, k)
    if not questions:
        return {}
    async with AsyncTypeSafeClient() as client:
        r = await client.system_one(state=state, questions=questions)
    return {keys[k]: r.nouls[k].noul for k in questions}


# Lease line vs the law it's quoted under. One yes/no per (lease line, law passage) pair worked far
# better than a four-way pick (which hovered near chance): on evals/run_lease_relation_eval.py,
# conflicts scored 0.68-0.92, non-conflicts at most 0.12, matches 0.82-0.90.
CONFLICT_MIN = 0.5
MATCH_MIN = 0.6


async def compare_terms(law: list[str], lease: list[str]) -> list[str | None]:
    """For each lease line: "conflicts", "matches" or None (neither is clear), against the law passages."""
    from typesafe_sdk import AsyncTypeSafeClient, Noul

    if not law or not lease:
        return [None] * len(lease)
    state = {f"law_{i}": t for i, t in enumerate(law, 1)} | {f"lease_{j}": t for j, t in enumerate(lease, 1)}
    questions = {}
    for j in range(1, len(lease) + 1):
        for i in range(1, len(law) + 1):
            questions[f"c{j}_{i}"] = Noul(instructions=f"Does lease_{j} allow or require something that law_{i} forbids, "
                                                       f"or contradict what law_{i} requires or protects?")
            questions[f"m{j}_{i}"] = Noul(instructions=f"Does lease_{j} say the same thing as law_{i}, or apply it, "
                                                       "without contradicting it?")
    async with AsyncTypeSafeClient() as client:
        r = await client.system_one(state=state, questions=questions)
    out = []
    for j in range(1, len(lease) + 1):
        con = max(r.nouls[f"c{j}_{i}"].noul for i in range(1, len(law) + 1))
        mat = max(r.nouls[f"m{j}_{i}"].noul for i in range(1, len(law) + 1))
        out.append("conflicts" if con >= CONFLICT_MIN else "matches" if mat >= MATCH_MIN else None)
    return out


SHORT_PICK_MIN = 0.7  # stricter than routing: below this the general short answer (which covers every case) is used


async def pick_short(question: str, options: dict[str, dict]) -> tuple[str | None, float]:
    """Which fact-specific version of a guide's short answer fits the question, if Jev is confident."""
    from typesafe_sdk import AsyncTypeSafeClient, Choice

    if not options:
        return None, 0.0
    criteria = {k: o["when"] for k, o in options.items()}
    criteria["general"] = "None of these specifically, or the question doesn't say."
    async with AsyncTypeSafeClient() as client:
        r = await client.system_one(state={"question": question}, questions={"facts": Choice(
            instructions="Which of these describes the situation in the question? Pick general unless the question clearly says so.",
            criteria=criteria)})
    ans = r.choices["facts"]
    if ans.choice == "general" or ans.confidence < SHORT_PICK_MIN:
        return None, ans.confidence
    return ans.choice, ans.confidence


async def case_checks(question: str, cases: list[tuple[str, str, str]]) -> list[tuple[float, float]]:
    """For each example decision (summary, reason shown, outcome): does it fit the question, and
    does the reason support the outcome? A decision often covers several claims, and a finding on
    one of them can point the other way ("not satisfied the tenant caused the damage" in a case the
    landlord won overall)."""
    from typesafe_sdk import AsyncTypeSafeClient, Noul

    if not cases:
        return []
    state, questions = {"question": question}, {}
    for i, (summary, why, outcome) in enumerate(cases, 1):
        state[f"case_{i}"] = summary
        questions[f"p{i}"] = Noul(instructions=f"Does case_{i} contain information that helps answer the question "
                                               "(a rule, right, duty or outcome that applies to it)?")
        if why:
            state[f"reason_{i}"], state[f"outcome_{i}"] = why, outcome
            questions[f"k{i}"] = Noul(instructions=f"Does reason_{i} support outcome_{i}, rather than describe a finding "
                                                   "that went the other way?")
    async with AsyncTypeSafeClient() as client:
        r = await client.system_one(state=state, questions=questions)
    return [(r.nouls[f"p{i}"].noul, r.nouls[f"k{i}"].noul if f"k{i}" in questions else 1.0)
            for i in range(1, len(cases) + 1)]
