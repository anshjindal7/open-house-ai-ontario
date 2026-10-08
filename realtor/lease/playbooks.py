"""Known everyday scenarios ("playbooks") and their official sources.

train/scenarios.json lists situations like "my sink is broken" with branches, each tied to an
official source and the sentence in it that answers that branch. The same file serves two jobs:

  training   train/scenarios.py builds answers from it
  search     when a question closely matches a known scenario, the scenario's sources are added
             to what the model reads, so it has the same sources it was trained with

Picking a quote is shared code, so training and search always see the same sentence.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from .sources import Source, _embedder, load_sources, tokenize

SCENARIOS = Path(__file__).parents[2] / "train" / "scenarios.json"
MAX_QUOTE = 600
AMEND = re.compile(r"\s+\d{4}, c\. \d+.*$", re.S)  # trailing amendment history
# Cosine similarity to a scenario's example questions. Kept strict: true paraphrases score 0.72-0.82
# but unrelated questions reach 0.78 too, and a wrong match brings in the wrong sources. Below this,
# the model answers from ordinary search and states only the branches its sources support.
# (With a Jev key, routing by a typed Choice over scenario names would be the better matcher.)
MATCH_THRESHOLD = 0.86


def _units(text: str) -> list[str]:
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or re.fullmatch(r"\[[^\]]+\]", line):
            continue
        out += [u.strip() for u in re.split(r"(?<=[.;])\s+(?=[A-Z“\"(•])", line) if len(u.strip()) > 25]
    return out


def _score(about: set[str], unit: str) -> float:
    words = set(tokenize(unit))
    return len(about & words) / max(1, len(about)) - len(unit) / 20000


def pick(src_ref: str, about: str, extend: int = 0) -> tuple[Source, str]:
    """The source and the verbatim sentence in it that best matches `about`.

    `extend` adds that many following lines, for statute lead-ins like "…shall not require a
    tenant to," whose meaning is in the clauses below them.
    """
    sources = load_sources()
    if src_ref.startswith("guide:"):
        name = src_ref.split(":", 1)[1].lower()
        candidates = [s for s in sources.values() if s.kind == "guide" and name in s.label.lower()]
        if not candidates:
            raise KeyError(f"no guidance document matches {src_ref!r}")
    else:
        candidates = [sources[src_ref]]
    want = set(tokenize(about))
    src, unit = max(((s, u) for s in candidates for u in _units(s.text)), key=lambda su: _score(want, su[1]))
    if extend:
        start = src.text.find(unit)
        end = start + len(unit)
        for _ in range(extend):
            nxt = src.text.find("\n", end + 1)
            end = len(src.text) if nxt == -1 else nxt
        unit = AMEND.sub("", src.text[start:end].rstrip()).rstrip()
    if len(unit) > MAX_QUOTE:
        unit = unit[:MAX_QUOTE].rsplit(" ", 1)[0]
    focused = replace(src, focus=unit)
    assert unit in focused.shown, f"quote not visible for {src_ref}: {unit[:60]}"
    return focused, unit


@lru_cache(maxsize=1)
def scenarios() -> list[dict]:
    return json.loads(SCENARIOS.read_text())["scenarios"]


@lru_cache(maxsize=1)
def _question_index() -> tuple[np.ndarray, list[int]]:
    questions, owner = [], []
    for i, sc in enumerate(scenarios()):
        for q in sc["questions"] + sc.get("clarify", {}).get("questions", []):
            questions.append(q)
            owner.append(i)
    vectors = np.array(list(_embedder().embed(questions)))
    return vectors, owner


def match(question: str) -> tuple[dict, float] | None:
    """The known scenario this question is about, if it's a close match."""
    vectors, owner = _question_index()
    q = next(iter(_embedder().embed([question])))
    sims = vectors @ q
    best = int(np.argmax(sims))
    return (scenarios()[owner[best]], float(sims[best])) if sims[best] >= MATCH_THRESHOLD else None


@lru_cache(maxsize=64)
def _gold(scenario_id: str) -> tuple[Source, ...]:
    sc = next(s for s in scenarios() if s["id"] == scenario_id)
    merged: dict[str, Source] = {}
    for b in sc["branches"]:
        src, quote = pick(b["src"], b["about"], b.get("extend", 0))
        merged.setdefault(src.id, src)
    return tuple(merged.values())


def gold_sources(scenario: dict) -> list[Source]:
    return list(_gold(scenario["id"]))
