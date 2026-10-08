"""Helpers shared by the dataset builders."""

from __future__ import annotations

import hashlib
import json
import random
import re
from functools import lru_cache

from realtor.lease.local_model import build_messages
from realtor.lease.pointers import pointerize
from realtor.lease.sources import CORPUS, OUTCOME, Source

rng = random.Random(17)


def split_for(key: str) -> str:
    bucket = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 20
    return "test" if bucket == 0 else "valid" if bucket == 1 else "train"


def case_key(c: dict) -> str:
    """One decision can cover several LTB files (and one file several orders): split by citation."""
    return f"ltb:{c.get('citation') or c['file_number']}"


@lru_cache(maxsize=1)
def held_out_cases() -> frozenset[str]:
    """Decisions in the valid/test splits. Training answers must never cite them."""
    out = set()
    for c in map(json.loads, (CORPUS / "ltb_cases.jsonl").open()):
        if split_for(case_key(c)) != "train":
            out.add(f"case:{c['doc_id']}")
    return frozenset(out)


def quote(src: Source) -> str:
    return src.shown.removeprefix("… ").removesuffix(" …").strip()


def with_gold(sources: list[Source], gold: list[Source]) -> list[Source]:
    """Put each gold source into a slot of its own kind, replacing a weaker retrieved one."""
    gold_ids = {g.id for g in gold}
    out = [s for s in sources if s.id not in gold_ids]
    for g in gold:
        slots = [i for i, s in enumerate(out) if s.kind == g.kind and s.id not in gold_ids]
        if slots:
            out[rng.choice(slots[-2:])] = g
        else:
            out.append(g)
    return out


def num(sources: list[Source], src: Source) -> int:
    return [s.id for s in sources].index(src.id) + 1


def example(question: str, sources: list[Source], answer: str, kind: str, key: str,
            role: str | None = None, history: list[tuple[str, str]] | None = None) -> dict:
    # Quotes become pointers ({2.3}); the app inserts the exact text, so the model never retypes it.
    answer = pointerize(answer, sources)
    history = [(q, a) for q, a in (history or [])]
    messages = build_messages(question, sources, role, history) + [{"role": "assistant", "content": answer}]
    return {"messages": messages, "kind": kind, "split": split_for(key)}


def shown_field(src: Source, field: str) -> str:
    m = re.search(rf"^{field}: (.*)$", src.shown, re.M)
    return m.group(1).removesuffix(" …").strip() if m else ""


def case_line(ctx: list[Source], src: Source) -> str:
    """One decision, quoted from what the model can see."""
    result = OUTCOME.get(src.side, OUTCOME["unclear"]).replace("decided", "decided the case")
    why = shown_field(src, "Key reasons")
    return f'In [{num(ctx, src)}] {src.label}, the Board {result}. Why: "{why}"'
