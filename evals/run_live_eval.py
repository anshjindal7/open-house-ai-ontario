"""End-to-end check: real questions through search → model → fact-checker.

    python -m evals.run_live_eval

The held-out test in train/compare.py guarantees the right sources are in the prompt. This one
doesn't: it uses ordinary search, like the app, so it catches the failure where a branch heading
("If the tenant damaged it") is paired with the wrong source because the right one wasn't found.

For each question it reports:
  verified     the fact-checker found nothing wrong (pointers, facts, case outcomes)
  coverage     of the sources the situation needs (and search found), how many the answer used
  looped       the model started repeating itself and was stopped

Headings on quotes come from the sources' own titles (pointers.restructure), so they can't
mismatch the quote; what's left to measure is whether the right passages were chosen.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from realtor.lease.local_model import answer
from realtor.lease.playbooks import pick, scenarios
from realtor.lease.sources import retrieve
from realtor.lease.verify import verify

OUT = Path(__file__).parent / f"live_eval_{os.environ.get('EVAL_TAG', 'results')}.jsonl"

# Paraphrases the model never saw in training, plus the app's starters.
QUESTIONS = [
    ("broken-fixture", "My sink is broken. What should I do?"),
    ("broken-fixture", "My toilet is leaking, who has to fix it?"),
    ("broken-fixture", "The oven in my apartment doesn't work anymore."),
    ("pests", "I found bed bugs in my bedroom. What now?"),
    ("entry", "My landlord came into my unit while I was at work without telling me."),
    ("rent-increase", "Can the rent go up in the middle of a lease?"),
    ("late-rent", "The tenant is two weeks late on rent. What happens next?"),
    ("own-use", "My landlord says his son is moving in so I have to leave."),
    ("damage-deposit", "The landlord is asking for a $1,000 damage deposit. Can they?"),
    ("locks", "I came home and the locks were changed."),
    (None, "Can my landlord evict me for having a baby?"),
    (None, "Can I paint the walls in my rental?"),
]


def branch_sources(scenario_id: str) -> dict[str, str]:
    sc = next(s for s in scenarios() if s["id"] == scenario_id)
    return {b["when"]: pick(b["src"], b["about"], b.get("extend", 0))[0].id for b in sc["branches"]}


def main() -> None:
    from realtor.lease.local_model import repeating
    from realtor.lease.pointers import POINTER_RE, restructure

    rows, verified, covered, expected_total, loops = [], 0, 0, 0, 0
    for scenario_id, q in QUESTIONS:
        sources = retrieve(q, "question")
        raw, _ = answer(q, sources)
        text = restructure(raw, sources)
        report = verify(text, sources, question=q)
        pointed = {sources[int(m.group(1)) - 1].id for m in POINTER_RE.finditer(text) if int(m.group(1)) <= len(sources)}
        missing = []
        if scenario_id:
            # The sources this situation needs: each branch's own source, if search found it.
            needed = {sid for sid in branch_sources(scenario_id).values() if sid in {x.id for x in sources}}
            expected_total += len(needed)
            covered += len(needed & pointed)
            missing = sorted(needed - pointed)
        looped = repeating(raw + "\n")
        loops += looped
        verified += report.ok
        rows.append({"q": q, "scenario": scenario_id, "answer": text, "raw": raw, "verified": report.ok,
                     "issues": [vars(i) for i in report.issues], "missing_sources": missing, "looped": looped})
        ok = report.ok and not missing and not looped
        print(f"{'ok  ' if ok else 'FAIL'} {q[:70]}")
        if missing:
            print(f"       didn't use: {', '.join(missing)}")
        if looped:
            print("       looped (stopped early)")
        for i in report.issues:
            print(f"       {i.kind}: {i.detail}")
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"\nverified by the fact-checker: {verified}/{len(rows)}")
    print(f"needed sources the answer used: {covered}/{expected_total}")
    print(f"answers that looped: {loops}/{len(rows)}")
    print(f"full answers: {OUT}")


if __name__ == "__main__":
    main()
