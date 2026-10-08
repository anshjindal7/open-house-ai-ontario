"""Build training data from real Ontario sources only.

    python -m train.build_dataset

Inputs (built by pipeline/):
  corpus/rta_sections.jsonl   Residential Tenancies Act, 2006, by section
  corpus/standard_lease.jsonl Ontario Standard Form of Lease (sections + official explanations)
  corpus/ltb_cases.jsonl      2026 LTB decisions, redacted, with outcomes and quoted lease terms

Every quote in an answer comes from a source that appears in its prompt, so the model learns
to find and cite the right source rather than to recall law from memory. Legal content is
never written by hand or by another AI model. The hand-written parts are question wording,
short connecting lines (openers, branch labels like "If it broke from normal use"), and the
"suggestions only" framing for negotiations; see train/scenarios.json.

Writes train/data/{train,valid,test}.jsonl. The split is by source, so a test case or statute
section never appears in training.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from dataclasses import replace

from realtor.lease.local_model import clause_question
from realtor.lease.sources import CORE_PARTS, OUTCOME, CORPUS, Source, case_summary, load_sources, retrieve, tidy

from . import scenarios
from .common import case_key, example, num, quote, rng, shown_field, split_for, with_gold

OUT = Path(__file__).parent / "data"

# Lease sections whose text doesn't name a Part, mapped to the Part that explains them.
EXTRA_PARTS = {6: ["K"], 10: ["S"], 13: ["J"], 15: ["Q", "R", "H"]}


# --- RTA sections ---------------------------------------------------------------------------

RTA_QUESTIONS = [
    "What does Ontario's Residential Tenancies Act say about {t}?",
    "Is there an Ontario rule about {t}?",
    "The lease mentions {t}. What does the law say?",
    "What are the rules for landlords and tenants in Ontario about {t}?",
]


def rta_examples(all_sources: dict[str, Source], rta_meta: dict[str, dict]) -> list[dict]:
    out = []
    for sid, src in all_sources.items():
        if src.kind != "rta":
            continue
        meta = rta_meta[sid.split(":")[1]]
        if meta["part"] not in CORE_PARTS and rng.random() > 0.3:
            continue
        topic = re.sub(r"[“”\"]", "", meta["title"]).lower()
        for template in rng.sample(RTA_QUESTIONS, 2):
            q = template.format(t=topic)
            ctx = with_gold(retrieve(q), [src])
            # "RTA s. 14 (“No pet” provisions void)": the title goes in brackets, because a title with
            # quote marks right after a colon would look like the start of a quote.
            a = f'[{num(ctx, src)}] RTA s. {meta["number"]} ({meta["title"]}) says:\n"{quote(src)}"'
            out.append(example(q, ctx, a, "rta", sid))
    return out


def abstain_examples(all_sources: dict[str, Source], rta_meta: dict[str, dict], n: int) -> list[dict]:
    """Teach the model to say so when the sources don't cover the question."""
    by_kind = {k: [s for s in all_sources.values() if s.kind == k] for k in ("rta", "lease", "case")}
    core = [s for s in by_kind["rta"] if rta_meta[s.id.split(":")[1]]["part"] in CORE_PARTS]
    far = [s for s in by_kind["rta"] if rta_meta[s.id.split(":")[1]]["part"] in
           {"Care Homes", "Mobile Home Parks And Land Lease Communities",
            "Termination Of Occupancy — Non-Profit Housing Co-Operatives", "The Landlord And Tenant Board"}]
    out = []
    for src in rng.sample(core, min(n, len(core))):
        topic = re.sub(r"[“”\"]", "", rta_meta[src.id.split(":")[1]]["title"]).lower()
        q = rng.choice(RTA_QUESTIONS).format(t=topic)
        ctx = rng.sample(far, 3) + rng.sample(by_kind["lease"][:2], 1) + [rng.choice(by_kind["case"])]
        a = ("The sources provided don't answer this question, so I can't say what the law requires "
             "here. Check the Residential Tenancies Act, 2006 directly, or contact the Landlord and "
             "Tenant Board or a local legal clinic.")
        out.append(example(q, ctx, a, "abstain", src.id))
    return out


# --- Standard Form of Lease -----------------------------------------------------------------

LEASE_QUESTIONS = ["What does this part of my lease mean?", "What should I know about this section?",
                   "Explain this part of my lease."]


def lease_examples(all_sources: dict[str, Source]) -> list[dict]:
    rows = [json.loads(l) for l in (CORPUS / "standard_lease.jsonl").open()]
    out = []
    for s in (r for r in rows if r["kind"] == "section"):
        letters = sorted(set(s["parts"]) | set(EXTRA_PARTS.get(s["number"], [])))
        gold = [all_sources[f"lease:{L}"] for L in letters][:2]
        if not gold:
            continue
        excerpt = re.sub(r"\s+", " ", s["text"])[:450].rsplit(" ", 1)[0]
        questions = [clause_question(excerpt)] + [f'The lease has this section: "{excerpt}" {t}' for t in LEASE_QUESTIONS]
        for q in questions:
            ctx = with_gold(retrieve(excerpt), gold)
            parts = "\n".join(f'[{num(ctx, g)}] {g.label} explains:\n"{quote(g)}"' for g in gold)
            # No section number or title here: the model only sees the clause text, so it can't know them.
            a = f"Ontario's Standard Form of Lease explains this part of a lease.\n{parts}"
            out.append(example(q, ctx, a, "lease_section", f"lease-section:{s['number']}"))
    for s in (r for r in rows if r["kind"] == "part"):
        src = all_sources[f"lease:{s['letter']}"]
        topic = re.sub(r"\s*\(.*\)\s*$", "", s["title"]).lower()
        for q in (f"What does Ontario's standard lease say about {topic}?",
                  f"What are the rules in Ontario for {topic}?"):
            ctx = with_gold(retrieve(q), [src])
            a = f'[{num(ctx, src)}] {src.label} explains:\n"{quote(src)}"'
            out.append(example(q, ctx, a, "lease_part", src.id))
    return out


# --- LTB cases ------------------------------------------------------------------------------

CASE_QUESTIONS = [
    "Here is a real Landlord and Tenant Board application: {app} How did the Board decide?",
    "{app} Who won at the LTB, and why?",
    "What did the LTB order in this case? {app}",
]


def law_line(ctx: list[Source], case: dict) -> str:
    cited = [s for s in ctx if s.kind == "rta" and s.id.split(":")[1] in case["rta_sections"]]
    return ("\nThe law: " + "; ".join(f"[{num(ctx, s)}] {s.label}" for s in cited)) if cited else ""


def case_answer(ctx: list[Source], src: Source, case: dict, lead: str = "") -> str:
    result = OUTCOME[case["favoured"]].replace("decided", "decided the case")
    return (f"{lead}In [{num(ctx, src)}] {src.label}, the Board {result}.\n"
            f'What was ordered: "{shown_field(src, "Ordered")}"\n'
            f'Why: "{shown_field(src, "Key reasons")}"' + law_line(ctx, case))


def surface_variants(clause: str) -> list[str]:
    """Same meaning, different drafting styles real leases use."""
    variants = {clause,
                re.sub(r"\bshall\b", "must", clause),
                re.sub(r"\bshall\b", "will", clause),
                re.sub(r"\b(Tenant|Landlord)(s?)\b", lambda m: f"the {m.group(1).lower()}{m.group(2)}", clause),
                clause.upper() if len(clause) < 160 else clause}
    return sorted(variants)


def case_examples(all_sources: dict[str, Source]) -> list[dict]:
    out = []
    for case in map(json.loads, (CORPUS / "ltb_cases.jsonl").open()):
        src = all_sources[f"case:{case['doc_id']}"]
        if not shown_field(src, "Key reasons"):
            continue
        app = shown_field(src, "Application")
        q = rng.choice(CASE_QUESTIONS).format(app=app)
        ctx = with_gold(retrieve(app, exclude={src.id}), [src])
        # Split by LTB file, not document: amended and review orders share a file and near-identical text.
        key = case_key(case)
        out.append(example(q, ctx, case_answer(ctx, src, case), "case", key))

        for clause in case["lease_quotes"]:
            # Show the paragraph where the Board discusses this wording, plus the one after it.
            idx = next((i for i, p in enumerate(case["reasons"]) if clause[:40] in p), None)
            if idx is None:
                continue
            focused = {**case, "reasons": [tidy(p) for p in case["reasons"][idx:idx + 2]]}
            focused_text = case_summary(focused).replace(
                "Key reasons: ", "Key reasons: ", 1)
            qsrc = replace(src, text=re.sub(r"Key reasons: .*", "Key reasons: " + " ".join(focused["reasons"])[:700],
                                            focused_text, flags=re.S))
            variants = surface_variants(clause)
            questions = [clause_question(variants[0])] + [
                f'The lease says: "{v}" {t}' for v, t in zip(
                    variants[1:] + variants, ["Can my landlord enforce this?", "What does this mean for me?",
                                              "Is this clause legal in Ontario?"])]
            for q in questions:
                ctx = with_gold(retrieve(clause, exclude={src.id}), [qsrc])
                lead = "A Landlord and Tenant Board decision dealt with this lease wording. "
                out.append(example(q, ctx, case_answer(ctx, qsrc, case, lead), "lease_quote", key))
    return out


def main() -> None:
    all_sources = load_sources()
    rta_meta = {s["number"]: s for s in map(json.loads, (CORPUS / "rta_sections.jsonl").open())}

    rows = (rta_examples(all_sources, rta_meta) + lease_examples(all_sources)
            + case_examples(all_sources) + scenarios.build())
    # Rebalance: case summaries would otherwise be ~75% of the data and crowd out the statute
    # and clause skills the app depends on. Keep a sample of cases; repeat the scarce
    # real-clause examples. Valid/test sets are left untouched.
    cases = [r for r in rows if r["kind"] == "case" and r["split"] == "train"]
    rows = [r for r in rows if not (r["kind"] == "case" and r["split"] == "train")] + rng.sample(cases, min(600, len(cases)))
    # Scarce, hand-curated kinds are repeated so the model sees them often enough.
    rows += [r for r in rows if r["kind"] in ("lease_section", "lease_quote") and r["split"] == "train"]
    rows += [r for r in rows if r["kind"] in ("scenario", "negotiation", "followup", "clarify", "clarify_reply", "scenario_partial") and r["split"] == "train"]
    rows += abstain_examples(all_sources, rta_meta, n=len(rows) // 15)
    rng.shuffle(rows)

    OUT.mkdir(exist_ok=True)
    for split in ("train", "valid", "test"):
        part = [r for r in rows if r["split"] == split]
        with (OUT / f"{split}.jsonl").open("w") as f:
            for r in part:
                f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
        print(f"{split:5} {len(part):5}  {dict(Counter(r['kind'] for r in part))}")


if __name__ == "__main__":
    main()
