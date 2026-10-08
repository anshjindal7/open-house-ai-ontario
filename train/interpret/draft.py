"""Draft "short answers" (plain-language interpretation) for a person to review.

    python -m train.interpret.draft --limit 50 [--model mlx-community/Qwen3-8B-4bit]

Each item is a real situation with its real outcome:
  case      a 2026 LTB decision: the facts from the application, and how the Board ruled
  scenario  a situation from train/scenarios.json, one item per branch

An open model (never Claude) drafts 2-4 sentences applying the law in the sources to the facts.
Every draft is run through the fact-checker, and for cases we compare its lean with the real
ruling. Nothing here is training data until a person approves it on the review page
(/review in the app). Drafts go to train/interpret/drafts.jsonl.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from realtor.lease.local_model import BASE_MODEL, CHAT_ARGS
from realtor.lease.sources import CORPUS, Source, load_sources, render, retrieve, tidy
from realtor.lease.verify import verify

from ..common import held_out_cases

HERE = Path(__file__).parent
DRAFTS = HERE / "drafts.jsonl"

DRAFT_SYSTEM = (
    "You write the short answer at the top of a reply about an Ontario tenancy. In 2 to 4 plain "
    "sentences, apply the law in the numbered sources to the facts given. End every sentence with the "
    "number of the source it relies on, like [2]; a sentence without one will be rejected. Use only what the sources say; don't add numbers, deadlines or rules "
    "that aren't in them. Be neutral: if the facts favour one side, say so plainly, whichever side "
    "it is. If the outcome depends on a fact that isn't given, say which fact. Don't quote at length; "
    "the quotes come after your short answer."
)


FINDING = re.compile(r"\b(I find|I am satisfied|I am not satisfied|dismiss|granted|is ordered|order(ed)? that|"
                     r"for the reasons|on a balance of probabilities)\b", re.I)


def _facts(case: dict) -> str:
    """What was claimed, plus the evidence paragraphs that come before any finding (no spoilers)."""
    claim = re.sub(r"\[name\]\s*\(the ['‘]?(Tenants?|Landlords?)['’]?\)", lambda m: f"the {m.group(1).lower()}", tidy(case["issues"]))
    claim = claim[0].upper() + claim[1:]
    evidence = []
    for p in case["reasons"][:6]:
        if FINDING.search(p):
            break
        evidence.append(tidy(p))
    text = claim + (" " + " ".join(evidence) if evidence else "")
    return text[:700].rsplit(" ", 1)[0]


def lean(text: str) -> str | None:
    """Which side a short answer comes down on, if it says."""
    t = text.lower()
    tenant = re.search(r"favou?rs? the tenant|tenant (is|would be|is likely) (entitled|right)|landlord (must|has to|is responsible)", t)
    landlord = re.search(r"favou?rs? the landlord|landlord (is|would be|is likely) (entitled|right)|tenant (must|has to|is responsible)", t)
    if tenant and not landlord:
        return "tenant"
    if landlord and not tenant:
        return "landlord"
    return None


def items(limit: int) -> list[dict]:
    rng = random.Random(5)
    held_out = held_out_cases()
    cases = [c for c in map(json.loads, (CORPUS / "ltb_cases.jsonl").open())
             if f"case:{c['doc_id']}" not in held_out and c["favoured"] in ("tenant", "landlord")]
    out = []
    for c in rng.sample(cases, min(limit, len(cases))):
        facts = _facts(c)
        out.append({"id": f"case:{c['doc_id']}", "kind": "case", "question": f"Here's what happened: {facts} What does the law say?",
                    "query": facts, "exclude": [f"case:{c['doc_id']}"], "truth": c["favoured"],
                    "truth_note": f"The Board {('decided in the ' + c['favoured'] + chr(39) + 's favour')}. Ordered: {tidy(c['ordered'][0])[:200]}"})
    return out


def draft(limit: int, model_name: str) -> None:
    from mlx_lm import generate, load

    model, tok = load(model_name)
    done = {json.loads(l)["id"] for l in DRAFTS.open()} if DRAFTS.exists() else set()
    with DRAFTS.open("a") as f:
        for it in items(limit):
            if it["id"] in done:
                continue
            sources = retrieve(it["query"], "question", exclude=set(it["exclude"]))
            messages = [{"role": "system", "content": DRAFT_SYSTEM},
                        {"role": "user", "content": f"Sources:\n\n{render(sources)}\n\nFacts and question: {it['question']}"}]
            prompt = tok.apply_chat_template(messages, **CHAT_ARGS)
            text = generate(model, tok, prompt=prompt, max_tokens=220).strip()
            if not re.search(r"\[\d\]", text):  # one retry with a reminder, then it's flagged for review
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": "Rewrite it so every sentence ends with its source number, like [2]."})
                text = generate(model, tok, prompt=tok.apply_chat_template(messages, **CHAT_ARGS), max_tokens=220).strip()
            report = verify(text, sources, question=it["question"])
            row = {**it, "model": model_name, "draft": text, "check": report.to_dict(),
                   "draft_lean": lean(text), "sources": [{"id": s.id, "label": s.label, "text": s.shown} for s in sources],
                   "status": "pending"}
            row["flags"] = [f for f in (
                not report.ok and "fact-check issues",
                it.get("truth") and row["draft_lean"] and row["draft_lean"] != it["truth"] and "leans opposite to the real ruling",
                not re.search(r"\[\d\]", text) and "no citations",
            ) if f]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{it['id']}: {'flags: ' + ', '.join(row['flags']) if row['flags'] else 'clean'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--model", default=BASE_MODEL)
    args = parser.parse_args()
    draft(args.limit, args.model)
