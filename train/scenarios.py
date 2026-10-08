"""Turn train/scenarios.json into conversational, two-sided training answers.

    python -m train.scenarios --show     # print every branch's quote for review

Each branch names an official source and a few keywords. We pick the sentence in that source's
real text that best matches the keywords, and make sure the model's view of the source contains
it, so every quote in an answer is verbatim and visible. Four kinds of examples:

  scenario     a situation ("my sink is broken"), answered branch by branch, plus decisions each
               side won. Sometimes the asker says which side they're on; the answer stays neutral.
  negotiation  the asker gives their side and asks for help: suggestions for them, labelled as
               suggestions, plus the strongest point for the other side so nothing is hidden.
  followup     a second turn ("what if I broke it by accident?") answered from the right branch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

from realtor.lease.local_model import search_query
from realtor.lease.playbooks import pick
from realtor.lease.sources import Source, load_sources, retrieve, tokenize

from .common import case_line, example, held_out_cases, num, rng, with_gold

SCENARIOS = Path(__file__).parent / "scenarios.json"

CLOSERS = [
    "If you share the exact wording in your lease or more about what happened, I can look at it more closely.",
    "Every situation is a little different. For advice on yours, the LTB (1-888-332-3234) or a community legal clinic can help.",
    "Want me to go through what either side can do next?",
]
NEGOTIATION_OPEN = ("Here are some suggestions for your side as the {role}. These are suggestions only, not legal "
                    "advice, and the other side may see it differently.")
NEGOTIATION_TIPS = ("**For the conversation itself.** Keep it in writing where you can, so there's a record of what "
                    "was agreed. If you can't agree, either of you can apply to the Landlord and Tenant Board.")
OTHER = {"tenant": "landlord", "landlord": "tenant"}


# Quote picking lives in realtor/lease/playbooks.py, so search and training share it.


def short_name(src: Source) -> str:
    if src.kind == "rta":
        return "RTA s. " + src.id.split(":")[1]
    if src.kind == "lease":
        return f"The Standard Form of Lease (Part {src.id.split(':')[1]})"
    doc = src.label.split(", \"")[0]
    m = re.match(r"LTB Interpretation Guideline (\d+)", doc)
    return f"LTB Interpretation Guideline {m.group(1)}" if m else "The LTB's \"" + doc.removeprefix("LTB brochure: ") + "\" brochure"


def branch_line(ctx: list[Source], src: Source, when: str, quote_text: str, bullet: bool = False) -> str:
    return f'{"- " if bullet else ""}**{when}.** {short_name(src)} [{num(ctx, src)}] says: "{quote_text}"'


# --- building examples ----------------------------------------------------------------------

def _resolve(branches: list[dict]) -> list[tuple[dict, Source, str]]:
    return [(b, *pick(b["src"], b["about"], b.get("extend", 0))) for b in branches]


def _context(query: str, gold: list[Source], split: str = "train") -> list[Source]:
    """Retrieved sources with the gold ones guaranteed in. The same source can't appear twice.

    Training examples never see held-out decisions, so test scores aren't inflated.
    """
    unique = list({g.id: g for g in gold}.values())
    exclude = {g.id for g in unique} | (held_out_cases() if split == "train" else set())
    return with_gold(retrieve(query, "question", exclude=exclude, use_playbooks=False), unique)


def _split(sid: str) -> str:
    ids = sorted(json.loads(SCENARIOS.read_text())["scenarios"], key=lambda s: hashlib.sha256(s["id"].encode()).hexdigest())
    order = [s["id"] for s in ids]
    return "test" if sid in order[:2] else "valid" if sid in order[2:3] else "train"


def _merge_gold(resolved: list[tuple[dict, Source, str]]) -> list[Source]:
    """Two branches can quote the same source; the model sees one view that contains both quotes."""
    merged: dict[str, Source] = {}
    for _, src, q in resolved:
        merged.setdefault(src.id, src)
    out = []
    for sid, src in merged.items():
        quotes = [q for _, s, q in resolved if s.id == sid]
        view = replace(src, focus=quotes[0])
        if all(q in view.shown for q in quotes):
            out.append(view)
        else:  # quotes too far apart to fit one view: keep the first, drop branches quoting the rest
            out.append(view)
    return out


def scenario_examples(sc: dict) -> list[dict]:
    resolved = _resolve(sc["branches"])
    split = _split(sc["id"])
    rows = []
    for i, q in enumerate(sc["questions"]):
        for role in (None, rng.choice(["tenant", "landlord"])):
            gold = _merge_gold(resolved)
            ctx = _context(q, gold, split)
            visible = {s.id: s for s in ctx}
            lines = [sc["opener"], ""]
            for b, src, qt in resolved:
                if qt in visible[src.id].shown:
                    lines.append(branch_line(ctx, visible[src.id], b["when"], qt))
            cases = [s for s in ctx if s.kind == "case"]
            if cases:
                intro = ("Two real decisions on similar issues, one each side won:"
                         if len({c.side for c in cases} & {"tenant", "landlord"}) == 2
                         else "Real decisions on similar issues:" if len(cases) > 1
                         else "A real decision on a similar issue:")
                lines += ["", f"**At the Landlord and Tenant Board.** {intro}"]
                lines += [case_line(ctx, c) for c in cases]
            lines += ["", CLOSERS[(i + (role is not None)) % len(CLOSERS)]]
            row = example(q, ctx, "\n".join(lines), "scenario", f"scenario:{sc['id']}", role=role)
            row["split"] = split
            rows.append(row)

        # The same question with ordinary search only: some branch sources won't be there, and the
        # answer must state only the branches whose source it can see. Without this the model learns
        # the shape of the answer and fills a missing branch with whatever source is in that slot.
        natural = retrieve(q, "question", exclude=held_out_cases() if split == "train" else set(), use_playbooks=False)
        visible = {x.id: x for x in natural}
        present = [(b, visible[src.id], qt) for b, src, qt in resolved
                   if src.id in visible and qt in visible[src.id].shown]
        if present and len(present) < len(resolved):
            lines = [sc["opener"], ""] + [branch_line(natural, s, b["when"], qt) for b, s, qt in present]
            missing = [b["when"].lower().rstrip(".") for b, src, qt in resolved if src.id not in visible or qt not in visible[src.id].shown]
            lines += ["", "The sources I found don't cover every case here (for example: " + missing[0] + "). "
                          "Tell me more about the situation and I'll look for what applies."]
            row = example(q, natural, "\n".join(lines), "scenario_partial", f"scenario:{sc['id']}")
            row["split"] = split
            rows.append(row)

        # And with each branch's source deliberately removed in turn.
        for drop in {src.id for _, src, _ in resolved}:
            keep = [r for r in resolved if r[1].id != drop]
            if not keep:
                continue
            ctx = with_gold(retrieve(q, "question", exclude={drop} | (held_out_cases() if split == "train" else set()),
                                     use_playbooks=False), _merge_gold(keep))
            visible = {x.id: x for x in ctx}
            lines = [sc["opener"], ""] + [branch_line(ctx, visible[s.id], b["when"], qt) for b, s, qt in keep
                                          if qt in visible[s.id].shown]
            row = example(q, ctx, "\n".join(lines), "scenario_partial", f"scenario:{sc['id']}")
            row["split"] = split
            rows.append(row)
            break  # one dropped-source variant per question keeps the mix balanced
    return rows


def negotiation_examples(sc: dict, templates: list[str]) -> list[dict]:
    resolved = _resolve(sc["branches"])
    split = _split(sc["id"])
    rows = []
    for role in ("tenant", "landlord"):
        mine = [r for r in resolved if r[0]["side"] in (role, "both")]
        theirs = [r for r in resolved if r[0]["side"] == OTHER[role]]
        if not mine:
            continue
        for t in rng.sample(templates, 2):
            q = t.format(role=role, topic=sc["topic"])
            ctx = _context(q, _merge_gold(mine + theirs[:1]), split)
            visible = {s.id: s for s in ctx}
            lines = [NEGOTIATION_OPEN.format(role=role), "", "**Points in your favour**"]
            lines += [branch_line(ctx, visible[s.id], b["when"], qt, bullet=True)
                      for b, s, qt in mine if qt in visible[s.id].shown]
            won = [s for s in ctx if s.kind == "case" and s.side == role]
            if won:
                lines.append("- " + case_line(ctx, won[0]))
            if theirs:
                b, s, qt = theirs[0]
                if qt in visible[s.id].shown:
                    lines += ["", "**Be ready for this point from the other side**",
                              branch_line(ctx, visible[s.id], b["when"], qt, bullet=True)]
            lines += ["", NEGOTIATION_TIPS]
            row = example(q, ctx, "\n".join(lines), "negotiation", f"scenario:{sc['id']}", role=role)
            row["split"] = split
            rows.append(row)
    return rows


def followup_examples(sc: dict, first_rows: list[dict]) -> list[dict]:
    rows = []
    neutral = [r for r in first_rows if "About me:" not in r["messages"][-2]["content"]]
    for fu in sc.get("followups", []):
        if "branch" in fu:
            b = sc["branches"][fu["branch"]]
        else:
            b = fu
        src, qt = pick(b["src"], b["about"], b.get("extend", 0))
        for prev in neutral[:2]:
            q1 = prev["messages"][-2]["content"].split("Question: ")[-1]
            a1 = prev["messages"][-1]["content"]
            history = [(q1, a1)]
            ctx = _context(search_query(fu["q"], history), [src], _split(sc["id"]))
            visible = {s.id: s for s in ctx}
            if qt not in visible[src.id].shown:
                continue
            answer = branch_line(ctx, visible[src.id], b["when"], qt)
            row = example(fu["q"], ctx, answer, "followup", f"scenario:{sc['id']}", history=history)
            row["split"] = _split(sc["id"])
            rows.append(row)
    return rows


ACKS = ["Thanks, that helps.", "Got it.", "Okay, that makes it clearer."]


def clarify_examples(sc: dict) -> list[dict]:
    """Ask the one deciding question first, then answer the branch the reply points to."""
    cl = sc.get("clarify")
    if not cl:
        return []
    resolved = _resolve(sc["branches"])
    split = _split(sc["id"])
    either = resolved[cl["either_way"]] if "either_way" in cl else None
    rows = []
    for q in cl["questions"]:
        gold = _merge_gold([resolved[o["branch"]] for o in cl["options"]] + ([either] if either else []))
        ctx = _context(q, gold, split)
        visible = {x.id: x for x in ctx}
        lines = [cl["ask"]] + [f"- {o['label']}" for o in cl["options"]]
        if either and either[2] in visible[either[1].id].shown:
            lines += ["", branch_line(ctx, visible[either[1].id], "Either way", either[2])]
        lines += ["", "Tell me which fits, and I'll go through what applies."]
        ask_answer = "\n".join(lines)
        row = example(q, ctx, ask_answer, "clarify", f"scenario:{sc['id']}")
        row["split"] = split
        rows.append(row)

        for k, opt in enumerate(cl["options"]):
            b, src, qt = resolved[opt["branch"]]
            for reply in rng.sample(opt["replies"], min(2, len(opt["replies"]))):
                history = [(q, ask_answer)]
                ctx2 = _context(search_query(reply, history), [replace(src, focus=qt)], split)
                vis2 = {x.id: x for x in ctx2}
                if qt not in vis2[src.id].shown:
                    continue
                lines2 = [f"{ACKS[k % len(ACKS)]} That sounds like this one: {opt['label'].lower()}.", "",
                          branch_line(ctx2, vis2[src.id], b["when"], qt)]
                cases = [c for c in ctx2 if c.kind == "case"]
                if cases:
                    lines2 += ["", "**A real decision on a similar issue:**", case_line(ctx2, cases[0])]
                row = example(reply, ctx2, "\n".join(lines2), "clarify_reply", f"scenario:{sc['id']}", history=history)
                row["split"] = split
                rows.append(row)
    return rows


def build() -> list[dict]:
    data = json.loads(SCENARIOS.read_text())
    rows = []
    for sc in data["scenarios"]:
        first = scenario_examples(sc)
        rows += (first + negotiation_examples(sc, data["negotiation"]["questions"]) + followup_examples(sc, first)
                 + clarify_examples(sc))
    return rows


def show() -> None:
    for sc in json.loads(SCENARIOS.read_text())["scenarios"]:
        print(f"\n=== {sc['id']}  ({_split(sc['id'])})")
        for b, src, qt in _resolve(sc["branches"]):
            print(f"  [{b['side']:8}] {b['when']}\n      {src.label[:90]}\n      \"{qt}\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.show:
        show()
    else:
        rows = build()
        print(len(rows), "examples")
