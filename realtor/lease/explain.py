"""Build the renter's report from Jev's findings and our own fine-tuned model.

The lease wording in the report is quoted by code, never generated. The model answers from
retrieved Ontario sources (statute, standard lease, LTB decisions), which are listed under
each answer so the renter can check them.
"""

from __future__ import annotations

from .checks import CHECKS
from .scanner import ClauseFinding, fit_label

SEVERITY = {f.key: f.severity for f in CHECKS}
RANK = {"high": 0, "medium": 1, "low": 2}


def format_findings(findings: list[ClauseFinding]) -> str:
    lines = []
    for f in findings:
        if not f.needs_attention:
            continue
        lines.append(f"### Clause {f.clause.index}: {f.clause.heading}")
        lines.append(f"Topic: {f.topic} | Fit with the RTA: {fit_label(f.rta_fit)}")
        for key, p in sorted(f.flags.items(), key=lambda kv: -kv[1]):
            lines.append(f"- FLAGGED {key} ({SEVERITY[key]} severity, p={p:.2f})")
        for key, p in f.unsure.items():
            lines.append(f"- UNSURE {key} (p={p:.2f}): check this one yourself")
        lines.append("")
    return "\n".join(lines) or "The scanner found nothing notable."


def _worst(f: ClauseFinding) -> int:
    return min((RANK[SEVERITY[k]] for k in f.flags), default=3)


def explain(findings: list[ClauseFinding]) -> str:
    from .local_model import explain_clause
    from .pointers import expand

    flagged = sorted((f for f in findings if f.flags), key=lambda f: (_worst(f), -f.rta_fit))
    unsure = [f for f in findings if f.unsure and not f.flags]

    out = ["# Lease review", ""]
    if not flagged:
        out.append("No terms flagged as conflicting with Ontario law. Both parties should still read the whole lease.")
    else:
        high = sum(_worst(f) == 0 for f in flagged)
        out.append(f"**{len(flagged)} clauses may conflict with Ontario law, {high} of them likely void.**")

    for f in flagged:
        flags = sorted(f.flags, key=lambda k: RANK[SEVERITY[k]])
        answer, sources = explain_clause(f.clause.text)
        answer = expand(answer, sources)
        out += ["", f"## {f.clause.heading} ({SEVERITY[flags[0]]})", "", f"> {f.clause.text}", "", answer,
                "", "Sources: " + "; ".join(f"[{i + 1}] {s.label}" for i, s in enumerate(sources))]

    if unsure:
        out += ["", "## Worth a second look", ""]
        out += [f"- {f.clause.heading}: possibly {', '.join(f.unsure)}" for f in unsure]

    out += ["", "_Answers are based on Ontario's Residential Tenancies Act, 2006, the Ontario Standard Form "
                "of Lease, and 2026 Landlord and Tenant Board decisions. This is not legal advice; for your "
                "situation, contact the LTB (1-888-332-3234) or a community legal clinic._"]
    return "\n".join(out)
