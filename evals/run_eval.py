"""Score Jev's red-flag checks against hand-labeled clauses.

    python -m evals.run_eval

Labels follow Ontario law (e.g. pet and damage deposits are not allowed: Standard Form of Lease, Part H).

For each flag it prints precision and recall at the current threshold, plus the threshold
that would have scored best on this set. Edit the question wording or thresholds in
realtor/lease/checks.py, re-run, and keep the changes that improve the numbers.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import AsyncTypeSafeClient, Noul

from realtor.lease.checks import RED_FLAGS

DATA = Path(__file__).parent / "lease_clauses.jsonl"


async def collect_probs(examples: list[dict]) -> list[dict[str, float]]:
    questions = {f.key: Noul(instructions=f.question) for f in RED_FLAGS}
    sem = asyncio.Semaphore(8)
    async with AsyncTypeSafeClient() as client:

        async def run(ex: dict) -> dict[str, float]:
            async with sem:
                r = await client.system_one(state={"lease_clause": ex["text"]}, questions=questions)
                return {k: r.nouls[k].noul for k in questions}

        return await asyncio.gather(*(run(ex) for ex in examples))


def f1_at(probs: list[float], truth: list[bool], t: float) -> tuple[float, float, float]:
    tp = sum(p >= t and y for p, y in zip(probs, truth))
    fp = sum(p >= t and not y for p, y in zip(probs, truth))
    fn = sum(p < t and y for p, y in zip(probs, truth))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main() -> None:
    load_dotenv()
    examples = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    all_probs = asyncio.run(collect_probs(examples))

    print(f"{'flag':32} {'prec':>5} {'rec':>5}  {'best t':>6} {'best f1':>7}")
    misses = []
    for flag in RED_FLAGS:
        probs = [p[flag.key] for p in all_probs]
        truth = [flag.key in ex["flags"] for ex in examples]
        precision, recall, _ = f1_at(probs, truth, flag.threshold)
        best_t = max((t / 20 for t in range(2, 19)), key=lambda t: f1_at(probs, truth, t)[2])
        print(f"{flag.key:32} {precision:5.2f} {recall:5.2f}  {best_t:6.2f} {f1_at(probs, truth, best_t)[2]:7.2f}")
        for ex, p, y in zip(examples, probs, truth):
            if (p >= flag.threshold) != y:
                kind = "missed" if y else "false alarm"
                misses.append(f"  [{kind}] {flag.key} p={p:.2f}: {ex['text'][:90]}")

    if misses:
        print("\nMistakes at current thresholds:")
        print("\n".join(misses))


if __name__ == "__main__":
    main()
