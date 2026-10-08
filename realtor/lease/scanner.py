"""Fan every clause out to Jev in parallel and collect typed findings."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from typesafe_sdk import AsyncTypeSafeClient

from .checks import CHECKS, RTA_FIT_LEVELS, build_questions
from .clauses import Clause

MAX_CONCURRENT = 8
UNSURE_MARGIN = 0.15  # probabilities this close below the threshold are "worth a second look"


@dataclass
class ClauseFinding:
    clause: Clause
    topic: str
    rta_fit: int  # 0 = consistent with the RTA … 3 = likely void (see checks.RTA_FIT_LEVELS)
    flags: dict[str, float] = field(default_factory=dict)  # flagged key -> probability
    unsure: dict[str, float] = field(default_factory=dict)  # near-threshold key -> probability

    @property
    def needs_attention(self) -> bool:
        return bool(self.flags or self.unsure) or self.rta_fit >= 2


async def scan_clause(client: AsyncTypeSafeClient, clause: Clause) -> ClauseFinding:
    response = await client.system_one(
        state={"lease_clause": clause.text, "section_heading": clause.heading},
        questions=build_questions(),
    )
    fit = response.scores["rta_fit"]
    finding = ClauseFinding(
        clause=clause,
        topic=response.choices["topic"].choice,
        # score is a probability-weighted level (may fall between levels); legend says where levels start.
        rta_fit=round(fit.score - min(fit.legend)),
    )
    for flag in CHECKS:
        p = response.nouls[flag.key].noul
        if p >= flag.threshold:
            finding.flags[flag.key] = p
        elif p >= flag.threshold - UNSURE_MARGIN:
            finding.unsure[flag.key] = p
    return finding


async def scan_lease(clauses: list[Clause]) -> list[ClauseFinding]:
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    async with AsyncTypeSafeClient() as client:

        async def run(clause: Clause) -> ClauseFinding:
            async with sem:
                return await scan_clause(client, clause)

        return await asyncio.gather(*(run(c) for c in clauses))


def fit_label(level: int) -> str:
    return RTA_FIT_LEVELS[max(0, min(level, len(RTA_FIT_LEVELS) - 1))]
