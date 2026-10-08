import asyncio
from types import SimpleNamespace

from realtor.lease.checks import CHECKS
from realtor.lease.clauses import Clause
from realtor.lease.explain import format_findings
from realtor.lease.scanner import scan_clause


class FakeJev:
    def __init__(self, probs: dict[str, float], risk: float):
        self.probs, self.risk = probs, risk

    async def system_one(self, state, questions):
        return SimpleNamespace(
            choices={"topic": SimpleNamespace(choice="access_privacy")},
            scores={"rta_fit": SimpleNamespace(score=self.risk, legend={1: "a", 2: "b", 3: "c", 4: "d"})},
            nouls={f.key: SimpleNamespace(noul=self.probs.get(f.key, 0.02)) for f in CHECKS},
        )


def test_flags_unsure_and_risk_level():
    clause = Clause(0, "7. LANDLORD ACCESS", "Landlord may enter at any time.")
    jev = FakeJev({"entry_without_notice": 0.93, "waives_the_act": 0.41}, risk=3.8)
    finding = asyncio.run(scan_clause(jev, clause))

    assert finding.flags == {"entry_without_notice": 0.93}
    assert finding.unsure == {"waives_the_act": 0.41}
    assert finding.rta_fit == 3  # 1-based legend -> 0-based "Likely void"
    assert "FLAGGED entry_without_notice (high severity" in format_findings([finding])


def test_clean_clause_is_skipped_in_report():
    clause = Clause(1, "14. QUIET ENJOYMENT", "Tenant may peacefully enjoy the premises.")
    finding = asyncio.run(scan_clause(FakeJev({}, risk=1.1), clause))
    assert not finding.needs_attention
    assert format_findings([finding]) == "The scanner found nothing notable."
