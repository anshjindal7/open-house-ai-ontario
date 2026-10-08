"""Jev integration, tested against a stand-in client (no key or network needed)."""

import asyncio
from types import SimpleNamespace

import pytest

from realtor.lease import jev
from realtor.lease.sources import Source
from realtor.lease.verify import Report, jev_check

S20 = Source("rta:20", "rta", "Residential Tenancies Act, 2006, s. 20: Landlord's responsibility to repair",
             "(1)  A landlord is responsible for providing and maintaining a residential complex in a good state of repair.")
S41 = Source("rta:41", "rta", "Residential Tenancies Act, 2006, s. 41: Disposal of abandoned property",
             "(1)  A landlord may sell, retain or otherwise dispose of property in a rental unit if it has been vacated.")


CALLS = []


class FakeClient:
    """Answers Nouls from a table keyed by a word in the passage; Choices from a fixed pick."""

    def __init__(self, noul_by_word=None, choice=("none", 0.9)):
        self.noul_by_word, self.choice = noul_by_word or {}, choice

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def system_one(self, state, questions):
        CALLS.append(sorted(questions))
        nouls, choices = {}, {}
        for key, q in questions.items():
            if type(q).__name__ == "Choice":
                choices[key] = SimpleNamespace(choice=self.choice[0], confidence=self.choice[1])
            else:
                idx = key[1:]
                text = state.get(f"passage_{idx}") or state.get(f"source_{idx}", "")
                p = next((v for w, v in self.noul_by_word.items() if w in text), 0.9)
                nouls[key] = SimpleNamespace(noul=p)
        return SimpleNamespace(nouls=nouls, choices=choices)


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-not-a-real-key")
    def install(**kw):
        import typesafe_sdk
        monkeypatch.setattr(typesafe_sdk, "AsyncTypeSafeClient", lambda: FakeClient(**kw))
    return install


def test_off_topic_passage_is_flagged(fake):
    fake(noul_by_word={"dispose": 0.1})  # Jev says the abandoned-property passage doesn't help
    answer = "It depends.\n**Repairs.** {1.1}\n**Property.** {2.1}"
    report = asyncio.run(jev_check(answer, [S20, S41], Report(), question="My sink is broken"))
    assert [i.kind for i in report.issues] == ["relevance"]
    assert "{2.1}" in report.issues[0].detail and report.jev_used


def test_relevant_answer_passes(fake):
    fake()
    report = asyncio.run(jev_check("**Repairs.** {1.1}", [S20, S41], Report(), question="My sink is broken"))
    assert report.ok and report.jev_used


def test_router_respects_none_and_low_confidence(fake):
    fake(choice=("broken-fixture", 0.45))
    assert asyncio.run(jev.route("My sink is broken"))[0] is None  # not confident enough
    fake(choice=("broken-fixture", 0.92))
    assert asyncio.run(jev.route("My sink is broken"))[0]["id"] == "broken-fixture"
    fake(choice=("none", 0.95))
    assert asyncio.run(jev.route("Can I smoke on my balcony?"))[0] is None


def test_without_a_key_nothing_is_sent(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    report = asyncio.run(jev_check("**Repairs.** {1.1}", [S20], Report(), question="q"))
    assert not report.jev_used and report.ok


def test_supported_and_relevance_share_one_request(fake):
    fake()
    CALLS.clear()
    answer = "In [1] the landlord must keep the unit in good repair for the tenant. {1.1}\n**Property.** {2.1}"
    report = asyncio.run(jev_check(answer, [S20, S41], Report(), question="My sink is broken"))
    assert len(CALLS) == 1, CALLS
    assert {"s1", "p1", "p2"} <= set(CALLS[0]) and report.jev_used
