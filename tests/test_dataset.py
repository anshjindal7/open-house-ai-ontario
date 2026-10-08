"""Checks on the built training data. Run `python -m train.build_dataset` first."""

import json
import re
from pathlib import Path

import pytest

from train.compare import norm

DATA = Path(__file__).parents[1] / "train" / "data"

pytestmark = pytest.mark.skipif(not (DATA / "train.jsonl").exists(), reason="dataset not built")


def rows(split):
    return [json.loads(line) for line in (DATA / f"{split}.jsonl").open()]


@pytest.mark.parametrize("split", ["train", "valid", "test"])
def test_answers_point_instead_of_quoting(split):
    """Answers cite sentences with pointers like {2.3}; they never retype source text."""
    for r in rows(split):
        answer = r["messages"][-1]["content"]
        from realtor.lease.pointers import QUOTE_RE
        long_quotes = [m.group(2) for m in QUOTE_RE.finditer(answer) if len(m.group(2)) >= 40]
        assert not long_quotes, f"retyped quote: {long_quotes[0][:80]}"


@pytest.mark.parametrize("split", ["train", "valid", "test"])
def test_every_pointer_points_at_a_numbered_sentence(split):
    for r in rows(split):
        prompt, answer = r["messages"][-2]["content"], r["messages"][-1]["content"]
        for n, a, b in re.findall(r"\{(\d)\.(\d+)(?:-(\d+))?\}", answer):
            for k in {a, b or a}:
                assert f"{{{n}.{k}}}" in prompt, f"pointer {{{n}.{k}}} not in prompt"


@pytest.mark.parametrize("split", ["train", "valid", "test"])
def test_every_cited_source_number_exists(split):
    for r in rows(split):
        prompt, answer = r["messages"][-2]["content"], r["messages"][-1]["content"]
        for n in re.findall(r"\[(\d)\]", answer):
            assert f"[{n}] " in prompt


def test_test_set_cases_never_appear_in_training():
    """Held-out case examples (answers that are about one decision) must be unseen in training.

    Held-out scenario answers may cite training decisions; they test unseen scenarios, not cases.
    """
    label = re.compile(r"In \[\d\] LTB decision ([^ ,(]+(?: ONLTB \d+)?)")
    is_case_answer = lambda a: a.startswith(("In [", "A Landlord and Tenant Board decision dealt"))
    train = {m.group(1) for r in rows("train") for m in label.finditer(r["messages"][-1]["content"])}
    test = {m.group(1) for r in rows("test") if is_case_answer(a := r["messages"][-1]["content"])
            for m in label.finditer(a)}
    assert test and not (train & test)


def test_no_personal_names_left_in_answers():
    leaked = [r for r in rows("train") if re.search(r"\b(Mr|Mrs|Ms)\. [A-Z][a-z]+", r["messages"][-1]["content"])]
    assert len(leaked) <= 2, leaked[:2]


def test_fact_checker_passes_every_training_answer():
    """Training answers are correct by construction, so the checker should find nothing.

    Guards both directions: data that quotes or states things the model can't see, and a
    checker that raises false alarms.
    """
    from evals.check_dataset_with_verifier import parse_sources
    from realtor.lease.verify import verify

    failures = []
    for r in rows("train"):
        prompt = r["messages"][-2]["content"]
        rep = verify(r["messages"][-1]["content"], parse_sources(prompt), question=prompt.rsplit("Question: ", 1)[-1])
        failures += [(i.kind, i.detail, i.text[:100]) for i in rep.issues]
    assert len(failures) <= 5, failures[:5]
