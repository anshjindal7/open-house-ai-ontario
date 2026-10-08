"""Can Jev tell a relevant passage from a real-but-off-topic one?

    python -m evals.run_relevance_eval      (needs TYPESAFE_API_KEY in .env)

Each case pairs a question with an official passage, labelled by hand: does the passage help
answer the question? Half are off-topic on purpose, including near misses that share words with
the question. Reports agreement at the current threshold and the threshold that would be best.
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv

from realtor.lease import jev
from realtor.lease.playbooks import pick

# (question, source, keywords to pick the passage, relevant?)
CASES = [
    ("My toilet is leaking, who has to fix it?", "rta:20", "landlord responsible maintaining good state of repair", True),
    ("My toilet is leaking, who has to fix it?", "rta:34", "tenant responsible undue damage wilful negligent", True),
    ("My toilet is leaking, who has to fix it?", "rta:41", "landlord may sell retain dispose property vacated", False),
    ("My toilet is leaking, who has to fix it?", "rta:108", "neither landlord nor tenancy agreement require post-dated cheques", False),
    ("Can my landlord evict me for having a baby?", "lease:M", "discriminates prohibited grounds Human Rights Code", True),
    ("Can my landlord evict me for having a baby?", "rta:48", "landlord good faith requires possession residential occupation", False),
    ("Can my landlord evict me for having a baby?", "rta:41", "landlord may sell retain dispose property vacated", False),
    ("Can the rent go up in the middle of a lease?", "rta:116", "shall not increase rent without 90 days written notice", True),
    ("Can the rent go up in the middle of a lease?", "rta:119", "only if at least 12 months have elapsed", True),
    ("Can the rent go up in the middle of a lease?", "rta:105", "only security deposit landlord may collect rent deposit", False),
    ("The landlord wants a $1,000 damage deposit. Can they?", "rta:105", "only security deposit landlord may collect rent deposit", True),
    ("The landlord wants a $1,000 damage deposit. Can they?", "lease:H", "landlord can only collect deposit last month's rent key deposit", True),
    ("The landlord wants a $1,000 damage deposit. Can they?", "rta:89", "landlord may apply order tenant pay reasonable costs repair", False),
    ("My landlord came in without telling me.", "rta:27", "enter written notice at least 24 hours", True),
    ("My landlord came in without telling me.", "rta:26", "enter without written notice emergency consents", True),
    ("My landlord came in without telling me.", "rta:24", "landlord shall not alter locking system replacement keys", False),
    ("I came home and the locks were changed.", "rta:24", "landlord shall not alter locking system replacement keys", True),
    ("I came home and the locks were changed.", "rta:35", "tenant shall not alter locking system consent landlord", False),
    ("Can I keep a cat even though the lease says no pets?", "rta:14", "provision prohibiting presence of animals void", True),
    ("Can I keep a cat even though the lease says no pets?", "rta:15", "remaining rent becomes due upon default void", False),
    ("My roommate won't pay their share.", "guide:Occupants and Roommates", "roommate shares rental unit not a tenant", True),
    ("My roommate won't pay their share.", "rta:59", "tenant fails to pay rent landlord may give notice termination", False),
    ("Do I have to move out when my lease ends?", "rta:38", "fixed term ends deemed renewed monthly tenancy", True),
    ("Do I have to move out when my lease ends?", "rta:53", "right of first refusal repairs renovations completed", False),
]


def main() -> None:
    load_dotenv()
    assert jev.enabled(), "set TYPESAFE_API_KEY in .env"
    scored = []
    for q, ref, about, label in CASES:
        _, passage = pick(ref, about)
        p = asyncio.run(jev.relevance(q, [passage]))[0]
        scored.append((p, label))
        ok = (p >= jev.RELEVANT_MIN) == label
        print(f"{'ok  ' if ok else 'MISS'} {p:.2f} {'relevant' if label else 'off-topic':9} {ref:32} | {q}")
    def acc(t):
        return sum((p >= t) == lab for p, lab in scored) / len(scored)
    best = max((t / 20 for t in range(2, 19)), key=acc)
    print(f"\nagreement at {jev.RELEVANT_MIN:.2f}: {acc(jev.RELEVANT_MIN):.0%} · best threshold {best:.2f}: {acc(best):.0%}")


if __name__ == "__main__":
    main()
