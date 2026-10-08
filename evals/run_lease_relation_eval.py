"""How Jev compares a lease line with the law quoted above it: conflicts, matches, or neither.

    python -m evals.run_lease_relation_eval        (needs TYPESAFE_API_KEY)

Made-up lease lines, labelled before the first run. Add cases as you find mistakes; the cut-offs
are jev.CONFLICT_MIN and jev.MATCH_MIN.
"""

import asyncio

from dotenv import load_dotenv

from realtor.lease import jev
from realtor.lease.sources import load_sources


def law(src_id: str, start: str) -> str:
    text = load_sources()[src_id].text
    i = text.index(start)
    return text[i:].split("\n")[0]


R20 = law("rta:20", "(1)  A landlord is responsible")
R34 = law("rta:34", "The tenant is responsible")
R106 = law("rta:106", "(10)")
R89 = law("rta:89", "(1)  A landlord may apply")
R14 = law("rta:14", "A provision")
R27 = law("rta:27", "(1)  A landlord may enter")
FIRST_STEP = "If a tenant has a maintenance problem, the first thing they should do is ask the landlord - in writing - to fix the problem."

CASES = [
    ("The Landlord may deduct from the rent deposit the cost of repairing any damage caused by the Tenant or the Tenant's guests.", [R89, R34, R106], "conflicts"),
    ("The Tenant shall be responsible for the repair or replacement of all appliances, including the refrigerator and stove, regardless of the cause of the breakdown.", [R20, R34, FIRST_STEP], "conflicts"),
    ("No dogs or cats are permitted in the unit at any time.", [R14], "conflicts"),
    ("The Landlord may enter the unit at any time to inspect it.", [R27], "conflicts"),
    ("The Landlord shall maintain the Rented Premises in a good state of repair and fit for habitation during the term.", [R20, R34, FIRST_STEP], "matches"),
    ("The Tenant shall notify the Landlord in writing of any repairs required and shall allow the Landlord reasonable access to carry out the repairs.", [R20, R34, FIRST_STEP], "matches"),
    ("The Tenant shall be liable for costs of repair of damage caused by the willful or negligent conduct of the Tenant or their guests.", [R20, R34, FIRST_STEP], "matches"),
    ("Garbage is to be enclosed in plastic bags and placed in the containers provided.", [R20, R34, FIRST_STEP], None),
]


async def main() -> None:
    right = 0
    for lease, passages, want in CASES:
        got = (await jev.compare_terms(passages, [lease]))[0]
        right += got == want
        print(f"{'ok  ' if got == want else 'MISS'} want={str(want):9} got={str(got):9} | {lease[:72]}")
    print(f"\n{right}/{len(CASES)} right")


if __name__ == "__main__":
    load_dotenv(".env")
    assert jev.enabled(), "set TYPESAFE_API_KEY in .env"
    asyncio.run(main())
