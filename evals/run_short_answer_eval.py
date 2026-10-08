"""Does Jev pick the right version of a guide's short answer? (None = the general version.)

    python -m evals.run_short_answer_eval        (needs TYPESAFE_API_KEY)

Labelled before the first run. A wrong specific pick is the costly mistake (a confident answer
for facts the person didn't state); falling back to the general version is always safe.
"""

import asyncio

from dotenv import load_dotenv

from realtor.lease import jev
from realtor.lease.playbooks import scenarios
from realtor.lease.router import short_options

CASES = [
    ("broken-fixture", "My sink is broken.", None),
    ("broken-fixture", "The fridge stopped working last night.", None),
    ("broken-fixture", "I dropped a pan and cracked the glass stove top.", "damaged"),
    ("broken-fixture", "My friend broke the toilet seat when he was over.", "damaged"),
    ("no-heat", "The heat isn't working and it's cold.", None),
    ("no-heat", "My landlord turned off the heat because I'm late on rent.", "landlord_cut"),
    ("no-heat", "Hydro cut off my power because I didn't pay my electricity bill.", "tenant_bill"),
    ("entry", "How much notice does my landlord have to give before coming in?", None),
    ("entry", "My landlord walked into my apartment yesterday without telling me.", "already_entered"),
    ("locks", "My front door lock is broken.", None),
    ("locks", "My landlord changed the locks and now I can't get in.", "locked_out"),
    ("locks", "Can I put a new lock on my door?", "tenant_changing"),
    ("move-out-damage", "My landlord says they'll take the cost of damage out of my deposit.", None),
    ("late-rent", "I can't pay my rent this month. What happens?", None),
    ("late-rent", "My cheque bounced, how much can the landlord charge me?", "nsf"),
    ("move-out-damage", "My former tenant moved out and left holes in the walls. How do I get paid?", "moved_out"),
]


async def main() -> None:
    by_id = {s["id"]: s for s in scenarios()}
    right = wrong_specific = 0
    for sid, q, want in CASES:
        got, conf = await jev.pick_short(q, short_options(by_id[sid]))
        right += got == want
        wrong_specific += got is not None and got != want
        print(f"{'ok  ' if got == want else 'MISS'} {sid:16} want={str(want):16} got={str(got):16} {conf:.2f} | {q[:60]}")
    print(f"\n{right}/{len(CASES)} right · wrong specific picks (the costly kind): {wrong_specific}"
          f" · cut-off {jev.SHORT_PICK_MIN}")


if __name__ == "__main__":
    load_dotenv(".env")
    assert jev.enabled(), "set TYPESAFE_API_KEY in .env"
    asyncio.run(main())
