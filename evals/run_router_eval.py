"""Does the router send questions to the right scenario, and leave the rest alone?

    python -m evals.run_router_eval            # local router (base 8B)
    JEV=1 python -m evals.run_router_eval      # Jev router

Wording here is deliberately different from train/scenarios.json. A wrong route is worse than
none (it answers a different question), so wrong routes are counted separately from misses.
"""

from __future__ import annotations

from realtor.lease.local_model import CHAT_ARGS, _load
from realtor.lease.router import route

CASES = [
    ("broken-fixture", "My toilet is leaking, who has to fix it?"),
    ("broken-fixture", "The oven in my apartment doesn't work anymore."),
    ("broken-fixture", "Our washing machine broke, does the landlord have to fix it?"),
    ("broken-fixture", "The tenant says the dishwasher is broken, do I pay for it?"),
    ("pests", "I found bed bugs in my bedroom. What now?"),
    ("pests", "There are rats in the basement of my rental house."),
    ("no-heat", "It's January and the heating has been off for three days."),
    ("entry", "My landlord came into my unit while I was at work without telling me."),
    ("entry", "How much notice do I have to give before inspecting my rental?"),
    ("rent-increase", "Can the rent go up in the middle of a lease?"),
    ("rent-increase", "My landlord wants to raise the rent by 10% next month."),
    ("late-rent", "The tenant is two weeks late on rent. What happens next?"),
    ("own-use", "My landlord says his son is moving in so I have to leave."),
    ("damage-deposit", "The landlord is asking for a $1,000 damage deposit. Can they?"),
    ("locks", "I came home and the locks were changed."),
    ("pets", "Can my landlord make me get rid of my cat?"),
    ("leave-early", "I got a job in another city and need to break my lease."),
    ("lease-ends", "My lease is up in June. Do I have to sign a new one or leave?"),
    ("sublet", "Can I rent out my room while I'm away for four months?"),
    ("renoviction", "Landlord says I have to leave because he's renovating the whole unit."),
    ("discrimination", "Can my landlord evict me for having a baby?"),
    ("discrimination", "The landlord refused my application because I use a wheelchair."),
    ("roommates", "My roommate won't pay their share. What can I do?"),
    (None, "Can my landlord charge me for parking?"),
    (None, "Can I paint the walls in my rental?"),
    (None, "Is my condo fee tax deductible?"),
    (None, "How do I file an application with the LTB?"),
    (None, "Can I smoke on my balcony?"),
]


def main() -> None:
    import asyncio
    import os

    if os.environ.get("JEV") == "1":  # route with Jev (needs TYPESAFE_API_KEY)
        from dotenv import load_dotenv

        from realtor.lease import jev
        load_dotenv()
        assert jev.enabled(), "set TYPESAFE_API_KEY in .env"
        router = lambda q: asyncio.run(jev.route(q))[0]
    elif os.environ.get("ROUTER_BASE") == "1":  # the plain base model, without our answer-writing adapter
        from mlx_lm import load

        from realtor.lease.local_model import BASE_MODEL
        model, tok = load(BASE_MODEL)
    else:
        model, tok = _load()
    if os.environ.get("JEV") != "1":
        router = lambda q: route(q, model, tok, CHAT_ARGS)
    right = wrong = missed = false_route = 0
    for want, q in CASES:
        sc = router(q)
        got = sc["id"] if sc else None
        if want and got == want:
            right += 1; mark = "ok  "
        elif want and got is None:
            missed += 1; mark = "miss"
        elif want:
            wrong += 1; mark = "WRONG"
        elif got:
            false_route += 1; mark = "WRONG"
        else:
            right += 1; mark = "ok  "
        print(f"{mark} want={want or '-':15} got={got or '-':15} | {q}")
    n_known = sum(1 for w, _ in CASES if w)
    print(f"\ncorrect: {right}/{len(CASES)} · known questions missed (falls back to the model): {missed}/{n_known}")
    print(f"wrong routes (the bad kind): {wrong + false_route}  ({false_route} of them on questions with no scenario)")


if __name__ == "__main__":
    main()
