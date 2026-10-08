"""End-to-end check of the app's answer path with Jev on, on questions written fresh (labelled
before the first run): route → tailored guide, or route → none → Jev-picked passages.

    python -m evals.run_app_eval        (needs TYPESAFE_API_KEY)

Each case says what a good answer must do:
  ("guide", id)        be answered from that reviewed guide
  ("source", [ids])    quote at least one of these sources
  ("nothing", None)    say plainly that the official sources don't cover it
"""

from __future__ import annotations

import asyncio
import re

from dotenv import load_dotenv

from realtor.lease import jev
from realtor.lease.pointers import POINTER_RE
from realtor.lease.router import playbook_answer, selected_answer, tailor
from realtor.lease.sources import retrieve

CASES = [
    ("The fridge in my apartment died. Who replaces it?", "guide", "broken-fixture"),
    ("Mice keep getting into the kitchen.", "guide", "pests"),
    ("The hot water hasn't worked for a week.", "guide", "no-heat"),
    ("Can my landlord let himself in to check the smoke detector without telling me?", "guide", "entry"),
    ("How much can my rent go up next year?", "guide", "rent-increase"),
    ("I'm a landlord and my tenant hasn't paid for March.", "guide", "late-rent"),
    ("The new owner of my building says they want to move in.", "guide", "selling"),
    ("Can a landlord ask for first and last month's rent plus a security deposit?", "guide", "damage-deposit"),
    ("My tenant left their couch behind when they moved out.", "guide", "property-left"),
    ("Can I get out of my lease if I'm moving away for school?", "guide", "leave-early"),
    ("The landlord won't rent to me because I have kids.", "guide", "discrimination"),
    ("Can I put up shelves on the wall?", "source", ["lease-section:12"]),
    ("Do I have to get tenant insurance?", "source", ["lease-section:11"]),
    ("Can my landlord stop me from vaping on my balcony?", "source", ["lease-section:10", "lease:S"]),
    ("Does my landlord have to give me a receipt for rent?", "source", ["lease:F"]),
    ("Who has to replace the smoke alarm batteries?", "source", ["lease:T"]),
    ("Can my landlord charge extra rent because my partner stays over a lot?", "source", ["lease:Q"]),
    ("Can a tenant keep a dog if the building is a condo that bans pets?", "source", ["lease:R"]),
    ("What's the best neighbourhood in Toronto for students?", "nothing", None),
    ("Can you recommend a moving company?", "nothing", None),
]


def main() -> None:
    load_dotenv(".env")
    assert jev.enabled(), "set TYPESAFE_API_KEY in .env"
    ok_total = 0
    for q, kind, want in CASES:
        sc, conf = asyncio.run(jev.route(q))
        if sc:
            keep = asyncio.run(tailor(sc, q))
            body, sources = playbook_answer(sc, q, keep)
            got = f"guide:{sc['id']}"
        else:
            sources = retrieve(q, "question")
            body, info = asyncio.run(selected_answer(q, sources))
            got = "nothing" if not info.get("selected") else "passages"
        pointed = {sources[int(m.group(1)) - 1].id for m in POINTER_RE.finditer(body) if int(m.group(1)) <= len(sources)}
        if kind == "guide":
            ok = got == f"guide:{want}"
        elif kind == "source":
            ok = bool(pointed & set(want))
        else:
            ok = got == "nothing"
        ok_total += ok
        detail = got if kind != "source" else f"{got}, quoted {sorted(pointed)[:4]}"
        print(f"{'ok  ' if ok else 'MISS'} {q[:62]:62} → {detail}")
    print(f"\n{ok_total}/{len(CASES)} answered the way a good answer should")


if __name__ == "__main__":
    main()
