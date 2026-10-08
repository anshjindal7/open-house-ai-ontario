"""How often does the right RTA section show up in the top 3 retrieved? python -m evals.run_retrieval_eval"""

import json
from pathlib import Path

from realtor.lease.sources import search

CASES = [json.loads(l) for l in (Path(__file__).parent / "retrieval_cases.jsonl").open()]


def main() -> None:
    hits = 0
    for c in CASES:
        got = [s.id.split(":")[1] for s in search(c["q"], "rta", 3)]
        ok = any(w in got for w in c["want"])
        hits += ok
        print(f"{'ok  ' if ok else 'MISS'} {c['q'][:60]:60} got {got} want {c['want']}")
    print(f"\nrecall@3: {hits}/{len(CASES)}")


if __name__ == "__main__":
    main()
