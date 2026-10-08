"""Ask the Ontario lease model about a clause or question, no Jev key needed.

    python -m realtor.lease.ask "Tenant shall pay a $500 damage deposit"
    python -m realtor.lease.ask --question "Can my landlord enter without notice?"
"""

from __future__ import annotations

import argparse

from .local_model import answer, clause_question
from .pointers import expand
from .sources import retrieve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clause", nargs="?", help="lease wording to check")
    parser.add_argument("--question", help="a free-form question instead of a clause")
    args = parser.parse_args()
    if not (args.clause or args.question):
        parser.error("give a clause or --question")

    query = args.question or args.clause
    question = args.question or clause_question(args.clause)
    text, sources = answer(question, retrieve(query, "clause" if args.clause else "question"))
    text = expand(text, sources)
    print(text, "\n\nSources:")
    for i, s in enumerate(sources, 1):
        print(f"  [{i}] {s.label}")


if __name__ == "__main__":
    main()
