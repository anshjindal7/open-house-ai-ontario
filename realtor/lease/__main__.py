"""Usage: python -m realtor.lease path/to/lease.(txt|pdf) [--scan-only]   (Ontario leases)"""

from __future__ import annotations

import argparse
import asyncio

from dotenv import load_dotenv

from .clauses import load_lease, split_clauses
from .explain import explain, format_findings
from .scanner import scan_lease


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Read a lease and flag the fine print.")
    parser.add_argument("lease")
    parser.add_argument("--scan-only", action="store_true", help="Skip the local model; print Jev's raw findings")
    args = parser.parse_args()

    text = load_lease(args.lease)
    clauses = split_clauses(text)
    print(f"Scanning {len(clauses)} clauses with Jev...\n")
    findings = asyncio.run(scan_lease(clauses))

    if args.scan_only:
        print(format_findings(findings))
        return
    print(explain(findings))


if __name__ == "__main__":
    main()
