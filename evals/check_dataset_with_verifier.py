"""Run the fact-checker over every training answer (all correct by construction).

    python -m evals.check_dataset_with_verifier

Any issue it reports is a false alarm in the checker, so this measures its false-alarm rate.
"""

import collections
import json
import re
from pathlib import Path

from realtor.lease.sources import Source
from realtor.lease.verify import verify

DATA = Path(__file__).parents[1] / "train" / "data" / "train.jsonl"
KIND = (("Residential Tenancies Act", "rta"), ("Ontario Standard Form", "lease"), ("LTB decision", "case"))


def parse_sources(prompt: str) -> list[Source]:
    body = prompt.split("Sources:\n\n", 1)[1].rsplit("\n\nQuestion: ", 1)[0]
    body = re.sub(r"\n\nAbout me: I'm the \w+\.\s*$", "", body)
    out = []
    for block in re.split(r"\n\n(?=\[\d\] )", body):
        m = re.match(r"\[(\d)\] (.*?)\n([\s\S]*)", block)
        if m:
            kind = next((k for prefix, k in KIND if m.group(2).startswith(prefix)), "guide")
            body = re.sub(r"\{\d\.\d+\} ", "", m.group(3))  # undo sentence numbering
            # The prompt shows the visible window already; keep it whole.
            # The prompt already shows the visible window; keep it whole (no further cutting).
            out.append(Source(f"s{m.group(1)}", "case", m.group(2), body) if len(body) > 600
                       else Source(f"s{m.group(1)}", kind, m.group(2), body))
    return out


def main() -> None:
    rows = [json.loads(line) for line in DATA.open()]
    issues, examples = collections.Counter(), []
    for r in rows:
        prompt = r["messages"][-2]["content"]
        rep = verify(r["messages"][-1]["content"], parse_sources(prompt), question=prompt.rsplit("Question: ", 1)[-1])
        for i in rep.issues:
            issues[i.kind] += 1
            if len(examples) < 8 and i.kind != "fact" or len(examples) < 3:
                examples.append((i.kind, i.detail, i.text[:150]))
    print(f"{len(rows)} correct answers; false alarms by kind: {dict(issues) or 'none'}")
    for e in examples:
        print("  ", e)


if __name__ == "__main__":
    main()
