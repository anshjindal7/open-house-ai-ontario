"""Score the base model and the fine-tuned model on the held-out test set.

    python -m train.compare            # all test examples
    python -m train.compare --limit 40 # quicker

Metrics, per model:
  right source   the answer cites the source number the reference answer cites first
  faithful       every "quoted" passage in the answer appears word for word in the sources
  right outcome  for LTB cases, says who the decision favoured correctly
  abstains       when the sources don't cover the question, says so instead of answering
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from mlx_lm import generate, load

from realtor.lease.local_model import ADAPTER_DIR, BASE_MODEL, CHAT_ARGS

TEST = Path(__file__).parent / "data" / "test.jsonl"
ABSTAIN = re.compile(r"don't answer|do not answer|not (covered|addressed)|no information|cannot find", re.I)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("“", '"').replace("”", '"')).strip().lower()


def score(answer: str, reference: str, prompt: str) -> dict[str, bool | None]:
    ref_cite = re.search(r"\[(\d)\]", reference)
    quotes = [q for q in re.findall(r'"([^"]{20,})"', answer.replace("“", '"').replace("”", '"'))]
    ref_outcome = re.search(r"(tenant|landlord)'s favour", reference)
    ref_abstains = bool(ABSTAIN.search(reference))
    return {
        "right source": None if ref_abstains or not ref_cite else f"[{ref_cite.group(1)}]" in answer,
        "faithful": None if not quotes else all(norm(q.rstrip(" …")) in norm(prompt) for q in quotes),
        "right outcome": None if not ref_outcome else ref_outcome.group(0) in answer,
        "abstains": bool(ABSTAIN.search(answer)) if ref_abstains else None,
    }


def run(model, tok, messages) -> str:
    prompt = tok.apply_chat_template(messages, **CHAT_ARGS)
    out = generate(model, tok, prompt=prompt, max_tokens=700)
    return re.sub(r"^\s*<think>[\s\S]*?</think>\s*", "", out).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--show", type=int, default=3, help="print this many side-by-side examples")
    args = parser.parse_args()

    rows = [json.loads(l) for l in TEST.open()][: args.limit]
    models = {"base": load(BASE_MODEL), "tuned": load(BASE_MODEL, adapter_path=str(ADAPTER_DIR))}
    totals = {m: {} for m in models}

    for i, row in enumerate(rows):
        msgs, reference = row["messages"][:-1], row["messages"][-1]["content"]
        prompt_text = msgs[-1]["content"]
        outs = {}
        for name, (model, tok) in models.items():
            outs[name] = run(model, tok, msgs)
            for metric, ok in score(outs[name], reference, prompt_text).items():
                if ok is not None:
                    hit, n = totals[name].get(metric, (0, 0))
                    totals[name][metric] = (hit + ok, n + 1)
        if i < args.show:
            print("=" * 90, "\nQ:", prompt_text.split("Question: ")[-1][:300])
            for name, out in outs.items():
                print(f"\n--- {name} ---\n{out[:700]}")
            print(f"\n--- reference ---\n{reference[:500]}\n")
        print(f"{i + 1}/{len(rows)}", end="\r", flush=True)

    print(f"\n{'metric':15} " + " ".join(f"{m:>14}" for m in models))
    for metric in ("right source", "faithful", "right outcome", "abstains"):
        cells = []
        for m in models:
            hit, n = totals[m].get(metric, (0, 0))
            cells.append(f"{hit:>4}/{n:<4} {hit / n:>4.0%}" if n else f"{'-':>14}")
        print(f"{metric:15} " + " ".join(cells))


if __name__ == "__main__":
    main()
