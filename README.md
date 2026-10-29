# Open House

Plain answers about renting in Ontario, for landlords and tenants alike.

Ask a question ("my fridge isn't working", "can my landlord keep my deposit for damage?") or
paste your lease, and Open House answers from Ontario's official sources: the Residential
Tenancies Act, the Standard Form of Lease, Landlord and Tenant Board (LTB) guidance, and about
2,400 real 2026 LTB decisions. Every quote is the source's exact wording and links to it.

> **Status:** a personal project in active development, running locally. General information,
> not legal advice. For help with a specific situation, contact the LTB (1-888-332-3234) or a
> community legal clinic.

## What it does

- **A short answer first.** The 23 most common situations (repairs, deposits, entry, rent
  increases, evictions, locks, pets, guests and more) open with a plain conclusion. The version
  that fits your facts is picked for you, so "a guest broke it" gets a different answer from
  "it stopped working".
- **The law behind it, word for word.** The exact sections of the Act, the standard lease and
  LTB guidance, each linked to its source.
- **Real decisions, one each side won.** Example LTB decisions on similar issues, labelled with
  who the Board sided with and why.
- **Your lease, read for you.** Paste your lease into the chat and every answer adds what your
  lease says: its numbers checked against the legal limits ("your rent deposit is one month's
  rent, the most allowed"; "your NSF charge is $35, more than the $20 allowed"), and any term
  that may conflict with the Act, flagged.
- **Lease review.** Upload a lease and every term is checked against 19 Ontario rules (no-pets
  clauses, extra deposits, entry without notice, early-exit penalties, liability waivers and more).
- **What to do next.** Practical steps for your side, the right LTB form, and an editable
  written request (for a repair, for example) to start from.
- **Neutral.** Answers favour neither landlords nor tenants. If you say which side you're on
  and ask for help with a negotiation, you get suggestions for your side, labelled as suggestions.

## How an answer is put together

```
your question ──▶ names, emails, phones and addresses removed
              ──▶ which situation is this?                    (Jev)
                     │ one of the 23 guides                        │ none of them
                     ▼                                             ▼
        short answer, version picked for your facts    search the official sources; Jev
        guide passages that apply to your question     picks the sentences that answer it
                     │                                             │
                     └──────▶ your lease: its numbers vs the legal limits, terms that match or conflict
                              example decisions that fit, checked against their outcomes
                              fact-checker: every quote, number and citation against its source
```

**No model writes the answer.** Answers are assembled from three things:
- **Reviewed text:** the guides' short answers and headings.
- **Official text:** the sources' own sentences.
- **Code:** the lease figures, compared with the legal limits.

[Jev](https://typesafe.ai) (TypeSafe AI) makes the judgment calls: which guide, which version,
which passages, whether a lease term conflicts. It returns probabilities, not text, so it can't
make up an answer.

**Why not just ask an LLM?** When a language model writes freely, it can misquote the Act or cite
the wrong section, and nothing warns you. Here every sentence can be checked against its source.

A fine-tuned Qwen3-8B (`train/`) answers only when Jev isn't available.

## Sources

| Source | Coverage |
|---|---|
| [Residential Tenancies Act, 2006](https://www.ontario.ca/laws/statute/06r17) | 307 sections |
| Residential Tenancy Agreement (Standard Form of Lease), form 2229E | sections 4–16 and the 21 explanation parts |
| LTB Interpretation Guidelines and brochures | 280 sections |
| LTB decisions (Order Catalogue), January–July 2026 | ~2,400 decided cases, with people's names and companies removed |

Credits and terms for each source are in [NOTICE.md](NOTICE.md). Licensing details for a future
commercial launch are in [docs/licensing.md](docs/licensing.md).

## Getting started

Needs a Mac with Apple Silicon (the local model runs on MLX) and Python 3.12 or newer.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env        # then add your TYPESAFE_API_KEY (optional, see below)
```

```bash
uvicorn realtor.web.app:app --port 8000
```

Then open http://localhost:8000.

**With a Jev key** (`TYPESAFE_API_KEY` in `.env`): the full experience described above.

**Without one:** the app still runs, but the local model answers, and its answers need more
caution. That requires the trained adapter in `train/adapters/`, which isn't in the repo; see
[Training the fallback model](#training-the-fallback-model).

To try the interface without any model, run:

```bash
OPEN_HOUSE_PREVIEW=1 uvicorn realtor.web.app:app --port 8000
```

### Using it

- **Ask a question:** type it in the chat. Pick "Tenant" or "Landlord" if you want next steps for
  your side only.
- **Ask about your lease:** click **Attach a lease** (PDF or text), drop the file on the chat box,
  or paste the whole lease. It's kept for that conversation only, with personal details removed.
- **Review a lease:** use the Review tab to upload a PDF or text file, or paste it.
- **From the terminal:**

  ```bash
  python -m realtor.lease path/to/lease.pdf
  ```

## Rebuilding the data

The processed sources are in `corpus/`, so this is only needed to refresh them. The raw downloads
go in `sources/`, which isn't committed.

```bash
python -m pipeline.fetch_ltb && python -m pipeline.ltb        # LTB decisions: download, then redact
python -m pipeline.fetch_guidance && python -m pipeline.guidance  # LTB guidelines and brochures
python -m pipeline.rta              # reads sources/rta.txt
python -m pipeline.standard_lease   # reads sources/2229e_standard_lease.txt
```

- **The Act:** download the Word version from e-Laws to `sources/rta.doc`, then convert it with
  `textutil -convert txt sources/rta.doc`.
- **The standard lease:** download the PDF (form 2229E) and save its text as
  `sources/2229e_standard_lease.txt`.

## Training the fallback model

```bash
python -m train.build_dataset       # builds train/data/ from the sources and scenarios
bash train/train.sh                 # LoRA fine-tune of Qwen3-8B with MLX (ITERS=10 for a smoke test)
```

Training answers quote the official sources verbatim. Known issue: the guides' connecting wording
(openers, headings and example questions in `train/scenarios.json`) was written with an AI assistant,
and the current model was trained on it. The next retrain leaves it out (see `ROADMAP.md`).

## Hosting

The hosted version runs in Docker (`Dockerfile`, `requirements-server.txt`) with
`OPEN_HOUSE_HOSTED=1`. In that mode:
- Jev answers every question, and the local model is never loaded.
- The draft-review pages are switched off.
- Each visitor gets a few questions and lease scans per hour, and the whole site has a daily cap.
  The limits are environment variables listed in `realtor/web/hosting.py`.
- A beta notice and a privacy note are shown.
- Answers get thumbs up/down feedback.
- Nothing visitors type or upload is saved, apart from the feedback they choose to send.

To put it on a [Hugging Face Space](https://huggingface.co/docs/hub/spaces-sdks-docker):

```bash
pip install huggingface_hub
hf auth login
python deploy/upload_space.py YOUR-HF-NAME/open-house-ai-ontario
```

`hf auth login` asks for a write token. Then, in the Space's **Settings → Variables and secrets**,
add the secret `TYPESAFE_API_KEY`.

To keep feedback across restarts, also add:
- `HF_TOKEN`: a write token.
- `FEEDBACK_DATASET`: for example `YOUR-HF-NAME/open-house-feedback`. A private dataset is created
  on first use.

## Checks

```bash
pytest                              # 37 tests
```

| Command | What it measures | Latest |
|---|---|---|
| `python -m evals.run_app_eval` | 20 fresh questions through the real answer path | 19/20 |
| `JEV=1 python -m evals.run_router_eval` | picking the right guide | 28/28 |
| `python -m evals.run_short_answer_eval` | picking the right version of a short answer | 15/16, no wrong specific picks |
| `python -m evals.run_lease_relation_eval` | lease line vs the law: conflicts, matches or neither | 8/8 |
| `python -m evals.run_relevance_eval` | Jev's relevance judgments vs hand labels | 92% |
| `python -m evals.run_eval` | 19 lease checks on 36 labelled clauses | 17 of 19 perfect |
| `python -m evals.check_dataset_with_verifier` | fact-checker false alarms on training data | none |

The Jev checks need `TYPESAFE_API_KEY`.

## Project layout

| Folder | What's in it |
|---|---|
| `realtor/lease/` | Search, guides and routing, Jev calls, short answers, lease reading and checks, lease figures, fact-checker, privacy scrub |
| `realtor/web/` | The app: chat, lease review, `/review` for checking drafted answers, and the hosting rules (`hosting.py`) |
| `deploy/` | Uploading to a Hugging Face Space |
| `pipeline/` | Downloading and parsing the official sources |
| `corpus/` | The processed sources the app searches |
| `train/` | The guides (`scenarios.json`), short answers, next steps, dataset builder, training, Phase 4 drafting |
| `evals/`, `tests/` | The checks above |
| `docs/` | Licensing and privacy notes |
| `ROADMAP.md` | What's done and what's next |

## Privacy

- Questions and leases are scrubbed of names, emails, phone numbers and addresses before
  anything is sent to Jev.
- The lease figures used in answers stay on your machine.
- Decisions in the corpus have people's names and companies removed.

Details are in [docs/privacy.md](docs/privacy.md).

## What's next

- **Short answers for every question, not only the 23 guides.** Our model drafts them and a person
  reviews the drafts; the reviewed ones become its training data (Phase 4 in the roadmap).
- **A person's review** of the guides' short answers and next steps.
- **A licence from the King's Printer** for the standard lease and LTB guidance, before any
  commercial launch.
