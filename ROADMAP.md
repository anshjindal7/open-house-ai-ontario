# Open House roadmap

Goal: an assistant for Ontario landlords and tenants that's accurate (every fact traceable to an
official source), neutral by default, on your side only when you ask for negotiation help, and
pleasant to talk to.

Status key: [x] done · [~] in progress · [ ] to do. Times are working time, not calendar days.

## Phase 1: Grounded explainer (v2), today

- [x] Official sources: RTA (307 sections), Standard Form of Lease, LTB guidance (280 sections),
      2,050 redacted 2026 LTB decisions
- [x] Retrieval: hybrid search, everyday-to-legal word mapping, both-sides decisions, follow-up context
- [x] Neutral prompt and neutral lease checks (15 Ontario checks, each with its RTA section)
- [x] 21 scenarios with branches; negotiation suggestions labelled "suggestions only"
- [x] Chat frontend: side selector, starters, follow-ups, source drawer
- [x] Project moved to Desktop/Open House
- [x] Train v2: stopped at step 100 (validation loss 0.245 → 0.078) to fold in the Phase 2 data
      and two data fixes rather than train twice
- [x] Train v3 = v2 + Phase 2 data (350 steps from the v2 checkpoint). Validation loss 0.017 and
      still falling at the last check (no overfitting). Installed in `train/adapters/`; backup in
      `train/checkpoints/v3_final.safetensors`
- [x] Score base vs. trained on 40 held-out examples:
      right source 72% → 97%, word-for-word quotes 44% → 87%, right case outcome 28% → 97%,
      declines when sources don't cover it 25% → 50%
- [x] First live test caught a real failure: with ordinary search, a branch heading was paired with
      the wrong source ("tenant damaged it" quoted s. 30 instead of s. 34). The fact-checker flagged it.
- [x] Fix part 1: playbooks. Close matches to a known scenario get that scenario's sources (kept strict,
      since loose matching pulled in wrong scenarios).
- [x] Fix part 2: 174 training examples where a source is missing and the answer leaves that branch
      out and says so
- [x] Train v4 from v3 with those examples (250 steps). Validation loss rose (0.049 → 0.107).
- [x] End-to-end check with unseen paraphrases (`python -m evals.run_live_eval`, 12 questions):
      v3 6/12 verified · v4 step 200 8/12 · v4 5/12. All versions invent branch headings, pair
      headings with the wrong source, and misquote long passages. Templates also bleed together
      ("Thanks, that helps" at the start of a first answer). The 97% held-out score was optimistic
      because those prompts always contained the right sources and familiar wording.
- [x] Decision: pointer-based answers on Qwen3-8B. App stays on v3 until the 8B passes the live check.
- [x] ~~Switch the app to v4~~ (superseded: the 8B with pointers replaced v4)

## Phase 2: Ask before guessing, today (about 3 h, including 45 min of training)

When the answer depends on a fact the person hasn't given, the model asks one short question
instead of guessing.

- [x] Clarifying questions for 4 scenarios (broken fixture, pests, move-out damage, leaving early): 18 examples
- [x] Replies to those questions answered from the right branch: 72 two-turn examples
- [x] ~~More follow-up training examples~~ (not needed while Jev answers follow-ups; revisit if the model is used again)
- [x] Training: part of v3 (see Phase 1)

## Phase 3: Fact-checker, today or tomorrow (about 2 h, no training)

Every answer is checked before it's shown.

- [x] Quotes must appear word for word in the cited source
- [x] Numbers, deadlines and section numbers in the answer must appear in the sources (or the question)
- [x] Citations must point at sources that exist
- [x] With a Jev key: Jev judges each plain-language sentence against its source (code ready; needs a key to run)
- [x] UI: a "Checked" note under every answer; unverified parts are listed and their quotes outlined
- [x] Zero false alarms across all 2,100 training answers (now a test)
- [x] Caught two real flaws in the training data, now fixed: lease answers stated section numbers the
      model can't see, and one quote-detection edge case (titles containing quote marks)

## Proposed change: pointer-based answers (replaces generated quotes)

The model stops typing quotes and headings. It writes short plain sentences and points at source
sentences like {2.3} (source 2, sentence 3); the app inserts the exact text under the right
source label. Misquotes and wrong-source pairing become impossible by construction, and the
fact-checker only has to judge the plain sentences. For known scenarios, the branch headings come
from the playbook, not the model.

- [x] Number the sentences of every source the model sees ({2.1}, {2.2}, …)
- [x] Re-express all 1,969 training answers with pointers; tests check every pointer exists and
      no answer retypes a quote; the fact-checker finds nothing to flag
- [x] Renderer: pointers become exact quotes labelled with their source (app, CLI, lease report)
- [x] Qwen3-8B downloaded; prompts ask it not to "think" so training and use match
- [x] Train the 8B (360 steps). Validation loss lowest at step 240 (0.060), 0.111 at step 360
- [x] Live check, first pass: every answer verified (12/12 at both checkpoints, up from 6/12 on v3).
      Remaining problem: model-written headings ("The first step" everywhere; one 30-line loop)
- [x] Headings now come from each quoted source's own title, so they can't contradict the quote;
      repeated lines are dropped and generation stops if the model loops
- [x] Fact-checker also checks a case's stated outcome against its real result
- [x] Live check, second pass (sources the situation needs that the answer used): 8B step 240:
      12/12 verified, 16/22 sources used, 0 loops · step 360: 10/22, 2 loops. Step 240 installed.
- [x] Router: a short list of 5 likely scenarios; the base 8B (adapter switched off) picks one or
      "none". On 28 differently-worded questions: 19/23 known situations recognised, 1 wrong route
      (no-heat → broken-fixture, a near miss), 0 unrelated questions routed. Misses fall back to the
      model with the fact-checker on.
- [x] Recognised situations get the reviewed playbook answer (headings and quotes reviewed in advance,
      one decision each side won), shown with a "From a reviewed guide" badge. Negotiation questions
      with a side chosen get suggestions for that side from the playbook. Follow-ups go to the model.
- [x] Two new scenarios found by the router test: discrimination (Human Rights Code, standard
      lease Part M, Guideline 17) and roommates (Guideline 21, Part Q). 23 scenarios in total.
- [x] App switched to 8B step 240 with routing; tested end to end in the browser
- [x] Gap closed with Jev: off-topic passages are scored, hidden, and guides tailored. Was: "verified" isn't "relevant". An answer can quote real but off-topic passages, and
      embedding similarity can't tell. Needs Jev (Noul: "does this passage address the question?")
      or reviewer spot-checks.

## Jev integration

- [x] Relevance check: Jev judges whether each quoted passage helps answer the question (one parallel
      request); off-topic passages are flagged under the answer and outlined
- [x] "Supported" check batched into one request instead of one call per sentence
- [x] Router: Jev picks one of the 23 scenarios or none; below 60% confidence counts as none; falls
      back to the local router if Jev is unreachable
- [x] Tested against a stand-in Jev (tests/test_jev.py); nothing is sent without a key
- [x] "Supported" and relevance checks merged into one request per answer
- [x] Key added (jev-1.13.0 responding)
- [x] Router: 28/28 (all 23 known situations, 0 wrong routes) vs the local router's 23/28. Caveat: the
      two newest scenarios were written from questions in this test; a fresh question set is needed.
      Jev is now the router; the local router is the fallback.
- [x] Relevance: 83% agreement with hand labels at 0.50, 92% at 0.65 (now the cut-off). It caught the
      hard case (eviction section for "evict me for having a baby": 0.22). In 2 of its 4
      disagreements Jev's call was arguably better than the label. Small set: re-check with real use.
- [x] Lease checks: 14/15 perfect on 28 clauses. Jev showed 6 of my labels were wrong under Ontario
      law (pet and damage deposits, late fees) and they were corrected, citing Standard Form Part H.
      Corrected after seeing the scores, so a fresh, independently labelled set is needed.
- [x] Reviewed guides are tailored with Jev: branches that don't fit the question are left out (at
      least one is always kept), with a note that the full guide is available
- [x] Sample lease scan: 10 of 15 clauses flagged with the right RTA sections; not covered by any
      check yet: auto-renewal (clause 1) and indemnity (clause 11)
- [~] Fresh sets: `evals/run_app_eval.py` (20 new questions, labelled before running: 19/20) and 8 new lease clauses. Still wanted: a set labelled by someone other than me
- [x] New checks: auto-renewal, liability waiver/indemnity, no rent reduction, landlord's legal costs (17/19 checks perfect on 36 clauses)
- [x] Your test found chats treating every new question as a follow-up (the baby question got
      repair law; the paint question got eviction passages and "It depends on how it broke"). Fixed:
      Jev decides whether a question is a follow-up; new topics start fresh and go through the router
- [x] Decision lines are rebuilt from the decision itself (the model had copied a citation from an
      earlier answer); lines with a broken pointer are dropped
- [x] Passages Jev rates off-topic are hidden (with "show anyway"); if nothing relevant is left the
      answer says plainly that the right part of the law wasn't found
- [x] Standard lease sections 4-16 are now searchable (Section 12 answers painting, 11 insurance)
- [x] With Jev on, answers without a guide are built from the sentences Jev rates most relevant;
      the model writes nothing. Your sequence (toilet → "what if I broke it?" → baby → paint) is
      now correct and clean end to end. The model is only loaded if Jev is unavailable.
- [x] Re-run on the new path with fresh questions: 19/20 (the miss is arguably a better answer than my label)
- [x] Form choices on the standard lease are separate lines ("• There are no tenant insurance requirements.")
- [x] Your real-lease test: most clauses failed with "Failed to fetch" because I restarted the app
      mid-review (my mistake; no restarts while you're testing from now on)
- [x] Splitter, for the standard form: drops the government appendix, the translations page and
      page footers; marks the cover page and Sections 1-3 and 17 as form details (not checked, not
      sent to Jev); joins headings split over lines ("COVENANTS ANNUAL INCREASE"). The blank form
      went from 36 pieces to 19 (14 terms to check). Tests use made-up text, not your lease.
- [x] Lease review with Jev: Jev scans every term; only flagged terms are explained (about 1-2 s
      each); the rest show "No issues found in the scan" and can be checked one by one
- [x] Second real-lease test: every explanation failed with a page bug (a "0 passages fitted" result
      crashed the page) and the browser ran a stale script. Fixed; the page now always loads the
      current script; failures retry once, then show a quiet "Retry" instead of error text
- [x] Time estimate uses the measured time per term (it was a fixed guess of 24 s per term)
- [x] Personal details (NAME: lines, emails, phones, postal codes, street addresses) are removed
      from lease text before it's scanned by Jev or explained
- [x] Sample lease: full review in under 6 s, 0 failures (10 explained, 5 clean)
- [x] Re-test with your lease (third time; fixes since: additional-terms splitting, 4 new checks)
- [x] Additional terms split at term openings ("The Tenant…", "If…"); stray letter runs removed. Was: split by
      length rather than by item

## Phase 4: Interpreter ("short answer" applied to your facts), multi-day, needs a reviewer

- [x] Compare Qwen3-8B with the 4B (done via the live checks; 8B kept as fallback)
- [x] Drafting tool (`python -m train.interpret.draft`): an open model drafts short answers for real
      LTB cases; each draft is fact-checked, and flagged if it leans against the real ruling
- [x] Review page (`/review`): approve, edit or reject; edits are fact-checked before saving
- [x] Drafts, second batch of 50: 37 clean (was 20), 1 without citations (was 25), 10 fact-check
      issues, 7 leaning against the real ruling. When a draft picks a winner it's usually wrong
      (7 of 10), so short answers should say what applies, not who wins; reviewers should reject
      drafts that predict an outcome.
- [ ] Review the drafts at /review (you or a paralegal)
- [ ] Train the interpreter; score it with the fact-checker on
- [ ] Ship: short answer, then the quoted law, then real decisions

## Phase 5: Before a public launch

- [x] Licence terms checked (docs/licensing.md): RTA and decisions are fine; the standard lease and LTB guidance likely need a licence. Draft request written, not sent. Footer credit fixed.
- [ ] Send the licence request (you); until then the standard lease and LTB guidance stay in (your call)
- [~] Older years not published yet; July 2026 was, and is now in the corpus (~2,400 decisions). Re-check monthly: `python -m pipeline.fetch_ltb && python -m pipeline.ltb`
- [x] Jev key: run `python -m evals.run_eval` and tune the checks
- [x] Jev key: route questions to scenarios with a typed Choice
- [x] Privacy review (docs/privacy.md); questions and clauses scrubbed before Jev, including free-text names
- [x] Company landlords' names shown as "[organization]" (your call); words inside company names
      ("Homes", "Woods") are no longer redacted as if they were people's names

## Log

- Oct 6: v2 training started; project moved to Desktop/Open House; roadmap written.
- Oct 6: fact-checker built and wired in; clarifying questions added; v2 stopped at step 100; v3 started.
  Note: macOS blocks the browser tool from running programs in the Desktop folder, so the app is
  started from a terminal and the browser attaches (`open-house-attach` in .claude/launch.json).
- Oct 6: redaction fixes found while building Phase 4 drafts. "Residential" was being redacted as a
  name (128 decisions), and one name written in lowercase across a line break slipped through.
  Official names are now shielded, and names match any case and spacing. A full sweep finds no
  individual's name left. Dataset rebuilt; v3 restarted on the clean data.
- Oct 6-7: pointer answers on Qwen3-8B; headings from source titles; loop guard; outcome check;
  scenario router with the base model; playbook answers and negotiation suggestions in the app.
- Oct 7: worked through the roadmap: smoking/changes search, form choices, one heading per source,
  additional-terms splitting, 4 new lease checks, fresh end-to-end test (19/20), licence check and
  draft request, privacy review and question scrubbing, July 2026 decisions, outcome-rule fix,
  first 50 Phase 4 drafts, README rewritten.

## 2026-10-06: Lease in the conversation
- [x] A lease pasted into the chat (or reviewed in the Review tab → "Ask about this lease") is kept for the conversation
- [x] Every answer adds "What your lease says": the matching lease sentences (picked by Jev), flagged terms marked, RTA s. 4 quoted when a term may conflict
- [x] Example decisions that don't fit the question are dropped (Jev); "Why" lines use a complete finding, never a cut-off lead-in
- [x] Locks guide covers broken locks and doors, and always keeps the "don't change the lock yourself" rule (RTA s. 35)
- [x] Re-tested with the real lease (you): good
- [x] Each quoted lease line compared with the law by Jev (matches / adds / may conflict), shown in fixed wording
- [x] Decisions dropped from an answer are removed from its sources, and the rest renumbered
- [x] "What you can do next" under 19 guides (train/next_steps.json), split by side; only your side if you picked one
- [x] Editable written requests (repair, entry, disturbance) to start from; nothing is sent from the app
- [ ] Review the next steps (you or a paralegal): every form named is in the official LTB guidance, but the
      wording is hand-written

## 2026-10-06: Decisive answers
- [x] Every guide answer opens with a **short answer**: a plain conclusion (train/short_answers.json, all 23
      guides), each sentence citing the guide passage behind it. Fact-specific versions ("a guest broke it",
      "the landlord shut it off") are picked by Jev only at 0.70+; otherwise the general version, which covers
      every case. evals/run_short_answer_eval.py: 15/16 right, 0 wrong specific picks
- [x] **In your lease**: values from the standard form (rent, rent and key deposits, NSF charge, term dates,
      payment method, on-time discount) compared with the legal limits in code (realtor/lease/facts.py), e.g.
      "your rent deposit is one month's rent, the most allowed"; plus whether a quoted lease term matches or
      may conflict with the law
- [x] Lease vs law is now one yes/no per (lease line, law passage) pair (the four-way pick was near chance):
      evals/run_lease_relation_eval.py 8/8. The lease checks run on the quoted lines only
- [x] Printed form options ("A rent deposit is not required") are no longer quoted as what the lease says:
      the PDF's ticked box doesn't survive as text
- [x] Follow-ups are routed too, read with the previous question; the same guide is used again only when the
      follow-up's facts change the short answer ("what if I broke it by accident?")
- [x] Damage guide covers a landlord deducting damage from the deposit (RTA s. 106(10)); heat guide covers
      breakdowns as well as shut-offs (RTA s. 20)
- [x] Example decisions whose reason points the other way from the outcome show what was ordered instead
- [ ] Review the short answers (you or a paralegal), same as the next steps
- [ ] Retrain the fallback model without the AI-written guide wording in train/scenarios.json
