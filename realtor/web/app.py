"""Open House lease reader: local web app.

    uvicorn realtor.web.app:app --port 8000          # real model
    OPEN_HOUSE_PREVIEW=1 uvicorn realtor.web.app:app      # UI preview: real retrieval, placeholder answers

The model runs on this machine. Requests are answered one at a time (one GPU), and answers
stream to the browser as server-sent events.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from realtor.lease.clauses import load_lease, split_clauses
from realtor.lease.facts import extract as lease_facts, lines as lease_lines
from realtor.lease.local_model import ADAPTER_DIR, CHAT_ARGS, build_messages, clause_question, repeating, search_query
from realtor.lease.sources import retrieve
from realtor.lease.verify import jev_check, verify

load_dotenv()
STATIC = Path(__file__).parent / "static"
SAMPLE = Path(__file__).parents[1] / "data" / "sample_lease.txt"
PREVIEW = os.environ.get("OPEN_HOUSE_PREVIEW") == "1"
MIN_CLAUSE_CHARS = 60  # skip headings, signature lines and the like

app = FastAPI(title="Open House lease reader")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
_gpu = ThreadPoolExecutor(max_workers=1)  # MLX work stays on one thread
_model_lock = threading.Lock()
_model = None


def _load_model():
    global _model
    with _model_lock:
        if _model is None:
            from mlx_lm import load

            from realtor.lease.local_model import BASE_MODEL

            adapter = str(ADAPTER_DIR) if (ADAPTER_DIR / "adapters.safetensors").exists() else None
            _model = load(BASE_MODEL, adapter_path=adapter)
    return _model


class Turn(BaseModel):
    question: str
    answer: str
    scenario: str | None = None  # the reviewed guide that answered it, if any


class LeaseTerm(BaseModel):
    heading: str = ""
    text: str


class AskRequest(BaseModel):
    text: str
    mode: str = "clause"  # "clause" (lease wording) | "question" (a situation or question)
    role: str | None = None  # "tenant" | "landlord" | None; only changes negotiation suggestions
    history: list[Turn] = []  # earlier turns in this conversation, oldest first
    lease: list[LeaseTerm] = []  # the user's own lease, if they shared one in this chat (terms, names removed)
    lease_facts: dict[str, str] = {}  # values from the lease's standard form (rent, deposits, dates); never sent to Jev


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _preview_tokens(sources):
    """Placeholder in the real answer layout, built from the retrieved sources (no model)."""
    def first_sentence(s):
        body = s.shown.removeprefix("… ")
        if s.kind == "case":
            body = re.search(r"Key reasons: (.*)", body).group(1) if "Key reasons:" in body else body
        return re.split(r"(?<=[.;])\s", body, maxsplit=1)[0][:300].rstrip(" …")
    lines = ["Preview mode: this is a placeholder built from the sources, not a model answer.", ""]
    for i, s in enumerate(sources[:3], 1):
        lines.append(f"**Source {i}.** [{i}] says: {{{i}.1}}")
    cases = [(i, s) for i, s in enumerate(sources, 1) if s.kind == "case"]
    if cases:
        lines += ["", "**At the Landlord and Tenant Board.** Real decisions on similar issues:"]
        lines += [f"- In [{i}] {s.label}, the Board decided. Why: {{{i}.{len(s.sentences)}}}" for i, s in cases]
    lines += ["", "Want me to go through what either side can do next?"]
    for word in "\n".join(lines).split(" "):
        yield word + " "


NEXT_STEPS = Path(__file__).parents[2] / "train" / "next_steps.json"


def _next_steps(scenario_id: str, role: str | None) -> dict | None:
    """The practical steps for a guide: only the asker's side if they said which, otherwise both."""
    steps = json.loads(NEXT_STEPS.read_text())["steps"].get(scenario_id) if NEXT_STEPS.exists() else None
    if not steps:
        return None
    sides = [role] if role else ["tenant", "landlord"]
    return {"sides": {s: steps[s] for s in sides if steps.get(s)}, "letter": steps.get("letter") if role != "landlord" else None}


def _lead(short: str, facts: list[str], summary: str) -> str:
    """The top of an answer: the short answer, then what the user's own lease says about it."""
    parts = [f"**Short answer.** {short}"] if short else []
    lease = " ".join([*facts, summary] if summary else facts)
    if lease:
        parts.append(f"**In your lease.** {lease}")
    return "\n\n".join(parts) + ("\n\n" if parts else "")


def _source_payload(sources) -> list[dict]:
    return [{"n": i + 1, "id": s.id, "kind": s.kind, "label": s.label, "text": s.shown, "side": s.side,
             "sentences": s.sentences} for i, s in enumerate(sources)]


@app.get("/")
def index():
    """The page, with its script and styles versioned by file time so a reload never runs stale code."""
    from fastapi.responses import HTMLResponse

    html = (STATIC / "index.html").read_text()
    for name in ("app.js", "app.css", "letters.js"):
        html = html.replace(f"/static/{name}", f"/static/{name}?v={int((STATIC / name).stat().st_mtime)}")
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


@app.get("/api/status")
def status():
    return {
        "preview": PREVIEW,
        "fine_tuned": (ADAPTER_DIR / "adapters.safetensors").exists(),
        "jev": bool(os.environ.get("TYPESAFE_API_KEY")),
        "model_loaded": _model is not None,
    }


@app.post("/api/ask")
async def ask(req: AskRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "Empty question")
    # Questions and lease wording can both carry names, emails, phone numbers and addresses;
    # none of it is needed to find the law, and some of it would otherwise go to Jev.
    from realtor.lease.privacy import scrub
    text = scrub(text)
    question = clause_question(text) if req.mode == "clause" else text
    history = [(t.question, t.answer) for t in req.history[-2:]]  # last two turns keep the prompt short
    last_guide = req.history[-1].scenario if req.history else None
    role = req.role if req.role in ("tenant", "landlord") else None
    # A new question in an ongoing chat isn't always a follow-up. If it starts a new topic, drop
    # the history so search and the model only see the new question.
    if history and req.mode == "question":
        from realtor.lease import jev
        from realtor.lease.local_model import looks_like_followup

        try:
            followup = (await jev.is_followup(history[-1][0], text)) >= 0.5 if jev.enabled() else looks_like_followup(text)
        except Exception:
            followup = looks_like_followup(text)
        if not followup:
            history = []
    loop = asyncio.get_running_loop()
    query = text if req.mode == "clause" else search_query(text, history)
    sources = await loop.run_in_executor(None, retrieve, query, "clause" if req.mode == "clause" else "question")
    # The user's own lease: the terms that match this question are quoted next to the law.
    from realtor.lease.sources import lease_terms
    terms = (await loop.run_in_executor(None, lease_terms, [t.model_dump() for t in req.lease], query)
             if req.lease and req.mode == "question" else [])
    # Terms can come from the review tab unscrubbed; the ones picked go to Jev, so clean them here.
    from dataclasses import replace as _replace
    terms = [_replace(t, text=scrub(t.text)) for t in terms]

    known: list[str] = []  # fixed-wording lines computed from the user's lease; their numbers count as known

    async def events():
        queue: asyncio.Queue = asyncio.Queue()

        def generate():
            nonlocal sources
            try:
                # Known situations are answered from the reviewed playbook (first turn only;
                # follow-ups go to the model, which sees the conversation so far).
                # New topics (first question, or a later one that isn't a follow-up) go through the
                # router; real follow-ups go to the model, which sees the conversation so far.
                # Follow-ups are routed too, read together with the previous question: a different guide
                # means a new topic in the same conversation; the same guide is used again only if the
                # follow-up adds facts that change the short answer ("what if I broke it by accident?").
                if not PREVIEW and req.mode == "question":
                    from realtor.lease.router import playbook_answer, playbook_negotiation, route, tailor, wants_negotiation_help

                    from realtor.lease import jev

                    route_q = f"{history[-1][0]} {text}" if history else text
                    sc = None
                    # The 8B model is only loaded when Jev isn't available to do the job.
                    if jev.enabled():
                        try:
                            sc, _ = asyncio.run(jev.route(route_q))
                        except Exception:  # Jev unreachable: fall back to the local router
                            sc = route(route_q, *_load_model(), CHAT_ARGS) if not history else None
                    elif not history:
                        sc = route(text, *_load_model(), CHAT_ARGS)
                    narrow = None  # same guide again: only the fact-specific short answer and what it cites
                    if sc and history and sc["id"] == last_guide:
                        from realtor.lease.router import short_options
                        try:
                            narrow = asyncio.run(jev.pick_short(text, short_options(sc)))[0] if jev.enabled() else None
                        except Exception:
                            narrow = None
                        if not narrow:
                            sc = None  # nothing new for this guide: answer from the passages instead
                    if sc:
                        negotiating = wants_negotiation_help(text, role)
                        keep, short = None, ""
                        if negotiating:
                            body, sources = playbook_negotiation(sc, role, text)
                        else:
                            from realtor.lease.router import cite_short, short_answer, short_options

                            option = narrow
                            if narrow:
                                short, need = short_answer(sc, narrow)
                                keep = set(need)
                            elif jev.enabled():
                                # Trim the guide to the question and pick the short answer's version, together.
                                async def judge():
                                    return await asyncio.gather(tailor(sc, route_q), jev.pick_short(text, short_options(sc)),
                                                                return_exceptions=True)
                                kept, picked = asyncio.run(judge())
                                keep = None if isinstance(kept, BaseException) else kept
                                option = None if isinstance(picked, BaseException) else picked[0]
                            short, need = short_answer(sc, option)
                            if keep is not None:
                                keep |= need  # every branch the short answer cites is shown
                            body, sources = playbook_answer(sc, text, keep)
                            short = cite_short(short, sc, sources) if short else ""
                            dropped = len(sc["branches"]) - len(keep) if keep is not None and not narrow else 0
                            if dropped:
                                body += f"\n\n_This guide has {dropped} more part{'s' if dropped > 1 else ''} that didn't fit your question; ask if you'd like the full guide._"
                        tailored_now = not negotiating and jev.enabled() and keep is not None
                        if jev.enabled() and not negotiating:
                            try:
                                from realtor.lease.router import drop_unrelated_cases
                                body, sources = asyncio.run(drop_unrelated_cases(text, body, sources))
                            except Exception:
                                pass
                        summary = ""
                        if terms and jev.enabled():
                            try:
                                from realtor.lease.router import lease_section
                                body, sources, summary = asyncio.run(lease_section(route_q, body, sources, terms))
                            except Exception:
                                pass  # the answer stands without the lease part
                        facts = [] if negotiating else lease_lines(req.lease_facts, sc["id"], route_q)
                        known.extend(facts)
                        body = _lead(short, facts, summary) + body
                        loop.call_soon_threadsafe(queue.put_nowait, ("route", {"scenario": sc["id"], "topic": sc["topic"],
                                                                               "tailored": tailored_now}))
                        loop.call_soon_threadsafe(queue.put_nowait, ("sources", _source_payload(sources)))
                        for piece in re.findall(r"\S+\s*|\n", body):
                            loop.call_soon_threadsafe(queue.put_nowait, ("token", piece))
                        if steps := _next_steps(sc["id"], role):
                            loop.call_soon_threadsafe(queue.put_nowait, ("steps", {**steps, "issue": text}))
                        loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
                        return
                # With Jev: everything without a reviewed guide is answered with the passages Jev
                # rates most relevant. The model writes nothing (on unseen questions its own
                # answers chose off-topic passages; Jev's typed judgments held up far better).
                from realtor.lease import jev as _jev
                if not PREVIEW and _jev.enabled():
                    from realtor.lease.router import selected_answer

                    try:
                        previous = history[-1][0] if history else ""
                        body, info = asyncio.run(selected_answer(question, sources, previous))
                        try:
                            from realtor.lease.router import drop_unrelated_cases
                            body, sources = asyncio.run(drop_unrelated_cases(text, body, sources))
                        except Exception:
                            pass
                        summary = ""
                        if terms:
                            from realtor.lease.router import lease_section
                            try:
                                n_before = len(sources)
                                body, sources, summary = asyncio.run(lease_section(question, body, sources, terms, previous))
                                info["lease"] = len(sources) - n_before
                            except Exception:
                                pass
                        facts = lease_lines(req.lease_facts, None, text)
                        known.extend(facts)
                        # After the opening line, so "this" in the lease note refers to the law below it.
                        intro, _, rest = body.partition("\n\n")
                        body = (intro + "\n\n" + _lead("", facts, summary) + rest) if rest else _lead("", facts, summary) + body
                        loop.call_soon_threadsafe(queue.put_nowait, ("route", {"selected": True, **info}))
                        loop.call_soon_threadsafe(queue.put_nowait, ("sources", _source_payload(sources)))
                        for piece in re.findall(r"\S+\s*|\n", body):
                            loop.call_soon_threadsafe(queue.put_nowait, ("token", piece))
                        loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
                        return
                    except Exception:
                        pass  # Jev unavailable: the model answers below
                loop.call_soon_threadsafe(queue.put_nowait, ("sources", _source_payload(sources)))
                if PREVIEW:
                    for piece in _preview_tokens(sources):
                        loop.call_soon_threadsafe(queue.put_nowait, ("token", piece))
                else:
                    from mlx_lm import stream_generate

                    model, tok = _load_model()
                    prompt = tok.apply_chat_template(build_messages(question, sources, role, history), **CHAT_ARGS)
                    # Qwen's chat format puts an empty <think></think> before every answer; hold
                    # tokens back until it has passed so it never reaches the screen.
                    pending, started, so_far = "", False, ""
                    for r in stream_generate(model, tok, prompt=prompt, max_tokens=700):
                        so_far += r.text
                        if repeating(so_far):
                            break
                        if started:
                            loop.call_soon_threadsafe(queue.put_nowait, ("token", r.text))
                            continue
                        pending += r.text
                        if "</think>" in pending or (len(pending) > 12 and not pending.lstrip().startswith("<")):
                            started = True
                            first = re.sub(r"^\s*<think>[\s\S]*?</think>\s*", "", pending).lstrip()
                            if first:
                                loop.call_soon_threadsafe(queue.put_nowait, ("token", first))
            except Exception as e:  # surface model errors in the UI instead of a dead stream
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

        _gpu.submit(generate)
        answer = []
        tailored = False
        while True:
            kind, value = await queue.get()
            if kind == "done":
                # Check the finished answer against the sources it was given before calling it done.
                report = verify("".join(answer), sources, question=" ".join([text, *known]))
                try:
                    # Tailored guide answers were already scored for relevance branch by branch;
                    # scoring the same passages again in a different batch can flip near the
                    # threshold, so only the "supported" check runs for those.
                    # Guide and Jev-picked answers are assembled by code from the sources (no model
                    # wording) and their relevance was already judged, so Jev has nothing left to
                    # check there; the deterministic checks above still ran.
                    if not tailored:
                        report = await jev_check("".join(answer), sources, report, question=text)
                except Exception:  # Jev unavailable: the deterministic checks still stand
                    pass
                yield _sse("check", report.to_dict())
                yield _sse("done", {})
                return
            if kind == "token":
                answer.append(value)
            if kind == "route":
                tailored = bool(value.get("tailored") or value.get("selected"))
                yield _sse("route", value)
                continue
            if kind == "sources":
                yield _sse("sources", value)
                continue
            yield _sse(kind, value)
            if PREVIEW:
                await asyncio.sleep(0.02)

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/split")
async def split(file: UploadFile | None = File(None), text: str | None = Form(None), sample: bool = Form(False)):
    if sample:
        lease_text = SAMPLE.read_text()
    elif file is not None:
        suffix = Path(file.filename or "lease.txt").suffix.lower()
        if suffix not in {".pdf", ".txt"}:
            raise HTTPException(400, "Upload a .pdf or .txt lease")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
        try:
            lease_text = load_lease(tmp.name)
        finally:
            os.unlink(tmp.name)
    elif text:
        lease_text = text
    else:
        raise HTTPException(400, "Send a file, text, or sample=true")

    clauses = split_clauses(lease_text)
    # The opening block (title, parties, address) isn't a term to check.
    is_preamble = lambda c: c.index == 0 and re.search(r"\b(between|made by|parties)\b", c.text, re.I) and len(clauses) > 1
    if not clauses:
        raise HTTPException(422, "No text found. If this is a scanned PDF, it needs OCR first.")
    return {"facts": lease_facts(lease_text),
            "clauses": [{"index": c.index, "heading": c.heading, "text": c.text, "form": c.form_detail,
                         "skip": len(c.text) < MIN_CLAUSE_CHARS or bool(is_preamble(c)) or c.form_detail}
                        for c in clauses]}


class LeaseText(BaseModel):
    text: str


@app.post("/api/lease")
def attach_lease(req: LeaseText):
    """Split a lease pasted into the chat into terms, with names and contact details removed.

    Nothing is stored here: the browser keeps the terms for this conversation and sends them
    with each question, so a new conversation starts without them.
    """
    from realtor.lease.privacy import scrub

    clauses = [c for c in split_clauses(req.text) if not c.form_detail and len(c.text) >= MIN_CLAUSE_CHARS]
    if not clauses:
        raise HTTPException(422, "Couldn't find any lease terms in that text.")
    return {"terms": [{"heading": c.heading, "text": scrub(c.text)} for c in clauses], "facts": lease_facts(req.text)}


class ScanRequest(BaseModel):
    clauses: list[dict]


@app.post("/api/scan")
async def scan(req: ScanRequest):
    """Jev red-flag scan. Only available with TYPESAFE_API_KEY set."""
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise HTTPException(400, "Jev isn't configured (set TYPESAFE_API_KEY in .env)")
    from realtor.lease.checks import CHECKS
    from realtor.lease.clauses import Clause
    from realtor.lease.scanner import fit_label, scan_lease

    checks = {c.key: c for c in CHECKS}
    from realtor.lease.privacy import scrub

    clauses = [Clause(c["index"], c["heading"], scrub(c["text"])) for c in req.clauses]
    findings = await scan_lease(clauses)
    return {"findings": [{"index": f.clause.index, "fit": fit_label(f.rta_fit),
                          "flags": [{"key": k, "p": p, "severity": checks[k].severity, "law": checks[k].law,
                                     "question": checks[k].question} for k, p in f.flags.items()],
                          "unsure": list(f.unsure)} for f in findings]}


# ---------- Phase 4: reviewing drafted short answers ----------
DRAFTS = Path(__file__).parents[2] / "train" / "interpret" / "drafts.jsonl"


def _read_drafts() -> list[dict]:
    return [json.loads(line) for line in DRAFTS.open()] if DRAFTS.exists() else []


@app.get("/review")
def review_page():
    return FileResponse(STATIC / "review.html")


@app.get("/api/drafts")
def list_drafts():
    rows = _read_drafts()
    counts = {s: sum(r["status"] == s for r in rows) for s in ("pending", "approved", "edited", "rejected")}
    return {"counts": counts, "drafts": rows}


class DraftDecision(BaseModel):
    status: str  # "approved" | "edited" | "rejected"
    final: str | None = None
    note: str | None = None


def _draft_sources(row: dict):
    from realtor.lease.sources import Source
    return [Source(s["id"], "case" if s["id"].startswith("case:") else "guide", s["label"], s["text"]) for s in row["sources"]]


class DraftCheck(BaseModel):
    text: str


@app.post("/api/drafts/check/{draft_id:path}")
def check_draft(draft_id: str, d: DraftCheck):
    """Fact-check an edit before it's saved."""
    row = next((r for r in _read_drafts() if r["id"] == draft_id), None)
    if row is None:
        raise HTTPException(404, "No such draft")
    return verify(d.text, _draft_sources(row), question=row["question"]).to_dict()


@app.post("/api/drafts/{draft_id:path}")
def decide_draft(draft_id: str, d: DraftDecision):
    if d.status not in ("approved", "edited", "rejected"):
        raise HTTPException(400, "status must be approved, edited or rejected")
    rows = _read_drafts()
    row = next((r for r in rows if r["id"] == draft_id), None)
    if row is None:
        raise HTTPException(404, "No such draft")
    row["status"], row["final"], row["note"] = d.status, d.final if d.status == "edited" else row["draft"], d.note
    if d.status == "edited":
        row["final_check"] = verify(d.final or "", _draft_sources(row), question=row["question"]).to_dict()
    tmp = DRAFTS.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(DRAFTS)
    return {"ok": True, "final_check": row.get("final_check")}
