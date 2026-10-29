"""Rules for running Open House on a public server (OPEN_HOUSE_HOSTED=1).

On your own machine none of this applies. When hosted:

  - the local model is never loaded (servers don't have Apple Silicon); if Jev can't answer, the
    visitor is asked to try again
  - the draft-review pages (/review, /api/drafts) are switched off: they change project files
  - each visitor gets a few questions and lease scans per hour, and the whole site a daily total,
    so a bot can't run up the Jev bill
  - request bodies are size-limited
  - nothing a visitor types or pastes is stored, except feedback they choose to send

Limits are set with environment variables (see LIMITS below).
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Request
from fastapi.responses import JSONResponse

HOSTED = os.environ.get("OPEN_HOUSE_HOSTED") == "1"

# path: (per visitor per hour, environment variable that overrides it)
LIMITS = {
    "/api/ask": (int(os.environ.get("ASK_PER_HOUR", 40)), "questions"),
    "/api/scan": (int(os.environ.get("SCAN_PER_HOUR", 6)), "lease scans"),
    "/api/split": (int(os.environ.get("SPLIT_PER_HOUR", 12)), "lease uploads"),
    "/api/lease": (int(os.environ.get("SPLIT_PER_HOUR", 12)), "lease uploads"),
    "/api/lease-file": (int(os.environ.get("SPLIT_PER_HOUR", 12)), "lease uploads"),
    "/api/feedback": (int(os.environ.get("FEEDBACK_PER_HOUR", 60)), "feedback messages"),
}
DAILY_ASKS = int(os.environ.get("DAILY_ASKS", 2000))  # whole site, questions per day
MAX_BODY = {"/api/split": 10_000_000, "/api/lease-file": 10_000_000}                   # bytes; uploads can be PDFs
DEFAULT_MAX_BODY = 1_000_000
BLOCKED = ("/review", "/api/drafts")


class Limiter:
    """Sliding one-hour window per (visitor, path), plus a daily total for questions."""

    def __init__(self) -> None:
        self._hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._day, self._day_count = "", 0
        self._lock = threading.Lock()

    def allow(self, visitor: str, path: str, per_hour: int) -> bool:
        now = time.time()
        with self._lock:
            q = self._hits[(visitor, path)]
            while q and now - q[0] > 3600:
                q.popleft()
            if len(q) >= per_hour:
                return False
            if path == "/api/ask":
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if today != self._day:
                    self._day, self._day_count = today, 0
                if self._day_count >= DAILY_ASKS:
                    return False
                self._day_count += 1
            q.append(now)
            if len(self._hits) > 50_000:  # forget idle visitors
                for key in [k for k, v in self._hits.items() if not v or now - v[-1] > 3600]:
                    del self._hits[key]
            return True


limiter = Limiter()


def visitor(request: Request) -> str:
    """The visitor's address. Behind the host's proxy it's the first X-Forwarded-For entry."""
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "?")


async def guard(request: Request, call_next):
    """Middleware: applies the hosting rules above. Does nothing unless HOSTED."""
    if not HOSTED:
        return await call_next(request)
    path = request.url.path
    if path.startswith(BLOCKED):
        return JSONResponse({"detail": "Not available on the hosted version."}, status_code=404)
    if request.method == "POST":
        size = int(request.headers.get("content-length") or 0)
        if size > MAX_BODY.get(path, DEFAULT_MAX_BODY):
            return JSONResponse({"detail": "That's too large to read here. Try a shorter text or a smaller file."},
                                status_code=413)
        if path in LIMITS:
            per_hour, what = LIMITS[path]
            if not limiter.allow(visitor(request), path, per_hour):
                return JSONResponse({"detail": f"You've reached the limit of {per_hour} {what} an hour on this test "
                                               "version (or today's total for the site). Please try again later."},
                                    status_code=429)
    return await call_next(request)


# ---------- feedback ----------
FEEDBACK_DIR = Path(os.environ.get("FEEDBACK_DIR", Path(__file__).parents[2] / "feedback"))
_feedback_lock = threading.Lock()
_scheduler = None


def start_feedback_sync() -> None:
    """On Hugging Face Spaces the disk is wiped on restart. With HF_TOKEN and FEEDBACK_DATASET set,
    feedback is copied every few minutes to a private dataset repo so it isn't lost."""
    global _scheduler
    repo, token = os.environ.get("FEEDBACK_DATASET"), os.environ.get("HF_TOKEN")
    if not (repo and token) or _scheduler is not None:
        return
    from huggingface_hub import CommitScheduler

    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    _scheduler = CommitScheduler(repo_id=repo, repo_type="dataset", folder_path=FEEDBACK_DIR,
                                 path_in_repo="feedback", every=5, private=True, token=token)


def save_feedback(row: dict) -> None:
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    row = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **row}
    lock = _scheduler.lock if _scheduler is not None else _feedback_lock
    with lock:
        with (FEEDBACK_DIR / "feedback.jsonl").open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
