"""Download LTB orders from Ontario's open-data LTB Order Catalogue.

    python -m pipeline.fetch_ltb

Source: https://data.ontario.ca/dataset/ltb-order-catalogue
Licence: Open Government Licence - Ontario. Attribution required:
"Contains information licensed under the Open Government Licence - Ontario."
Orders include personal names and addresses; pipeline/ltb.py redacts them before training.
"""

from __future__ import annotations

import asyncio
import os
import csv
import re
import sys
from pathlib import Path

import httpx

SOURCES = Path(__file__).parents[1] / "sources"
CSV_PATH = SOURCES / "ltb_orders_2026.csv"
# New downloads go here. LTB_PDF_DIR can name extra folders (separated by ":") that already hold
# PDFs from an earlier download, so they aren't fetched again.
PDF_DIR = SOURCES / "ltb_pdfs"
EXTRA_DIRS = [Path(p) for p in os.environ.get("LTB_PDF_DIR", "").split(":") if p]


def pdf_path(doc_id: str) -> Path:
    """Where a document's PDF is, or where it should be downloaded to."""
    for d in [PDF_DIR, *EXTRA_DIRS]:
        p = d / f"{doc_id}.pdf"
        if p.exists() and p.stat().st_size > 0:
            return p
    return PDF_DIR / f"{doc_id}.pdf"

# Tenant applications and A1/A2 are where lease terms get tested.
# T1 illegal charges, T2 tenant rights, T3 rent reduction, T5 bad-faith notice, T6 maintenance,
# A1 whether the Act applies, A2 sublet/unauthorized occupant.
APP_TYPES = {"T1", "T2", "T3", "T5", "T6", "A1", "A2"}
DOC_TYPES = {"Order", "Amended Order", "Review Order"}
URL_COL = "ContentDownload URL/URL de téléchargement du contenu"


def wanted_rows() -> list[dict]:
    csv.field_size_limit(sys.maxsize)
    rows = []
    for r in csv.DictReader(CSV_PATH.open(encoding="utf-8-sig")):
        apps = {a.strip() for a in r["Applications/Requêtes"].split(";")}
        m = re.search(r'"(https[^"]+\.pdf)"', r[URL_COL])
        if apps & APP_TYPES and r["Document Type/Type de document"] in DOC_TYPES and m:
            r["pdf_url"] = m.group(1)
            rows.append(r)
    return rows


async def fetch_all(rows: list[dict], concurrency: int = 6) -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    done = failed = 0

    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:

        async def one(r: dict) -> None:
            nonlocal done, failed
            out = pdf_path(r["Document ID/Identifiant du document"])
            if out.exists() and out.stat().st_size > 0:
                done += 1
                return
            async with sem:
                try:
                    resp = await client.get(r["pdf_url"])
                    resp.raise_for_status()
                    out.write_bytes(resp.content)
                    done += 1
                except httpx.HTTPError as e:
                    failed += 1
                    print(f"failed {out.name}: {e}", file=sys.stderr)
            if (done + failed) % 250 == 0:
                print(f"{done + failed}/{len(rows)} ({failed} failed)", flush=True)

        await asyncio.gather(*(one(r) for r in rows))
    print(f"finished: {done} downloaded, {failed} failed")


if __name__ == "__main__":
    rows = wanted_rows()
    print(f"{len(rows)} orders to fetch")
    asyncio.run(fetch_all(rows))
