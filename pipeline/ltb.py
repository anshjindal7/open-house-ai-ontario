"""Turn downloaded LTB orders into redacted, structured case records.

    python -m pipeline.ltb

Contains information licensed under the Open Government Licence - Ontario
(LTB Order Catalogue, https://data.ontario.ca/dataset/ltb-order-catalogue).

Each record keeps: issues raised, the Board's numbered reasons, what was ordered, RTA
sections cited, who the outcome favoured, and any lease wording the order quotes. Names,
addresses, phone numbers and emails are removed.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .fetch_ltb import pdf_path, wanted_rows

OUT = Path(__file__).parents[1] / "corpus" / "ltb_cases.jsonl"
NAME_COLS = [
    "Landlord Name/Nom du locateur", "Landlord Agent Name/Nom du représentant du locateur",
    "Tenant Name/Nom du locataire", "Former Tenant Name/Nom de l’ancien locataire",
    "Sub-Tenant Name/Nom du sous-locataire", "Occupant Names/Nom de l’occupant",
    "Co-op Member Name/Nom du membre de la cooperative",
]
ADDRESS_COLS = ["Rental Unit Address/Adresse du logement locatif", "Complex Address/Adresse du complexe"]

ABANDONED = re.compile(r"application (has been|is) abandoned|did not attend the hearing to support|"
                       r"dismissed as abandoned", re.I)
NOT_ENGLISH = re.compile(r"\bOrdonnance\b|\bLocataire\b|\bLocateur\b")
SECTION_CITE = re.compile(
    r"\b(?:sections?|subsections?|ss?\.)\s*(\d{1,3}(?:\.\d)?)(?:\s*\(\d+\))?(?=[^\d]|$)", re.I)
QUOTE = re.compile(r"[“\"]([^”\"]{40,700})[”\"]")
# The quote must be introduced as lease wording ("the lease states", "clause 12 provides", ...).
LEASE_INTRO = re.compile(
    r"\b(lease|tenancy agreement|rental agreement|addendum|clause|term)\b[^.“\"]{0,80}"
    r"\b(states?|stated|provides?|provided|says|said|reads?|includes?|included|contains?|required|requires|that)\b[^.“\"]{0,40}$",
    re.I)
NOT_A_TERM = re.compile(r"\bthis Act\b|\bthe Board\b|\bN12\b|\bI have\b|\bwe\b|\byou\b", re.I)
TERM_WORDS = re.compile(r"\b(tenants?|landlords?|lessee|lessor|shall|must|will|rent|premises|unit)\b", re.I)
PHONE = re.compile(r"\(?\b\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b")
EMAIL = re.compile(r"\S+@\S+\.\w+")
POSTAL = re.compile(r"\b[A-Z]\d[A-Z] ?\d[A-Z]\d\b")
# spaCy misses many "Ms. Surname" mentions; titles are a reliable signal on their own.
TITLED_NAME = re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Mx|Dr)\.?\s+[A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+)?")
PAGE_FURNITURE = re.compile(r"File Number: LTB-[\w-]+\s*|Order Page \d+ of \d+\s*|Tribunals Ontario\s*")

_nlp = None
# Official names spaCy sometimes mistakes for people ("Residential" → [name]).
_PROTECTED_NAMES = (r"Residential Tenancies Act|Tenant Protection Act|Landlord and Tenant Board|Human Rights Code|"
                       r"Divisional Court|Superior Court of Justice|Court of Appeal|Ontario Works|Ontario Disability Support|"
                       r"Statutory Powers Procedure Act|Interpretation Guideline|Rules of Procedure|Sheriff|Court Enforcement Office")
# PDF text breaks lines anywhere, so "Residential\nTenancies Act" must match too.
PROTECTED = re.compile(_PROTECTED_NAMES.replace(" ", r"\s+"), re.I)


def _spacy():
    global _nlp
    if _nlp is None:
        import spacy

        _nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    return _nlp


def redact(text: str, row: dict) -> str:
    # Shield official names from every step below, then put them back at the end.
    shielded: list[str] = []
    def _shield(m):
        shielded.append(m.group(0))
        return f"\x00{len(shielded) - 1}\x00"
    text = PROTECTED.sub(_shield, text)
    text = _redact(text, row)
    return re.sub(r"\x00(\d+)\x00", lambda m: shielded[int(m.group(1))], text)


# A party that is a business (incl. numbered Ontario companies and co-ops). Shown as
# "[organization]" so a reader can tell a company landlord from an individual.
COMPANY = re.compile(r"\b(?:inc|ltd|limited|corp|corporation|lp|llp|llc|co-?op(?:erative)?|homes|properties|"
                     r"property|management|holdings|realty|rentals?|developments?|group|trust|reit|housing|"
                     r"apartments|residences|living|capital|investments?|enterprises)\b\.?|^\d{6,8}\s+ontario\b", re.I)
NUMBERED_COMPANY = re.compile(r"\b\d{6,8}\s+Ontario\s+(?:Inc|Ltd|Limited|Corp|Corporation)\b\.?", re.I)
# One-word brand names with no "Inc"/"Ltd" (a property manager called, say, "NorthStarRez").
BRAND = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+){2,}\b")


def _redact(text: str, row: dict) -> str:
    names, companies = set(), set()
    for col in NAME_COLS:
        for name in re.split(r"[;,]| and ", row.get(col, "")):
            name = name.strip()
            if len(name) <= 2:
                continue
            if COMPANY.search(name):
                companies.add(name)  # whole name only: its words ("Homes") aren't anyone's name
            else:
                names.add(name)
                names.update(p for p in name.split() if len(p) > 2 and p[0].isupper())
    for name in sorted(companies, key=len, reverse=True):
        pattern = r"\s+".join(map(re.escape, name.split()))
        text = re.sub(rf"{pattern}", "[organization]", text, flags=re.I)
    text = NUMBERED_COMPANY.sub("[organization]", text)
    text = BRAND.sub("[organization]", text)
    for col in ADDRESS_COLS:
        addr = row.get(col, "").split(",")[0].strip()
        if addr:
            text = re.sub(re.escape(addr), "[address]", text, flags=re.I)
            street = re.sub(r"^[\w-]*\d[\w-]*\s+", "", addr)  # drop unit/street number
            if len(street) > 5:
                text = re.sub(re.escape(street), "[address]", text, flags=re.I)
    for name in sorted(names, key=len, reverse=True):
        if name.lower() not in {"the", "inc", "ltd", "corp", "limited", "landlord", "tenant"}:
            # Full names match any case and any spacing ("rayon\nmoodie"); single words only as capitalized.
            pattern = r"\s+".join(map(re.escape, name.split()))
            text = re.sub(rf"\b{pattern}\b", "[name]", text, flags=re.I if " " in name else 0)
    text = PHONE.sub("[phone]", EMAIL.sub("[email]", POSTAL.sub("", text)))
    # A company name whose first words were caught as a person ("[name] Ltd.") is a company.
    text = re.sub(r"\[name\](?:\s+\[name\])*\s+(?:Inc|Ltd|Limited|Corp|Corporation|LP|Homes|Properties|Management)\b\.?",
                  "[organization]", text)
    text = TITLED_NAME.sub("[name]", text)

    doc = _spacy()(text)
    for ent in reversed(doc.ents):
        if "\x00" in ent.text:
            continue
        if ent.label_ == "PERSON" and ent.text not in {"Tenant", "Landlord", "Tenants", "Landlords"}:
            text = text[:ent.start_char] + "[name]" + text[ent.end_char:]
    return text


def numbered_paragraphs(block: str) -> list[str]:
    parts = re.split(r"(?:^|\s)(\d{1,3})\.\s+(?=[A-Z“\"])", block)
    paras, expected = [], 1
    for i in range(1, len(parts) - 1, 2):
        if int(parts[i]) == expected:
            paras.append(re.sub(r"\s+", " ", parts[i + 1]).strip())
            expected += 1
        elif paras:
            paras[-1] += f" {parts[i]}. {re.sub(r'\s+', ' ', parts[i + 1]).strip()}"
    return paras


def favoured(ordered: str, applications: str) -> str:
    o = ordered.lower()
    tenant_app = all(a.strip().startswith("T") for a in applications.split(";"))
    # Whose application was dismissed decides it, when the order says so: a dismissed landlord's
    # application is a tenant win, and the reverse. (Orders can deal with both sides' applications.)
    landlord_lost = bool(re.search(r"landlords?['’]?s?['’]? (?:\w+ )?applications? (?:\S+ )?(?:is|are) dismissed", o))
    tenant_lost = bool(re.search(r"tenants?['’]?s?['’]? (?:\w+ )?applications? (?:\S+ )?(?:is|are) dismissed", o))
    if landlord_lost != tenant_lost:
        return "tenant" if landlord_lost else "landlord"
    if landlord_lost and tenant_lost:
        return "split"
    if "application is dismissed" in o or "applications are dismissed" in o:
        return "landlord" if tenant_app else "tenant"
    if re.search(r"withdrawn|request to review .* (is|are) denied|order is confirmed", o):
        return "none"
    if re.search(r"(landlords?|purchasers?) shall (pay|rebate|refund|complete|repair|not)|abatement|rent reduction", o):
        return "tenant"
    if re.search(r"tenants? shall pay|tenancy .* terminated|evict", o):
        return "landlord"
    if "dismissed" in o and ("granted" in o or "shall pay" in o):
        return "split"
    return "unclear"


def parse_order(args: tuple[Path, dict]) -> dict | None:
    from pypdf import PdfReader

    path, row = args
    try:
        raw = "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
    except Exception:
        return None
    if len(raw) < 3000 or ABANDONED.search(raw) or NOT_ENGLISH.search(raw[:1500]):
        return None

    text = PAGE_FURNITURE.sub(" ", raw)
    text = re.sub(r"Between:.*?(?=\S+ \(the ['‘]?(Tenant|Landlord))", "", text, count=1, flags=re.S)
    text = re.sub(r"Date Issued.*", "", text, flags=re.S)  # signature block and Board contact info

    m_reasons = re.search(r"(Determinations|Reasons):?", text)
    m_order = re.search(r"It is ordered that:?", text)
    if not (m_reasons and m_order and m_reasons.start() < m_order.start()):
        return None

    header = re.sub(r"\s+", " ", redact(text[:m_reasons.start()], row))
    reasons_text = redact(text[m_reasons.end():m_order.start()], row)
    ordered_text = redact(text[m_order.end():], row)
    citation = re.search(r"\d{4} ONLTB \d+", header)
    applied = re.search(r"\S+ \(the ['‘]?\w+['’]?\) applied.*?(?=This application was heard|$)", header)
    issues = (applied.group(0) if applied else header[-1200:]).strip()
    reasons = numbered_paragraphs(reasons_text)
    ordered = numbered_paragraphs(ordered_text) or [re.sub(r"\s+", " ", ordered_text)[:1500]]
    if len(reasons) < 3 or favoured(" ".join(ordered), row["Applications/Requêtes"]) == "none":
        return None

    body = " ".join(reasons)
    lease_quotes = []
    for m in QUOTE.finditer(body):
        quote, intro = m.group(1).strip(), body[max(0, m.start() - 160):m.start()]
        if LEASE_INTRO.search(intro) and TERM_WORDS.search(quote) and not NOT_A_TERM.search(quote):
            lease_quotes.append(quote)

    return {
        "file_number": row["File Number/Numéro de dossier"],
        "doc_id": row["Document ID/Identifiant du document"],
        "applications": row["Applications/Requêtes"],
        "order_date": row["Order Date/Date de l’ordonnance"],
        "citation": citation.group(0) if citation else None,
        "issues": issues[-1500:],
        "reasons": reasons,
        "ordered": ordered,
        "rta_sections": sorted({m.group(1) for m in SECTION_CITE.finditer(body)}, key=float),
        "favoured": favoured(" ".join(ordered), row["Applications/Requêtes"]),
        "lease_quotes": lease_quotes,
        "source": "LTB Order Catalogue, Open Government Licence - Ontario",
    }


def main() -> None:
    # The catalogue can list one document several times (one row per application or party).
    # Merge them so every name listed for the document is redacted, and parse it once.
    merged: dict[str, dict] = {}
    for r in wanted_rows():
        doc = r["Document ID/Identifiant du document"]
        if doc not in merged:
            merged[doc] = dict(r)
        else:
            for col in NAME_COLS + ADDRESS_COLS:
                if r.get(col) and r[col] not in merged[doc].get(col, ""):
                    merged[doc][col] = "; ".join(x for x in (merged[doc].get(col), r[col]) if x)
    jobs = [(pdf_path(doc), r) for doc, r in merged.items()]
    jobs = [j for j in jobs if j[0].exists()]
    OUT.parent.mkdir(exist_ok=True)
    kept = 0
    with ProcessPoolExecutor() as pool, OUT.open("w") as f:
        for rec in pool.map(parse_order, jobs, chunksize=16):
            if rec:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
    print(f"{kept} substantive cases from {len(jobs)} orders -> {OUT}")


if __name__ == "__main__":
    main()
