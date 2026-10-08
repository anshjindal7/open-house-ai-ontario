"""Numbers and dates from a filled-in Ontario Standard Form of Lease (2229E), and what they mean.

The form's field labels are fixed, so plain patterns find the values (rent, deposits, NSF charge,
term dates). Comparisons with the legal limits are done here in code, in fixed wording, so an
answer can say "your rent deposit is one month's rent, the most allowed" without a model
writing it. Nothing here is sent to Jev.

A value is used only if the form's field has one: a PDF's checkboxes don't survive as text,
so "a key deposit is not required" can't be told apart from the option that was ticked, but a
filled-in amount can.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal

MONEY = r"\$\s?(\d[\d,]*(?:\.\d{2})?)"
DATE = r"(\d{4})[-/](\d{2})[-/](\d{2})"
NSF_MAX = Decimal("20.00")       # standard form, Section 5(f) note
DISCOUNT_MAX = Decimal("0.02")   # standard form, Part G: up to two per cent of the lawful rent

PATTERNS = {
    "total_rent": r"Total Rent \(Lawful Rent\)\s*" + MONEY,
    "rent_deposit": r"will pay a rent deposit of\s*" + MONEY,
    "key_deposit": r"will pay a refundable key deposit of\s*" + MONEY,
    "nsf_charge": r"administration charge of\s*" + MONEY,
    "start": r"This tenancy starts on:?\s*" + DATE,
    "end": r"fixed length of time ending on:?\s*" + DATE,
    "discount": MONEY + r"\s+discount",
}
PAYMENT = re.compile(r"Rent will be paid using the following methods:\s*(.+?)\s*(?:\n\s*Note:|\n\s*e\)|$)", re.S)
MONTHLY = re.compile(r"day of each \(select one\):\s*Month\b", re.I)


def _money(s: str) -> Decimal:
    return Decimal(s.replace(",", ""))


def _fmt(d: Decimal) -> str:
    return f"${d:,.2f}"


def extract(text: str) -> dict:
    """The filled-in values the lease has, as strings (JSON-friendly). Missing fields are left out."""
    out: dict[str, str] = {}
    for key, pattern in PATTERNS.items():
        m = re.search(pattern, text, re.I)
        if not m:
            continue
        out[key] = "-".join(m.groups()) if key in ("start", "end") else str(_money(m.group(1)))
    if m := PAYMENT.search(text):
        method = re.sub(r"\s+", " ", m.group(1)).strip()
        if 3 <= len(method) <= 120:
            out["payment_method"] = method
    if MONTHLY.search(text):
        out["period"] = "month"
    return out


def _date(s: str) -> date | None:
    try:
        y, m, d = map(int, s.split("-"))
        return date(y, m, d)
    except (ValueError, AttributeError):
        return None


def _plus_year(d: date) -> date:
    try:
        return d.replace(year=d.year + 1)
    except ValueError:  # 29 February
        return d.replace(year=d.year + 1, day=28)


# Which facts bear on which answers: by guide, or by words in the question when no guide matched.
TOPICS = {
    "deposit": ({"damage-deposit", "move-out-damage"}, r"\bdeposits?\b|last month'?s rent"),
    "nsf": ({"late-rent"}, r"\bnsf\b|bounced|returned cheque|cheque (?:was )?returned|insufficient funds"),
    "discount": (set(), r"\bdiscount|\bon time\b|\blate (?:rent|payment|fee)"),
    "term": ({"leave-early", "lease-ends"}, r"break (?:my|the) lease|end (?:my|the) lease|lease (?:ends|is up)|move out|fixed term|early termination"),
    "increase": ({"rent-increase"}, r"\b(?:rent )?(?:increase|raise|go(?:es|ing)? up)\b"),
    "payment": ({"post-dated"}, r"post-?dated|pre-?authori[sz]ed|auto(?:matic)?[- ]?pay|pay (?:the |my )?rent by|payment method"),
}


def lines(facts: dict, scenario: str | None, question: str) -> list[str]:
    """Fixed-wording statements about the user's lease that bear on this question."""
    if not facts:
        return []
    q = question.lower()
    want = {t for t, (guides, words) in TOPICS.items() if scenario in guides or re.search(words, q)}
    out = []
    rent = _money(facts["total_rent"]) if "total_rent" in facts else None
    monthly = facts.get("period") == "month"

    if "deposit" in want and "rent_deposit" in facts:
        dep = _money(facts["rent_deposit"])
        if rent and monthly:
            if dep == rent:
                out.append(f"Your lease lists a rent deposit of {_fmt(dep)}, the same as one month's rent ({_fmt(rent)}). "
                           "That's the most a rent deposit can be.")
            elif dep < rent:
                out.append(f"Your lease lists a rent deposit of {_fmt(dep)}, less than one month's rent ({_fmt(rent)}), "
                           "so it's within the limit.")
            else:
                out.append(f"Your lease lists a rent deposit of {_fmt(dep)}, more than one month's rent ({_fmt(rent)}). "
                           "A rent deposit can't be more than one month's rent.")
        else:
            out.append(f"Your lease lists a rent deposit of {_fmt(dep)}.")
    if "deposit" in want and "key_deposit" in facts:
        out.append(f"It also lists a refundable key deposit of {_fmt(_money(facts['key_deposit']))}. A key deposit has to be "
                   "refundable and can't be more than the expected cost of replacing the keys.")
    if "nsf" in want and "nsf_charge" in facts:
        nsf = _money(facts["nsf_charge"])
        out.append(f"Your lease sets an NSF (returned payment) administration charge of {_fmt(nsf)}"
                   + (f", more than the {_fmt(NSF_MAX)} the law allows." if nsf > NSF_MAX
                      else f". The most allowed is {_fmt(NSF_MAX)}, so it's within the limit."))
    if "discount" in want and "discount" in facts and rent:
        disc = _money(facts["discount"])
        cap_low, cap_high = (rent * DISCOUNT_MAX).quantize(Decimal("0.01")), ((rent + disc) * DISCOUNT_MAX).quantize(Decimal("0.01"))
        if disc > cap_high:
            verdict = f"That's more than 2% of the rent ({_fmt(cap_high)} at most), the most a discount for paying on time can be."
        elif disc <= cap_low:
            verdict = "That's within the 2% a discount for paying on time can be."
        else:
            verdict = "That's right at the 2% limit for a discount for paying on time."
        out.append(f"Your lease describes a {_fmt(disc)} discount for paying rent on time, on rent of {_fmt(rent)}. {verdict}")
    if "term" in want and (end := _date(facts.get("end", ""))):
        out.append(f"Your lease is for a fixed term ending on {end.isoformat()}. A tenant's notice can't end the tenancy "
                   "before that date unless the landlord agrees, including through an early-exit term in the lease.")
    if "increase" in want and (start := _date(facts.get("start", ""))):
        out.append(f"Your tenancy started on {start.isoformat()}. If you moved in then, the first rent increase can take "
                   f"effect no earlier than {_plus_year(start).isoformat()}, and only with at least 90 days' written notice.")
    if "payment" in want and facts.get("payment_method"):
        out.append(f"Your lease lists rent payment by: {facts['payment_method']}. A tenant can choose to pay that way, "
                   "but can't be required to use post-dated cheques or automatic payments.")
    return out
