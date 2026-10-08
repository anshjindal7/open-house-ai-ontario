"""Download the LTB's official Interpretation Guidelines and brochures.

    python -m pipeline.fetch_guidance

Source: Tribunals Ontario, https://tribunalsontario.ca/ltb/rules-practice-directions-guidelines/
and https://tribunalsontario.ca/ltb/resources/. These are the Board's own explanations of how it
usually interprets the Residential Tenancies Act, written for both landlords and tenants.
Hearing-procedure documents (adjournments, reviews, public access) are skipped.
"""

from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import quote

import httpx

BASE = "https://tribunalsontario.ca/documents/ltb/"
OUT = Path(__file__).parents[1] / "sources" / "ltb_guidance"

GUIDELINES = [
    "02 - Payment into the Board", "04 - Abandonment of a Rental Unit", "05 - Breach of Maintenance Obligations",
    "06 - Tenants Rights", "07 - Relief from Eviction - Refusing or Delaying an Eviction",
    "09 - Eviction for an Illegal Act or Business", "10 - Procedural Issues Regarding Eviction Applications",
    "11 - Rent Arrears", "12 - Eviction for Personal Use", "14 - Applications for Rent Increases above the Guideline",
    "16 - Administrative Fines", "17 - Human Rights", "19 - The Landlords Right of Entry into a Rental Unit",
    "21 - Landlords Tenants Occupants and Residential Tenancies",
]
BROCHURES = [
    "A Guide to the Residential Tenancies Act", "Automatic Rent Reductions and Tax Decreases",
    "Collecting Money a Former Tenant Owes", "How a Landlord Can End a Tenancy", "How a Tenant Can End Their Tenancy",
    "If You are Locked out of Your Unit Information for Tenants", "If a Tenant Does Not Pay Rent",
    "Information about Applications for a Rent Increase Above the Guideline",
    "Information for Landlords and Tenants About Suite Meters", "Information for New Tenants",
    "Maintenance and Repairs", "Paying Money into the Landlord and Tenant Board",
    "Property Left Behind When a Tenant Moves Out", "Rules_for_Air_Conditioners_in_Rental_Units",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [("Interpretation Guidelines", g, "guideline") for g in GUIDELINES] + \
           [("Brochures", b, "brochure") for b in BROCHURES]
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for folder, name, kind in jobs:
            out = OUT / f"{kind}--{name}.html"
            if out.exists():
                continue
            r = client.get(BASE + quote(f"{folder}/{name}.html"))
            print(r.status_code, kind, name)
            if r.status_code == 200:
                out.write_bytes(r.content)
            time.sleep(0.5)  # be polite


if __name__ == "__main__":
    main()
