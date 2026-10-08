from pathlib import Path

from realtor.lease.clauses import MAX_CLAUSE_CHARS, split_clauses

SAMPLE = Path(__file__).parents[1] / "realtor" / "data" / "sample_lease.txt"


def test_sample_lease_splits_into_numbered_clauses():
    clauses = split_clauses(SAMPLE.read_text())
    headings = [c.heading for c in clauses]
    assert headings[0] == "RESIDENTIAL LEASE AGREEMENT"
    assert any(h.startswith("7. LANDLORD ACCESS") for h in headings)
    assert len(clauses) == 16  # title block + 15 numbered clauses


def test_long_clause_is_broken_up():
    text = "1. LONG. " + "Tenant agrees to a very long term. " * 100
    clauses = split_clauses(text)
    assert len(clauses) > 1
    assert all(len(c.text) <= MAX_CLAUSE_CHARS + 50 for c in clauses)


STANDARD_LIKE = """Residential Tenancy Agreement (Standard Form of Lease)
2229E (2020/12)
1. Parties to the Agreement
Landlord's Legal Name Example Landlord Inc.
5. Rent
The tenant will pay the following rent: Base rent $1,000.00
Page 2 of 14
15. Additional Terms
An additional term cannot take away a right under the Residential Tenancies Act, 2006.
17. Signatures
By signing this agreement, the landlord(s) and the tenant(s) agree to follow its terms.
Appendix: General Information
A. When to Use This Form
This form must be used for most residential tenancy agreements.
Guide to the Standard Lease
www.ontario.ca/standardlease
2229E (2020/12) Page 15 of 14
COVENANTS
ANNUAL
INCREASE
The Tenant agrees that rent will increase following each twelve-month period by the guideline amount.
"""


def test_standard_form_appendix_and_furniture_are_dropped():
    clauses = split_clauses(STANDARD_LIKE)
    text = " ".join(c.text for c in clauses)
    assert "When to Use This Form" not in text and "Page 2 of 14" not in text and "standardlease" not in text


def test_form_details_are_marked_and_terms_are_not():
    clauses = {c.heading.split(".")[0]: c for c in split_clauses(STANDARD_LIKE) if c.heading[:2].strip(". ").isdigit()}
    assert clauses["1"].form_detail and clauses["17"].form_detail
    assert not clauses["5"].form_detail and not clauses["15"].form_detail


def test_heading_fragments_join_their_paragraph():
    last = split_clauses(STANDARD_LIKE)[-1]
    assert last.heading.startswith("COVENANTS ANNUAL INCREASE") and "twelve-month" in last.text


ADDITIONAL_TERMS = """15. Additional Terms
A. B. C. D. The Tenant agrees to abide by the rules and regulations of the building and is deemed to have given notice of them to every person living in the unit, including guests who stay over.
The Tenant shall be responsible for all legal costs on a solicitor and own client basis incurred by the Landlord as a result of any breach by the Tenant of any term of this agreement.
If the Tenant locks themselves out, the Tenant shall be responsible for the cost of re-entry including locksmith charges and any service charge payable to the Landlord for attending.
i. ii. iii. The Tenant agrees not to allow guests to stay more than three consecutive nights in any month without the written approval of the Landlord and all other named tenants.
"""


def test_additional_terms_split_into_separate_terms():
    pieces = split_clauses(ADDITIONAL_TERMS)
    texts = [p.text for p in pieces]
    assert len(pieces) >= 3
    assert not any("A. B. C." in t or "i. ii." in t for t in texts)
    assert any(t.startswith("The Tenant shall be responsible for all legal costs") for t in texts)
