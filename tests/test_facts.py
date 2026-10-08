"""Values from a filled-in standard form (made-up values), and the fixed-wording comparisons."""

from realtor.lease.facts import extract, lines

FORM = """4. Term of Tenancy Agreement
This tenancy starts on: 2025-05-01
Date (yyyy/mm/dd)
a fixed length of time ending on: 2026-04-30
5. Rent
a) Rent is to be paid on the first (e.g., first, second, last) day of each (select one):
Month
Base rent for the rental unit $1,900.00
Total Rent (Lawful Rent) $2,000.00
d) Rent will be paid using the following methods:
E-transfer or cheque
Note:
The tenant cannot be required to pay rent by post-dated cheques or automatic payments, but can choose to do so.
f) If the tenant's cheque is returned because of non-sufficient funds (NSF), the tenant will have to pay the landlord's
administration charge of $35.00 plus any NSF charges made by the landlord's bank.
8. Rent Deposit
Select one:
A rent deposit is not required
or
The tenant will pay a rent deposit of $2,000.00. This can only be applied to the rent for the last rental period of the tenancy.
9. Key Deposit
The tenant will pay a refundable key deposit of $50 to cover the cost of replacing the keys
7. Rent Discounts
The lawful rent includes a $50 discount provided that the rent is received on or before the date the rent is due."""


def test_extract_reads_the_filled_in_fields():
    f = extract(FORM)
    assert f["total_rent"] == "2000.00" and f["rent_deposit"] == "2000.00" and f["key_deposit"] == "50"
    assert f["nsf_charge"] == "35.00" and f["start"] == "2025-05-01" and f["end"] == "2026-04-30"
    assert f["payment_method"] == "E-transfer or cheque" and f["period"] == "month" and f["discount"] == "50"


def test_lines_compare_with_the_legal_limits():
    f = extract(FORM)
    dep = " ".join(lines(f, "damage-deposit", "How much deposit can my landlord ask for?"))
    assert "the same as one month's rent ($2,000.00)" in dep and "refundable key deposit of $50.00" in dep
    assert "more than the $20.00 the law allows" in " ".join(lines(f, "late-rent", "My rent is late"))
    assert "no earlier than 2026-05-01" in " ".join(lines(f, "rent-increase", "Can my rent go up?"))
    assert "fixed term ending on 2026-04-30" in " ".join(lines(f, None, "Can I break my lease?"))
    assert "more than 2% of the rent" in " ".join(lines(f, None, "Is the discount for paying on time allowed?"))
    assert lines(f, None, "My sink is broken") == []  # nothing in the lease's numbers bears on this


def test_a_blank_form_gives_no_values():
    assert extract("The tenant will pay a rent deposit of $ . Total Rent (Lawful Rent)") == {}
