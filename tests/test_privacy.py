from realtor.lease.privacy import scrub

SAMPLE = """TENANT NAME: Alex Example
GUARANTOR NAME: Sam Example
RENTED ADDRESS: unit 12 100 Queen St N Waterloo, ON A1A 1A1
Email: someone@example.com Phone: +1 555-555-0123
This Amendment is entered into on 2026-01-01 between Alex Example and Some Landlord LP is binding.
The Tenant shall pay rent of $1,000.00 on the first day of each month."""


def test_personal_details_are_removed_and_terms_kept():
    out = scrub(SAMPLE)
    for gone in ("Alex Example", "Sam Example", "someone@example.com", "555-555-0123", "A1A 1A1", "100 Queen St"):
        assert gone not in out, gone
    assert "$1,000.00 on the first day of each month" in out


def test_free_text_names_in_a_question_are_removed():
    out = scrub("My landlord Priya Sharma said I have to pay for the broken fridge. Is that right?")
    assert "Priya" not in out and "broken fridge" in out
