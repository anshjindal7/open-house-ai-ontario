"""The user's own lease in chat: finding the matching terms, and usable decision reasons."""

from realtor.lease.sources import Source, case_why, lease_terms

TERMS = [  # made-up wording
    {"heading": "LOCKS", "text": "The Tenant agrees to notify the Landlord in writing should any door lock become inoperative."},
    {"heading": "PETS", "text": "No pets of any kind are permitted in the suite common areas without approval."},
    {"heading": "GUESTS", "text": "Guests may not stay more than three consecutive nights in any month without consent."},
]


def test_lease_terms_finds_the_matching_term():
    assert lease_terms(TERMS, "my front door lock is broken", k=1)[0].label == "Your lease · LOCKS"
    assert lease_terms(TERMS, "can my friend stay over for a week", k=1)[0].label == "Your lease · GUESTS"


def test_case_why_skips_cut_off_and_lead_in_sentences():
    text = ("Application: The tenant applied about repairs.\nResult: decided in the tenant's favour.\n"
            "Ordered: The landlord shall pay the tenant an abatement of rent of $500.\n"
            "Key reasons: This application is before the Board under section 20(1) of the Act which reads: "
            "I find that the landlord did not repair the door for three months after being told in writing.")
    src = Source("case:x", "case", "LTB decision X", text, side="tenant")
    k = case_why(src)
    assert src.sentences[k - 1].startswith("I find")
    lead_in = next(i for i, t in enumerate(src.sentences, 1) if t.endswith("reads:"))
    assert case_why(src, lead_in) == k  # a picked lead-in is replaced with a real reason
