"""The checklist Jev runs against every lease clause.

The checks are neutral: each one asks whether a term may conflict with Ontario's Residential
Tenancies Act (RTA). A void term is a problem for both sides: a tenant may not know their
rights, and a landlord can't enforce it. `law` names the rule each check is based on.

Jev isn't fine-tuned: improve results by sharpening these questions and thresholds, then
re-running `evals/run_eval.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

from typesafe_sdk import Choice, Noul, Score


@dataclass(frozen=True)
class Check:
    key: str
    severity: str  # "high": likely void under the RTA | "medium": often disputed | "low": worth discussing
    law: str
    question: str
    threshold: float = 0.5  # Noul probability at/above which we flag it


CHECKS: list[Check] = [
    Check("no_pets", "high", "RTA s. 14",
          "Does this clause prohibit pets or animals in the unit or the building?"),
    Check("extra_deposit", "high", "RTA ss. 105-106",
          "Does this clause require a deposit other than a last month's rent deposit or a refundable key "
          "deposit, such as a damage, security, pet or cleaning deposit?"),
    Check("extra_fees", "high", "RTA s. 134",
          "Does this clause charge money beyond rent and permitted deposits, such as an application, "
          "administration, move-in, amenity or key fee, or make a key deposit non-refundable?"),
    Check("rent_increase_rules", "high", "RTA ss. 116, 119, 120",
          "Does this clause allow the rent to be increased with less than 90 days' written notice, more "
          "than once in 12 months, or by more than the provincial guideline?"),
    Check("entry_without_notice", "high", "RTA ss. 25-27",
          "Does this clause let the landlord enter the unit without 24 hours' written notice, other than "
          "in an emergency or with the tenant's consent at the time?"),
    Check("early_exit_penalty", "high", "RTA ss. 15-16",
          "Does this clause make the rest of the rent, a set penalty, or the deposit due or forfeited if "
          "the tenant leaves early or breaks a term?"),
    Check("repairs_shifted", "high", "RTA ss. 20, 34",
          "Does this clause make the tenant responsible for repairs the landlord must do, such as major "
          "systems, appliances, or damage the tenant didn't cause?"),
    Check("post_dated_cheques", "high", "RTA s. 108",
          "Does this clause require post-dated cheques or automatic payments for rent?"),
    Check("must_move_out_at_end", "high", "RTA ss. 37-38",
          "Does this clause say the tenant must move out when the fixed term ends, or that the tenancy "
          "ends automatically without a notice or agreement?"),
    Check("waives_the_act", "high", "RTA ss. 3-4",
          "Does this clause ask either party to give up rights under the Residential Tenancies Act, or "
          "say the Act doesn't apply?"),
    Check("vital_services", "high", "RTA s. 21",
          "Does this clause allow heat, water, electricity or fuel to be shut off or reduced, for example "
          "if rent is late?"),
    Check("lock_changes", "medium", "RTA ss. 24, 35",
          "Does this clause allow the locks to be changed without giving the other party a key or "
          "getting their consent?"),
    Check("guest_limits", "medium", "Standard Form of Lease, Part Q",
          "Does this clause ban guests, limit how long guests may stay, require permission for guests, "
          "or charge for guests or roommates?"),
    Check("late_fees", "medium", "RTA s. 134",
          "Does this clause charge a late fee, penalty or interest on late rent, other than a returned "
          "cheque (NSF) charge?"),
    Check("sublet_ban", "medium", "RTA ss. 95, 97",
          "Does this clause ban subletting or assigning the lease outright?"),
    Check("auto_renewal", "medium", "RTA s. 38",
          "Does this clause automatically renew the lease for another fixed term (such as another year) "
          "unless someone gives notice, instead of the tenancy simply continuing month to month?"),
    Check("liability_waiver", "medium", "RTA ss. 4, 20",
          "Does this clause say the landlord is not liable for damage, injury or loss in general, or make "
          "the tenant indemnify the landlord, including for problems the landlord is responsible for?"),
    Check("no_rent_abatement", "high", "RTA ss. 4, 30",
          "Does this clause say the tenant gets no rent reduction or compensation while repairs, pest "
          "treatment or other problems the landlord is responsible for are happening?"),
    Check("legal_costs", "medium", "LTB Interpretation Guideline 3 (Costs)",
          "Does this clause make the tenant pay the landlord's legal fees or legal costs (for example on a "
          "solicitor and client basis) if there is a dispute or breach?"),
]

RED_FLAGS = CHECKS  # older name, kept for evals/run_eval.py

CLAUSE_TOPICS = {
    "rent_payment": "Rent amount, due dates, payment methods.",
    "fees": "Late fees, extra charges, fines.",
    "deposit": "Deposits and how they are returned.",
    "term_renewal": "Lease length, renewal, ending the lease.",
    "maintenance": "Repairs, upkeep, who fixes what.",
    "access_privacy": "Landlord entry, inspections, locks.",
    "use_rules": "Pets, guests, noise, smoking, subletting.",
    "liability_legal": "Insurance, liability, disputes, legal rights.",
    "utilities": "Who pays which utilities.",
    "other": "Anything else, including signatures and definitions.",
}

RTA_FIT_LEVELS = [
    "Consistent with the RTA: a standard term.",
    "Unusual: worth discussing before signing.",
    "Often disputed at the Landlord and Tenant Board.",
    "Likely void or unenforceable under the RTA.",
]


def build_questions() -> dict:
    questions: dict = {
        "topic": Choice(instructions="What is this lease clause mainly about?", criteria=CLAUSE_TOPICS),
        "rta_fit": Score(instructions="How well does this clause fit Ontario's Residential Tenancies Act?",
                         criteria=RTA_FIT_LEVELS),
    }
    for check in CHECKS:
        questions[check.key] = Noul(instructions=check.question)
    return questions

# Plain names for the checks, as the app shows them.
LABELS = {
    "no_pets": "No-pets rule", "extra_deposit": "Deposit not allowed", "extra_fees": "Extra fee",
    "rent_increase_rules": "Rent increase rules", "entry_without_notice": "Entry without notice",
    "early_exit_penalty": "Early-exit penalty", "repairs_shifted": "Repairs shifted to the tenant",
    "post_dated_cheques": "Required post-dated cheques", "must_move_out_at_end": "Move-out at term end",
    "waives_the_act": "Waives the Act", "vital_services": "Vital services", "lock_changes": "Lock changes",
    "guest_limits": "Guest limits", "late_fees": "Late fees", "sublet_ban": "Sublet ban",
    "auto_renewal": "Automatic renewal", "liability_waiver": "Liability waiver",
    "no_rent_abatement": "No rent reduction", "legal_costs": "Legal costs",
}
