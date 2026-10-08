"""Remove personal details from lease text before it leaves the app or is processed.

Real leases carry names, emails, phone numbers and addresses (guarantor pages, contact sections).
None of that is needed to check a term against the law, so it's replaced before a clause is
scanned by Jev or explained.
"""

from __future__ import annotations

import re

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE = re.compile(r"(?:\+?1[\s.-]?)?\(?\b\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")
POSTAL = re.compile(r"\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b")
NAME_FIELD = re.compile(r"\b((?:TENANT|GUARANTOR|LANDLORD|RESIDENT|OCCUPANT)['’]?S?\s+(?:LEGAL\s+)?NAME\s*:)[^\n]*", re.I)
ADDRESS = re.compile(r"\b\d{1,6}(?:-\d{1,6})?\s+(?:[A-Z][a-z]+\s){1,3}(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Dr|Drive|Cres|Crt|Court|Way|Lane|Ln|Pl)\b\.?(?:\s+[NSEW]\b)?", re.U)
BETWEEN = re.compile(r"(\bbetween\s+)([A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+){1,3})(\s+and\b)")


NOT_NAMES = {"tenant", "tenants", "landlord", "landlords", "guarantor", "board", "act", "ltb", "rta"}
_nlp = None


def _names(text: str) -> str:
    """People's names written in free text ("my landlord Priya said…"), found with spaCy."""
    global _nlp
    if _nlp is None:
        import spacy

        _nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    for ent in reversed(_nlp(text).ents):
        if ent.label_ == "PERSON" and ent.text.lower() not in NOT_NAMES and "[" not in ent.text:
            text = text[:ent.start_char] + "[name]" + text[ent.end_char:]
    return text


def scrub(text: str) -> str:
    text = NAME_FIELD.sub(r"\1 [name]", text)
    text = EMAIL.sub("[email]", text)
    text = PHONE.sub("[phone]", text)
    text = POSTAL.sub("[postal code]", text)
    text = ADDRESS.sub("[address]", text)
    text = BETWEEN.sub(r"\1[name]\3", text)
    return _names(text)
