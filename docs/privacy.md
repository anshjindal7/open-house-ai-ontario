# Privacy in Open House (reviewed Oct 7, 2026)

## What stays on this computer
- The app, search, the fact-checker and the fallback model all run locally.
- Uploaded leases are written to a temporary file only long enough to read the text, then deleted.
- The server log records request paths (e.g. `POST /api/ask`), never questions or lease text.
- Nothing about a conversation is saved; the chat lives in the browser tab.

## What is sent to Jev (TypeSafe AI), when a key is set
- The question, and for lease review the text of each term.
- Before sending, both are scrubbed (`realtor/lease/privacy.py`): name fields ("TENANT NAME: …"),
  emails, phone numbers, postal codes, street addresses and "between X and Y" party names.
- Standard-form sections with personal details (cover page, Sections 1-3 and 17) are never sent.
- Without a key, nothing is sent anywhere.

## The LTB decisions in the corpus
- Individuals' names, street addresses, postal codes, phones and emails are removed
  (party columns from the catalogue, spaCy name detection, titled names, any case and spacing).
- A full sweep finds no listed individual's name. Company landlords appear as "[organization]", so a
  reader can tell a company from an individual.
- The Open Government Licence doesn't cover personal information, so redaction is also a licence
  condition (see docs/licensing.md).

- Names written in free text ("my landlord Priya said…") are found with spaCy and removed too.
  Like any name detector it can miss unusual names, so it reduces rather than eliminates the risk.
