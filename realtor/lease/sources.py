"""Search the Ontario sources the model answers from.

Four kinds, all official or decided by the tribunal:
  rta    Residential Tenancies Act, 2006, by section
  lease  Ontario Standard Form of Lease: the government's plain-language explanations
  guide  Landlord and Tenant Board Interpretation Guidelines and brochures
  case   2026 LTB decisions (redacted), with what was ordered and why

The corpus files are built by `pipeline/` (see the README). Search is hybrid: BM25 keywords for
exact legal terms plus small embeddings for meaning.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

CORPUS = Path(__file__).parents[2] / "corpus"
MAX_SOURCE_CHARS = {"rta": 650, "lease": 650, "guide": 900, "case": 1000, "mine": 700}
STOPWORDS = set("""the a an of to and or in on for by is are be that this with as at any it its may shall
    what should do does did i me my mine we our you your they them their he she his her can could would will
    how if when which who whom why there here about from so but not no yes have has had been am was were
    just really now then than also any some get got need want""".split())

# Everyday words people use → the words the Act and the LTB use. Keeps keyword search from
# missing "my sink is broken" just because the law says "repair" and "maintenance".
PLAIN_TO_LEGAL = {
    "repair maintenance good state of repair": "sink faucet tap toilet shower bathtub fridge refrigerator stove oven "
        "dishwasher washer dryer appliance furnace boiler leak leaking mould mold broken broke ceiling window "
        "door elevator plumbing pipe",
    "vital service heat fuel hot water electricity": "heat heating cold freezing water electricity power hydro gas",
    "pest infestation maintenance": "cockroach cockroaches roach bedbug bedbugs mice mouse rats rat pests bugs",
    "undue damage wilful negligent conduct": "fault caused accident accidentally damaged",
    "notice of termination": "evict eviction kicked leave notice",
    "rent increase guideline": "raise raising increase hike",
    "entry notice 24 hours": "enter entering came walked showed",
    "arrears non-payment of rent": "late behind owe unpaid",
    "rent deposit last month": "deposit",
    # LTB form names
    "landlord personally requires residential occupation own use purchaser": "n12",
    "demolition conversion repairs renovation right of first refusal": "n13 renoviction renovict",
    "non-payment of rent arrears": "n4 l1",
    "substantial interference reasonable enjoyment undue damage": "n5",
    "tenant notice to terminate": "n9",
    "agreement to terminate tenancy": "n11",
    "maintenance obligations": "t6",
    "tenant rights harassment interference": "t2",
    "money collected illegally": "t1",
    "bad faith notice": "t5",
    "smoking rules": "smoke smoking smoker vape vaping cannabis cigarette cigarettes weed",
    "changes to the rental unit": "paint painting wallpaper renovate shelves holes nails mount",
    "tenant's insurance": "insurance insured",
}
_EXPAND = {w: legal for legal, plain in PLAIN_TO_LEGAL.items() for w in plain.split()}


def expand(query: str) -> str:
    extra = {_EXPAND[w] for w in re.findall(r"[a-z]+", query.lower()) if w in _EXPAND}
    return query + (" " + " ".join(sorted(extra)) if extra else "")

# Parts of the RTA that govern ordinary tenancies. Others (care homes, co-ops, Board procedure)
# still count, at half weight, so they surface only when clearly relevant.
CORE_PARTS = {"Introduction", "Tenancy Agreements", "Responsibilities Of Landlords",
              "Responsibilities Of Tenants", "Security Of Tenure And Termination Of Tenancies",
              "Assignment, Subletting And Unauthorized Occupancy", "Rules Relating To Rent",
              "Suite Meters And Apportionment Of Utility Costs", "Maintenance Standards"}

OUTCOME = {"tenant": "decided in the tenant's favour", "landlord": "decided in the landlord's favour",
           "split": "partly granted", "unclear": "made the order below"}


def _stem(w: str) -> str:
    for suffix in ("ies", "es", "s"):
        if len(w) > 3 and w.endswith(suffix) and not w.endswith("ss"):
            return w[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return w


def tokenize(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


@dataclass(frozen=True)
class Source:
    id: str
    kind: str  # "rta" | "lease" | "guide" | "case" | "mine" (the user's own lease)
    label: str
    text: str
    weight: float = 1.0
    side: str = ""   # cases only: who the decision favoured
    focus: str = ""  # what to centre the visible window on (a query, or an exact phrase)

    @property
    def shown(self) -> str:
        """Exactly the text the model sees. Training answers may only quote from this.

        Long sources show the window of lines that best matches `focus`. If `focus` is an exact
        phrase in the text, the window always contains it.
        """
        limit = MAX_SOURCE_CHARS[self.kind]
        if len(self.text) <= limit:
            return self.text
        if self.kind == "case":  # cases are pre-summarised: keep the head (application, result, order)
            return self.text[:limit].rsplit(" ", 1)[0] + " …"
        lines = self.text.split("\n")
        exact = bool(self.focus) and self.focus in self.text
        want = set(tokenize(self.focus))
        best, best_score = None, -1.0
        for start in range(len(lines)):
            window, size = [], 0
            for line in lines[start:]:
                if window and size + len(line) > limit:
                    break
                window.append(line)
                size += len(line) + 1
            text = "\n".join(window)
            if len(text) > limit or (exact and self.focus not in text):
                continue
            score = len(want & set(tokenize(text))) - 0.01 * start  # prefer earlier on ties
            if score > best_score:
                best, best_score = (start, len(window), text), score
        if best is None:  # a single line longer than the limit: cut around the focus
            i = max(0, self.text.find(self.focus) - 100) if exact else 0
            i = self.text.rfind(" ", 0, i) + 1 if i else 0
            piece = self.text[i:i + limit]
            piece = piece if i + limit >= len(self.text) else piece.rsplit(" ", 1)[0] + " …"
            return ("… " if i else "") + piece
        start, count, text = best
        return ("… " if start else "") + text + (" …" if start + count < len(lines) else "")

    def render(self, n: int) -> str:
        """The source as the model sees it: every sentence numbered {n.k} so answers can point at it."""
        return f"[{n}] {self.label}\n" + numbered(self.shown, n)

    @property
    def sentences(self) -> list[str]:
        """Sentence k (1-based) is sentences[k - 1]; the same numbering the model sees."""
        return [text for _, text in segments(self.shown)]


# --- sentence numbering --------------------------------------------------------------------
# Answers point at sentences ({2.3} = source 2, sentence 3) instead of retyping them, and the
# app inserts the exact text. These two functions are the single definition of a "sentence".

FIELD = re.compile(r"^(Application|Result|Ordered|Key reasons): ")
SENTENCE_BREAK = re.compile(r"(?<=[.;:?!])\s+(?=[A-Z“\"(•\d])")


def segments(text: str) -> list[tuple[int, str]]:
    """(line index, sentence) pairs. Marginal notes like "[Same]" and case field labels aren't sentences."""
    out = []
    for i, line in enumerate(text.split("\n")):
        line = FIELD.sub("", line.strip()).removeprefix("… ").removesuffix(" …").strip()
        if not line or re.fullmatch(r"\[[^\]]+\]", line) or line == "…":
            continue
        out += [(i, part.strip()) for part in SENTENCE_BREAK.split(line) if part.strip()]
    return out


def numbered(text: str, n: int) -> str:
    lines = text.split("\n")
    by_line: dict[int, list[str]] = {}
    for k, (i, sentence) in enumerate(segments(text), 1):
        by_line.setdefault(i, []).append(f"{{{n}.{k}}} {sentence}")
    out = []
    for i, line in enumerate(lines):
        if i in by_line:
            label = FIELD.match(line.strip())
            out.append((label.group(0) if label else "") + " ".join(by_line[i]))
        elif line.strip():
            out.append(line.strip())
    return "\n".join(out)


def tidy(text: str) -> str:
    text = re.sub(r"\[name\](?:[\s,]*(?:and\s+)?\[name\])+", "[name]", text)
    text = re.sub(r"\s*[A-Z][a-z]+ \d{1,2}, \d{4}\s*_{3,}.*$", "", text)  # signature line
    return re.sub(r"\s+", " ", text).strip()


def key_reasons(c: dict, limit: int = 520) -> str:
    """The Board's findings: paragraphs that say "I find" or cite the Act, in order."""
    reasons = [tidy(p) for p in c["reasons"]]
    picked = [p for p in reasons if re.search(r"\bI find\b|\bs\. ?\d|\bsection \d", p)] or reasons
    out = ""
    for p in picked:
        if len(out) + len(p) > limit:
            break
        out += p + " "
    return (out or picked[0][:limit].rsplit(" ", 1)[0]).strip()


def case_summary(c: dict) -> str:
    return (f"Application: {tidy(c['issues'])[:300].rsplit(' ', 1)[0]}\n"
            f"Result: {OUTCOME[c['favoured']]}.\n"
            f"Ordered: {tidy(c['ordered'][0])[:280]}\n"
            f"Key reasons: {key_reasons(c)}")


def case_label(c: dict) -> str:
    cite = c.get("citation") or c["file_number"]
    return f"LTB decision {cite} ({c['applications']}, {c['order_date']})"


@lru_cache(maxsize=1)
def load_sources() -> dict[str, Source]:
    out: dict[str, Source] = {}
    for s in map(json.loads, (CORPUS / "rta_sections.jsonl").open()):
        title = s["title"] or s["text"].split("\n")[0][:60]
        out[f"rta:{s['number']}"] = Source(f"rta:{s['number']}", "rta",
                                           f"Residential Tenancies Act, 2006, s. {s['number']}: {title}", s["text"],
                                           1.0 if s["part"] in CORE_PARTS else 0.5)
    for s in map(json.loads, (CORPUS / "standard_lease.jsonl").open()):
        # The lease's own sections 4-16 (the rules both sides sign); 1-3 and 17 are form fields.
        if s["kind"] == "section" and 4 <= s["number"] <= 16:
            text = re.sub(r"\s*2229E \(\d{4}/\d{2}\) Page \d+ of \d+\s*", " ", s["text"]).strip()
            # The form's checkbox choices run together ("…requirements. or The tenant must…"):
            # put each choice on its own line, keeping the form's wording exactly.
            text = re.sub(r"(?<=[.:])\s+or\s+(?=[A-Z])", "\n• ", text)
            text = re.sub(r"Select one:\s*", "Select one:\n• ", text)
            out[f"lease-section:{s['number']}"] = Source(f"lease-section:{s['number']}", "lease",
                                                         f"Ontario Standard Form of Lease, Section {s['number']}: {s['title']}", text)
        if s["kind"] == "part":
            out[f"lease:{s['letter']}"] = Source(f"lease:{s['letter']}", "lease",
                                                 f"Ontario Standard Form of Lease, General Information Part {s['letter']}: "
                                                 f"{s['title']}", s["text"])
    guide_path = CORPUS / "ltb_guidance.jsonl"
    if guide_path.exists():
        for g in map(json.loads, guide_path.open()):
            out[g["id"]] = Source(g["id"], "guide", f"{g['doc']}, \"{g['heading']}\"", g["text"])
    for c in map(json.loads, (CORPUS / "ltb_cases.jsonl").open()):
        out[f"case:{c['doc_id']}"] = Source(f"case:{c['doc_id']}", "case", case_label(c), case_summary(c),
                                            side=c["favoured"])
    return out


EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge's recommended prefix


@lru_cache(maxsize=1)
def _embedder():
    from fastembed import TextEmbedding

    return TextEmbedding(EMBED_MODEL)


@lru_cache(maxsize=4)
def _index(kind: str) -> tuple[BM25Okapi, np.ndarray, list[Source]]:
    docs = [s for s in load_sources().values() if s.kind == kind]
    bm25 = BM25Okapi([tokenize(f"{d.label} {d.text}") for d in docs])
    cache = CORPUS / f"embeddings_{kind}.npy"
    if cache.exists() and np.load(cache).shape[0] == len(docs):
        vectors = np.load(cache)
    else:
        vectors = np.array(list(_embedder().embed([f"{d.label}\n{d.text[:1500]}" for d in docs])))
        np.save(cache, vectors)
    return bm25, vectors, docs


def _minmax(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min() + 1e-9)


def ranked(query: str, kind: str, exclude: set[str] = frozenset()) -> list[Source]:
    bm25, vectors, docs = _index(kind)
    q = next(iter(_embedder().embed([QUERY_PREFIX + query])))
    scores = 0.4 * _minmax(np.asarray(bm25.get_scores(tokenize(expand(query))))) + 0.6 * _minmax(vectors @ q)
    scores *= np.array([d.weight for d in docs])
    return [replace(docs[i], focus=query) for i in np.argsort(-scores) if docs[i].id not in exclude]


def search(query: str, kind: str, k: int, exclude: set[str] = frozenset()) -> list[Source]:
    return ranked(query, kind, exclude)[:k]


def balanced_cases(query: str, exclude: set[str] = frozenset(), pool: int = 15) -> list[Source]:
    """The most relevant decision each side won, so examples never lean one way by default."""
    top = ranked(query, "case", exclude)[:pool]
    picks = [next((c for c in top if c.side == side), None) for side in ("tenant", "landlord")]
    return [c for c in picks if c] or top[:2]


def retrieve(query: str, mode: str = "clause", exclude: set[str] = frozenset(),
             use_playbooks: bool = True) -> list[Source]:
    """What the model reads.

    clause:   3 statute sections, 1 standard-lease note, 1 LTB guidance passage, 1 decision
    question: 2 statute sections, 2 LTB guidance passages, 2 standard-lease entries, and 2 decisions,
              one each side won (a scenario usually has two sides)
    """
    if mode == "question":
        found = (search(query, "rta", 2, exclude) + search(query, "guide", 2, exclude)
                 + search(query, "lease", 2, exclude) + balanced_cases(query, exclude))
        if use_playbooks:
            found = _add_playbook_sources(query, found, exclude)
        return found
    return (search(query, "rta", 3, exclude) + search(query, "lease", 1, exclude)
            + search(query, "guide", 1, exclude) + search(query, "case", 1, exclude))


def _add_playbook_sources(query: str, found: list[Source], exclude: set[str]) -> list[Source]:
    """If the question is a known scenario, make sure its official sources are in what the model reads.

    Each one replaces the lowest-ranked result of the same kind (or is added if none), mirroring
    how the model was trained.
    """
    from .playbooks import gold_sources, match

    hit = match(query)
    if not hit:
        return found
    out = list(found)
    have = {s.id for s in out}
    for g in gold_sources(hit[0]):
        if g.id in have or g.id in exclude:
            continue
        same = [i for i, s in enumerate(out) if s.kind == g.kind]
        protected = {s.id for s in gold_sources(hit[0])}
        same = [i for i in same if out[i].id not in protected]
        if same:
            out[same[-1]] = g
        else:
            out.insert(max((i for i, s in enumerate(out) if s.kind != "case"), default=-1) + 1, g)
        have = {s.id for s in out}
    return out


def render(sources: list[Source]) -> str:
    return "\n\n".join(s.render(i + 1) for i, s in enumerate(sources))


FINDING = re.compile(r"\b(I find|I am satisfied|I am not satisfied|I was satisfied|I accept|I do not accept|therefore|"
                     r"As a result|For these reasons|on a balance of probabilities)\b", re.I)


def case_why(src: Source, pick: int | None = None) -> int | None:
    """The sentence that best says why a decision went the way it did, or None.

    A usable reason is a complete sentence from "Key reasons" (not one cut off by the window, not
    a lead-in that ends in a colon). A sentence with a finding ("I find…") is preferred; failing
    that, what was ordered. `pick` is a sentence someone else chose (Jev); kept if it's usable.
    """
    sents = src.sentences
    reasons = src.shown.split("Key reasons:", 1)[-1] if "Key reasons:" in src.shown else ""
    cut = src.shown.endswith(" …")

    def usable(k: int) -> bool:
        t = sents[k - 1]
        return (t in reasons and len(t) >= 60 and t[-1] in ".!?\"”" and not (cut and k == len(sents))
                and not re.search(r"\b(s|ss|No|para)\.$", t))
    if pick and 1 <= pick <= len(sents) and usable(pick):
        return pick
    ks = [k for k in range(1, len(sents) + 1) if usable(k)]
    best = next((k for k in ks if FINDING.search(sents[k - 1])), None)
    if best:
        return best
    # No finding in view: what was ordered says more than a sentence that restates the Act.
    return case_ordered(src) or (ks[0] if ks else None)


def case_ordered(src: Source) -> int | None:
    """The sentence number of what the Board ordered, if the decision shows it."""
    ordered = re.search(r"^Ordered: (.*)$", src.shown, re.M)
    return next((k for k, t in enumerate(src.sentences, 1) if ordered and t and t in ordered.group(1) and len(t) > 25), None)


def _term_name(term: dict, i: int) -> str:
    """A short name for a lease term: its own capitalised heading word(s), else the given heading,
    else its first words. Form titles ("STANDARD FORM OF LEASE…") aren't names."""
    text = re.sub(r"^.*?ADDITIONAL TERMS\s*", "", term["text"].strip(), flags=re.S) if "STANDARD FORM" in term["text"][:200] else term["text"]
    m = re.match(r"\s*(?:[A-Z]\.|\d+\.|[ivx]+\.)?\s*([A-Z][A-Z&/'’ -]{2,40}[A-Z])(?=\s+[A-Z][a-z])", text)
    if m:
        return m.group(1).strip().capitalize()
    heading = (term.get("heading") or "").strip()
    if heading and len(heading) <= 40 and "STANDARD FORM" not in heading.upper():
        return heading
    words = text.split()[:6]
    return " ".join(words) + "…" if words else f"term {i + 1}"


def lease_terms(terms: list[dict], query: str, k: int = 3) -> list[Source]:
    """The terms of the user's own lease that best match the question (hybrid keyword + meaning)."""
    docs = [Source(f"mine:{i}", "mine", f"Your lease · {_term_name(t, i)}", t["text"], focus=query)
            for i, t in enumerate(terms) if len(t.get("text", "")) >= 60]
    if not docs:
        return []
    q = expand(query)
    bm = BM25Okapi([tokenize(d.text) for d in docs]).get_scores(tokenize(q))
    emb = _embedder()
    vecs = np.array(list(emb.embed([d.text[:1500] for d in docs])))
    qv = next(iter(emb.embed([QUERY_PREFIX + query])))
    score = 0.5 * _minmax(np.asarray(bm)) + 0.5 * _minmax(vecs @ qv)
    return [docs[i] for i in np.argsort(-score)[:k]]


def _plain(text: str) -> str:
    text = text.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"[^a-z0-9$]+", " ", text).strip()


@lru_cache(maxsize=1)
def _blank_form() -> str:
    """The standard form's own printed text (sections and explanations), for telling it apart from what was filled in."""
    return " ".join(_plain(s.text) for s in load_sources().values() if s.kind == "lease")


def is_form_text(sentence: str) -> bool:
    """True if a sentence from someone's lease is the standard form's printed wording, not something filled in.

    Printed choices ("A rent deposit is not required") are kept out of "what your lease says": the
    PDF's ticked box doesn't survive as text, so a printed option may not be the one chosen.
    """
    plain = _plain(sentence)
    return len(plain) >= 15 and plain in _blank_form()
