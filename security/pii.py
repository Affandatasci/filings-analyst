"""Step 3.3: personal data hiding, and the checks for personal data and secret words in an answer.

    from security.pii import mask_text, secret_words_in, fake_personal_data
    mask_text("Dear Maria Okafor, call (212) 555-0142 or write to maria.okafor@northwind.test.")
    # ("Dear [NAME], call [PHONE] or write to [EMAIL].", {"PERSON": 1, "PHONE": 1, "EMAIL": 1})

Three layers find personal data, and the first match of any layer wins:
  1. Patterns that are always on: emails (any ending, also .test), phone numbers, employee IDs such as EMP40213.
  2. Presidio with the small spaCy model: names, and phone numbers in other shapes. Presidio is optional: without it the other
     layers still work and presidio_status() says why. It misses emails that end in .test (it checks the ending against real ones),
     which is one reason layer 1 exists.
  3. The fake names in data\\fake_pdfs\\manifest.json, and their first and last names alone, found by exact match.

Masking is for pieces of internal and hr files when the role is below hr. Real filings are public and are never masked:
names, addresses and phone numbers of executives in a public filing are public, and answers need them.
Check it by hand:  python -m security.pii "Dear Maria Okafor, call (212) 555-0142"
"""
import argparse
import logging
import re
import sys
from dataclasses import replace
from functools import lru_cache

from security import fakes

PLACEHOLDER = {"PERSON": "[NAME]", "EMAIL": "[EMAIL]", "PHONE": "[PHONE]", "EMP_ID": "[EMP ID]"}

EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")
EMP_ID = re.compile(r"\bEMP[\s\-]?\d{5}\b", re.IGNORECASE)
PHONE = re.compile(
    r"(?<![\w.])(?:\+?1[\s.\-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.\-])\d{3}[\s.\-]\d{4}(?!\w)"  # (212) 555-0142, +1 415 555 0167, 312-555-0123
    r"|(?<![\w.])555[\s.\-]?01\d\d(?!\d)"  # 555-0142: the fiction range without an area code
)
# What the output guard looks for. Narrower than the masking patterns on purpose: real emails and phone numbers of a public filing are fine.
FAKE_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+\.test\b", re.IGNORECASE)
FAKE_PHONE = re.compile(r"(?<!\d)555[\s.\-]?01\d\d(?!\d)")

_presidio = {}


def presidio_status():
    """(True, "") when Presidio and the small spaCy model are ready, else (False, the reason)."""
    _analyzer()
    return _presidio["analyzer"] is not None, _presidio.get("problem", "")


def _analyzer():
    if "analyzer" in _presidio:
        return _presidio["analyzer"]
    try:
        from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
        engine = NlpEngineProvider(
            nlp_configuration={"nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]}
        ).create_engine()
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers(nlp_engine=engine, languages=["en"])
        _presidio["analyzer"] = AnalyzerEngine(nlp_engine=engine, registry=registry, supported_languages=["en"])
    except Exception as err:  # noqa: BLE001 - not installed, or the spaCy model is missing: the other layers still work
        _presidio["analyzer"] = None
        _presidio["problem"] = f"{type(err).__name__}: {str(err)[:160]}"
    return _presidio["analyzer"]


@lru_cache(maxsize=4)
def _name_pattern(names):
    """One pattern for all fake names, and for their first and last names alone (at least 4 letters)."""
    parts = set(names)
    for name in names:
        parts.update(token for token in name.split() if len(token) >= 4)
    if not parts:
        return None
    return re.compile(r"\b(?:" + "|".join(re.escape(part) for part in sorted(parts, key=len, reverse=True)) + r")\b", re.IGNORECASE)


def _merge(spans):
    """Longest first, no overlaps."""
    kept = []
    for start, end, label in sorted(spans, key=lambda span: (span[0], -(span[1] - span[0]))):
        if kept and start < kept[-1][1]:
            continue
        kept.append((start, end, label))
    return kept


def find_spans(text, use_presidio=True, use_known=True):
    """Where personal data sits in a text: a list of (start, end, label), label being PERSON, EMAIL, PHONE or EMP_ID."""
    spans = []
    for label, pattern in (("EMAIL", EMAIL), ("EMP_ID", EMP_ID), ("PHONE", PHONE)):
        spans += [(match.start(), match.end(), label) for match in pattern.finditer(text)]
    if use_known:
        pattern = _name_pattern(tuple(fakes.known_names()))
        if pattern:
            spans += [(match.start(), match.end(), "PERSON") for match in pattern.finditer(text)]
    analyzer = _analyzer() if use_presidio else None
    if analyzer is not None and text.strip():
        for found in analyzer.analyze(text=text, language="en", entities=["PERSON", "PHONE_NUMBER"]):
            spans.append((found.start, found.end, "PERSON" if found.entity_type == "PERSON" else "PHONE"))
    return _merge(spans)


def mask_text(text, use_presidio=True, use_known=True):
    """Replace personal data with [NAME], [EMAIL], [PHONE] and [EMP ID]. Returns (masked text, {label: how many})."""
    counts, out, last = {}, [], 0
    for start, end, label in find_spans(text, use_presidio, use_known):
        out += [text[last:start], PLACEHOLDER[label]]
        counts[label] = counts.get(label, 0) + 1
        last = end
    out.append(text[last:])
    return "".join(out), counts


def needs_masking(hit, role):
    """Pieces of internal and hr files are masked for every role below hr. Real public filings never are."""
    return role is not None and role != "hr" and hit.access in ("internal", "hr")


def mask_hits(hits, role, use_presidio=True):
    """Mask the pieces that need it. Returns (hits, events); a piece that was changed has an event with the counts."""
    out, events = [], []
    for hit in hits:
        if not needs_masking(hit, role):
            out.append(hit)
            continue
        masked, counts = mask_text(hit.text, use_presidio)
        out.append(replace(hit, text=masked) if counts else hit)
        if counts:
            events.append({"stage": "pii", "doc": hit.doc, "page": hit.page, "piece": hit.piece, "masked": counts})
    return out, events


def fake_personal_data(text):
    """The fake personal data found in a text: emails, phone numbers, employee IDs and full names from the fake files."""
    found = [match.group(0) for pattern in (FAKE_EMAIL, FAKE_PHONE, EMP_ID) for match in pattern.finditer(text)]
    lowered = text.lower()
    found += [value for value in fakes.personal_values() if value.lower() in lowered]
    found += [name for name in fakes.known_names() if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.IGNORECASE)]
    return sorted(set(found))


def secret_words_in(text):
    """The secret words found in a text. Also finds a word spelled with dashes or spaces between its letters."""
    found, squashed = [], re.sub(r"[^a-z]", "", text.lower())
    for word in fakes.secret_words():
        if re.search(r"(?<![A-Za-z0-9])" + re.escape(word) + r"(?![A-Za-z0-9])", text, re.IGNORECASE) or (len(word) >= 7 and word in squashed):
            found.append(word)
    return found


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Hide personal data in a text (step 3.3).")
    parser.add_argument("text", nargs="?", help="a text to mask")
    parser.add_argument("--status", action="store_true", help="say if Presidio and the spaCy model are ready")
    args = parser.parse_args(argv)
    ready, problem = presidio_status()
    if args.status or not args.text:
        print("Presidio and the spaCy small model: " + ("ready" if ready else f"NOT ready ({problem}). Only the pattern layer and the fake name list work."))
        if not args.text:
            return
    masked, counts = mask_text(args.text)
    print(masked)
    print("Masked:", counts or "nothing")


if __name__ == "__main__":
    main()
