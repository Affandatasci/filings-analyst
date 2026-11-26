"""Step 2.4: the free checks. No Claude call, no network, no cost.

    from evals.code_checks import run_checks
    marks = run_checks(case, record)          # record = Answer.to_dict(): a plain dict
    marks["number_match"].status              # pass, fail or skip

Each check returns a Mark: pass, fail or skip, and a short reason. Some marks also carry a number
(recall, precision, MRR, and the share of reference numbers found).

    found_page       is any evidence page among the pieces the bot was given (hit at 5)
    recall           how many of the evidence pages were found
    precision        how many of the pieces were on an evidence page
    mrr              1 divided by the place of the first right page (1 is best, 0 means not found)
    json_shape       the bot answered in the agreed format
    number_match     every number of the reference answer is in the bot's answer (units and rounding allowed)
    citation_valid   every cited page was in the pieces the bot was given (no invented citations)
    citation_hit     a cited page is an evidence page
    length           the answer stays short
    not_in_documents found is false (used on the trap questions in Section 3, not by run_checks)

Try the number reader by hand, from the project folder:
    python -m evals.code_checks "1.58 billion" "The amount was 1,577 million dollars."
    (the first text is the reference answer, the second is the bot's answer)
"""
import argparse
import re
import sys
from dataclasses import dataclass

PASS, FAIL, SKIP = "pass", "fail", "skip"

MAX_WORDS = 80  # the bot prompt asks for one to three sentences
SCALES = {"thousand": 1e3, "k": 1e3, "million": 1e6, "mn": 1e6, "mm": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9, "b": 1e9, "trillion": 1e12}
CASE_MARKS = (
    "found_page", "recall", "precision", "mrr", "json_shape", "number_match", "citation_valid", "citation_hit", "length",
)  # the order of run_checks

NUMBER = re.compile(
    r"""
    (?<![\w.])                          # not in the middle of a word, and not after a point
    (?P<open>\()?                       # a bracket before the number: (1,234) is negative
    (?P<minus>[-−–])?         # minus sign, unicode minus or en dash
    (?P<cur>US\$|\$|USD\ ?|€|£)?
    (?P<minus2>[-−–])?        # the minus may also come after the dollar sign
    (?P<digits>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)
    (?P<tail>
        \ ?%
      | \ ?(?:percent|per\ cent|pct)\b
      | \ ?(?:thousand|million|billion|trillion)s?\b
      | \ ?(?:bn|mn|mm)\b
      | [kmb]\b                         # glued to the number: 5M, 2.3B, 10k
    )?
    (?P<close>\))?
    """,
    re.VERBOSE | re.IGNORECASE,
)
MONTH_BEFORE = re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s*$", re.IGNORECASE)
LABEL_BEFORE = re.compile(r"\b(?:page|pages|pp?|note|notes|item|exhibit|figure|table|section|chapter|quarter)\.?\s*$", re.IGNORECASE)
NEGATIVE_WORD_BEFORE = re.compile(r"\b(?:negative|minus)\s+$", re.IGNORECASE)


@dataclass(frozen=True)
class Mark:
    name: str
    status: str  # pass, fail or skip
    reason: str = ""
    value: object = None  # a number for the checks that have one, otherwise None

    @property
    def passed(self):
        return self.status == PASS

    def to_dict(self):
        data = {"status": self.status, "reason": self.reason}
        if self.value is not None:
            data["value"] = round(float(self.value), 4)
        return data


# ---------- the number reader ----------

@dataclass(frozen=True)
class Quantity:
    text: str  # as written, for messages
    raw: float  # the number as written, without sign and without its unit
    scale: float  # 1, or 1000 for thousand, 1 000 000 for million, and so on
    percent: bool  # written as a share of 100: 12.5%, 12.5 percent
    negative: bool  # a minus sign, a bracket (1,234) or the word negative
    decimals: int  # digits after the point, as written
    year: bool  # looks like a year: four digits from 1900 to 2100 and nothing else

    @property
    def value(self):
        """The full value: 1.58 billion is 1580000000. A share of 100 keeps the number as written (12.5)."""
        number = self.raw if self.percent else self.raw * self.scale
        return -number if self.negative else number


def _decimals(digits, percent=False):
    """Digits after the point, as written. A number of 100 or more that ends in .00 (3215.00) counts as a whole number: the .00 is how a table or a
    question file writes it, not a claim that it is exact to the cent. A share of 100 keeps its decimals."""
    if "." not in digits:
        return 0
    whole, fraction = digits.split(".")
    if not percent and len(whole.replace(",", "")) >= 3 and set(fraction) == {"0"}:
        return 0
    return len(fraction)


def parse_numbers(text):
    """Every number in the text, in order, as Quantity objects.

    It reads $1,577 million, 1.58 billion, 12.5%, 12.5 percent, (1,234) as negative, -5.2 and 5M.
    It leaves out what only looks like a number: 3M (the company), 10-K, FY2018, Q3, 2nd, page 60, December 31.
    A year such as 2018 is kept but flagged as a year.
    """
    found = []
    text = str(text or "")
    for match in NUMBER.finditer(text):
        digits, tail = match.group("digits"), (match.group("tail") or "").strip().lower()
        currency = match.group("cur")
        before, after = text[: match.start()], text[match.end():]
        if after[:1].isalpha() and not (after[:1] in "xX" and not after[1:2].isalpha()):
            continue  # glued to a word: 2nd, 12abc. (2.5x is a multiple and stays.)
        if re.match(r"-[A-Za-z]", after) or after.lower().startswith("(k)"):
            continue  # 10-K, 8-K, 401(k)
        if len(tail) == 1 and tail.isalpha() and not currency and "." not in digits and "," not in digits:
            continue  # 3M is a company. 5M needs a dollar sign and 5.2M needs a decimal point.
        plain = not currency and not tail
        if plain and digits.isdigit() and len(digits) <= 2 and MONTH_BEFORE.search(before):
            continue  # the day in December 31
        if plain and LABEL_BEFORE.search(before):
            continue  # page 60, Note 12, Item 7
        if plain and match.group("open") and match.group("close") and len(digits) == 1:
            continue  # a list marker such as (1) or (2), not a loss of 1 (two digits or a dollar sign make it a number)
        percent = tail == "%" or tail.startswith(("percent", "per cent", "pct"))
        word = tail[:-1] if tail.endswith("s") and tail[:-1] in SCALES else tail
        scale = 1.0 if percent else SCALES.get(word, 1.0)
        bracket = bool(match.group("open") and match.group("close"))
        negative = bracket or bool(match.group("minus") or match.group("minus2")) or bool(NEGATIVE_WORD_BEFORE.search(before))
        found.append(
            Quantity(
                text=match.group(0).strip(),
                raw=float(digits.replace(",", "")),
                scale=scale,
                percent=percent,
                negative=negative,
                decimals=_decimals(digits, percent),
                year=digits.isdigit() and len(digits) == 4 and 1900 <= int(digits) <= 2100 and plain,
            )
        )
    return found


def key_numbers(text):
    """The numbers that carry the answer: every number except years."""
    return [quantity for quantity in parse_numbers(text) if not quantity.year]


def _forms(quantity, other):
    """The values this number may mean, each with its rounding: half a unit of the last written digit.

    A number without a unit next to a number with one may be in thousands, millions or billions
    ($1577.00 against 1,577 million). A share of 100 may be a fraction on the other side (12.5% against 0.125).
    """
    half = 0.5 * 10 ** (-quantity.decimals)
    if quantity.percent:
        forms = [(quantity.raw, half)]
        if not other.percent and other.scale == 1:
            forms.append((quantity.raw / 100, half / 100))
        return forms
    if quantity.scale != 1:
        return [(quantity.raw * quantity.scale, half * quantity.scale)]
    if not other.percent and other.scale != 1:
        return [(quantity.raw * scale, half * scale) for scale in (1.0, 1e3, 1e6, 1e9)]
    return [(quantity.raw, half)]


def numbers_agree(reference, got):
    """True when the two numbers are the same after rounding. The less precise number sets the rounding.

    1.58 billion and 1,577 million agree (1.58 stops at the second decimal, and 1,577 million rounds to it).
    1.60 billion and 1,577 million do not. Signs are not compared: the judge reads the wording.
    """
    for a, half_a in _forms(reference, got):
        for b, half_b in _forms(got, reference):
            if abs(a - b) <= max(half_a, half_b) + 1e-9 * max(abs(a), abs(b), 1.0):
                return True
    return False


def _in_question(quantity, asked):
    """True when the number is one the question itself states (a threshold such as the 20% of "more than 20%"). It is not part of the answer."""
    return any(
        quantity.percent == other.percent and abs(quantity.raw - other.raw) < 1e-9 and (quantity.scale == other.scale or 1.0 in (quantity.scale, other.scale))
        for other in asked
    )


def answer_part(text, skip=()):
    """(sentence, numbers): the first sentence of the text that has a key number the question did not state itself, and those numbers."""
    for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z$(\"'])", str(text or "")):
        numbers = [quantity for quantity in key_numbers(sentence) if not _in_question(quantity, skip)]
        if numbers:
            return sentence, numbers
    return "", []


def answer_numbers(text, skip=()):
    """The numbers an answer rests on: the key numbers of its first sentence that has one. Later sentences only explain.

    "36%. The answer assumes FY2023 is the 12 months to June." rests on 36%, not on 12.
    "No. Debt fell by $229 million." rests on $229 million.
    skip: the numbers of the question. A number the question states (the 20% of "more than 20%") is not part of the answer, so the next sentence counts.
    """
    return answer_part(text, skip)[1]


CHANGE_WORDS = re.compile(
    r"\b(?:improv\w*|increas\w*|decreas\w*|declin\w*|expand\w*|contract\w*|reduc\w*|rose|rise\w*|rising|fell|fall\w*|falling|drop\w*|gain\w*|grew|chang\w*|differ\w*|higher|lower)\b",
    re.IGNORECASE,
)


def _is_difference(quantity, others):
    """True when this number is the difference of two of the other numbers, within the rounding of all three.

    "improved by 13" against 155.5 and 142.7: the difference is 12.8, which rounds to 13. "declined by 0.8" against 18.5 and 19.4: 0.9 is within the
    rounding of the three numbers (0.05 each). It is only used when the reference words a change (CHANGE_WORDS).
    """
    pool = [other for other in others if not other.year and other.percent == quantity.percent]
    for index, first in enumerate(pool):
        for second in pool[index + 1:]:
            if quantity.percent:
                gap, wanted, unit = abs(first.raw - second.raw), quantity.raw, 1.0
            else:
                if first.scale != second.scale:
                    continue
                unit = first.scale if quantity.scale == 1 else quantity.scale
                gap, wanted = abs(first.value - second.value), quantity.raw * unit
            slack = (0.5 * 10 ** -quantity.decimals + 0.5 * 10 ** -first.decimals + 0.5 * 10 ** -second.decimals) * unit
            if abs(gap - wanted) <= slack + 1e-9 * max(abs(gap), abs(wanted), 1.0):
                return True
    return False


def number_match(reference, answer, question=None):
    """Is every number the reference answer rests on in the bot's answer? Skip when the reference holds no number.

    question: the question text. A number the question itself states (the 20% of "more than 20%") is not asked for. A change that the reference
    gives as one number ("improved by 13") is also found when the bot gives the two numbers it is the difference of.
    """
    sentence, wanted = answer_part(reference, key_numbers(question) if question else ())
    if not wanted:
        return Mark("number_match", SKIP, "the reference answer holds no number" if not key_numbers(reference) else "the only numbers of the reference answer are in the question")
    got = parse_numbers(answer)
    change = bool(CHANGE_WORDS.search(sentence))
    missing = [
        quantity.text for quantity in wanted
        if not any(numbers_agree(quantity, other) for other in got) and not (change and _is_difference(quantity, got))
    ]
    share = (len(wanted) - len(missing)) / len(wanted)
    if not missing:
        return Mark("number_match", PASS, f"all {len(wanted)} number{'s' if len(wanted) != 1 else ''} found", share)
    return Mark("number_match", FAIL, f"{len(wanted) - len(missing)} of {len(wanted)} found; missing: {', '.join(missing[:3])}", share)


# ---------- search marks (also used by experiments.py, which scores search alone) ----------

def search_marks(pages, hit_pairs):
    """Score one search. pages: the evidence as (doc, page) pairs. hit_pairs: (doc, page) of each piece, best first."""
    wanted, hit_pairs = set(pages), list(hit_pairs)
    place = next((n for n, pair in enumerate(hit_pairs, start=1) if pair in wanted), None)
    found = wanted & set(hit_pairs)
    useful = sum(pair in wanted for pair in hit_pairs)
    return {
        "hit": place is not None,
        "place": place,
        "mrr": 1.0 / place if place else 0.0,
        "recall": len(found) / len(wanted) if wanted else 0.0,
        "precision": useful / len(hit_pairs) if hit_pairs else 0.0,
        "all_pages": bool(wanted) and found == wanted,
        "found": len(found),
        "wanted": len(wanted),
    }


def _pairs(items):
    return [(item["doc"], item["page"]) for item in items or []]


def _listed(pairs, limit=3):
    shown = ", ".join(f"{doc} p{page}" for doc, page in pairs[:limit])
    return shown + (f" and {len(pairs) - limit} more" if len(pairs) > limit else "")


# ---------- the checks ----------

def check_search(case, record):
    """found_page, recall, precision and mrr, from the pieces the bot was given."""
    pieces = _pairs(record.get("pieces"))
    s = search_marks(case.pages, pieces)
    where = f"first right page at place {s['place']} of {len(pieces)}" if s["hit"] else f"no evidence page among the {len(pieces)} pieces"
    return {
        "found_page": Mark("found_page", PASS if s["hit"] else FAIL, where),
        "recall": Mark("recall", PASS if s["all_pages"] else FAIL, f"{s['found']} of {s['wanted']} evidence pages found", s["recall"]),
        "precision": Mark("precision", PASS if s["precision"] > 0 else FAIL, f"{round(s['precision'] * len(pieces))} of {len(pieces)} pieces on an evidence page", s["precision"]),
        "mrr": Mark("mrr", PASS if s["hit"] else FAIL, where, s["mrr"]),
    }


def check_json_shape(record):
    if record.get("format_ok"):
        return Mark("json_shape", PASS, "JSON with answer, citations and found")
    return Mark("json_shape", FAIL, record.get("format_note") or "the reply was not in the agreed format")


def check_citation_valid(record):
    cited, sent = _pairs(record.get("citations")), set(_pairs(record.get("pieces")))
    if not cited:
        if record.get("found"):
            return Mark("citation_valid", FAIL, "found is true but nothing is cited")
        return Mark("citation_valid", PASS, "nothing cited and nothing invented")
    invented = [pair for pair in cited if pair not in sent]
    if invented:
        return Mark("citation_valid", FAIL, "cited a page that was not sent: " + _listed(invented))
    return Mark("citation_valid", PASS, f"{len(cited)} cited, all were sent")


def check_citation_hit(case, record):
    cited = _pairs(record.get("citations"))
    if not cited:
        if not record.get("found"):
            return Mark("citation_hit", SKIP, "the bot said the pieces do not hold the answer: there is nothing to cite")
        return Mark("citation_hit", FAIL, "nothing cited")
    wanted = set(case.pages)
    if any(pair in wanted for pair in cited):
        return Mark("citation_hit", PASS, "a cited page is an evidence page")
    return Mark("citation_hit", FAIL, "no cited page is an evidence page")


def check_length(record):
    words = len(str(record.get("answer") or "").split())
    if words == 0:
        return Mark("length", FAIL, "the answer is empty")
    if words > MAX_WORDS:
        return Mark("length", FAIL, f"{words} words, the limit is {MAX_WORDS}")
    return Mark("length", PASS, f"{words} words")


def check_not_in_documents(record, should_be_found=False):
    """For a trap question (the answer is not in the documents) found must be false."""
    found = bool(record.get("found"))
    if found == should_be_found:
        return Mark("not_in_documents", PASS, f"found is {str(found).lower()}, as it should be")
    return Mark("not_in_documents", FAIL, f"found is {str(found).lower()}, it should be {str(should_be_found).lower()}")


def run_checks(case, record):
    """All the free marks for one answer. Returns {name: Mark} in the order of CASE_MARKS."""
    marks = check_search(case, record)
    marks["json_shape"] = check_json_shape(record)
    marks["number_match"] = number_match(case.answer, record.get("answer", ""), getattr(case, "question", None))
    marks["citation_valid"] = check_citation_valid(record)
    marks["citation_hit"] = check_citation_hit(case, record)
    marks["length"] = check_length(record)
    return {name: marks[name] for name in CASE_MARKS}


def marks_line(marks):
    """One short line for the screen: f for a fail, p for a pass, - for a skip, for each mark."""
    sign = {PASS: "ok", FAIL: "NO", SKIP: "--"}
    return " ".join(f"{name}={sign[mark.status]}" for name, mark in marks.items())


# ---------- check it by hand ----------

def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Try the number reader on two texts (step 2.4).")
    parser.add_argument("reference", help="the reference answer, in quotes")
    parser.add_argument("answer", help="the bot's answer, in quotes")
    args = parser.parse_args(argv)
    for label, text in (("Reference", args.reference), ("Answer", args.answer)):
        numbers = parse_numbers(text)
        shown = ", ".join(f"{q.text} = {q.value:g}{'%' if q.percent else ''}{' (year)' if q.year else ''}" for q in numbers)
        print(f"{label}: {shown or 'no numbers'}")
    mark = number_match(args.reference, args.answer)
    print(f"number_match: {mark.status.upper()} ({mark.reason})")


if __name__ == "__main__":
    main()
