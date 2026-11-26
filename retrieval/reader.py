"""Turn 6: read the company and the year out of a question.

Code only: no Claude call, no model, no cost.

    from retrieval.reader import read_question
    reading = read_question("What was the FY2018 capital expenditure of 3M?")
    reading.company_filter      # ("3M",)
    reading.year_filter         # (2018,)
    reading.mode                # "company+year"

search() uses this when config.FILTER_MODE is "company" or "company+year" (experiment 4).

The safe rules. A wrong filter hides the right page, so when the reader is not sure it filters less:
  1. No known company named in the question   -> no filter at all.
  2. One or more known companies named       -> keep only the pieces of those companies.
  3. Years are used only when a company is named, and only the years for which we hold a filing
     of that company. If no named year matches a filing, the years are ignored.
"""
import re
from dataclasses import dataclass
from functools import lru_cache

import config
from ingest.names import COMPANIES, parse_name

MODES = ("off", "company", "company+year")

# The names a question may use for each company, in lower case. A new company in
# ingest/names.py needs a line here (a test checks this); without one only its full name is used.
ALIASES = {
    "3M": ["3m"],
    "AES Corporation": ["aes corporation", "aes"],
    "AMD": ["amd", "advanced micro devices"],
    "Activision Blizzard": ["activision blizzard", "activision", "blizzard"],
    "Adobe": ["adobe"],
    "Amazon": ["amazon", "amzn"],
    "Amcor": ["amcor"],
    "American Express": ["american express", "amex"],
    "Best Buy": ["best buy"],
    "Block": [],  # also a common word: matched only as "Block" with a capital B, see BLOCK below
    "Boeing": ["boeing"],
    "CVS Health": ["cvs health", "cvs"],
    "Corning": ["corning"],
    "Costco": ["costco"],
    "General Mills": ["general mills"],
    "JPMorgan": ["jpmorgan", "jp morgan", "jpm"],
    "Johnson & Johnson": ["johnson & johnson", "johnson and johnson", "j&j", "j & j", "jnj"],
    "Lockheed Martin": ["lockheed martin", "lockheed"],
    "MGM Resorts": ["mgm resorts", "mgm"],
    "Microsoft": ["microsoft", "msft"],
    "Netflix": ["netflix", "nflx"],
    "Nike": ["nike"],
    "PepsiCo": ["pepsico", "pepsi"],
    "Pfizer": ["pfizer"],
    "Ulta Beauty": ["ulta beauty", "ulta"],
    "Verizon": ["verizon"],
}


def _word(alias):
    """A pattern that finds the alias as whole words, so 'aes' is not found inside 'daesh'."""
    return re.compile(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])")


PATTERNS = {company: [_word(a) for a in ALIASES.get(company, [company.lower()])] for company in COMPANIES.values()}
BLOCK = re.compile(r"(?<![A-Za-z0-9])Block(?![A-Za-z0-9])")
YEAR = re.compile(r"(?<![\d$,.])((?:19|20)\d{2})(?!\d)")  # 2018, but not 2,018 or $2018 or 0.2018
SHORT_YEAR = re.compile(r"\bFY\s?'?(\d{2})\b", re.IGNORECASE)  # FY18 or FY'18 means 2018


@dataclass(frozen=True)
class Reading:
    companies: tuple  # the companies named in the question, for example ("3M",)
    years: tuple  # the years named in the question, for example (2018,)
    company_filter: tuple | None  # companies to keep, or None for no company filter
    year_filter: tuple | None  # years to keep, or None for no year filter
    mode: str  # what was really used: off, company or company+year
    note: str  # one plain line on why


def companies_named(question):
    """The known companies a question names, as a sorted tuple."""
    text = " ".join(question.split())
    low = text.lower()
    found = {company for company, patterns in PATTERNS.items() if any(p.search(low) for p in patterns)}
    if BLOCK.search(text):
        found.add("Block")
    return tuple(sorted(found))


def years_named(question):
    """The years a question names (2018, FY2018, FY18 ...), as a sorted tuple."""
    text = " ".join(question.split())
    years = {int(y) for y in YEAR.findall(text)} | {2000 + int(y) for y in SHORT_YEAR.findall(text)}
    return tuple(sorted(years))


@lru_cache(maxsize=1)
def known_filings():
    """The filings we hold, from data/pdf_list.txt, as DocInfo objects. Empty when the list is missing."""
    try:
        lines = config.PDF_LIST.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ()
    return tuple(parse_name(line.strip()) for line in lines if line.strip())


def read_question(question, mode="company+year", filings=None):
    """Read a question and say which filters are safe to use.

    mode      the most the reader may use: off, company or company+year
    filings   the filings we hold (default: data/pdf_list.txt). Only the years are checked against them.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown filter mode '{mode}'. Use one of: {', '.join(MODES)}.")
    companies, years = companies_named(question), years_named(question)
    if not companies:
        return Reading(companies, years, None, None, "off", "no known company named")
    if mode == "off":
        return Reading(companies, years, None, None, "off", "mode is off")
    if mode == "company":
        return Reading(companies, years, companies, None, "company", "company only (mode company)")
    held = known_filings() if filings is None else tuple(filings)
    if not held:
        return Reading(companies, years, companies, None, "company", "no filing list, so the years are not checked")
    usable = tuple(sorted({d.year for d in held if d.company in companies} & set(years)))
    if usable:
        return Reading(companies, years, companies, usable, "company+year", "company and year")
    return Reading(companies, years, companies, None, "company", "years matched no filing" if years else "no year named")
