"""Step 1.2: the facts that sit inside a PDF file name.

    3M_2023Q2_10Q                             ->  3M, 2023, Q2, 10Q
    JOHNSON_JOHNSON_2023_8K_dated-2023-08-30  ->  Johnson & Johnson, 2023, 8K, 2023-08-30

The company names are the same as in the "company" field of the FinanceBench
questions, so a company filter can use them later.
"""
import re
from dataclasses import dataclass

COMPANIES = {
    "3M": "3M",
    "ACTIVISIONBLIZZARD": "Activision Blizzard",
    "ADOBE": "Adobe",
    "AES": "AES Corporation",
    "AMAZON": "Amazon",
    "AMCOR": "Amcor",
    "AMD": "AMD",
    "AMERICANEXPRESS": "American Express",
    "BESTBUY": "Best Buy",
    "BLOCK": "Block",
    "BOEING": "Boeing",
    "CORNING": "Corning",
    "COSTCO": "Costco",
    "CVSHEALTH": "CVS Health",
    "GENERALMILLS": "General Mills",
    "JOHNSON_JOHNSON": "Johnson & Johnson",
    "JPMORGAN": "JPMorgan",
    "LOCKHEEDMARTIN": "Lockheed Martin",
    "MGMRESORTS": "MGM Resorts",
    "MICROSOFT": "Microsoft",
    "NETFLIX": "Netflix",
    "NIKE": "Nike",
    "PEPSICO": "PepsiCo",
    "PFIZER": "Pfizer",
    "ULTABEAUTY": "Ulta Beauty",
    "VERIZON": "Verizon",
}

NAME_PATTERN = re.compile(
    r"^(?P<company>[A-Za-z0-9_]+?)_(?P<year>(?:19|20)\d{2})(?P<quarter>Q[1-4])?"
    r"_(?P<type>[A-Za-z0-9]+)(?:_dated-(?P<date>\d{4}-\d{2}-\d{2}))?$"
)


@dataclass(frozen=True)
class DocInfo:
    name: str  # the file name without .pdf
    company: str
    year: int
    period: str  # FY for a 10K, Q1 to Q4 for a quarter, empty for an 8K
    type: str  # 10K, 10Q, 8K or EARNINGS
    date: str | None = None  # only for 8K files that carry a date

    def label(self):
        """Short words for the header line, for example 'FY2018 10-K'."""
        if self.type == "10K":
            return f"FY{self.year} 10-K"
        if self.type == "10Q":
            return f"{self.period} {self.year} 10-Q"
        if self.type == "8K":
            return f"{self.year} 8-K" + (f" dated {self.date}" if self.date else "")
        if self.type == "EARNINGS":
            return f"{self.period} {self.year} earnings release".strip()
        return f"{self.year} {self.type}"

    def header(self, page):
        """The first line of every piece, for example '3M FY2018 10-K, page 58'."""
        return f"{self.company} {self.label()}, page {page}"


def parse_name(name):
    """Turn a PDF file name into a DocInfo. Stops with a clear message if it is unknown."""
    name = name.removesuffix(".pdf")
    match = NAME_PATTERN.match(name)
    if not match:
        raise ValueError(f"Cannot read the file name '{name}'. Expected COMPANY_2022_10K or similar.")
    key = match["company"].upper()
    if key not in COMPANIES:
        raise ValueError(f"Unknown company '{match['company']}' in '{name}'. Add it to COMPANIES in ingest/names.py.")
    doc_type = match["type"].upper()
    period = match["quarter"] or ("FY" if doc_type == "10K" else "")
    return DocInfo(
        name=name,
        company=COMPANIES[key],
        year=int(match["year"]),
        period=period,
        type=doc_type,
        date=match["date"],
    )
