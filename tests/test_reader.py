"""Tests for turn 6: the question reader, the company and year filter, and the filter comparison.

No network and no model download: a fake embedder and an in memory Qdrant, as in test_search.py.
Run from the project folder:
    python -m pytest -q
"""
import math
import re
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import add_year_index  # noqa: E402
import compare_filters  # noqa: E402
import config  # noqa: E402
from ingest.names import COMPANIES, parse_name  # noqa: E402
from retrieval import reader  # noqa: E402
from retrieval.search import search  # noqa: E402

FILES = tuple(
    parse_name(name)
    for name in (
        "3M_2018_10K",
        "3M_2019_10K",
        "ADOBE_2021_10K",
        "ADOBE_2022_10K",
        "AES_2022_10K",
        "BLOCK_2020_10K",
        "MICROSOFT_2016_10K",
    )
)


def bag_of_words(texts):
    """A fake embedder: every word goes into one of 384 slots."""
    vectors = []
    for text in texts:
        vector = [0.0] * config.EMBED_DIM
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vector[zlib.crc32(word.encode()) % config.EMBED_DIM] += 1.0
        length = math.sqrt(sum(value * value for value in vector)) or 1.0
        vectors.append([value / length for value in vector])
    return vectors


# ---------- the reader: companies ----------

def test_one_company_and_one_year():
    reading = reader.read_question("What was the FY2018 capital expenditure of 3M?", filings=FILES)
    assert reading.companies == ("3M",)
    assert reading.years == (2018,)
    assert (reading.company_filter, reading.year_filter, reading.mode) == (("3M",), (2018,), "company+year")


def test_no_company_named_means_no_filter_at_all():
    reading = reader.read_question("What drove the reduction in SG&A expense in FY2023?", filings=FILES)
    assert (reading.company_filter, reading.year_filter, reading.mode) == (None, None, "off")
    assert reading.years == (2023,)  # the year is read, but never used without a company


@pytest.mark.parametrize(
    "question, company",
    [
        ("How did Amazon do?", "Amazon"),
        ("AMZN revenue growth", "Amazon"),
        ("J&J results for the year", "Johnson & Johnson"),
        ("Johnson and Johnson net sales", "Johnson & Johnson"),
        ("Johnson & Johnson net sales", "Johnson & Johnson"),
        ("American Express's net income", "American Express"),
        ("AMEX card fees", "American Express"),
        ("JPM trading revenue", "JPMorgan"),
        ("MGM room revenue", "MGM Resorts"),
        ("Lockheed backlog", "Lockheed Martin"),
        ("Pepsi snack sales", "PepsiCo"),
        ("Ulta store count", "Ulta Beauty"),
        ("Best Buy's revenue", "Best Buy"),
        ("3M's net income", "3M"),
        ("Activision Blizzard bookings", "Activision Blizzard"),
        ("advanced micro devices gross margin", "AMD"),
    ],
)
def test_nicknames_and_possessives(question, company):
    assert reader.companies_named(question) == (company,)


def test_block_is_found_only_with_a_capital_b():
    assert reader.companies_named("What was Block's gross profit?") == ("Block",)
    assert reader.companies_named("the block of shares and a block diagram") == ()


def test_a_name_inside_another_word_does_not_count():
    assert reader.companies_named("Which taxes did the aestheticians pay in $13M?") == ()


def test_several_companies_are_all_kept():
    reading = reader.read_question("Compare Microsoft and Adobe in FY2022", filings=FILES)
    assert reading.companies == ("Adobe", "Microsoft")
    assert reading.company_filter == ("Adobe", "Microsoft")
    assert reading.year_filter == (2022,)  # Adobe has a 2022 filing


def test_every_company_is_found_by_its_own_full_name():
    for company in COMPANIES.values():
        assert company in reader.companies_named(f"What was the revenue of {company} in 2020?"), company


def test_the_alias_table_covers_exactly_the_company_list():
    assert set(reader.ALIASES) == set(COMPANIES.values())
    assert set(reader.PATTERNS) == set(COMPANIES.values())
    for aliases in reader.ALIASES.values():
        assert all(alias == alias.lower() for alias in aliases)


# ---------- the reader: years ----------

@pytest.mark.parametrize(
    "question, years",
    [
        ("FY2018 and FY19 and fiscal 2020", (2018, 2019, 2020)),
        ("in the year 2020.", (2020,)),
        ("FY'21 revenue", (2021,)),
        ("fy22 revenue", (2022,)),
        ("capital expenditure of $2,018 million", ()),
        ("a price of $2018 and 0.2019", ()),
        ("the 12 months of 2016-2017", (2016, 2017)),
        ("no year here", ()),
    ],
)
def test_years(question, years):
    assert reader.years_named(question) == years


def test_a_year_without_a_matching_filing_is_ignored():
    reading = reader.read_question("What was the revenue of 3M in 2005?", filings=FILES)
    assert (reading.company_filter, reading.year_filter, reading.mode) == (("3M",), None, "company")
    assert reading.note == "years matched no filing"


def test_a_year_is_checked_against_the_filings_of_the_named_company():
    reading = reader.read_question("What was the revenue of Adobe in 2018?", filings=FILES)  # 2018 exists only for 3M
    assert reading.year_filter is None and reading.mode == "company"


def test_only_the_years_we_hold_are_kept():
    reading = reader.read_question("3M revenue in FY2018, FY2019 and FY2025", filings=FILES)
    assert reading.years == (2018, 2019, 2025)
    assert reading.year_filter == (2018, 2019)


def test_no_year_named_means_company_only():
    reading = reader.read_question("What is the revenue of 3M?", filings=FILES)
    assert (reading.year_filter, reading.mode, reading.note) == (None, "company", "no year named")


def test_without_a_filing_list_the_years_are_not_used():
    reading = reader.read_question("3M revenue in 2018", filings=())
    assert (reading.company_filter, reading.year_filter, reading.mode) == (("3M",), None, "company")


# ---------- the reader: modes ----------

def test_the_modes_cap_what_the_reader_may_use():
    question = "3M revenue in 2018"
    off = reader.read_question(question, mode="off", filings=FILES)
    company = reader.read_question(question, mode="company", filings=FILES)
    both = reader.read_question(question, mode="company+year", filings=FILES)
    assert (off.company_filter, off.year_filter, off.mode) == (None, None, "off")
    assert (company.company_filter, company.year_filter, company.mode) == (("3M",), None, "company")
    assert (both.company_filter, both.year_filter, both.mode) == (("3M",), (2018,), "company+year")


def test_an_unknown_mode_stops_with_the_allowed_names():
    with pytest.raises(ValueError, match="off, company, company\\+year"):
        reader.read_question("3M revenue", mode="everything", filings=FILES)


def test_known_filings_reads_the_pdf_list(tmp_path, monkeypatch):
    listing = tmp_path / "pdf_list.txt"
    listing.write_text("3M_2018_10K\n\nADOBE_2022_10K\n", encoding="utf-8")
    monkeypatch.setattr(config, "PDF_LIST", listing)
    reader.known_filings.cache_clear()
    try:
        assert [d.name for d in reader.known_filings()] == ["3M_2018_10K", "ADOBE_2022_10K"]
        monkeypatch.setattr(config, "PDF_LIST", tmp_path / "missing.txt")
        reader.known_filings.cache_clear()
        assert reader.known_filings() == ()
    finally:
        reader.known_filings.cache_clear()


# ---------- search with the filter ----------

TEXT = "dividends paid to shareholders were declared every quarter"  # the same text in every filing: only the filter can tell them apart


@pytest.fixture()
def index(monkeypatch):
    client = QdrantClient(":memory:")
    client.create_collection(config.COLLECTION, vectors_config=VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE))
    points = []
    for number, (name, page) in enumerate([("ADOBE_2022_10K", 1), ("ADOBE_2022_10K", 2), ("ADOBE_2021_10K", 1), ("AES_2022_10K", 1), ("3M_2018_10K", 1)]):
        info = parse_name(name)
        text = f"{info.header(page)}\n{TEXT}" if page == 1 else f"{info.header(page)}\nrestructuring charges and severance costs"
        payload = {
            "doc": name, "company": info.company, "year": info.year, "period": info.period, "type": info.type,
            "page": page, "piece": 1, "text": text, "source": "real", "access": "public",
        }
        points.append(PointStruct(id=number + 1, vector=bag_of_words([text])[0], payload=payload))
    client.upsert(config.COLLECTION, points=points)
    monkeypatch.setattr(reader, "known_filings", lambda: FILES)
    return client


def docs_of(hits):
    return {hit.doc for hit in hits}


QUESTION = "What dividends were paid to shareholders by Adobe in FY2022?"


def test_search_is_unchanged_by_default(index, monkeypatch):
    def must_not_run(*args, **kwargs):
        raise AssertionError("the reader must not run when the mode is off")

    monkeypatch.setattr(reader, "read_question", must_not_run)
    monkeypatch.setattr(config, "FILTER_MODE", "off")
    hits = search(QUESTION, k=5, client=index, embed=bag_of_words)
    assert docs_of(hits) >= {"AES_2022_10K", "3M_2018_10K", "ADOBE_2021_10K"}


def test_mode_company_keeps_every_year_of_the_company(index):
    hits = search(QUESTION, k=5, client=index, embed=bag_of_words, mode="company")
    assert docs_of(hits) == {"ADOBE_2022_10K", "ADOBE_2021_10K"}


def test_mode_company_and_year_keeps_one_filing(index):
    hits = search(QUESTION, k=5, client=index, embed=bag_of_words, mode="company+year")
    assert docs_of(hits) == {"ADOBE_2022_10K"}
    assert hits[0].page == 1


def test_the_default_mode_comes_from_config(index, monkeypatch):
    monkeypatch.setattr(config, "FILTER_MODE", "company+year")
    assert docs_of(search(QUESTION, k=5, client=index, embed=bag_of_words)) == {"ADOBE_2022_10K"}
    monkeypatch.setattr(config, "FILTER_MODE", "off")
    assert len(docs_of(search(QUESTION, k=5, client=index, embed=bag_of_words))) > 1


def test_a_question_without_a_company_is_searched_everywhere_in_every_mode(index):
    question = "What dividends were paid to shareholders in FY2022?"
    for mode in ("off", "company", "company+year"):
        assert len(docs_of(search(question, k=5, client=index, embed=bag_of_words, mode=mode))) > 1


def test_an_explicit_company_wins_over_the_reader(index):
    hits = search(QUESTION, k=5, client=index, embed=bag_of_words, company="AES Corporation", mode="company+year")
    assert docs_of(hits) == {"AES_2022_10K"}


def test_year_alone_one_year_or_several(index):
    assert docs_of(search(TEXT, k=5, client=index, embed=bag_of_words, year=2021)) == {"ADOBE_2021_10K"}
    assert docs_of(search(TEXT, k=5, client=index, embed=bag_of_words, year=(2021, 2018))) == {"ADOBE_2021_10K", "3M_2018_10K"}


def test_company_as_a_list(index):
    hits = search(TEXT, k=5, client=index, embed=bag_of_words, company=["3M", "AES Corporation"])
    assert docs_of(hits) == {"3M_2018_10K", "AES_2022_10K"}


def test_an_unknown_mode_in_search_stops_with_a_clear_message(index):
    with pytest.raises(ValueError, match="Unknown filter mode"):
        search(QUESTION, client=index, embed=bag_of_words, mode="everything")


# ---------- the filter comparison ----------

def case(case_id, doc, question, split="test", pages=None):
    return SimpleNamespace(id=case_id, doc=doc, question=question, split=split, pages=pages or [(doc, 1)])


CASES = [
    case("financebench_id_00001", "ADOBE_2022_10K", "What dividends did Adobe pay in FY2022?"),
    case("financebench_id_00002", "3M_2018_10K", "What was the revenue of 3M in FY2018?"),
    case("financebench_id_00003", "AES_2022_10K", "What drove the change in expenses in FY2022?"),  # names no company
    case("financebench_id_00004", "AES_2022_10K", "What did Microsoft say about AES?", pages=[("AES_2022_10K", 1), ("AES_2022_10K", 2)]),  # names the wrong company too
]


def test_the_holdout_is_left_out(monkeypatch):
    everything = CASES + [case("financebench_id_00099", "ADOBE_2022_10K", "Adobe?", split="holdout")]
    monkeypatch.setattr(compare_filters, "load_cases", lambda: everything)
    assert [c.id for c in compare_filters.usable_cases()] == [c.id for c in CASES]
    assert [c.id for c in compare_filters.usable_cases(first=2)] == [c.id for c in CASES[:2]]


def test_reader_check_counts():
    stats = compare_filters.reader_check(CASES, filings=FILES)
    assert stats["named"] == {"none": 1, "one": 2, "several": 1}
    assert stats["true_named"] == 3  # Adobe, 3M, and AES in the last question
    assert stats["wrong_filter"] == []  # the last question names AES as well, so the true company is still kept
    assert stats["allowed"] == {"off": 4, "company": 4, "company+year": 4}
    assert stats["sizes"]["off"] == [len(FILES)] * 4
    assert stats["sizes"]["company+year"][:2] == [1, 1]


def test_reader_check_flags_a_filter_that_hides_the_right_company():
    wrong = [case("financebench_id_00010", "AES_2022_10K", "What did Microsoft pay in FY2022?")]
    stats = compare_filters.reader_check(wrong, filings=FILES)
    assert stats["wrong_filter"] == ["financebench_id_00010"]
    assert stats["allowed"]["company"] == 0 and stats["allowed"]["off"] == 1
    assert [mode for _, _, mode in stats["lost"]] == ["company", "company+year"]


def test_first_place_and_pages_found():
    hits = [SimpleNamespace(doc="A", page=3), SimpleNamespace(doc="A", page=1), SimpleNamespace(doc="B", page=1)]
    assert compare_filters.first_place(hits, [("A", 1), ("B", 1)]) == (2, 2)
    assert compare_filters.first_place(hits, [("C", 9)]) == (None, 0)
    assert compare_filters.first_place([], [("A", 1)]) == (None, 0)


def test_run_modes_searches_each_question_once_per_mode():
    calls = []

    def fake_search(question, mode=None, collection=None):
        calls.append((question, mode, collection))
        return [SimpleNamespace(doc=cases_doc[question], page=1)]

    cases_doc = {c.question: c.doc for c in CASES}
    results = compare_filters.run_modes(CASES[:2], fake_search, collection="other")
    assert len(calls) == 2 * 3
    assert {mode for _, mode, _ in calls} == {"off", "company", "company+year"}
    assert {collection for _, _, collection in calls} == {"other"}
    assert results["off"] == [(1, 1), (1, 1)]


def test_summarize_and_flips():
    cases = [case("a", "D1", "q", pages=[("D1", 1), ("D1", 2)]), case("b", "D2", "q"), case("c", "D3", "q")]
    results = {
        "off": [(None, 0), (2, 1), (1, 1)],
        "company": [(3, 1), (2, 1), (None, 0)],
        "company+year": [(1, 2), (1, 1), (None, 0)],
    }
    summary, several = compare_filters.summarize(cases, results)
    assert several == 1
    assert summary["off"] == {"top5": 2, "first": 1, "all_pages": 0}
    assert summary["company+year"] == {"top5": 2, "first": 2, "all_pages": 1}
    assert compare_filters.flips(cases, results, "off", "company") == (["a"], ["c"])
    assert compare_filters.flips(cases, results, "company", "company+year") == ([], [])


def test_the_report_names_the_best_mode_and_prefers_the_simpler_on_a_tie(capsys):
    cases = [case("financebench_id_00001", "D1", "What did Adobe pay in FY2022?"), case("financebench_id_00002", "D2", "q")]
    results = {"off": [(1, 1), (None, 0)], "company": [(1, 1), (None, 0)], "company+year": [(1, 1), (2, 1)]}
    compare_filters.print_search_report(cases, results)
    out = capsys.readouterr().out
    assert "Most evidence pages in the top 5: company+year (2 of 2)" in out
    results["company+year"] = [(1, 1), (None, 0)]
    compare_filters.print_search_report(cases, results)
    assert "Most evidence pages in the top 5: off (1 of 2)" in capsys.readouterr().out


def test_main_reader_only_needs_no_model_and_no_qdrant(monkeypatch, capsys):
    monkeypatch.setattr(compare_filters, "load_cases", lambda: CASES + [case("financebench_id_00099", "ADOBE_2022_10K", "Adobe?", split="holdout")])
    monkeypatch.setattr(reader, "known_filings", lambda: FILES)
    compare_filters.main(["--reader-only"])
    out = capsys.readouterr().out
    assert "Reader check on 4 questions (the holdout is not used)" in out
    assert "a filter that would hide the right company: 0" in out


# ---------- the one time year index ----------

class FakeClient:
    def __init__(self, schema):
        self.schema, self.made = schema, []

    def get_collection(self, name):
        return SimpleNamespace(payload_schema=self.schema, points_count=11125)

    def create_payload_index(self, name, field, kind):
        self.made.append((name, field, str(kind)))
        self.schema = {**self.schema, field: "made"}


def test_the_year_index_is_made_once():
    client = FakeClient({"company": "keyword"})
    assert add_year_index.add_year_index(client, "filings_v1") == "made"
    assert client.made == [("filings_v1", "year", "integer")]
    assert add_year_index.add_year_index(client, "filings_v1") == "already there"
    assert len(client.made) == 1


def test_add_year_index_main_reports_the_state(monkeypatch, capsys):
    client = FakeClient({"company": "keyword", "doc": "keyword"})
    import retrieval.search as search_module

    monkeypatch.setattr(search_module, "get_client", lambda: client)
    add_year_index.main(["--collection", "filings_v2"])
    out = capsys.readouterr().out
    assert "Collection filings_v2: 11125 pieces. Index on 'year': made." in out
    assert "company, doc, year" in out


# ---------- the full run of the comparison, with a fake search ----------

def fake_world(monkeypatch, search_fn):
    import retrieval.search as search_module

    everything = CASES + [case("financebench_id_00099", "ADOBE_2022_10K", "Holdout question about Adobe", split="holdout")]
    monkeypatch.setattr(compare_filters, "load_cases", lambda: everything)
    monkeypatch.setattr(reader, "known_filings", lambda: FILES)
    monkeypatch.setattr(compare_filters, "cached_embedder", lambda: bag_of_words)
    monkeypatch.setattr(search_module, "search", search_fn)


def test_main_runs_all_modes_and_never_searches_the_holdout(monkeypatch, capsys):
    asked = []
    answer_of = {c.question: c.pages for c in CASES}

    def fake_search(question, mode=None, collection=None, embed=None):
        asked.append((question, mode))
        # the right page is found only when the company and the year are used
        return [SimpleNamespace(doc=doc, page=page) for doc, page in answer_of[question]] if mode == "company+year" else []

    fake_world(monkeypatch, fake_search)
    compare_filters.main([])
    out = capsys.readouterr().out
    assert len(asked) == 4 * 3
    assert all("Holdout" not in question for question, _ in asked)
    assert "Search on 4 questions (the holdout is not used)" in out
    assert "off -> company: found now 0 more, lost 0" in out
    assert "company -> company+year: found now 4 more, lost 0" in out
    assert "Most evidence pages in the top 5: company+year (4 of 4)" in out
    assert "12 searches in" in out


def test_main_says_what_to_do_when_the_year_index_is_missing(monkeypatch):
    def fake_search(question, mode=None, collection=None, embed=None):
        raise RuntimeError('Qdrant search failed.\n  query_points: UnexpectedResponse: Index required but not found for "year"')

    fake_world(monkeypatch, fake_search)
    with pytest.raises(SystemExit) as stop:
        compare_filters.main(["--first", "2"])
    assert "add_year_index.py" in str(stop.value)
