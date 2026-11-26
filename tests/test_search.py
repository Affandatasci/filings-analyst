"""Tests for step 1.5: search and the index check.

No network and no model download: the tests use a fake embedder (words go into 384 slots,
so texts that share words are close) and an in memory Qdrant. Run from the project folder:
    python -m pytest -q
"""
import contextlib
import io
import json
import math
import re
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import check_index  # noqa: E402
import config  # noqa: E402
from ingest import build_index  # noqa: E402
from ingest.names import parse_name  # noqa: E402
from ingest.read_pdf import Document, Page  # noqa: E402
from retrieval.search import Hit, search  # noqa: E402

# Qdrant in memory says payload indexes do nothing. True, and harmless here.
pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes")

TOPICS = {  # one sentence per page
    "ADOBE_2022_10K": [
        "dividends paid to shareholders were declared every quarter and the board approved the payment",
        "restructuring charges and severance costs were recorded for the year after the reorganization",
        "cloud subscription revenue increased because of new customers and higher renewal rates",
    ],
    "AES_2022_10K": [
        "wind and solar power plants started producing electricity for the national grid this year",
        "coal plants were retired and replaced with natural gas and battery storage projects",
    ],
    "MICROSOFT_2016_10K": ["secret merger plan and the board vote are described in this internal memo"],
}


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


def words(texts):
    return [len(text.split()) for text in texts]  # one token per word


def make_pdf(path, pages):
    """Write a PDF. pages is a list of pages, each a list of text lines."""
    pdf = pymupdf.open()
    for lines in pages:
        page = pdf.new_page()
        for i, text in enumerate(lines):
            page.insert_text((72, 60 + 14 * i), text, fontsize=10)
    pdf.save(str(path))
    pdf.close()


def make_document(folder, name, sentences, access="public"):
    path = folder / f"{name}.pdf"
    make_pdf(path, [[s, s + " again", s + " as well"] for s in sentences])
    return Document(parse_name(name), path, access=access)


@pytest.fixture()
def index(tmp_path):
    documents = [
        make_document(tmp_path, name, sentences, access="internal" if name.startswith("MICROSOFT") else "public")
        for name, sentences in TOPICS.items()
    ]
    client = QdrantClient(":memory:")
    build_index.build(documents, client, bag_of_words, words, config.COLLECTION)
    return client


# ---------- search ----------

def test_search_finds_the_right_page_and_returns_plain_hits(index):
    hits = search("dividends paid to shareholders each quarter", client=index, embed=bag_of_words)
    first = hits[0]
    assert isinstance(first, Hit)
    assert (first.doc, first.page) == ("ADOBE_2022_10K", 1)
    assert (first.company, first.year, first.type, first.piece) == ("Adobe", 2022, "10K", 1)
    assert (first.source, first.access) == ("real", "public")
    assert first.text.startswith("Adobe FY2022 10-K, page 1")
    assert first.label == "ADOBE_2022_10K, page 1"
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_the_question_is_embedded_once_with_the_given_function(index):
    asked = []

    def recording(texts):
        asked.append(list(texts))
        return bag_of_words(texts)

    search("solar power plants", client=index, embed=recording)
    assert asked == [["solar power plants"]]


def test_k_and_the_default_number_of_pieces(index):
    assert len(search("revenue", k=2, client=index, embed=bag_of_words)) == 2
    assert len(search("revenue", client=index, embed=bag_of_words)) == config.TOP_K


def test_company_filter(index):
    aes = parse_name("AES_2022_10K").company
    hits = search("revenue and plants", company=aes, client=index, embed=bag_of_words)
    assert len(hits) == 2 and {hit.company for hit in hits} == {aes}


def test_access_filter(index):
    everything = search("secret merger plan board vote", k=6, client=index, embed=bag_of_words)
    assert everything[0].doc == "MICROSOFT_2016_10K"
    public = search("secret merger plan board vote", k=6, access=["public"], client=index, embed=bag_of_words)
    assert public and all(hit.access == "public" for hit in public)
    internal = search("secret merger plan board vote", access=["internal"], client=index, embed=bag_of_words)
    assert [hit.doc for hit in internal] == ["MICROSOFT_2016_10K"]


def test_another_collection_and_a_missing_one(index, tmp_path):
    documents = [make_document(tmp_path, "AES_2022_10K", TOPICS["AES_2022_10K"])]
    build_index.build(documents, index, bag_of_words, words, "other")
    hits = search("solar power", collection="other", client=index, embed=bag_of_words)
    assert {hit.doc for hit in hits} == {"AES_2022_10K"}
    with pytest.raises(RuntimeError, match="Qdrant search failed"):
        search("solar power", collection="does_not_exist", client=index, embed=bag_of_words)


def test_an_empty_question_is_refused(index):
    for text in ("", "   ", None):
        with pytest.raises(ValueError):
            search(text, client=index, embed=bag_of_words)


# ---------- query_points or search ----------

def scored(page):
    return ScoredPoint(id=1, version=0, score=0.9, payload={"doc": "ADOBE_2022_10K", "page": page, "text": "t"})


class OnlySearch:
    """Like an old qdrant-client: it only has search()."""

    def __init__(self):
        self.calls = []

    def search(self, **kwargs):
        self.calls.append("search")
        return [scored(7)]


class OnlyQueryPoints:
    """Like a new qdrant-client: it only has query_points()."""

    def __init__(self):
        self.calls = []

    def query_points(self, **kwargs):
        self.calls.append("query_points")
        return SimpleNamespace(points=[scored(8)])


class FirstOneFails:
    """Has both methods, but query_points fails."""

    def __init__(self, fail_both=False):
        self.calls, self.fail_both = [], fail_both

    def query_points(self, **kwargs):
        self.calls.append("query_points")
        raise RuntimeError("rejected by the server")

    def search(self, **kwargs):
        self.calls.append("search")
        if self.fail_both:
            raise RuntimeError("also rejected")
        return [scored(9)]


def test_it_uses_the_method_the_client_has(monkeypatch):
    monkeypatch.setattr(config, "QDRANT_SEARCH", "query_points")
    old = OnlySearch()
    assert search("x", client=old, embed=bag_of_words)[0].page == 7 and old.calls == ["search"]
    monkeypatch.setattr(config, "QDRANT_SEARCH", "search")
    new = OnlyQueryPoints()
    assert search("x", client=new, embed=bag_of_words)[0].page == 8 and new.calls == ["query_points"]


def test_it_tries_the_other_method_when_the_first_fails(monkeypatch):
    monkeypatch.setattr(config, "QDRANT_SEARCH", "query_points")
    client = FirstOneFails()
    assert search("x", client=client, embed=bag_of_words)[0].page == 9
    assert client.calls == ["query_points", "search"]
    both = FirstOneFails(fail_both=True)
    with pytest.raises(RuntimeError) as error:
        search("x", client=both, embed=bag_of_words)
    assert "query_points" in str(error.value) and "search" in str(error.value)


def test_a_point_without_labels_does_not_crash():
    class Bare:
        def query_points(self, **kwargs):
            return SimpleNamespace(points=[ScoredPoint(id=1, version=0, score=0.5, payload=None)])

    hit = search("x", client=Bare(), embed=bag_of_words)[0]
    assert (hit.doc, hit.page, hit.text) == ("", 0, "")


# ---------- check_index ----------

def printed(function, *args, **kwargs):
    """Run a function and return (its result, everything it printed)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = function(*args, **kwargs)
    return result, buffer.getvalue()


def evidence(doc, page, text="text"):
    return {"doc_name": doc, "evidence_page_num": page, "evidence_text": text}


def test_evidence_pages_use_the_page_offset(monkeypatch):
    question = {"evidence": [evidence("ADOBE_2022_10K", 0), evidence("ADOBE_2022_10K", 4)]}
    monkeypatch.setattr(config, "PAGE_OFFSET", 1)
    assert check_index.evidence_pages(question) == [("ADOBE_2022_10K", 1), ("ADOBE_2022_10K", 5)]
    monkeypatch.setattr(config, "PAGE_OFFSET", 0)
    assert check_index.evidence_pages(question) == [("ADOBE_2022_10K", 0), ("ADOBE_2022_10K", 4)]


def test_find_page_matches_text_from_another_reader():
    pages = [
        Page(1, ["Consolidated Balance Sheet", "Total assets | $36,500 | $37,987"], False),
        Page(2, ["Net sales were 5,363 million in 2018", "Operating income rose to 7,207 million"], False),
    ]
    groups = check_index.page_groups(pages)
    number, share = check_index.find_page("NET SALES  were 5,363\n \nmillion in 2018 and\nOperating income rose to 7,207 million", groups)
    assert number == 2 and share > 0.6  # the extra word "and" breaks one group, the rest still match
    assert check_index.find_page("a text that is on no page at all, not even close", groups)[0] is None
    assert check_index.find_page("", groups)[0] is None


def hit(doc, page):
    return Hit(0.5, doc, "Adobe", 2022, "FY", "10K", page, 1, "text", "real", "public")


def test_evidence_found_reports_the_place_and_the_count(monkeypatch):
    monkeypatch.setattr(config, "PAGE_OFFSET", 1)
    question = {"evidence": [evidence("ADOBE_2022_10K", 4), evidence("ADOBE_2022_10K", 6)]}  # our pages 5 and 7
    hits = [hit("ADOBE_2022_10K", 1), hit("AES_2022_10K", 5), hit("ADOBE_2022_10K", 7), hit("ADOBE_2022_10K", 5)]
    assert check_index.evidence_found(question, hits) == (3, 2)  # the AES page 5 does not count
    assert check_index.evidence_found(question, hits[:2]) == (None, 0)


def write_questions(tmp_path, monkeypatch, questions, names=("ADOBE_2022_10K",)):
    (tmp_path / "pdf_list.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    (tmp_path / "questions.jsonl").write_text("\n".join(json.dumps(q) for q in questions) + "\n", encoding="utf-8")
    monkeypatch.setattr(config, "PDF_LIST", tmp_path / "pdf_list.txt")
    monkeypatch.setattr(config, "QUESTIONS_FILE", tmp_path / "questions.jsonl")


def question(number, doc, page, text="text"):
    return {"financebench_id": f"id_{number}", "doc_name": doc, "question": f"Question number {number}?", "evidence": [evidence(doc, page, text)]}


def test_load_questions_keeps_only_our_documents(tmp_path, monkeypatch):
    write_questions(tmp_path, monkeypatch, [question(1, "ADOBE_2022_10K", 0), question(2, "OTHER_2020_10K", 0), question(3, "ADOBE_2022_10K", 1)])
    assert [q["financebench_id"] for q in check_index.load_questions()] == ["id_1", "id_3"]


def test_check_counts_hits_and_prints_a_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PAGE_OFFSET", 1)
    write_questions(tmp_path, monkeypatch, [question(n, "ADOBE_2022_10K", n - 1) for n in (1, 2, 3)])  # our pages 1, 2, 3
    answers = {
        "Question number 1?": [hit("ADOBE_2022_10K", 9), hit("ADOBE_2022_10K", 1)],  # hit at place 2
        "Question number 2?": [hit("ADOBE_2022_10K", 9)],  # miss
        "Question number 3?": [hit("ADOBE_2022_10K", 3)],  # hit at place 1
    }
    (found, total), out = printed(check_index.check, 3, search_fn=lambda text, collection=None: answers[text])
    assert (found, total) == (2, 3)
    assert "HIT: an evidence page is at place 2" in out and "MISS" in out
    assert f"Evidence page in the top {config.TOP_K}: 2 of 3" in out


def test_check_with_many_questions_prints_one_line_each(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PAGE_OFFSET", 1)
    write_questions(tmp_path, monkeypatch, [question(n, "ADOBE_2022_10K", 0) for n in range(1, 13)])
    (found, total), out = printed(check_index.check, 12, search_fn=lambda text, collection=None: [hit("ADOBE_2022_10K", 1)])
    lines = [line for line in out.splitlines() if line.startswith("[")]
    assert (found, total) == (12, 12) and len(lines) == 12 and all("HIT 1" in line for line in lines)


def test_main_says_pass_or_what_to_do_next(monkeypatch):
    monkeypatch.setattr(check_index, "check", lambda first, collection: (4, 10))
    _, out = printed(check_index.main, [])
    assert "PASS" in out
    monkeypatch.setattr(check_index, "check", lambda first, collection: (3, 10))
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), pytest.raises(SystemExit) as stop:
        check_index.main([])
    assert stop.value.code == 1
    assert "check_index.py --pages" in buffer.getvalue()


def test_the_page_test_finds_the_offset(tmp_path, monkeypatch):
    sentences = [
        "The company reported net sales of 5,363 million dollars in the year",
        "Operating income rose to 7,207 million dollars because of lower costs",
        "Cash flow from operations reached 6,439 million dollars this year",
        "Total assets were 36,500 million dollars at the end of the period",
    ]
    make_pdf(tmp_path / "ADOBE_2022_10K.pdf", [["Cover page of the filing"]] + [[s] for s in sentences])  # sentences are on pages 2 to 5
    write_questions(tmp_path, monkeypatch, [question(n + 1, "ADOBE_2022_10K", n, s) for n, s in enumerate(sentences, start=1)])  # the file says 1 to 4
    monkeypatch.setattr(config, "PDF_DIR", tmp_path)

    monkeypatch.setattr(config, "PAGE_OFFSET", 0)
    result, out = printed(check_index.page_test)
    assert result == 1 and "All 4 agree" in out and "Set PAGE_OFFSET = 1" in out

    monkeypatch.setattr(config, "PAGE_OFFSET", 1)
    result, out = printed(check_index.page_test)
    assert result == 0 and "Good." in out
