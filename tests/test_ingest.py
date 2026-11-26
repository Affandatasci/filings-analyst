"""Tests for steps 1.2 to 1.4: file names, page reader, piece cutter and the index upload.

No network and no model download: the tests use a fake embedder, a word counter and an
in memory Qdrant. Run from the project folder:  python -m pytest -q
"""
import json
import sys
from pathlib import Path

import pymupdf
import pytest
from qdrant_client import QdrantClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from ingest import build_index, chunker  # noqa: E402
from ingest.names import COMPANIES, parse_name  # noqa: E402
from ingest.read_pdf import Document, Page, is_table_line, read_pdf  # noqa: E402

INFO = parse_name("3M_2018_10K")

# Qdrant in memory says payload indexes do nothing. True, and harmless here.
pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes")


def words(texts):
    """A simple token counter for the tests: one token per word."""
    return [len(text.split()) for text in texts]


class FakeEmbedder:
    """Gives every text a vector of 384 numbers and remembers what it was asked to embed."""

    def __init__(self):
        self.texts = []

    def __call__(self, texts):
        self.texts.extend(texts)
        return [[1.0 + len(text) % 7] * config.EMBED_DIM for text in texts]


def make_pdf(path, pages):
    """Write a PDF. pages is a list of pages, each a list of (x, y, text) to draw."""
    doc = pymupdf.open()
    for items in pages:
        page = doc.new_page()
        for x, y, text in items:
            page.insert_text((x, y), text, fontsize=10)
    doc.save(str(path))
    doc.close()


def prose_page(tag, lines=12):
    return [(72, 60 + 14 * i, f"{tag} line {i} has some plain words about the company results and plans") for i in range(lines)]


def make_document(folder, name, pages):
    path = folder / f"{name}.pdf"
    make_pdf(path, pages)
    return Document(parse_name(name), path)


# ---------- step 1.2: names ----------

@pytest.mark.parametrize(
    "name, company, year, period, doc_type, date",
    [
        ("3M_2018_10K", "3M", 2018, "FY", "10K", None),
        ("3M_2023Q2_10Q", "3M", 2023, "Q2", "10Q", None),
        ("Pfizer_2023Q2_10Q", "Pfizer", 2023, "Q2", "10Q", None),
        ("AMCOR_2023Q4_EARNINGS", "Amcor", 2023, "Q4", "EARNINGS", None),
        ("JOHNSON_JOHNSON_2023_8K_dated-2023-08-30", "Johnson & Johnson", 2023, "", "8K", "2023-08-30"),
    ],
)
def test_parse_name(name, company, year, period, doc_type, date):
    info = parse_name(name)
    assert (info.company, info.year, info.period, info.type, info.date) == (company, year, period, doc_type, date)


def test_header_line():
    assert INFO.header(58) == "3M FY2018 10-K, page 58"
    assert parse_name("AMCOR_2023Q4_EARNINGS").header(3) == "Amcor Q4 2023 earnings release, page 3"
    assert parse_name("BESTBUY_2024Q2_10Q").header(7) == "Best Buy Q2 2024 10-Q, page 7"
    assert parse_name("JOHNSON_JOHNSON_2023_8K_dated-2023-08-30").header(1) == "Johnson & Johnson 2023 8-K dated 2023-08-30, page 1"


def test_unknown_names_stop_with_a_clear_message():
    with pytest.raises(ValueError, match="Unknown company"):
        parse_name("ACME_2020_10K")
    with pytest.raises(ValueError, match="Cannot read"):
        parse_name("hello")


def test_all_listed_names_parse_and_match_the_question_file():
    if not config.PDF_LIST.exists():
        pytest.skip("data/pdf_list.txt is missing")
    names = [line.strip() for line in config.PDF_LIST.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(names) == 37
    infos = [parse_name(name) for name in names]
    assert len({info.company for info in infos}) == len(COMPANIES) == 26
    if config.QUESTIONS_FILE.exists():
        in_questions = {}
        for line in config.QUESTIONS_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                in_questions[record["doc_name"]] = record["company"]
        for info in infos:
            if info.name in in_questions:
                assert info.company == in_questions[info.name]


# ---------- step 1.2: reader ----------

def test_reader_keeps_a_table_line_together(tmp_path):
    table = [
        (72, 60, "Table of Contents"),
        (72, 80, "Consolidated Balance Sheet of the example company, with all amounts shown in millions of dollars."),
        (72, 110, "(Millions)"), (300, 110, "2018"), (400, 110, "2017"),
        (72, 130, "Total assets"), (290, 130, "$"), (298, 130, "36,500"), (390, 130, "$"), (398, 130, "37,987"),
        (72, 150, "Less: Accumulated depreciation"), (300, 150, "(16,135)"), (400, 150, "(16,048)"),
    ]
    path = tmp_path / "t.pdf"
    make_pdf(path, [table, [(72, 72, "Hello")]])
    first, second = read_pdf(path)
    assert (first.number, second.number) == (1, 2)  # page numbers start at 1
    assert "(Millions) | 2018 | 2017" in first.lines
    assert "Total assets | $36,500 | $37,987" in first.lines
    assert "Less: Accumulated depreciation | (16,135) | (16,048)" in first.lines
    assert "Table of Contents" not in first.lines
    assert not first.no_text
    assert second.no_text  # fewer than 100 letters and digits


def test_reader_can_read_one_page_only(tmp_path):
    path = tmp_path / "t.pdf"
    make_pdf(path, [prose_page("a"), prose_page("b"), prose_page("c")])
    pages = read_pdf(path, only_pages={2})
    assert [page.number for page in pages] == [2]
    assert pages[0].lines[0].startswith("b line 0")


def test_is_table_line():
    assert is_table_line("Total assets | $36,500 | $37,987")
    assert not is_table_line("Total assets are shown in the table")
    assert not is_table_line("Business Segment | Major Products")


def test_real_balance_sheet_if_the_pdf_is_there():
    path = config.PDF_DIR / "3M_2018_10K.pdf"
    if not path.exists():
        pytest.skip("3M_2018_10K.pdf is not in the PDF folder")
    page = read_pdf(path, only_pages={58})[0]
    assert "Total assets | $36,500 | $37,987" in page.lines
    assert not page.no_text


# ---------- step 1.3: cutting ----------

def page_of(lines, number=1):
    return Page(number, lines, False)


def text_lines(count, tag="x"):
    """Lines of exactly 10 words each."""
    return [f"{tag}{i} " + " ".join(["word"] * 9) for i in range(count)]


def test_a_short_page_is_one_piece_with_a_header():
    pieces = chunker.cut_document(INFO, [page_of(text_lines(30), 5)], words)
    assert len(pieces) == 1
    assert pieces[0].text.splitlines()[0] == "3M FY2018 10-K, page 5"
    assert (pieces[0].doc, pieces[0].page, pieces[0].piece) == ("3M_2018_10K", 5, 1)


def test_a_page_up_to_the_hard_stop_stays_whole():
    pieces = chunker.cut_document(INFO, [page_of(text_lines(47))], words)  # 470 words + 5 header words
    assert len(pieces) == 1 and pieces[0].tokens == 475


def test_a_long_page_is_cut_with_overlap_and_never_above_the_hard_stop():
    lines = text_lines(150)
    pieces = chunker.cut_document(INFO, [page_of(lines)], words)
    assert len(pieces) >= 3
    assert all(piece.tokens <= config.PIECE_HARD_STOP for piece in pieces)
    assert all(piece.text.splitlines()[0] == "3M FY2018 10-K, page 1" for piece in pieces)
    seen = {line for piece in pieces for line in piece.text.splitlines()[1:]}
    assert seen == set(lines)  # nothing is lost
    shared = set(pieces[0].text.splitlines()[1:]) & set(pieces[1].text.splitlines()[1:])
    assert 1 <= len(shared) <= 5  # about 50 tokens of overlap, so up to 5 lines of 10 words


def test_a_tiny_last_piece_joins_the_piece_before_it():
    pieces = chunker.cut_document(INFO, [page_of(text_lines(75))], words)
    assert len(pieces) == 2
    assert all(piece.tokens <= config.PIECE_HARD_STOP for piece in pieces)


def test_pieces_never_cross_a_page():
    pages = [page_of(text_lines(60, "alpha"), 1), page_of(text_lines(60, "beta"), 2)]
    for piece in chunker.cut_document(INFO, pages, words):
        other = "beta" if piece.page == 1 else "alpha"
        assert other not in piece.text


def test_pages_without_text_give_no_pieces():
    pages = [Page(1, ["Hello"], True), page_of(text_lines(10), 2)]
    pieces = chunker.cut_document(INFO, pages, words)
    assert {piece.page for piece in pieces} == {2}


def test_a_very_long_line_is_split():
    lines = ["start of the page", " ".join(f"w{i}" for i in range(700)), "end of the page"]
    pieces = chunker.cut_document(INFO, [page_of(lines)], words)
    assert len(pieces) >= 2
    assert all(piece.tokens <= config.PIECE_HARD_STOP for piece in pieces)
    assert "w699" in " ".join(piece.text for piece in pieces)


def test_a_long_table_repeats_the_table_header():
    column_line = "(Millions) | 2018 | 2017"
    body = [f"Long item name number {i} for the table | {7 * i + 3},500 | {3 * i + 1},250" for i in range(100)]
    lines = ["Consolidated Balance Sheet", column_line] + body
    pieces = chunker.cut_document(INFO, [page_of(lines)], words)
    assert len(pieces) >= 3
    for piece in pieces:
        assert column_line in piece.text.splitlines()  # also in every piece after the first
    assert all(piece.tokens <= config.PIECE_HARD_STOP for piece in pieces)


def test_text_after_a_table_does_not_get_a_table_header():
    lines = ["(Millions) | 2018 | 2017"] + [f"Item {i} | {7 * i + 3},500 | {3 * i + 1},250" for i in range(20)] + text_lines(120)
    pieces = chunker.cut_document(INFO, [page_of(lines)], words)
    assert all("(Millions) | 2018 | 2017" not in piece.text for piece in pieces[2:])


def test_without_the_header_line():
    pieces = chunker.cut_document(INFO, [page_of(text_lines(30))], words, with_header=False)
    assert pieces[0].text.splitlines()[0].startswith("x0 ")


def test_column_line_detection():
    assert chunker.is_column_line("(Millions) | 2018 | 2017 | 2016")
    assert chunker.is_column_line("Three Months Ended | December 31, 2023 | December 31, 2022")
    assert chunker.is_column_line("2018 | 2017")
    assert not chunker.is_column_line("Total assets | $36,500 | $37,987")
    assert not chunker.is_column_line("In 2018 we grew")


# ---------- step 1.4: ids and upload ----------

def test_piece_ids_are_stable_and_different():
    first = build_index.piece_id("3M_2018_10K", 5, 1)
    assert first == build_index.piece_id("3M_2018_10K", 5, 1)
    assert first != build_index.piece_id("3M_2018_10K", 5, 2)
    assert first != build_index.piece_id("3M_2018_10K", 6, 1)
    assert len(first) == 36  # a UUID


def test_build_labels_and_second_run_adds_nothing(tmp_path):
    document = make_document(tmp_path, "ADOBE_2022_10K", [prose_page("a", 40), prose_page("b"), [(72, 72, "Hello")]])
    client, embedder = QdrantClient(":memory:"), FakeEmbedder()

    first = build_index.build([document], client, embedder, words, "test")
    assert first["pieces"] > 2
    assert first["new"] == first["pieces"] and first["changed"] == 0 and first["removed"] == 0
    assert first["qdrant_these"] == first["pieces"]  # Qdrant shows the same count the script printed
    embedded_once = len(embedder.texts)

    second = build_index.build([document], client, embedder, words, "test")
    assert second["new"] == 0 and second["changed"] == 0 and second["same"] == first["pieces"]
    assert len(embedder.texts) == embedded_once  # nothing was embedded again
    assert second["qdrant_all"] == first["qdrant_all"]

    points, _ = client.scroll("test", limit=100, with_payload=True)
    one = points[0].payload
    assert (one["doc"], one["company"], one["year"], one["type"]) == ("ADOBE_2022_10K", "Adobe", 2022, "10K")
    assert (one["source"], one["access"]) == ("real", "public")
    assert one["text"].startswith("Adobe FY2022 10-K, page ")
    assert {point.payload["page"] for point in points} == {1, 2}  # page 3 has no text, so no pieces


def test_a_changed_document_is_updated_and_old_pieces_are_removed(tmp_path):
    document = make_document(tmp_path, "ADOBE_2022_10K", [prose_page("a", 40), prose_page("b", 40)])
    client, embedder = QdrantClient(":memory:"), FakeEmbedder()
    first = build_index.build([document], client, embedder, words, "test")

    make_pdf(document.path, [prose_page("c", 40)])  # same file name, new content, fewer pages
    second = build_index.build([document], client, embedder, words, "test")
    assert second["changed"] > 0 and second["removed"] > 0
    assert second["pieces"] < first["pieces"]
    assert second["qdrant_these"] == second["pieces"]  # no old pieces are left


def test_the_collection_and_indexes_are_made_once(tmp_path):
    client = QdrantClient(":memory:")
    build_index.ensure_collection(client, "test")
    build_index.ensure_collection(client, "test")  # a second call must not fail
    info = client.get_collection("test")
    assert info.config.params.vectors.size == config.EMBED_DIM
