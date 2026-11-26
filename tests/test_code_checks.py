"""Tests for step 2.4: the free checks and the number reader.

No network, no key, no model. Run from the project folder:
    python -m pytest -q
"""
import contextlib
import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals import code_checks as cc  # noqa: E402
from evals.cases import Case, Evidence  # noqa: E402

DOC = "3M_2018_10K"


def make_case(answer="$1,577 million", pages=(60,)):
    return Case(
        id="financebench_id_00001", split="test", company="3M", doc=DOC, doc_type="10K",
        question_type="metrics-generated", reasoning_type="Numerical reasoning", question="What was the capex?",
        answer=answer, justification="", evidence=tuple(Evidence(page=page, text="evidence") for page in pages),
    )


def make_record(answer="Capex was 1,577 million dollars.", citations=((DOC, 60),), found=True, pieces=((DOC, 60), (DOC, 12)),
                format_ok=True, format_note=""):
    return {
        "answer": answer,
        "citations": [{"doc": doc, "page": page} for doc, page in citations],
        "found": found,
        "pieces": [{"doc": doc, "page": page, "piece": 1, "score": 0.5} for doc, page in pieces],
        "format_ok": format_ok,
        "format_note": format_note,
    }


def values(text):
    return [q.value for q in cc.parse_numbers(text)]


# ---------- the number reader: 12 hand written tests ----------

def test_parse_1_dollar_sign_commas_and_million():
    [q] = cc.parse_numbers("Capex was $1,577 million.")
    assert q.value == 1577e6 and q.scale == 1e6 and q.decimals == 0 and not q.percent and not q.negative


def test_parse_2_billion_with_decimals():
    [q] = cc.parse_numbers("about 1.58 billion")
    assert q.value == pytest.approx(1.58e9) and q.decimals == 2


def test_parse_3_brackets_are_negative():
    [q] = cc.parse_numbers("net loss of (1,234)")
    assert q.negative and q.value == -1234


def test_parse_4_minus_sign_and_share_of_100():
    [q] = cc.parse_numbers("margin fell to -5.2%")
    assert q.percent and q.negative and q.raw == 5.2 and q.value == -5.2


def test_parse_5_the_word_percent():
    [q] = cc.parse_numbers("grew 12.5 percent")
    assert q.percent and q.raw == 12.5


def test_parse_6_dollars_and_cents_keep_their_decimals():
    [q] = cc.parse_numbers("EPS of $8.70")
    assert q.raw == 8.7 and q.decimals == 2


def test_parse_7_years_are_flagged_and_fy_is_ignored():
    assert cc.parse_numbers("FY2018 revenue") == []
    [q] = cc.parse_numbers("in 2018")
    assert q.year
    assert cc.key_numbers("in 2018") == []


def test_parse_8_form_names_are_not_numbers():
    assert cc.parse_numbers("the 10-K, the 10-Q and the 8-K") == []


def test_parse_9_the_company_3m_is_not_three_million():
    assert values("3M's net income was $5.3 billion") == [5.3e9]


def test_parse_10_glued_units_need_a_dollar_sign_or_a_decimal_point():
    assert values("$5M and 2.3B and 10k") == [5e6, 2.3e9]  # 10k has no dollar sign and no point: left out below
    assert values("10k") == []


def test_parse_11_negative_in_words():
    assert values("negative 4.5 million") == [-4.5e6]
    assert values("minus 3") == [-3]


def test_parse_12_dates_pages_and_labels_are_left_out():
    found = cc.parse_numbers("On December 31, 2018 (page 60, Note 12, Item 7) it was 2nd, 401(k)")
    assert [q.text for q in found] == ["2018"] and found[0].year


def test_parse_list_markers_are_not_losses():
    assert values("(1) cash and (2) debt of $4.3 billion") == [4.3e9]
    assert values("a loss of (15) and ($2)") == [-15, -2]  # two digits or a dollar sign: a number


def test_parse_unicode_minus_multiples_and_currency_codes():
    assert values("−7") == [-7]
    assert values("2.5x the cost") == [2.5]
    assert values("USD 3 million") == [3e6]
    assert values("it cost €4 thousand") == [4e3]


def test_parse_ignores_numbers_inside_doc_names_and_words():
    assert cc.parse_numbers("see 3M_2018_10K for details") == []
    assert cc.parse_numbers("a 12abc code") == []


def test_parse_percent_sign_with_space_and_plural_units():
    assert [(q.percent, q.raw) for q in cc.parse_numbers("12.5 % and 3 per cent")] == [(True, 12.5), (True, 3.0)]
    assert values("2 millions") == [2e6]


def test_parse_empty_and_none():
    assert cc.parse_numbers("") == []
    assert cc.parse_numbers(None) == []


# ---------- do two numbers agree? ----------

def agree(a, b):
    return cc.numbers_agree(cc.parse_numbers(a)[0], cc.parse_numbers(b)[0])


def test_rounding_billion_against_million():
    assert agree("1.58 billion", "1,577 million") and agree("1,577 million", "1.58 billion")
    assert not agree("1.60 billion", "1,577 million")


def test_no_unit_may_be_in_millions():
    assert agree("$1577.00", "1,577 million")
    assert agree("$1577.00", "$1.58 billion")
    assert not agree("$1577.00", "1,677 million")


def test_share_of_100_against_a_fraction():
    assert agree("12.5%", "0.125") and agree("0.66", "66%")
    assert not agree("12.5%", "12.4%")
    assert agree("12.46%", "12.5%")


def test_trailing_zeros_and_scale_mismatch():
    assert agree("8.7", "8.70")
    assert not agree("5 million", "5 billion")
    assert not agree("5", "5000")  # neither has a unit: no guessing


def test_signs_are_not_compared():
    assert agree("(1,234)", "1,234")


# ---------- number_match ----------

def test_number_match_pass_with_units_and_rounding():
    mark = cc.number_match("$1577.00", "Capital expenditure was 1,577 million dollars in FY2018.")
    assert mark.status == cc.PASS and mark.value == 1.0
    assert cc.number_match("$8.70", "EPS was $8.7.").status == cc.PASS
    assert cc.number_match("24.26%", "about 24.3 percent").status == cc.PASS


def test_number_match_fail_names_the_missing_number():
    mark = cc.number_match("$1.2 billion", "It was $1.5 billion.")
    assert mark.status == cc.FAIL and "1.2 billion" in mark.reason and mark.value == 0.0


def test_number_match_needs_every_number_and_reports_the_share():
    mark = cc.number_match("FY2018: $5.2 billion, FY2019: $6.1 billion", "In FY2018 it was $5.2 billion.")
    assert mark.status == cc.FAIL and mark.value == 0.5 and "1 of 2" in mark.reason


def test_number_match_rests_on_the_first_sentence_with_a_number():
    assert [q.text for q in cc.answer_numbers("36%. The answer assumes FY2023 is the 12 months to June.")] == ["36%"]
    assert [q.text for q in cc.answer_numbers("Yes. Segments are 20%, 39% and 35%.")] == ["20%", "39%", "35%"]
    assert [q.text for q in cc.answer_numbers("No. Verizon's debt fell by $229 million.")] == ["$229 million"]
    assert cc.answer_numbers("Adj. EBITDA was $2,018mn in FY 2023")[0].value == 2018e6
    assert cc.number_match("36%. It assumes the 12 months to June.", "The rate was 36%.").status == cc.PASS
    assert cc.answer_numbers("") == [] and cc.answer_numbers("Yes.") == []


def test_number_match_skips_when_the_reference_has_no_number():
    assert cc.number_match("Yes, the company paid a dividend.", "Yes.").status == cc.SKIP
    assert cc.number_match("In 2018 it did.", "2018").status == cc.SKIP  # only a year


def test_number_match_empty_answer_fails():
    assert cc.number_match("$5 million", "").status == cc.FAIL


# ---------- search marks ----------

def test_search_marks_first_place_recall_precision():
    pages = [(DOC, 60), (DOC, 61)]
    s = cc.search_marks(pages, [(DOC, 1), (DOC, 61), (DOC, 61), (DOC, 9)])
    assert s["hit"] and s["place"] == 2 and s["mrr"] == 0.5
    assert s["recall"] == 0.5 and not s["all_pages"] and s["precision"] == 0.5
    both = cc.search_marks(pages, [(DOC, 60), (DOC, 61)])
    assert both["all_pages"] and both["recall"] == 1.0 and both["precision"] == 1.0


def test_search_marks_miss_and_empty():
    s = cc.search_marks([(DOC, 60)], [(DOC, 1), (DOC, 2)])
    assert not s["hit"] and s["place"] is None and s["mrr"] == 0.0 and s["recall"] == 0.0
    empty = cc.search_marks([(DOC, 60)], [])
    assert empty["precision"] == 0.0 and not empty["hit"]


# ---------- the checks on one answer ----------

def test_run_checks_all_good():
    marks = cc.run_checks(make_case(), make_record())
    assert list(marks) == list(cc.CASE_MARKS)
    assert all(mark.status == cc.PASS for mark in marks.values())
    assert marks["mrr"].value == 1.0 and marks["precision"].value == 0.5


def test_run_checks_search_failed_but_answer_right_is_visible():
    record = make_record(pieces=((DOC, 3), (DOC, 4)), citations=((DOC, 3),))
    marks = cc.run_checks(make_case(), record)
    assert marks["found_page"].status == cc.FAIL and marks["number_match"].status == cc.PASS
    assert marks["citation_valid"].status == cc.PASS and marks["citation_hit"].status == cc.FAIL


def test_citation_valid_catches_an_invented_page():
    mark = cc.check_citation_valid(make_record(citations=((DOC, 99),)))
    assert mark.status == cc.FAIL and "p99" in mark.reason
    other_doc = cc.check_citation_valid(make_record(citations=(("Adobe_2020_10K", 60),)))
    assert other_doc.status == cc.FAIL


def test_citation_valid_without_citations():
    assert cc.check_citation_valid(make_record(citations=(), found=True)).status == cc.FAIL
    assert cc.check_citation_valid(make_record(citations=(), found=False)).status == cc.PASS


def test_citation_hit_and_json_shape_and_length():
    assert cc.check_citation_hit(make_case(), make_record(citations=())).status == cc.FAIL
    assert cc.check_citation_hit(make_case(), make_record(citations=((DOC, 12),))).status == cc.FAIL
    assert cc.check_json_shape(make_record(format_ok=False, format_note="found is missing")).reason == "found is missing"
    assert cc.check_json_shape(make_record(format_ok=False)).status == cc.FAIL
    assert cc.check_length(make_record(answer="word " * 81)).status == cc.FAIL
    assert cc.check_length(make_record(answer="")).status == cc.FAIL
    assert cc.check_length(make_record(answer="word " * 80)).status == cc.PASS


def test_not_in_documents_for_trap_questions():
    assert cc.check_not_in_documents(make_record(found=False)).status == cc.PASS
    assert cc.check_not_in_documents(make_record(found=True)).status == cc.FAIL
    assert cc.check_not_in_documents(make_record(found=True), should_be_found=True).status == cc.PASS


def test_mark_to_dict_and_marks_line():
    marks = cc.run_checks(make_case(), make_record())
    data = marks["mrr"].to_dict()
    assert data == {"status": "pass", "reason": "first right page at place 1 of 2", "value": 1.0}
    assert "value" not in marks["json_shape"].to_dict()
    line = cc.marks_line(marks)
    assert "found_page=ok" in line and "number_match=ok" in line
    assert "number_match=--" in cc.marks_line(cc.run_checks(make_case(answer="Yes."), make_record()))


def test_main_prints_both_texts_and_the_verdict():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        cc.main(["1.58 billion", "The amount was 1,577 million dollars."])
    text = buffer.getvalue()
    assert "Reference: 1.58 billion" in text and "Answer: 1,577 million" in text and "number_match: PASS" in text
