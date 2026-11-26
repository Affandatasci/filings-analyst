"""Tests for step 2.3: the test set builder (evals/build_cases.py) and the loader (evals/cases.py).

No network and no key. Most tests use small made up questions. One test uses the real question file and
skips itself when that file is not there. Run from the project folder:
    python -m pytest -q
"""
import contextlib
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from evals import build_cases, cases  # noqa: E402

NAMES = {"ADOBE_2022_10K", "3M_2018_10K"}
GROUPS = [
    ("domain_relevant", "Information extraction"),
    ("domain_relevant", "Numerical reasoning"),
    ("metrics_generated", "Numerical reasoning"),
    ("novel_generated", None),
]


def printed(function, *args, **kwargs):
    """Run a function and return (its result, everything it printed)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = function(*args, **kwargs)
    return result, buffer.getvalue()


def raw_question(number, doc="ADOBE_2022_10K", kind="domain_relevant", reasoning="Information extraction", page=0):
    """One question as it is written in the FinanceBench question file."""
    return {
        "financebench_id": f"id_{number:03d}", "company": "Adobe", "doc_name": doc, "question_type": kind,
        "question_reasoning": reasoning, "question": f"Question {number}?", "answer": f"Answer {number}",
        "justification": f"Why {number}",
        "evidence": [{"doc_name": doc, "evidence_page_num": page, "evidence_text": f" Evidence text {number} "}],
    }


def many_questions(total=40):
    """Questions that go round the 4 groups in turn."""
    questions = []
    for number in range(total):
        kind, reasoning = GROUPS[divmod(number, 4)[1]]
        questions.append(raw_question(number, kind=kind, reasoning=reasoning))
    return questions


def case_dict(number, split="test", page=3):
    """One line of a cases file, as a dict."""
    return {
        "id": f"id_{number:03d}", "split": split, "company": "Adobe", "doc": "ADOBE_2022_10K", "doc_type": "10K",
        "question_type": "domain_relevant", "reasoning_type": "Information extraction", "question": f"Question {number}?",
        "answer": f"Answer {number}", "justification": "", "evidence": [{"page": page, "text": "evidence"}],
    }


def broken(**changes):
    line = case_dict(1)
    line.update(changes)
    return line


def write_cases(tmp_path, *lines):
    """Write lines (dicts, or text for a line that is not JSON) to a cases file and return its path."""
    path = tmp_path / "test_cases.jsonl"
    path.write_text("\n".join(line if isinstance(line, str) else json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    return path


def fake_cases(counts):
    made, number = [], 0
    for split, count in counts.items():
        for _ in range(count):
            number += 1
            made.append(cases.parse_case(case_dict(number, split), NAMES))
    return made


def make_pdf(path, pages):
    document = pymupdf.open()
    for number in range(pages):
        document.new_page().insert_text((72, 72), f"Page {number + 1}")
    document.save(path)
    document.close()


@pytest.fixture
def small_set(tmp_path, monkeypatch):
    """A cases file with the planned counts (103 small cases) in a temp folder."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "PDF_LIST", tmp_path / "pdf_list.txt")
    (tmp_path / "pdf_list.txt").write_text("ADOBE_2022_10K\n", encoding="utf-8")
    (tmp_path / "cases").mkdir()
    splits = ["practice"] * 10 + ["test"] * 75 + ["holdout"] * 18
    lines = [case_dict(number, split) for number, split in enumerate(splits, start=1)]
    cases.cases_path().write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    return tmp_path


def fake_project(tmp_path, monkeypatch, questions, pdf_pages=None):
    """Point the settings at a temp folder that holds a small question file and the plan counts for 40 questions."""
    (tmp_path / "pdf_list.txt").write_text("ADOBE_2022_10K\n", encoding="utf-8")
    (tmp_path / "questions.jsonl").write_text("\n".join(json.dumps(item) for item in questions) + "\n", encoding="utf-8")
    (tmp_path / "pdf").mkdir()
    if pdf_pages:
        make_pdf(tmp_path / "pdf" / "ADOBE_2022_10K.pdf", pdf_pages)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "PDF_LIST", tmp_path / "pdf_list.txt")
    monkeypatch.setattr(config, "QUESTIONS_FILE", tmp_path / "questions.jsonl")
    monkeypatch.setattr(config, "PDF_DIR", tmp_path / "pdf")
    monkeypatch.setattr(cases, "EXPECTED_COUNTS", {"practice": 4, "test": 28, "holdout": 8})
    monkeypatch.setattr(build_cases, "PRACTICE_COUNT", 4)
    monkeypatch.setattr(build_cases, "HOLDOUT_COUNT", 8)


# ---------- the loader ----------

def test_load_cases_reads_every_field(tmp_path):
    path = write_cases(tmp_path, case_dict(1, "practice"), case_dict(2, "test", page=7))
    loaded = cases.load_cases(path=path, names=NAMES)
    assert [case.id for case in loaded] == ["id_001", "id_002"]
    first = loaded[0]
    assert (first.split, first.company, first.doc, first.doc_type) == ("practice", "Adobe", "ADOBE_2022_10K", "10K")
    assert (first.question, first.answer, first.reasoning_type) == ("Question 1?", "Answer 1", "Information extraction")
    assert first.pages == [("ADOBE_2022_10K", 3)] and first.evidence[0].text == "evidence"
    assert loaded[1].pages == [("ADOBE_2022_10K", 7)]


def test_load_cases_can_pick_one_split(tmp_path):
    path = write_cases(tmp_path, case_dict(1, "practice"), case_dict(2, "test"), case_dict(3, "holdout"), case_dict(4, "test"))
    assert [case.id for case in cases.load_cases("test", path, NAMES)] == ["id_002", "id_004"]
    assert [case.id for case in cases.load_cases("holdout", path, NAMES)] == ["id_003"]
    with pytest.raises(cases.CaseError, match="Unknown split 'dev'"):
        cases.load_cases("dev", path, NAMES)


def test_a_missing_file_says_how_to_make_it(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    with pytest.raises(cases.CaseError, match=r"python -m evals\.build_cases"):
        cases.load_cases()


BAD_LINES = [
    ("this is not json", "not valid JSON"),
    ("[1, 2]", "not a JSON object"),
    (broken(question=""), "'question' is missing or empty"),
    (broken(split="dev"), "split 'dev' is not one of"),
    (broken(reasoning_type="Other"), "reasoning_type 'Other'"),
    (broken(doc="NOPE_2020_10K"), "unknown doc 'NOPE_2020_10K'"),
    (broken(evidence=[]), "no evidence page"),
    (broken(evidence="page 3"), "no evidence page"),
    (broken(evidence=[{"page": 0, "text": "x"}]), "pages start at 1"),
    (broken(evidence=[{"page": True, "text": "x"}]), "pages start at 1"),
    (broken(evidence=[{"page": "3", "text": "x"}]), "pages start at 1"),
    (broken(evidence=[{"page": 3, "text": " "}]), "has no text"),
    (broken(justification=5), "'justification' must be text"),
]


@pytest.mark.parametrize("line, problem", BAD_LINES)
def test_a_bad_line_stops_the_load_and_names_the_line(tmp_path, line, problem):
    path = write_cases(tmp_path, case_dict(0), line)  # the bad line is line 2
    with pytest.raises(cases.CaseError, match=rf"test_cases\.jsonl line 2: .*{re.escape(problem)}"):
        cases.load_cases(path=path, names=NAMES)


def test_a_repeated_id_is_refused(tmp_path):
    path = write_cases(tmp_path, case_dict(1), case_dict(2), case_dict(1))
    with pytest.raises(cases.CaseError, match="line 3: id 'id_001' is already used on line 1"):
        cases.load_cases(path=path, names=NAMES)


def test_blank_lines_are_skipped_and_an_empty_file_is_refused(tmp_path):
    assert len(cases.load_cases(path=write_cases(tmp_path, case_dict(1), "", "   ", case_dict(2)), names=NAMES)) == 2
    with pytest.raises(cases.CaseError, match="has no cases"):
        cases.load_cases(path=write_cases(tmp_path, ""), names=NAMES)


def test_count_problems_are_empty_when_the_counts_match_the_plan():
    assert cases.count_problems(fake_cases(cases.EXPECTED_COUNTS)) == []


def test_count_problems_say_which_split_is_wrong():
    problems = cases.count_problems(fake_cases({"practice": 10, "test": 74, "holdout": 19}))
    assert problems == ["test has 74 cases, the plan says 75", "holdout has 19 cases, the plan says 18"]


# ---------- the check and read commands ----------

def test_the_check_command_prints_the_counts_and_passes(small_set):
    _, out = printed(cases.main, [])
    assert "103 cases: practice 10, test 75, holdout 18" in out
    assert "Question type (practice / test / holdout):" in out
    assert "All 103 lines pass the checks and the counts match the plan." in out


def test_the_check_command_fails_when_the_counts_are_wrong(small_set):
    lines = cases.cases_path().read_text(encoding="utf-8").splitlines()
    cases.cases_path().write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")  # one holdout case less
    with pytest.raises(SystemExit, match="PROBLEM: holdout has 17 cases, the plan says 18"):
        printed(cases.main, [])


def test_the_check_command_stops_on_a_bad_line(small_set):
    with cases.cases_path().open("a", encoding="utf-8") as handle:
        handle.write("this is not json\n")
    with pytest.raises(SystemExit, match="line 104: not valid JSON"):
        printed(cases.main, [])


def test_read_writes_a_text_file_with_question_reference_and_evidence(small_set):
    _, out = printed(cases.main, ["--read", "practice"])
    text = (small_set / "cases" / "read_practice.txt").read_text(encoding="utf-8")
    assert "Wrote 10 questions" in out
    assert text.count("Reference: ") == 10 and "Question 1?" in text and "Reference: Answer 1" in text
    assert "Evidence on page 3:" in text and "Question 11?" not in text  # case 11 is the first test case


def test_read_refuses_the_holdout(small_set):
    with pytest.raises(SystemExit, match="opened once"):
        printed(cases.main, ["--read", "holdout"])
    assert not (small_set / "cases" / "read_holdout.txt").exists()


def test_the_reading_text_shows_the_why_line_only_when_there_is_one():
    plain = cases.parse_case(case_dict(1), NAMES)
    explained = cases.parse_case({**case_dict(2), "justification": "Because of the table"}, NAMES)
    assert "Why:" not in cases.reading_text([plain])
    assert "Why:       Because of the table" in cases.reading_text([explained])


# ---------- building the file ----------

@pytest.mark.parametrize("raw, clean", [
    (None, "Not labelled"),
    ("", "Not labelled"),
    ("Information extraction", "Information extraction"),
    ("Numerical reasoning", "Numerical reasoning"),
    ("Logical reasoning (based on numerical reasoning)", "Logical reasoning"),
    ("Logical reasoning (based on numerical reasoning) OR Numerical reasoning OR Logical reasoning", "Logical reasoning"),
    ("Logical reasoning (based on numerical reasoning) OR Logical reasoning", "Logical reasoning"),
    ("Numerical reasoning OR information extraction", "Numerical reasoning"),
    ("Numerical reasoning OR Logical reasoning", "Numerical reasoning"),
    ("Information extraction OR Logical reasoning OR", "Information extraction"),
    ("Information extraction OR Logical reasoning", "Information extraction"),
])
def test_the_reasoning_type_is_cleaned(raw, clean):
    assert build_cases.clean_reasoning(raw) == clean
    assert clean in cases.REASONING_TYPES


def test_an_unknown_reasoning_type_is_refused():
    with pytest.raises(ValueError, match="Unknown reasoning type 'Creative writing'"):
        build_cases.clean_reasoning("Creative writing")


def test_holdout_quotas_follow_the_mix_and_add_up():
    quotas = build_cases.holdout_quotas({"a": 40, "b": 39, "c": 14}, total=18)
    assert quotas == {"a": 8, "b": 7, "c": 3}  # a and c had the biggest left over share
    assert sum(quotas.values()) == 18


def test_holdout_quotas_never_ask_for_more_than_a_group_has():
    counts = {"a": 1, "b": 1, "c": 50}
    quotas = build_cases.holdout_quotas(counts, total=6)
    assert sum(quotas.values()) == 6 and all(quotas[group] <= counts[group] for group in counts)


def test_the_first_questions_are_practice_and_the_counts_are_exact():
    splits = build_cases.assign_splits(many_questions(40), practice=4, holdout=8)
    assert splits[:4] == ["practice"] * 4
    assert [splits.count(split) for split in cases.SPLITS] == [4, 28, 8]


def test_the_split_is_the_same_every_time_and_follows_the_seed(monkeypatch):
    questions = many_questions(40)
    first = build_cases.assign_splits(questions, practice=4, holdout=8)
    assert build_cases.assign_splits(questions, practice=4, holdout=8) == first
    monkeypatch.setattr(build_cases, "SPLIT_SEED", 999)
    assert build_cases.assign_splits(questions, practice=4, holdout=8) != first


def test_the_holdout_does_not_move_when_the_order_changes():
    def holdout_ids(questions):
        splits = build_cases.assign_splits(questions, practice=4, holdout=8)
        return {question["financebench_id"] for question, split in zip(questions, splits) if split == "holdout"}

    questions = many_questions(40)
    reordered = questions[:4] + questions[4:][::-1]  # the same practice questions, the others in the other order
    assert holdout_ids(reordered) == holdout_ids(questions)


def test_every_group_gives_its_share_to_the_holdout():
    questions = many_questions(40)  # after 4 practice questions there are 9 in each of the 4 groups
    splits = build_cases.assign_splits(questions, practice=4, holdout=8)
    groups = Counter(build_cases.stratum(question) for question, split in zip(questions, splits) if split == "holdout")
    assert sorted(groups.values()) == [2, 2, 2, 2]


def test_a_case_line_has_the_pages_moved_to_start_at_one():
    line = build_cases.make_case(raw_question(5, page=0), "test")
    assert line["evidence"] == [{"page": 0 + config.PAGE_OFFSET, "text": "Evidence text 5"}]
    assert (line["id"], line["split"], line["company"], line["doc"], line["doc_type"]) == ("id_005", "test", "Adobe", "ADOBE_2022_10K", "10K")
    assert (line["question"], line["answer"], line["justification"]) == ("Question 5?", "Answer 5", "Why 5")
    assert line["reasoning_type"] == "Information extraction"


def test_a_missing_justification_becomes_empty_text():
    question = raw_question(1)
    question["justification"] = None
    assert build_cases.make_case(question, "test")["justification"] == ""


def test_a_company_that_does_not_match_the_name_list_is_refused():
    question = raw_question(1)
    question["company"] = "Adobe Inc"
    with pytest.raises(ValueError, match="ingest/names.py says 'Adobe'"):
        build_cases.make_case(question, "test")


def test_evidence_from_another_document_is_refused():
    question = raw_question(1)
    question["evidence"][0]["doc_name"] = "3M_2018_10K"
    with pytest.raises(ValueError, match="another document"):
        build_cases.make_case(question, "test")


def test_the_file_is_the_same_every_time_and_the_folder_hides_it_from_git(tmp_path):
    questions = many_questions(40)
    accented = "Caf" + chr(233) + " revenue in USD?"  # text that is not plain ASCII
    questions[7]["question"] = accented
    lines = build_cases.build_lines(questions, practice=4, holdout=8)
    path = tmp_path / "cases" / "test_cases.jsonl"
    build_cases.write_lines(lines, path)
    first = path.read_bytes()
    build_cases.write_lines(lines, path)
    assert path.read_bytes() == first and first.count(b"\n") == 40 and b"\r" not in first
    assert (path.parent / ".gitignore").read_bytes().endswith(b"*\n")
    assert not list(path.parent.glob("*.tmp"))
    loaded = cases.load_cases(path=path, names={"ADOBE_2022_10K"})
    assert [case.id for case in loaded] == [f"id_{number:03d}" for number in range(40)]
    assert loaded[7].question == accented
    assert chr(233).encode("utf-8") in first  # written as real UTF-8 text, not as an escape


def test_check_pages_finds_a_page_outside_its_pdf(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PDF_DIR", tmp_path)
    make_pdf(tmp_path / "ADOBE_2022_10K.pdf", 3)
    good = build_cases.make_case(raw_question(1, page=2), "test")  # our page 3, the last page
    bad = build_cases.make_case(raw_question(2, page=3), "test")  # our page 4, one too far
    problems, checked = build_cases.check_pages([good, bad])
    assert checked == 1 and len(problems) == 1
    assert "id_002: page 4 is outside ADOBE_2022_10K, which has 3 pages" in problems[0]


def test_check_pages_skips_quietly_when_the_pdfs_are_not_there(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PDF_DIR", tmp_path)
    assert build_cases.check_pages([build_cases.make_case(raw_question(1), "test")]) is None


def test_the_build_command_end_to_end(tmp_path, monkeypatch):
    fake_project(tmp_path, monkeypatch, many_questions(40))  # no PDFs: the page check is skipped
    _, out = printed(build_cases.main, [])
    assert "40 questions point to our 1 PDFs." in out and "Pages not checked" in out
    assert "40 cases: practice 4, test 28, holdout 8" in out
    assert "All 40 lines pass the checks and the counts match the plan." in out
    assert len(cases.load_cases()) == 40


def test_the_build_command_writes_nothing_when_a_page_is_outside_its_pdf(tmp_path, monkeypatch):
    questions = many_questions(40)
    questions[20]["evidence"][0]["evidence_page_num"] = 9
    fake_project(tmp_path, monkeypatch, questions, pdf_pages=2)
    with pytest.raises(SystemExit, match=r"(?s)Nothing written.*id_020: page 10 is outside ADOBE_2022_10K, which has 2 pages"):
        printed(build_cases.main, [])
    assert not cases.cases_path().exists()


def test_the_build_command_says_when_the_question_file_is_missing(tmp_path, monkeypatch):
    fake_project(tmp_path, monkeypatch, many_questions(40))
    (tmp_path / "questions.jsonl").unlink()
    with pytest.raises(SystemExit, match="download_pdfs"):
        printed(build_cases.main, [])


# ---------- the real question file ----------

def test_the_real_question_file_gives_the_planned_103_cases(tmp_path):
    if not config.QUESTIONS_FILE.exists() or not config.PDF_LIST.exists():
        pytest.skip("the question file or data/pdf_list.txt is missing")
    questions = build_cases.read_questions()
    lines = build_cases.build_lines(questions)
    assert len(lines) == sum(cases.EXPECTED_COUNTS.values()) == 103
    assert Counter(line["split"] for line in lines) == cases.EXPECTED_COUNTS
    assert len({line["id"] for line in lines}) == 103
    assert [line["id"] for line in lines[:10]] == [question["financebench_id"] for question in questions[:10]]
    assert all(line["split"] == "practice" for line in lines[:10])
    for line, question in zip(lines, questions):
        assert [item["page"] for item in line["evidence"]] == [item["evidence_page_num"] + config.PAGE_OFFSET for item in question["evidence"]]
    path = build_cases.write_lines(lines, tmp_path / "test_cases.jsonl")
    loaded = cases.load_cases(path=path, names=cases.read_names())
    assert len(loaded) == 103 and cases.count_problems(loaded) == []
    checked = build_cases.check_pages(lines)
    if checked is not None:  # the PDFs are in the PDF folder
        assert checked == ([], 37)
