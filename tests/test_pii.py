"""Tests for step 3.3: hiding personal data, and the checks for personal data and secret words in an answer.

30 test strings. The exact rules (emails, phone numbers, employee IDs, the known fake names) are checked strictly.
The name finder of Presidio is a model, so it is checked loosely: it must be wired in and find some names, and it may miss others.
Without Presidio and the small spaCy model those tests are skipped and the rest still runs.

To see every miss and every wrong mask in a table (no pytest):
    python -m tests.test_pii
"""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from scripts import make_fake_pdfs  # noqa: E402
from security import fakes, pii  # noqa: E402


@pytest.fixture(autouse=True)
def fake_data(tmp_path, monkeypatch):
    """Every test gets its own copy of the 20 fake files and their manifest."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    make_fake_pdfs.build(tmp_path / "fake_pdfs")


# ---------- the 30 strings ----------
# 15 strings for the exact patterns: (text, the strings that must be gone, the placeholder that must be there)
PATTERN_CASES = [
    ("Write to maria.okafor@northwind.test for details.", ["maria.okafor@northwind.test"], "[EMAIL]"),
    ("Contact: JOHN.SMITH@NORTHWIND.TEST.", ["JOHN.SMITH@NORTHWIND.TEST"], "[EMAIL]"),
    ("Send it to a.b+hr@northwind.test, or to c@northwind.test.", ["a.b+hr@northwind.test", "c@northwind.test"], "[EMAIL]"),
    ("Reply to <ir@3m.com> before Friday.", ["ir@3m.com"], "[EMAIL]"),
    ("Mail: first.last@sub.example.co.uk", ["first.last@sub.example.co.uk"], "[EMAIL]"),
    ("Call (212) 555-0142 today.", ["(212) 555-0142"], "[PHONE]"),
    ("Call +1 415 555 0167 today.", ["415 555 0167"], "[PHONE]"),
    ("Call 312-555-0123 today.", ["312-555-0123"], "[PHONE]"),
    ("Call 555-0142 today.", ["555-0142"], "[PHONE]"),
    ("Call 1 (800) 555-0199 today.", ["(800) 555-0199"], "[PHONE]"),
    ("Call 212.555.0100 today.", ["212.555.0100"], "[PHONE]"),
    ("Employee EMP40213 joined in May.", ["EMP40213"], "[EMP ID]"),
    ("Employee emp 40213 joined in May.", ["40213"], "[EMP ID]"),
    ("Employee EMP-40213 joined in May.", ["40213"], "[EMP ID]"),
    ("EMP40213 and EMP10001 are both listed.", ["EMP40213", "EMP10001"], "[EMP ID]"),
]
# 5 strings with a known fake name (templates: {full}, {first}, {last} come from the manifest)
KNOWN_NAME_CASES = [
    "{full} signed the letter.",
    "Dear {first}, welcome to the team.",
    "Regards, Mr {last}.",
    "EMP40213 | {full} | Engineer | 85,000",
    "please ask {full_lower} about the budget",
]
# 3 strings with names that are not in the manifest: only a name finder can mask these, and it may miss some
UNKNOWN_NAME_CASES = [
    ("Dr. Alan Whitcombe will present the results.", "Alan Whitcombe"),
    ("Dear Margaret Thompson, welcome.", "Margaret Thompson"),
    ("Please ask Hiroshi Watanabe about the budget.", "Hiroshi Watanabe"),
]
# 7 strings that hold no personal data and must come out exactly as they went in
KEEP_CASES = [
    "Capital expenditures were $1,577 million in fiscal year 2018.",
    "Net sales rose to 46,100 million, up 4.5 percent.",
    "The 10-K for the period ended December 31, 2022 is on page 60.",
    "Form 8-K filed 2022-06-30 reports a ratio of 1.84.",
    "EMPLOYEES 120,000 worked in 3M plants.",
    "Account 12345678 was credited 10,000,000,000 dollars.",
    "Boeing and Adobe reported revenue of $1,577 million in 2018 on page 60.",
]
assert len(PATTERN_CASES) + len(KNOWN_NAME_CASES) + len(UNKNOWN_NAME_CASES) + len(KEEP_CASES) == 30


def known_name_texts():
    full = fakes.known_names()[0]
    first, last = full.split(" ", 1)
    texts = [template.format(full=full, first=first, last=last, full_lower=full.lower()) for template in KNOWN_NAME_CASES]
    wanted = [full, first, last, full, full.lower()]
    return list(zip(texts, wanted))


# ---------- the exact layers ----------

@pytest.mark.parametrize("text, gone, label", PATTERN_CASES)
def test_the_patterns_mask_every_email_phone_number_and_employee_id(text, gone, label):
    masked, counts = pii.mask_text(text, use_presidio=False, use_known=False)
    for value in gone:
        assert value not in masked
    assert label in masked
    assert sum(counts.values()) >= 1


def test_the_known_name_cases_use_names_from_the_manifest():
    assert len(known_name_texts()) == 5
    assert all(name in text.lower() or name.lower() in text.lower() for text, name in known_name_texts())


@pytest.mark.parametrize("index", range(5))
def test_the_layer_of_known_fake_names_hides_full_names_first_names_and_last_names(index):
    text, name = known_name_texts()[index]
    masked, counts = pii.mask_text(text, use_presidio=False, use_known=True)
    assert name.lower() not in masked.lower()
    assert "[NAME]" in masked and counts.get("PERSON", 0) >= 1
    assert pii.mask_text(text, use_presidio=False, use_known=False)[0].lower().count(name.lower()) >= 1  # without the layer the name stays


@pytest.mark.parametrize("text", KEEP_CASES)
def test_text_without_personal_data_comes_out_unchanged(text):
    assert pii.mask_text(text, use_presidio=False) == (text, {})


def test_masking_is_the_same_every_time():
    text = f"{fakes.known_names()[1]} (EMP40213) wrote to a.b@northwind.test from (212) 555-0142."
    assert pii.mask_text(text, use_presidio=False) == pii.mask_text(text, use_presidio=False)
    masked, counts = pii.mask_text(text, use_presidio=False)
    assert masked == "[NAME] ([EMP ID]) wrote to [EMAIL] from [PHONE]." and counts == {"PERSON": 1, "EMP_ID": 1, "EMAIL": 1, "PHONE": 1}


def test_a_name_inside_a_longer_word_is_left_alone():
    last = fakes.known_names()[0].split(" ")[1]
    text = f"The {last.lower()}ification of the plan was approved."
    assert pii.mask_text(text, use_presidio=False)[0] == text


def test_the_manifest_values_of_every_file_are_masked():
    # everything the manifest lists as personal data is found by the layers that need no model
    for item in fakes.files():
        for value in item["emails"] + item["phones"] + item["emp_ids"] + item["people"]:
            masked, counts = pii.mask_text(f"see {value} here", use_presidio=False)
            assert value not in masked and counts, value


# ---------- the Presidio layer (loose) ----------

@pytest.fixture()
def presidio():
    pytest.importorskip("presidio_analyzer")
    ready, why = pii.presidio_status()
    if not ready:
        pytest.skip(f"Presidio is not ready: {why}")


def test_presidio_finds_some_unknown_names_and_keeps_the_numbers(presidio):
    found = sum(name not in pii.mask_text(text, use_presidio=True, use_known=False)[0] for text, name in UNKNOWN_NAME_CASES)
    assert found >= 1, "the name finder found none of the 3 unknown names: is the spaCy model the right one?"
    changed = [text for text in KEEP_CASES if pii.mask_text(text, use_presidio=True, use_known=False)[0] != text]
    assert len(changed) <= 2, changed  # a model may be wrong now and then, but not often


def test_presidio_status_says_ready_or_gives_a_reason():
    ready, why = pii.presidio_status()
    assert (ready and why == "") or (not ready and why)


def test_without_presidio_the_other_layers_still_work(monkeypatch):
    monkeypatch.setitem(pii._presidio, "analyzer", None)
    monkeypatch.setitem(pii._presidio, "problem", "not installed")
    assert pii.presidio_status() == (False, "not installed")
    assert pii.mask_text("Mail me at a@northwind.test", use_presidio=True)[0] == "Mail me at [EMAIL]"


# ---------- which pieces are masked ----------

def hit(access, text="Mail a.b@northwind.test or call (212) 555-0142.", doc="MEMO_BOARD_01"):
    from retrieval.search import Hit

    return Hit(0.9, doc, "Northwind", 2024, "FY", "MEMO", 1, 1, text, "fake", access)


@pytest.mark.parametrize("role, access, expected", [
    ("guest", "public", False), ("guest", "internal", True), ("guest", "hr", True),
    ("analyst", "public", False), ("analyst", "internal", True), ("analyst", "hr", True),
    ("hr", "public", False), ("hr", "internal", False), ("hr", "hr", False),
    (None, "internal", False), (None, "hr", False),
])
def test_which_pieces_need_masking_for_which_role(role, access, expected):
    assert pii.needs_masking(hit(access), role) is expected


def test_mask_hits_masks_internal_and_hr_pieces_for_roles_below_hr_and_reports_it():
    hits = [hit("public", "Call (212) 555-0142."), hit("internal"), hit("hr")]
    out, events = pii.mask_hits(hits, "analyst", use_presidio=False)
    assert out[0].text == "Call (212) 555-0142."  # a public piece is never masked
    assert out[1].text == out[2].text == "Mail [EMAIL] or call [PHONE]."
    assert [event["stage"] for event in events] == ["pii", "pii"]
    assert events[0]["masked"] == {"EMAIL": 1, "PHONE": 1} and events[0]["doc"] == "MEMO_BOARD_01"
    assert hits[1].text.startswith("Mail a.b@")  # the original is not changed


def test_mask_hits_leaves_everything_for_hr_and_for_no_role():
    hits = [hit("internal"), hit("hr")]
    for role in ("hr", None):
        out, events = pii.mask_hits(hits, role, use_presidio=False)
        assert out == hits and events == []


def test_a_piece_with_nothing_to_mask_gets_no_event():
    out, events = pii.mask_hits([hit("internal", "The board met on Monday.")], "guest", use_presidio=False)
    assert out[0].text == "The board met on Monday." and events == []


# ---------- what the output guard looks for ----------

@pytest.mark.parametrize("variant", ["daxarton", "DAXARTON", "The word is daxarton.", "d-a-x-a-r-t-o-n", "d a x a r t o n", "daxarton,", "(daxarton)"])
def test_a_secret_word_is_found_also_when_it_is_spelled_with_spaces_or_dashes(variant):
    assert "daxarton" in pii.secret_words_in(variant)


@pytest.mark.parametrize("text", ["Revenue was 12 billion dollars.", "The sack of grain was ordered.", "", "dax"])
def test_normal_text_holds_no_secret_word(text):
    assert pii.secret_words_in(text) == []


def test_every_secret_word_of_the_manifest_is_found():
    words = fakes.secret_words()
    assert len(words) >= 6
    for word in words:
        assert pii.secret_words_in(f"it is {word}") == [word]


def test_fake_personal_data_finds_emails_phones_ids_and_names():
    name = fakes.known_names()[2]
    found = pii.fake_personal_data(f"{name} at jo@northwind.test, 555-0142, EMP12345.")
    assert {name, "jo@northwind.test", "555-0142", "EMP12345"} <= set(found)


def test_fake_personal_data_leaves_public_filing_text_alone():
    for text in KEEP_CASES + ["Call investor relations on 1-800-555-9999 or write to ir@3m.com."]:
        assert pii.fake_personal_data(text) == []


def test_a_real_email_ending_is_not_a_fake_one():
    assert pii.fake_personal_data("Write to ir@3m.com") == []
    assert pii.fake_personal_data("Write to ir@northwind.test") == ["ir@northwind.test"]


# ---------- the command ----------

def test_the_command_masks_a_sentence_and_prints_no_percent_sign(capsys):
    pii.main(["Dear Aisha Tanaka, call (212) 555-0142 or mail a@northwind.test."])
    out = capsys.readouterr().out
    assert "[PHONE]" in out and "[EMAIL]" in out
    assert "%" not in out


def test_the_status_command_prints_one_line(capsys):
    pii.main(["--status"])
    assert capsys.readouterr().out.strip()


# ---------- the table: python -m tests.test_pii ----------

def table_rows():
    """(number of strings, [(layer name, problems)]) for the 3 layers, over the same 30 strings.

    A layer is only held to what it should find: the patterns must catch the 15 pattern strings, the known names layer must also
    catch the 5 known names, and only the name finder (Presidio) is asked for the 3 unknown names.
    """
    cases = []  # (text, strings that must be gone, kind, the first layer that should catch it)
    cases += [(text, gone, "mask", 1) for text, gone, _ in PATTERN_CASES]
    cases += [(text, [name], "mask", 2) for text, name in known_name_texts()]
    cases += [(text, [name], "mask", 3) for text, name in UNKNOWN_NAME_CASES]
    cases += [(text, [], "keep", 1) for text in KEEP_CASES]
    layers = [("patterns only", dict(use_presidio=False, use_known=False)), ("patterns and known names", dict(use_presidio=False, use_known=True))]
    if pii.presidio_status()[0]:
        layers.append(("patterns, known names and Presidio", dict(use_presidio=True, use_known=True)))
    report = []
    for number, (layer, options) in enumerate(layers, start=1):
        problems = []
        for text, gone, kind, first_layer in cases:
            masked = pii.mask_text(text, **options)[0]
            if kind == "mask" and first_layer <= number:
                missed = [value for value in gone if value.lower() in masked.lower()]
                if missed:
                    problems.append(("MISS", text, f"still there: {', '.join(missed)}", masked))
            elif kind == "keep" and masked != text:
                problems.append(("WRONG", text, "this text holds no personal data but was changed", masked))
        report.append((layer, problems))
    return len(cases), report


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    with tempfile.TemporaryDirectory() as folder:
        previous = config.DATA_DIR
        config.DATA_DIR = Path(folder)
        try:
            make_fake_pdfs.build(Path(folder) / "fake_pdfs")
            total, report = table_rows()
        finally:
            config.DATA_DIR = previous
    print(f"{total} test strings: 15 patterns, 5 known names, 3 unknown names, 7 texts with no personal data.\n")
    for layer, problems in report:
        misses = [item for item in problems if item[0] == "MISS"]
        wrong = [item for item in problems if item[0] == "WRONG"]
        print(f"Layer: {layer}: {len(misses)} miss(es), {len(wrong)} wrong mask(s)")
        for kind, text, why, masked in problems:
            print(f"  {kind}: {text}\n        {why}\n        came out as: {masked}")
    ready, why = pii.presidio_status()
    print("\nPresidio:", "ready" if ready else f"not used ({why})")
    print("A miss or a wrong mask in the first two layers is a bug to fix. A miss in the Presidio layer is the limit of the model: write it in your notes.")


if __name__ == "__main__":
    main()
