"""Tests for steps 4.4 and 4.5: the chat page and the dashboard, run headless with Streamlit's own test runner.

No browser, no network, no key, no model: a fake bot stands in. These tests are skipped when Streamlit is not installed
(the push check on GitHub installs only light libraries). Run from the project folder:
    python -m pytest -q
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from bot.answer import Answer  # noqa: E402
from monitor import flow, store  # noqa: E402
from scripts import check_scores  # noqa: E402

CHAT, BOARD = str(ROOT / "app" / "streamlit_app.py"), str(ROOT / "app" / "dashboard.py")


def bot(question, role="guest", mode="plain"):
    hit = NS(doc="A_2022_10K", page=3, score=0.91, text="Adobe | 2022 | 10K | page 3\nrevenue was 100")
    return Answer(question, "Revenue was 100.", [{"doc": "A_2022_10K", "page": 3}], True, [hit], tokens_in=900, tokens_out=60, cost_cents=0.04, seconds=1.5, role=role)


@pytest.fixture
def keys(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"):
        monkeypatch.setattr(config, name, "x")
    real = flow.ask
    monkeypatch.setattr(flow, "ask", lambda *args, **kwargs: real(*args, ask_fn=bot, **kwargs))


def texts(at):
    return [item.value for item in at.markdown] + [item.value for item in at.caption] + [item.value for item in at.warning]


def test_the_chat_page_shows_the_notice_and_the_example_before_any_question(keys):
    at = AppTest.from_file(CHAT, default_timeout=30).run()
    assert not at.exception and at.title[0].value == "Filings Analyst" and len(at.chat_input) == 1
    assert config.DEMO_NOTICE in [item.value for item in at.info]
    assert any("Do not type personal data" in text for text in texts(at))
    assert at.selectbox[0].label == "Role" and at.selectbox[0].value == "guest" and len(at.selectbox[0].options) == 3
    assert at.radio[0].value == "plain" and any("Questions left in this visit: 10" in item.value for item in at.markdown)


def test_a_question_gets_an_answer_with_citations_pieces_cost_and_a_log_line(keys):
    at = AppTest.from_file(CHAT, default_timeout=30).run()
    at.chat_input[0].set_value("What was revenue?").run()
    assert not at.exception
    shown = " | ".join(texts(at))
    assert "Revenue was 100." in shown and "Cited: A_2022_10K page 3" in shown and "0.040 cents" in shown and "plain mode, role guest" in shown
    assert len(at.expander) == 1 and "1 pieces" in at.expander[0].label
    saved = store.get_client().select(store.REQUESTS)
    assert len(saved) == 1 and saved[0]["question"] == "What was revenue?" and saved[0]["role"] == "guest" and saved[0]["mode"] == "plain"


def test_the_role_and_the_mode_reach_the_bot(keys):
    at = AppTest.from_file(CHAT, default_timeout=30).run()
    at.selectbox[0].select("hr")
    at.radio[0].set_value("agent")
    at.chat_input[0].set_value("Who works in finance?").run()
    saved = store.get_client().select(store.REQUESTS)[0]
    assert saved["role"] == "hr" and saved["mode"] == "agent"


def test_a_thumb_is_written_to_the_log(keys):
    at = AppTest.from_file(CHAT, default_timeout=30).run()
    at.chat_input[0].set_value("What was revenue?").run()
    at.button(key="bad0").click().run()
    assert store.get_request(1)["thumb"] == -1 and any("rating is saved" in text for text in texts(at))
    assert not [button for button in at.button if button.key in ("good0", "bad0")]  # one rating per answer


def test_the_visit_limit_shows_a_plain_sentence_and_spends_nothing(keys, monkeypatch):
    monkeypatch.setattr(config, "DEMO_SESSION_QUESTIONS", 2)
    at = AppTest.from_file(CHAT, default_timeout=30).run()
    for number in range(3):
        at.chat_input[0].set_value(f"question {number}").run()
    assert any("used its 2 questions" in text for text in texts(at))
    assert len(store.get_client().select(store.REQUESTS)) == 2


def test_the_daily_limit_closes_the_page_for_the_rest_of_the_day(keys):
    store.get_client().insert(store.REQUESTS, {"source": "app", "cost_cents": 1.95, "question": "q", "answer": "a"})
    at = AppTest.from_file(CHAT, default_timeout=30).run()
    at.chat_input[0].set_value("What was revenue?").run()
    assert any("money for today" in text for text in texts(at)) and len(store.get_client().select(store.REQUESTS)) == 1


def test_missing_keys_are_named_without_their_values(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"):
        monkeypatch.setattr(config, name, None)
    at = AppTest.from_file(CHAT, default_timeout=30).run()
    assert not at.exception and len(at.chat_input) == 0
    assert "ANTHROPIC_API_KEY, QDRANT_URL, QDRANT_API_KEY" in at.error[0].value


# ---------- the dashboard ----------

def case(number, correct=True, company="3M"):
    return {"id": f"financebench_id_{number:05d}", "company": company, "doc_type": "10K", "question_type": "a", "reasoning_type": "r", "error": "",
            "marks": {"found_page": {"status": "pass"}, "number_match": {"status": "pass"}},
            "judge": {"correct": {"passed": correct}, "supported": {"passed": True}}}


def write_scores(folder, name, right, split="test"):
    run = {"tag": name, "split": split, "partial": False, "created": "2026-11-23T12:00:00", "settings": {}, "totals": {},
           "cases": [case(index, correct=index < right, company="3M" if index % 2 else "Adobe") for index in range(20)]}
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}_scores.json").write_text(json.dumps(check_scores.score_file(run)), encoding="utf-8")


@pytest.fixture
def results(tmp_path, monkeypatch):
    folder = tmp_path / "results"
    monkeypatch.setattr(config, "RESULTS_DIR", folder)
    monkeypatch.setattr(config, "PASS_LINES", {"correct": 75.0})
    write_scores(folder, "baseline", 16)
    write_scores(folder, "latest", 14)
    (folder / "attack_table_final.txt").write_text("ATTACKS 40  got through 3 of 40\n", encoding="utf-8")
    (folder / "false_blocks.json").write_text(json.dumps({"run": "base_test", "questions": 75, "blocked": 2, "per_100": 3}), encoding="utf-8")
    return folder


def test_the_dashboard_shows_scores_against_the_lines_the_attack_table_and_live_numbers(results):
    store.log_request(Answer("q", "a", [], True, [], role="guest", seconds=2.0, cost_cents=0.05))
    at = AppTest.from_file(BOARD, default_timeout=30).run()
    assert not at.exception
    assert [item.value for item in at.header][:3] == ["Scores against the pass lines", "Scores by slice", "The attack table"]
    assert "ATTACKS 40  got through 3 of 40" in at.code[0].value
    scores = at.dataframe[0].value
    entry = scores[scores["Score"] == "answer is correct (judge)"].iloc[0]
    assert (entry["Baseline"], entry["Latest"], entry["Pass line"], entry["Result"]) == (80.0, 70.0, 75.0, "dropped more than 5 points")
    assert any("under its pass line 75.0" in item.value for item in at.error)
    assert any("blocked by mistake: 2 of 75" in item.value for item in at.markdown)
    assert [item.label for item in at.metric][:2] == ["Requests today", "Spend today (cents)"] and at.metric[0].value == "1"


def test_the_dashboard_without_any_file_says_what_to_do_next(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "empty")
    at = AppTest.from_file(BOARD, default_timeout=30).run()
    assert not at.exception and any("no baseline yet" in item.value.lower() for item in at.info)
    assert any("No requests in the log yet" in item.value for item in at.info)


def test_what_visitors_typed_is_hidden_on_the_public_page(results, monkeypatch):
    store.log_request(Answer("my private question", "no", [], False, [], role="guest", blocked="output:advice"))
    hidden = AppTest.from_file(BOARD, default_timeout=30).run()
    failures = [item.value for item in hidden.dataframe if "Why" in item.value.columns][0]
    assert failures.iloc[0]["Why"] == "blocked: output:advice" and failures.iloc[0]["Question"] == "(hidden on the public page)"
    monkeypatch.setattr(config, "SHOW_QUESTIONS", True)
    shown = AppTest.from_file(BOARD, default_timeout=30).run()
    assert [item.value for item in shown.dataframe if "Why" in item.value.columns][0].iloc[0]["Question"] == "my private question"


def test_the_dashboard_opens_as_the_second_page_of_the_app(keys, results):
    at = AppTest.from_file(CHAT, default_timeout=30).run()
    at.switch_page(BOARD).run()
    assert not at.exception and at.title[0].value == "Scores and monitor" and at.header[0].value == "Scores against the pass lines"


def test_the_helper_functions_of_the_dashboard():
    from app import dashboard

    moment = datetime(2026, 11, 23, 12, 0, tzinfo=timezone.utc)
    records = [{"created_at": store.iso(moment), "role": "guest", "mode": "plain", "question": "q1", "blocked": "", "no_citation": True, "thumb": None},
               {"created_at": store.iso(moment), "role": "hr", "mode": "agent", "question": "q2", "blocked": "", "no_citation": False, "thumb": 1}]
    found = dashboard.failure_table(records, show_questions=True)
    assert len(found) == 1 and found[0]["Why"] == "no citation" and found[0]["Question"] == "q1"
    assert dashboard.newest_attack_table("/nonexistent") is None
