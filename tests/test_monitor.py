"""Tests for steps 4.1 to 4.6: the request log, traces, daily numbers and alerts, the question path and the push check.

No network, no key, no model: a local log folder, a fake Supabase and a fake LangSmith stand in. Run from the project folder:
    python -m pytest -q
"""
import contextlib
import io
import json
import logging
import re
import sys
import types
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest

import config
from bot.answer import Answer
from evals import report
from monitor import flow, metrics, store, tracing
from scripts import check_scores

NOW = datetime(2026, 11, 23, 12, 0, tzinfo=timezone.utc)


def answer(question="What was revenue?", text="Revenue was 100.", citations=((("A_2022_10K", 3)),), found=True, seconds=2.0, cents=0.05, blocked="", events=None, role="guest"):
    return Answer(question, text, [{"doc": doc, "page": page} for doc, page in citations], found, [], tokens_in=900, tokens_out=60, cost_cents=cents,
                  seconds=seconds, role=role, events=events or [], blocked=blocked, raw="RAW REPLY NOT FOR THE LOG")


def record(minutes_ago=5, **fields):
    base = {"id": 1, "created_at": store.iso(NOW - timedelta(minutes=minutes_ago)), "source": "app", "role": "guest", "mode": "plain", "question": "q", "answer": "a",
            "citations": [{"doc": "A", "page": 1}], "found": True, "cost_cents": 0.05, "seconds": 2.0, "blocked": "", "no_citation": False, "thumb": None}
    base.update(fields)
    return base


def run_command(main, *args, **kwargs):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = main(list(args), **kwargs)
    return code, out.getvalue()


# ---------- the log (4.1) ----------

def test_a_request_record_holds_what_the_plan_says_and_never_the_raw_reply():
    events = [{"stage": "pii", "doc": "X", "masked": {"[NAME]": 1}}, {"stage": "injection", "doc": "N"}, {"stage": "agent", "step": 1, "tool": "search_filings", "args": {"query": "x" * 900}}]
    saved = store.request_record(answer(citations=(), events=events), mode="agent", session="s1")
    assert saved["pii_hidden"] and saved["injection"] and saved["no_citation"] and saved["mode"] == "agent" and saved["session"] == "s1"
    assert len(saved["steps"]) == 1 and len(saved["steps"][0]["args"]["query"]) < 400
    assert "RAW REPLY" not in json.dumps(saved) and "raw" not in saved


def test_personal_data_typed_into_the_question_is_hidden_before_it_is_logged():
    saved = store.request_record(answer(question="Mail ana.perez@northwind.test about her pay"))
    assert "@" not in saved["question"] and "[EMAIL]" in saved["question"]


def test_a_blocked_answer_is_logged_with_its_reason_and_is_not_a_missing_citation():
    saved = store.request_record(answer(text="No.", citations=(), found=False, blocked="output:secret_word"))
    assert saved["blocked"] == "output:secret_word" and not saved["no_citation"]


def test_log_request_saves_one_line_and_the_thumb_updates_it():
    request_id = store.log_request(answer())
    assert request_id == 1 and store.set_thumb(request_id, -1) and not store.set_thumb(request_id, 5)
    assert store.get_client().select(store.REQUESTS)[0]["thumb"] == -1
    assert store.set_review(request_id, "search missed", "the page was in FY2021")
    assert store.get_request(request_id)["failure_type"] == "search missed"


def test_a_broken_log_never_stops_an_answer_and_never_prints_the_key(caplog):
    class Down:
        def insert(self, *args, **kwargs):
            raise ConnectionError("https://secret-project.supabase.co refused KEYVALUE123")

    with caplog.at_level(logging.WARNING):
        assert store.log_request(answer(), client=Down()) is None
        assert store.log_run({"tag": "x", "cases": []}, client=Down()) is None
        assert store.requests_between(client=Down()) == []
    assert "KEYVALUE123" not in caplog.text and "ConnectionError" in caplog.text


def test_without_keys_the_log_goes_to_local_files_and_with_keys_to_supabase(monkeypatch):
    store.set_client(None)
    monkeypatch.setattr(config, "SUPABASE_URL", None)
    monkeypatch.setattr(config, "SUPABASE_KEY", None)
    assert isinstance(store.get_client(), store.FileClient)
    store.set_client(None)
    monkeypatch.setattr(config, "SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setattr(config, "SUPABASE_KEY", "k")
    assert isinstance(store.get_client(), store.SupabaseClient)


class FakeHttp:
    def __init__(self, data=None):
        self.calls, self.data = [], data if data is not None else []

    def request(self, method, url, headers=None, timeout=None, params=None, json=None):
        self.calls.append({"method": method, "url": url, "headers": headers, "params": params, "json": json})
        data = self.data
        return NS(json=lambda: data, raise_for_status=lambda: None)


def test_supabase_calls_have_the_right_shape():
    http = FakeHttp([{"id": 7}])
    client = store.SupabaseClient("https://x.supabase.co/", "KEY", http=http)
    assert client.insert("requests", {"a": 1}) == {"id": 7}
    assert client.select("requests", [("source", "eq", "app"), ("cost_cents", "gt", 0)], order="created_at.desc", limit=5) == [{"id": 7}]
    assert client.update("requests", 7, {"thumb": 1})
    insert, select, update = http.calls
    assert insert["method"] == "POST" and insert["url"] == "https://x.supabase.co/rest/v1/requests" and insert["headers"]["Prefer"] == "return=representation"
    assert insert["headers"]["apikey"] == "KEY" and insert["json"] == {"a": 1}
    assert ("source", "eq.app") in select["params"] and ("cost_cents", "gt.0") in select["params"] and ("limit", "5") in select["params"]
    assert update["method"] == "PATCH" and update["params"] == {"id": "eq.7"}


def test_times_with_5_digits_after_the_dot_are_read_on_every_python():
    assert store.parse_time("2026-11-23T10:31:22.12345+00:00").microsecond == 123450
    assert store.parse_time("2026-11-23T10:31:22Z").tzinfo is not None


def test_the_local_log_orders_newest_first_and_filters():
    client = store.get_client()
    for number in range(3):
        client.insert(store.REQUESTS, {"source": "app" if number < 2 else "eval", "cost_cents": number})
    assert [item["id"] for item in client.select(store.REQUESTS, order="created_at.desc")] == [3, 2, 1]
    assert [item["id"] for item in client.select(store.REQUESTS, [("source", "eq", "app"), ("cost_cents", "gt", 0)])] == [2]


def test_an_eval_run_is_saved_with_its_scores():
    run = {"tag": "base", "split": "test", "partial": False, "settings": {"k": 5}, "scores": {"correct": {"good": 3, "of": 4, "points": 75.0}},
           "totals": {"questions": 4, "cost_cents": 0.4}, "cases": []}
    assert store.log_run(run) == 1
    saved = store.recent_runs()[0]
    assert saved["tag"] == "base" and saved["scores"]["correct"]["points"] == 75.0 and saved["questions"] == 4


# ---------- the question path ----------

def fake_bot(**fixed):
    def ask_fn(question, role="guest", mode="plain"):
        return answer(question=question, role=role, **fixed)

    return ask_fn


def test_each_question_leaves_one_line_in_the_log():
    for number in range(10):
        outcome = flow.ask(f"question {number}", ask_fn=fake_bot(), session="s", now=NOW)
        assert outcome.request_id == number + 1
    assert len(store.get_client().select(store.REQUESTS)) == 10


def test_a_crash_becomes_one_plain_sentence_and_a_log_line():
    def broken(question, role="guest", mode="plain"):
        raise RuntimeError("Qdrant is down")

    outcome = flow.ask("q", ask_fn=broken, now=NOW)
    assert outcome.result.answer.startswith("Something went wrong") and "Qdrant" not in outcome.result.answer
    assert store.get_request(outcome.request_id)["blocked"] == "error:RuntimeError"


def test_the_spend_fuse_is_named_in_plain_words():
    from bot import llm

    def stopped(question, role="guest", mode="plain"):
        raise llm.BudgetStop("fuse")

    assert "budget" in flow.ask("q", ask_fn=stopped, now=NOW).result.answer


def test_the_app_still_answers_when_the_log_is_switched_off():
    class Down:
        def select(self, *args, **kwargs):
            return []

        def insert(self, *args, **kwargs):
            raise ConnectionError("down")

    outcome = flow.ask("q", ask_fn=fake_bot(), client=Down(), now=NOW)
    assert outcome.result.answer == "Revenue was 100." and outcome.request_id is None


def test_the_command_logs_the_practice_questions(monkeypatch):
    monkeypatch.setattr("evals.cases.load_cases", lambda split: [NS(question="one"), NS(question="two")])
    code, out = run_command(flow.main, "--practice", "2", ask_fn=fake_bot())
    assert code == 0 and "2 of 2 questions were saved" in out and "%" not in out and out.startswith("COLAB: no | MODEL: unchanged")
    assert len(store.get_client().select(store.REQUESTS)) == 2


# ---------- traces (4.2) ----------

def test_sampling_sends_1_in_5_from_the_app_and_everything_from_eval_runs(monkeypatch):
    monkeypatch.setattr(tracing, "enabled", lambda: True)
    assert tracing.wanted("app", sampler=lambda: 0.1) and not tracing.wanted("app", sampler=lambda: 0.3)
    assert tracing.wanted("eval", sampler=lambda: 0.99)
    monkeypatch.setattr(tracing, "enabled", lambda: False)
    assert not tracing.wanted("eval")


def test_a_trace_gets_the_tags_and_the_outcome(monkeypatch):
    seen = {}

    class Run:
        def add_outputs(self, outputs):
            seen["outputs"] = outputs

    @contextlib.contextmanager
    def fake_trace(name, **options):
        seen["trace"] = options
        yield Run()

    @contextlib.contextmanager
    def fake_context(**options):
        seen["enabled"] = options
        yield

    monkeypatch.setitem(sys.modules, "langsmith", types.SimpleNamespace(trace=fake_trace, tracing_context=fake_context))
    monkeypatch.setattr(tracing, "enabled", lambda: True)
    with tracing.trace(role="analyst", mode="agent", question="q", source="eval", tag="base") as span:
        span.finish(answer(events=[{"stage": "agent"}, {"stage": "agent"}]))
    assert {"role:analyst", "mode:agent", "source:eval", "run:base"} <= set(seen["trace"]["tags"]) and seen["enabled"] == {"enabled": True}
    assert seen["outputs"]["steps"] == 2 and seen["outputs"]["cost_cents"] == 0.05


def test_a_failing_trace_never_stops_the_answer(monkeypatch):
    def broken(name, **options):
        raise RuntimeError("LangSmith is down")

    monkeypatch.setitem(sys.modules, "langsmith", types.SimpleNamespace(trace=broken, tracing_context=lambda **o: contextlib.nullcontext()))
    monkeypatch.setattr(tracing, "enabled", lambda: True)
    with tracing.trace(source="eval") as span:
        span.finish(answer())
    assert span.run is None


def test_without_a_key_tracing_does_nothing_and_errors_still_come_out():
    assert not tracing.enabled()
    with pytest.raises(ZeroDivisionError):
        with tracing.trace():
            1 / 0
    wrapped = tracing.traced(lambda question: answer(question=question), tag="base")
    assert wrapped("q").question == "q"


# ---------- daily numbers, alerts, weekly review (4.3) ----------

def day_of(count, **fields):
    return [record(minutes_ago=index + 1, id=index + 1, **fields) for index in range(count)]


def test_the_daily_numbers_count_the_day():
    records = day_of(8) + [record(30, id=90, blocked="input:too long", cost_cents=0.0), record(60 * 30, id=91, cost_cents=0.5)]  # the last is yesterday
    numbers = metrics.daily_numbers(records, NOW)
    assert numbers["requests_today"] == 9 and numbers["blocked_today"] == 1 and numbers["block_rate"] == 11
    assert numbers["spend_today_cents"] == pytest.approx(0.4) and numbers["answers"] == 9 and numbers["slowest_seconds"] == 2.0


def test_ten_slow_requests_fire_the_slow_alert_and_a_quiet_day_fires_nothing():
    assert [alert.name for alert in metrics.alerts(metrics.daily_numbers(day_of(10, seconds=30.0), NOW))] == ["slow"]
    assert metrics.alerts(metrics.daily_numbers(day_of(30), NOW)) == []


def test_the_command_prints_the_slow_alert_for_10_logged_requests():
    for _ in range(10):
        store.log_request(answer(seconds=31.0))
    code, out = run_command(metrics.main)
    assert code == 1 and "ALERT slow" in out and "%" not in out and "The rules:" in out


def test_each_alert_rule_has_its_own_line():
    def fired(records):
        return {alert.name for alert in metrics.alerts(metrics.daily_numbers(records, NOW), daily_limit=2.0)}

    assert "spend" in fired(day_of(5, cost_cents=0.34)) and "spend" not in fired(day_of(5, cost_cents=0.3))  # 1.7 and 1.5 of 2.0
    assert "no citation" in fired(day_of(10)[:8] + day_of(12, no_citation=True)[:2]) and "no citation" not in fired(day_of(10)[:9] + day_of(12, no_citation=True)[:1])
    rated = [record(index + 1, id=index + 1, thumb=-1 if index < 3 else 1) for index in range(10)]
    assert "thumbs down" in fired(rated) and "thumbs down" not in fired(rated[:1] + [dict(item, thumb=1) for item in rated[1:]])
    assert "blocked" in fired([record(index + 1, id=index + 1, blocked="input:x") for index in range(5)])
    assert "blocked" not in fired([record(index + 1, id=index + 1, blocked="input:x") for index in range(4)])


def test_blocked_must_be_more_than_double_the_usual():
    usual = [record(60 * 24 * day + 10, id=100 + day, blocked="input:x") for day in range(1, 4) for _ in range(1)] * 3  # 3 a day on 3 past days
    for index, item in enumerate(usual):
        item["id"] = 200 + index
    today = lambda count: [record(index + 1, id=index + 1, blocked="input:x") for index in range(count)]  # noqa: E731
    assert metrics.daily_numbers(usual, NOW)["usual_blocked"] == pytest.approx(3.0)
    assert "blocked" not in {a.name for a in metrics.alerts(metrics.daily_numbers(usual + today(6), NOW))}
    assert "blocked" in {a.name for a in metrics.alerts(metrics.daily_numbers(usual + today(7), NOW))}


def test_the_weekly_sample_is_10_random_plus_every_thumbs_down_and_the_same_all_week():
    records = [record(60 * 6 * index, id=index + 1, thumb=-1 if index in (3, 9) else None) for index in range(30)]  # 30 requests over 7.5 days
    first, second = metrics.weekly_sample(records, NOW), metrics.weekly_sample(records, NOW)
    week = [item for item in records if metrics.when(item) >= NOW - timedelta(days=7)]
    assert [item["id"] for item in first] == [item["id"] for item in second]
    assert sum(item["why"] == "random" for item in first) == 10
    assert {item["id"] for item in first if item["why"] == "thumbs down"} == {item["id"] for item in week if item["thumb"] == -1}
    assert all(metrics.when(item) >= NOW - timedelta(days=7) for item in first)


def test_a_flagged_request_becomes_a_candidate_test_question_once(tmp_path):
    path = tmp_path / "cases" / "new_failures.jsonl"
    saved = record(id=12, question="What was the margin?", failure_type="search missed")
    assert metrics.add_case(saved, path) == (path, True) and metrics.add_case(saved, path) == (path, False)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["id"] == "request_12" and json.loads(lines[0])["failure_type"] == "search missed"


def test_the_review_commands_write_to_the_log_and_the_cases_file(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    request_id = store.log_request(answer())
    code, out = run_command(metrics.main, "--review", str(request_id), "search missed", "wrong", "year")
    assert code == 0 and store.get_request(request_id)["review_note"] == "wrong year"
    code, out = run_command(metrics.main, "--to-case", str(request_id))
    assert code == 0 and (tmp_path / "data" / "cases" / "new_failures.jsonl").exists()
    code, out = run_command(metrics.main, "--week")
    assert code == 0 and (tmp_path / "data" / "logs" / "weekly_review.md").exists() and "%" not in out


# ---------- keys (4.4) ----------

def test_the_secrets_box_comes_before_env(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "from-env")
    assert config._secret("QDRANT_URL") == "from-env"
    monkeypatch.setitem(sys.modules, "streamlit", types.SimpleNamespace(secrets={"QDRANT_URL": "from-secrets"}))
    assert config._secret("QDRANT_URL") == "from-secrets"

    class NoSecrets:
        def get(self, name):
            raise FileNotFoundError("no secrets.toml")

    monkeypatch.setitem(sys.modules, "streamlit", types.SimpleNamespace(secrets=NoSecrets()))
    assert config._secret("QDRANT_URL") == "from-env"


# ---------- the push check (4.6) ----------

def case(number, found=True, correct=True, company="3M"):
    return {"id": f"financebench_id_{number:05d}", "company": company, "doc_type": "10K", "question_type": "a", "reasoning_type": "r", "error": "",
            "marks": {"found_page": {"status": "pass" if found else "fail"}, "number_match": {"status": "pass"}},
            "judge": {"correct": {"passed": correct}, "supported": {"passed": True}}}


def make_run(right, total=20, tag="base", split="test", **extra):
    items = [case(index, correct=index < right, company="3M" if index % 2 else "Adobe") for index in range(total)]
    run = {"tag": tag, "split": split, "partial": False, "stopped": "", "created": "2026-11-23T12:00:00", "settings": {"k": 5}, "totals": {"cost_cents": 1.2}, "cases": items}
    run.update(extra)
    return run


def write_run(tmp_path, name, run):
    path = tmp_path / name
    path.write_text(json.dumps(run), encoding="utf-8")
    return str(path)


@pytest.fixture
def results(tmp_path, monkeypatch):
    folder = tmp_path / "results"
    monkeypatch.setattr(config, "PASS_LINES", {})
    return folder


def save(results, tmp_path, flag, run):
    return run_command(check_scores.main, flag, write_run(tmp_path, f"{flag}.json", run), results=results)


def test_the_score_file_keeps_numbers_only_and_cuts_by_slice():
    summary = check_scores.score_file(make_run(16))
    assert summary["scores"]["correct"]["points"] == 80.0 and summary["questions"] == 20 and not summary["partial"]
    assert summary["slices"]["company"]["3M"]["n"] == 10 and '"question":' not in json.dumps(summary)
    assert check_scores.score_file(make_run(16, partial=True))["partial"] and check_scores.score_file(make_run(16, stopped="fuse"))["partial"]


def test_the_check_passes_for_the_baseline_and_a_small_move(results, tmp_path, monkeypatch):
    assert run_command(check_scores.main, results=results)[0] == 0  # nothing to check yet
    save(results, tmp_path, "--baseline", make_run(16))
    code, out = run_command(check_scores.main, results=results)
    assert code == 0 and "OK" in out and "%" not in out
    save(results, tmp_path, "--latest", make_run(15, tag="final"))  # 75 against 80: a drop of exactly 5 is allowed
    assert run_command(check_scores.main, results=results)[0] == 0


def test_a_drop_of_more_than_5_points_fails(results, tmp_path):
    save(results, tmp_path, "--baseline", make_run(16))
    save(results, tmp_path, "--latest", make_run(14, tag="final"))
    code, out = run_command(check_scores.main, results=results)
    assert code == 1 and "FAIL" in out and "more than 5 points under the baseline" in out


def test_a_score_under_its_pass_line_fails(results, tmp_path, monkeypatch):
    save(results, tmp_path, "--baseline", make_run(16))
    code, out = run_command(check_scores.main, "--show-lines", results=results)
    assert '"correct": 75.0' in out and "%" not in out
    monkeypatch.setattr(config, "PASS_LINES", {"correct": 78.0})
    save(results, tmp_path, "--latest", make_run(15, tag="final"))
    code, out = run_command(check_scores.main, results=results)
    assert code == 1 and "under its pass line 78.0" in out


def test_a_partial_run_or_another_split_fails(results, tmp_path):
    save(results, tmp_path, "--baseline", make_run(16))
    code, out = save(results, tmp_path, "--latest", make_run(16, partial=True))
    assert "WARNING" in out and run_command(check_scores.main, results=results)[0] == 1
    save(results, tmp_path, "--latest", make_run(16, split="practice"))
    code, out = run_command(check_scores.main, results=results)
    assert code == 1 and "cannot be compared" in out


def test_the_baseline_must_be_a_full_test_run(results, tmp_path):
    with pytest.raises(SystemExit):
        save(results, tmp_path, "--baseline", make_run(16, partial=True))
    with pytest.raises(SystemExit):
        save(results, tmp_path, "--baseline", make_run(16, split="practice"))
    assert not (results / "baseline_scores.json").exists()


def test_pass_lines_without_a_baseline_fail(results, monkeypatch):
    monkeypatch.setattr(config, "PASS_LINES", {"correct": 70.0})
    assert run_command(check_scores.main, results=results)[0] == 1


def test_the_new_files_name_no_key_and_print_no_percent_sign():
    from pathlib import Path

    root = Path(config.ROOT)
    for name in ("monitor/store.py", "monitor/metrics.py", "monitor/tracing.py", "monitor/flow.py", "scripts/check_scores.py", "app/streamlit_app.py", "app/dashboard.py"):
        path = root / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        assert "sk-ant" not in text and "eyJ" not in text, name
        assert "%" not in re.sub(r"%[sd]", "", text), name  # a %s in a log line is a blank to fill, not a printed sign
