"""Tests for steps 2.5 to 2.8: the judge, the runner, the report, compare runs, the judge check and the experiments.

No network, no key, no model, no Qdrant server: a fake Claude, a fake search and an in-memory Qdrant stand in.
Run from the project folder:
    python -m pytest -q
"""
import contextlib
import csv
import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import evals.cases as cases_module  # noqa: E402
from bot import llm  # noqa: E402
from bot.answer import Answer  # noqa: E402
from evals import calibrate_judge as cal  # noqa: E402
from evals import code_checks as cc  # noqa: E402
from evals import compare_runs, experiments, judge, report, run_eval  # noqa: E402
from evals.cases import Case, Evidence  # noqa: E402
from retrieval.search import Hit  # noqa: E402

DOC = "3M_2018_10K"
YES = '{"verdict": "yes", "reason": "same facts"}'
NO = '{"verdict": "no", "reason": "wrong number"}'


@pytest.fixture(autouse=True)
def clean_state(tmp_path, monkeypatch):
    """Every test gets its own empty data and results folders, and the filter off."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "FILTER_MODE", "off")


def printed(function, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = function(*args, **kwargs)
    return result, buffer.getvalue()


# ---------- fakes ----------

def make_case(number=1, split="test", doc=DOC, answer="$1,577 million", pages=(60,), company="3M"):
    return Case(
        id=f"financebench_id_{number:05d}", split=split, company=company, doc=doc, doc_type="10K", question_type="metrics-generated",
        reasoning_type="Numerical reasoning", question=f"What was the capex in case {number}?", answer=answer, justification="",
        evidence=tuple(Evidence(page=page, text=f"evidence text of page {page}") for page in pages),
    )


def make_hit(doc=DOC, page=60, piece=1, text="Capex was 1,577 million.", score=0.5, company="3M", year=2018):
    return Hit(score=score, doc=doc, company=company, year=year, period="FY", type="10K", page=page, piece=piece, text=text, source="real", access="public")


def make_answer(question="q", answer="Capex was 1,577 million dollars.", citations=((DOC, 60),), found=True, pieces=((DOC, 60),), cost=0.03):
    result = Answer(
        question=question, answer=answer, citations=[{"doc": d, "page": p} for d, p in citations], found=found,
        hits=[make_hit(d, p) for d, p in pieces], tokens_in=2000, tokens_out=100, cost_cents=cost, seconds=0.5, search_seconds=0.1,
        calls=1, format_ok=True, prompt="bot_v1", model="claude-test",
    )
    result.raw = "RAW reply with the bot's reasoning"
    return result


class FakeAsk:
    """Stands in for llm.ask: gives the texts one by one, then repeats the last one."""

    def __init__(self, *texts, cents=0.01):
        self.texts, self.cents, self.calls = list(texts), cents, []

    def __call__(self, system, user, max_tokens=500):
        self.calls.append((system, user, max_tokens))
        text = self.texts.pop(0) if len(self.texts) > 1 else self.texts[0]
        return llm.Reply(text, 1000, 50, self.cents, 0.1, "claude-test", "end_turn", 1)


def verdict(kind, passed, cents=0.01):
    return judge.Verdict(kind, f"{kind}_v1", "claude", passed, "reason", cost_cents=cents)


def fake_judge(case, record):
    return {"correct": verdict("correct", True), "supported": verdict("supported", True)}


# ====================================================================================
# the judge (2.5)
# ====================================================================================

def test_read_verdict_accepts_yes_no_and_booleans():
    assert judge.read_verdict(YES) == (True, "same facts")
    assert judge.read_verdict('```json\n{"verdict": "No", "reason": "x"}\n```') == (False, "x")
    assert judge.read_verdict('{"verdict": true, "reason": ""}') == (True, "")
    assert judge.read_verdict('{"verdict": "pass"}')[0] is True
    assert judge.read_verdict('Here: {"verdict": "unsupported", "reason": "r"} done')[0] is False


def test_read_verdict_errors_are_clear():
    with pytest.raises(judge.JudgeError, match="JSON"):
        judge.read_verdict("I think it is fine")
    with pytest.raises(judge.JudgeError, match="yes or no"):
        judge.read_verdict('{"verdict": "maybe"}')
    with pytest.raises(judge.JudgeError):
        judge.read_verdict("")


def test_judge_correct_builds_the_prompt_and_reads_the_verdict():
    ask = FakeAsk(YES)
    result = judge.judge_correct("What was capex?", "$1,577 million", "It was 1,577 million.", model="claude", ask_fn=ask)
    assert result.passed and result.reason == "same facts" and not result.cached
    assert result.version == "correct_v2" and result.model == "claude" and result.cost_cents == 0.01
    system, user, max_tokens = ask.calls[0]
    assert "reference answer" in system and "JSON" in system
    assert "<question>\nWhat was capex?\n</question>" in user
    assert "<reference_answer>\n$1,577 million\n</reference_answer>" in user
    assert "<bot_answer>\nIt was 1,577 million.\n</bot_answer>" in user
    assert max_tokens == judge.CLAUDE_MAX_TOKENS


def test_a_repeat_costs_nothing_and_a_changed_rule_is_judged_again(monkeypatch):
    ask = FakeAsk(YES)
    judge.judge_correct("q", "r", "a", model="claude", ask_fn=ask)
    again = judge.judge_correct("q", "r", "a", model="claude", ask_fn=ask)
    assert len(ask.calls) == 1 and again.cached and again.passed and again.cost_cents == 0
    judge.judge_correct("q", "r", "another answer", model="claude", ask_fn=ask)
    assert len(ask.calls) == 2
    rule = judge.Rule("correct", "correct_v2", judge.CORRECT_V1_SYSTEM + "\nBe strict.", judge.CORRECT_V1_USER)
    monkeypatch.setitem(judge.RULES["correct"], "correct_v2", rule)
    third = judge.judge_correct("q", "r", "a", model="claude", ask_fn=ask, version="correct_v2")
    assert len(ask.calls) == 3 and third.version == "correct_v2"
    judge.judge_correct("q", "r", "a", model="claude", ask_fn=ask, use_cache=False)
    assert len(ask.calls) == 4


def test_the_cache_file_survives_and_ignores_a_broken_line(tmp_path):
    path = tmp_path / "cache.jsonl"
    cache = judge.JsonlCache(path)
    assert cache.get("a") is None
    cache.put("a", {"x": 1})
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"key": "b", "val')  # half written
    again = judge.JsonlCache(path)
    assert again.get("a") == {"x": 1} and again.get("b") is None


def test_the_judge_error_is_not_cached():
    ask = FakeAsk("not json", YES)
    with pytest.raises(judge.JudgeError):
        judge.judge_correct("q", "r", "a", model="claude", ask_fn=ask)
    assert judge.judge_correct("q", "r", "a", model="claude", ask_fn=ask).passed
    assert len(ask.calls) == 2


def test_judge_record_runs_both_rules_and_never_sends_the_raw_reply():
    ask = FakeAsk(YES, NO)
    record = make_answer().to_dict(include_text=True)
    verdicts = judge.judge_record(make_case(), record, model="claude", ask_fn=ask)
    assert verdicts["correct"].passed and not verdicts["supported"].passed
    assert len(ask.calls) == 2
    supported_user = ask.calls[1][1]
    assert '<piece doc="3M_2018_10K" page="60">' in supported_user and "Capex was 1,577 million." in supported_user
    assert "<bot_answer>\nCapex was 1,577 million dollars.\n</bot_answer>" in supported_user
    assert all("reasoning" not in call[1] for call in ask.calls)  # the raw reply holds the bot's reasoning


def test_an_empty_answer_fails_both_rules_without_a_call():
    ask = FakeAsk(YES)
    verdicts = judge.judge_record(make_case(), {"answer": "  ", "pieces": []}, model="claude", ask_fn=ask)
    assert not verdicts["correct"].passed and not verdicts["supported"].passed and ask.calls == []
    assert verdicts["correct"].reason == "the bot gave no answer"


def test_default_model_and_unknown_model(monkeypatch):
    assert judge.default_model() == "claude"
    monkeypatch.setattr(config, "JUDGE_MODEL", "groq", raising=False)
    assert judge.default_model() == "groq"
    monkeypatch.setattr(config, "JUDGE_MODEL", "nope", raising=False)
    with pytest.raises(judge.JudgeError, match="JUDGE_MODEL"):
        judge.default_model()
    with pytest.raises(judge.JudgeError, match="Unknown judge model"):
        judge.call_model("nope", "s", "u")
    with pytest.raises(judge.JudgeError, match="Unknown rule"):
        judge.judge("nope", {}, model="claude")


def test_gemini_request_and_reply(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key-not-real")
    seen = {}

    def post(url, headers, body, service):
        seen.update(url=url, headers=headers, body=body, service=service)
        return {"candidates": [{"content": {"parts": [{"text": YES}]}}], "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 7}}

    result = judge.judge_correct("q", "r", "a", model="gemini", post_fn=post)
    assert result.passed and result.cost_cents == 0 and (result.tokens_in, result.tokens_out) == (11, 7)
    assert seen["url"].endswith(f"{judge.GEMINI_MODEL}:generateContent") and "key=" not in seen["url"]
    assert seen["headers"] == {"x-goog-api-key": "test-key-not-real"} and seen["service"] == "Gemini"
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == judge.CORRECT_V2_SYSTEM
    assert "<bot_answer>" in seen["body"]["contents"][0]["parts"][0]["text"]


def test_groq_request_and_reply(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key-not-real")

    def post(url, headers, body, service):
        assert url == judge.GROQ_URL and service == "Groq"
        assert headers == {"Authorization": "Bearer test-key-not-real"}
        assert body["model"] == judge.GROQ_MODEL and [m["role"] for m in body["messages"]] == ["system", "user"]
        return {"choices": [{"message": {"content": NO}}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}

    result = judge.judge_correct("q", "r", "a", model="groq", post_fn=post)
    assert not result.passed and result.tokens_in == 5 and result.cost_cents == 0


def test_free_services_give_clear_errors(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", None)
    monkeypatch.setattr(config, "GROQ_API_KEY", None)
    with pytest.raises(judge.JudgeError, match="Missing GEMINI_API_KEY"):
        judge.judge_correct("q", "r", "a", model="gemini", post_fn=lambda *a: {})
    with pytest.raises(judge.JudgeError, match="Missing GROQ_API_KEY"):
        judge.judge_correct("q", "r", "a", model="groq", post_fn=lambda *a: {})
    monkeypatch.setattr(config, "GEMINI_API_KEY", "k")
    monkeypatch.setattr(config, "GROQ_API_KEY", "k")
    with pytest.raises(judge.JudgeError, match="Gemini gave no answer"):
        judge.judge_correct("q", "r", "a", model="gemini", post_fn=lambda *a: {"candidates": []})
    with pytest.raises(judge.JudgeError, match="Groq gave no answer"):
        judge.judge_correct("q", "r", "a", model="groq", post_fn=lambda *a: {})


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.data


def http_error(code):
    return urllib.error.HTTPError("http://x", code, "message", {}, None)


def test_post_json_sends_json_with_a_user_agent():
    seen = []

    def opener(request, timeout):
        seen.append((dict(request.header_items()), request.data, timeout))
        return FakeResponse(b'{"ok": 1}')

    assert judge.post_json("http://x/y", {"A": "b"}, {"q": 1}, "Svc", opener=opener) == {"ok": 1}
    headers, data, timeout = seen[0]
    assert headers["User-agent"] == "filings-analyst/1.0" and headers["A"] == "b" and json.loads(data) == {"q": 1} and timeout == 60


def test_post_json_waits_and_tries_again_on_a_rate_limit():
    replies = [http_error(429), http_error(503), FakeResponse(b'{"ok": 2}')]
    waits = []

    def opener(request, timeout):
        item = replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    assert judge.post_json("http://x", {}, {}, "Svc", opener=opener, sleep=waits.append) == {"ok": 2}
    assert waits == [2, 5]


def test_post_json_errors_are_plain():
    def raising(error):
        def opener(request, timeout):
            raise error
        return opener

    quiet = {"sleep": lambda seconds: None}
    with pytest.raises(judge.JudgeError, match="Groq answered 401.*key"):
        judge.post_json("http://x", {}, {}, "Groq", opener=raising(http_error(401)), **quiet)
    with pytest.raises(judge.JudgeError, match="answered 404.*model name"):
        judge.post_json("http://x", {}, {}, "Groq", opener=raising(http_error(404)), **quiet)
    with pytest.raises(judge.JudgeError, match="answered 429.*too many"):
        judge.post_json("http://x", {}, {}, "Groq", opener=raising(http_error(429)), **quiet)
    with pytest.raises(judge.JudgeError, match="could not be reached"):
        judge.post_json("http://x", {}, {}, "Groq", opener=raising(urllib.error.URLError("down")), **quiet)
    with pytest.raises(judge.JudgeError, match="not JSON"):
        judge.post_json("http://x", {}, {}, "Groq", opener=lambda request, timeout: FakeResponse(b"<html>"), **quiet)


def test_judge_main_judges_a_saved_run_and_a_second_run_is_free(tmp_path, monkeypatch):
    case = make_case()
    monkeypatch.setattr(cases_module, "load_cases", lambda *args, **kwargs: [case])
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"cases": [{"id": case.id, "answer": make_answer().to_dict(include_text=True)}, {"id": case.id, "answer": None}]}), encoding="utf-8")
    ask = FakeAsk(YES)
    _, first = printed(judge.main, [str(path)], ask_fn=ask)
    assert "correct yes" in first and "supported yes" in first and "(new)" in first and "2 judge calls made" in first
    _, second = printed(judge.main, [str(path)], ask_fn=ask)
    assert "(cached)" in second and "0 judge calls made" in second and len(ask.calls) == 2
    with pytest.raises(SystemExit, match="is missing"):
        judge.main([str(tmp_path / "nope.json")])


# ====================================================================================
# the runner (2.6)
# ====================================================================================

def patch_cases(monkeypatch, cases):
    monkeypatch.setattr(run_eval, "load_cases", lambda split=None: [case for case in cases if split in (None, case.split)])


def test_estimate_cents_and_names():
    total, bot, judged = run_eval.estimate_cents(10, "claude")
    assert total == pytest.approx(bot + judged) and bot > 0 and judged > 0
    assert run_eval.estimate_cents(10, "groq")[2] == 0 and run_eval.estimate_cents(10, None)[2] == 0
    assert run_eval.run_name("base", "test", None) == "base_test.json"
    assert run_eval.run_name("base", "test", 10) == "base_test_first10.json"


def test_run_cases_collects_marks_judge_and_keeps_text_out():
    lines = []
    items, stopped, seconds = run_eval.run_cases([make_case(1), make_case(2)], lambda q: make_answer(q), fake_judge, say=lines.append)
    assert stopped == "" and len(items) == 2 and len(lines) == 2
    item = items[0]
    assert item["marks"]["found_page"]["status"] == "pass" and item["judge"]["correct"]["passed"]
    assert "question" not in item["answer"] and "raw" not in item["answer"]
    assert item["answer"]["pieces"][0]["text"] and item["evidence"] == [[DOC, 60]] and item["error"] == ""
    assert "page ok" in lines[0] and "correct yes" in lines[0] and "supported yes" in lines[0]


def test_one_bad_question_does_not_stop_the_run_but_three_in_a_row_do():
    def flaky(question):
        if "case 2" in question:
            raise RuntimeError("Claude did not answer")
        return make_answer(question)

    items, stopped, _ = run_eval.run_cases([make_case(1), make_case(2), make_case(3)], flaky, None, say=lambda text: None)
    assert [bool(item["error"]) for item in items] == [False, True, False] and stopped == ""
    assert "RuntimeError: Claude did not answer" in items[1]["error"] and items[1]["answer"] is None

    def always(question):
        raise ValueError("no index")

    items, stopped, _ = run_eval.run_cases([make_case(i) for i in range(1, 7)], always, None, say=lambda text: None)
    assert len(items) == 3 and "3 errors one after another" in stopped and "no index" in stopped


def test_the_spend_fuse_stops_the_run_and_keeps_what_is_done():
    def answer_fn(question):
        if "case 2" in question:
            raise llm.BudgetStop("Spend fuse: 90 cents")
        return make_answer(question)

    items, stopped, _ = run_eval.run_cases([make_case(1), make_case(2), make_case(3)], answer_fn, None, say=lambda text: None)
    assert len(items) == 1 and stopped == "Spend fuse: 90 cents"

    def judge_fuse(case, record):
        raise llm.BudgetStop("Spend fuse: 90 cents")

    items, stopped, _ = run_eval.run_cases([make_case(1), make_case(2)], lambda q: make_answer(q), judge_fuse, say=lambda text: None)
    assert len(items) == 1 and items[0]["answer"] and items[0]["judge"] is None and stopped


def test_a_judge_error_keeps_the_answer():
    def bad_judge(case, record):
        raise judge.JudgeError("the judge did not reply with JSON")

    items, stopped, _ = run_eval.run_cases([make_case(1)], lambda q: make_answer(q), bad_judge, say=lambda text: None)
    assert stopped == "" and items[0]["answer"] and items[0]["judge"] is None and "JudgeError" in items[0]["judge_error"]
    run = run_eval.build_run("t", "test", None, False, "", {}, items, 1.0)
    assert run["totals"]["judge_errors"] == 1 and run["totals"]["errors"] == 0
    assert "could not be judged" in "\n".join(report.build_report(run))


def test_build_run_adds_up_the_cost():
    items, _, _ = run_eval.run_cases([make_case(1), make_case(2)], lambda q: make_answer(q, cost=0.05), fake_judge, say=lambda text: None)
    run = run_eval.build_run("base", "test", None, False, "", {"filter": "off"}, items, 2.0)
    assert run["totals"]["bot_cents"] == pytest.approx(0.10) and run["totals"]["judge_cents"] == pytest.approx(0.04)
    assert run["totals"]["cost_cents"] == pytest.approx(0.14) and run["totals"]["questions"] == 2
    assert run["scores"]["correct"]["good"] == 2


def test_main_saves_the_run_and_the_report_and_keeps_question_text_out(monkeypatch):
    patch_cases(monkeypatch, [make_case(i) for i in range(1, 4)])
    _, out = printed(run_eval.main, ["--split", "test", "--tag", "base", "--limit", "2"], answer_fn=lambda q: make_answer(q), judge_fn=fake_judge)
    path = report.runs_dir() / "base_test_first2.json"
    text = path.read_text(encoding="utf-8")
    run = json.loads(text)
    assert run["partial"] is True and len(run["cases"]) == 2 and run["limit"] == 2 and run["scores"]["found_page"]["good"] == 2
    assert "What was the capex" not in text and "$1,577 million" not in text and "RAW reply" not in text
    assert (path.parent / ".gitignore").read_text(encoding="utf-8").startswith("*")
    assert (Path(config.RESULTS_DIR) / "report_base_test.txt").exists()
    assert "COLAB: no" in out and "CLAUDE COST: about" in out and "REPORT base / test" in out and "Saved:" in out


def test_a_full_split_is_not_partial(monkeypatch):
    patch_cases(monkeypatch, [make_case(i) for i in range(1, 4)])
    printed(run_eval.main, ["--split", "test", "--tag", "full"], answer_fn=lambda q: make_answer(q), judge_fn=fake_judge)
    run = json.loads((report.runs_dir() / "full_test.json").read_text(encoding="utf-8"))
    assert run["partial"] is False and run["limit"] is None


def test_a_big_run_needs_spend_and_a_dry_run_asks_nothing(monkeypatch):
    patch_cases(monkeypatch, [make_case(i) for i in range(1, 6)])
    monkeypatch.setattr(run_eval, "SPEND_FREE_CENTS", 0.1)

    def boom(question):
        raise AssertionError("no question may be asked")

    with pytest.raises(SystemExit, match="--spend"):
        printed(run_eval.main, ["--split", "test", "--tag", "t"], answer_fn=boom, judge_fn=fake_judge)
    _, out = printed(run_eval.main, ["--split", "test", "--tag", "t", "--dry-run"], answer_fn=boom, judge_fn=fake_judge)
    assert "Dry run" in out and "CLAUDE COST: about" in out and not report.runs_dir().exists()
    printed(run_eval.main, ["--split", "test", "--tag", "t", "--spend"], answer_fn=lambda q: make_answer(q), judge_fn=fake_judge)
    assert (report.runs_dir() / "t_test.json").exists()


def test_the_holdout_is_opened_once(monkeypatch):
    patch_cases(monkeypatch, [make_case(1, split="holdout")])
    args = ["--split", "holdout", "--tag", "h"]
    fakes = {"answer_fn": lambda q: make_answer(q), "judge_fn": fake_judge}
    with pytest.raises(SystemExit, match="--final"):
        printed(run_eval.main, args, **fakes)
    printed(run_eval.main, args + ["--final"], **fakes)
    with pytest.raises(SystemExit, match="already opened"):
        printed(run_eval.main, args + ["--final"], **fakes)
    printed(run_eval.main, args + ["--final", "--again"], **fakes)


def test_bad_tag_and_bad_limit_are_refused(monkeypatch):
    patch_cases(monkeypatch, [make_case(1)])
    with pytest.raises(SystemExit, match="tag"):
        printed(run_eval.main, ["--split", "test", "--tag", "a b"])
    with pytest.raises(SystemExit, match="--limit"):
        printed(run_eval.main, ["--split", "test", "--tag", "ok", "--limit", "0"])


def test_the_filter_setting_is_used_for_the_run_and_put_back(monkeypatch):
    patch_cases(monkeypatch, [make_case(1), make_case(2)])
    seen = []

    def answer_fn(question):
        seen.append(config.FILTER_MODE)
        return make_answer(question)

    printed(run_eval.main, ["--split", "test", "--tag", "f", "--filter", "company"], answer_fn=answer_fn, judge_fn=fake_judge)
    assert seen == ["company", "company"] and config.FILTER_MODE == "off"
    run = json.loads((report.runs_dir() / "f_test.json").read_text(encoding="utf-8"))
    assert run["settings"]["filter"] == "company" and run["settings"]["judge_rules"] == {"correct": "correct_v2", "supported": "supported_v2"}


def test_no_judge_means_code_checks_only(monkeypatch):
    patch_cases(monkeypatch, [make_case(1)])

    def boom(case, record):
        raise AssertionError("the judge must not be called")

    _, out = printed(run_eval.main, ["--split", "test", "--tag", "n", "--no-judge"], answer_fn=lambda q: make_answer(q), judge_fn=boom)
    run = json.loads((report.runs_dir() / "n_test.json").read_text(encoding="utf-8"))
    assert run["settings"]["judge"] == "none" and run["cases"][0]["judge"] is None
    assert "Judge: none" in out and "no judge in this run" in out


def test_a_stopped_run_is_saved_and_exits_with_an_error(monkeypatch):
    patch_cases(monkeypatch, [make_case(1), make_case(2)])

    def answer_fn(question):
        if "case 2" in question:
            raise llm.BudgetStop("Spend fuse: 90 cents")
        return make_answer(question)

    with pytest.raises(SystemExit) as stop:
        printed(run_eval.main, ["--split", "test", "--tag", "s"], answer_fn=answer_fn, judge_fn=fake_judge)
    assert stop.value.code == 1
    run = json.loads((report.runs_dir() / "s_test.json").read_text(encoding="utf-8"))
    assert run["partial"] is True and "Spend fuse" in run["stopped"] and len(run["cases"]) == 1


# ====================================================================================
# the report (2.6)
# ====================================================================================

def item(number, found=True, numbers="pass", correct=None, supported=None, company="3M", doc_type="10K", question_type="a", error=""):
    marks = {name: {"status": "pass", "reason": ""} for name in cc.CASE_MARKS}
    marks["found_page"]["status"] = "pass" if found else "fail"
    marks["number_match"]["status"] = numbers
    verdicts = None
    if correct is not None:
        verdicts = {
            "correct": {"passed": correct, "reason": "r", "cost_cents": 0.01},
            "supported": {"passed": True if supported is None else supported, "reason": "r", "cost_cents": 0.01},
        }
    return {
        "id": f"financebench_id_{number:05d}", "split": "test", "company": company, "doc": DOC, "doc_type": doc_type,
        "question_type": question_type, "reasoning_type": "Numerical reasoning", "evidence": [[DOC, 60]],
        "error": error, "judge_error": "", "answer": None if error else {"cost_cents": 0.03, "seconds": 1.0},
        "marks": None if error else marks, "judge": verdicts,
    }


def make_run(items, **extra):
    run = {
        "version": 1, "tag": "base", "split": "test", "limit": None, "partial": False, "stopped": "", "created": "2026-11-23T12:00:00",
        "settings": {"filter": "off", "k": 5, "collection": "filings_v1", "prompt": "bot_v1", "claude_model": "m", "judge": "claude"},
        "totals": {"questions": len(items), "cost_cents": 0.4, "bot_cents": 0.3, "judge_cents": 0.1, "seconds": 4.0}, "cases": items,
    }
    run.update(extra)
    return run


def test_metric_and_is_right():
    assert report.metric(item(1, correct=True), "correct") is True
    assert report.metric(item(1), "correct") is None  # no judge
    assert report.metric(item(1, numbers="skip"), "number_match") is None
    assert report.metric(item(1, error="boom"), "found_page") is None
    assert report.is_right(item(1, correct=False, numbers="pass")) is False  # the judge's word first
    assert report.is_right(item(1, numbers="fail")) is False  # no judge: the number match
    assert report.is_right(item(1, numbers="skip")) is None


def test_compute_scores_and_the_range():
    items = [item(1, found=True, correct=True), item(2, found=True, correct=False), item(3, found=False, correct=True), item(4, found=False, correct=False)]
    scores = report.compute_scores(items)
    assert scores["found_page"] == {"good": 2, "of": 4, "points": 50.0}
    assert scores["correct"]["good"] == 2 and scores["supported"]["of"] == 4
    assert report.wilson(0, 0) == (0.0, 0.0)
    low, high = report.wilson(50, 100)
    assert 0.39 < low < 0.41 and 0.59 < high < 0.61
    assert report.wilson(0, 10)[0] == 0.0 and report.wilson(10, 10)[1] == 1.0


def test_where_it_breaks_reads_the_first_two_marks_together():
    items = [item(1, True, correct=True), item(2, True, correct=False), item(3, False, correct=True), item(4, False, correct=False), item(5, False, correct=False)]
    table = report.where_it_breaks(items)
    assert table[("yes", "yes")] == 1 and table[("yes", "no")] == 1 and table[("no", "yes")] == 1 and table[("no", "no")] == 2


def test_the_report_has_every_part_and_no_percent_sign():
    items = [item(1, correct=True, company="3M"), item(2, found=False, correct=False, company="Adobe"), item(3, correct=False, company="Adobe")]
    # pass_lines={}: this test must not depend on the PASS_LINES that you pasted into config.py (step 2.9)
    text = "\n".join(report.build_report(make_run(items), notes=(report.Counter({"search missed": 2}), 2), pass_lines={}))
    for part in ("REPORT base / test", "Settings:", "SCORES", "WHERE THE ANSWERS BREAK", "search failed", "BY SLICE", "by company:", "Adobe", "YOUR FAILURE TYPES", "search missed", "COST:", "PASS LINES", "not set yet"):
        assert part in text
    assert "%" not in text
    assert "fail   (range" in text and "out of 100" in text or "SCORES (how many fail out of every 100" in text


def test_the_report_warns_about_partial_stopped_and_error_runs():
    run = make_run([item(1, correct=True), item(2, error="Boom")], partial=True, stopped="Spend fuse")
    text = "\n".join(report.build_report(run, notes=(report.Counter(), 0)))
    assert "NOT A FULL RUN" in text and "STOPPED: Spend fuse" in text and "1 question had an error" in text
    assert "none yet" in text and "(rough)" in text


def test_the_report_without_a_judge_says_so():
    text = "\n".join(report.build_report(make_run([item(1), item(2, numbers="fail")]), notes=(report.Counter(), 0)))
    assert "no judge in this run" in text and "numbers match the reference" in text


def test_pass_lines_show_ok_and_below(monkeypatch):
    monkeypatch.setattr(config, "PASS_LINES", {"found_page": 40, "correct": 90, "supported": 50}, raising=False)
    items = [item(1, found=True, correct=True), item(2, found=False, correct=False)]
    text = "\n".join(report.build_report(make_run(items), notes=(report.Counter(), 0)))
    assert "found_page" in text and "ok" in text and "BELOW" in text
    assert "not set yet" not in text


def test_failure_notes_are_counted_by_type(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text(
        "# notes\n- 00012 [Search missed] the evidence page ranked ninth\n- financebench_id_00013 [bot misread] x\n* 00014 [search missed] y\nrandom text\n",
        encoding="utf-8",
    )
    types, count = report.read_notes(path)
    assert count == 3 and types["search missed"] == 2 and types["bot misread"] == 1
    assert report.read_notes(tmp_path / "missing.md") == (report.Counter(), 0)


def test_report_main_prints_the_newest_run_and_saves_a_file():
    with pytest.raises(SystemExit, match="no run yet"):
        report.main([])
    run_eval.save_run(make_run([item(1, correct=True)]), report.runs_dir() / "base_test.json")
    _, out = printed(report.main, [])
    assert "REPORT base / test" in out and "Saved to" in out
    assert (Path(config.RESULTS_DIR) / "report_base_test.txt").exists()
    with pytest.raises(SystemExit, match="not a run file"):
        (report.runs_dir() / "bad.json").write_text("{}", encoding="utf-8")
        report.main([str(report.runs_dir() / "bad.json")])


# ====================================================================================
# compare runs (2.8)
# ====================================================================================

def test_paired_uses_only_questions_both_runs_measured():
    run_a = make_run([item(1, correct=True), item(2, correct=False), item(3, correct=True)])
    run_b = make_run([item(1, correct=True), item(2, correct=True), item(4, correct=True)])
    ids, a, b = compare_runs.paired(run_a, run_b, "right")
    assert ids == ["financebench_id_00001", "financebench_id_00002"] and a == [1, 0] and b == [1, 1]
    assert compare_runs.flips(ids, a, b) == (["financebench_id_00002"], [])


def test_the_range_is_the_same_every_time_and_holds_the_difference():
    a = [1, 0, 1, 0, 1, 0, 1, 1, 0, 0] * 3
    b = [1, 1, 1, 0, 1, 1, 1, 1, 0, 0] * 3
    first = compare_runs.bootstrap_difference(a, b)
    assert first == compare_runs.bootstrap_difference(a, b)
    point, low, high = first
    assert point == pytest.approx(20.0) and low <= point <= high
    assert compare_runs.bootstrap_difference(a, a) == (0.0, 0.0, 0.0)
    assert compare_runs.bootstrap_difference([0] * 30, [1] * 30) == (100.0, 100.0, 100.0)
    assert compare_runs.bootstrap_difference([], []) == (0.0, 0.0, 0.0)
    assert compare_runs.bootstrap_difference([0], [1]) == (100.0, 100.0, 100.0)


def test_compare_lines_say_which_questions_flipped_and_whether_it_is_clear():
    run_a = make_run([item(i, correct=(i % 2 == 0)) for i in range(1, 7)])
    run_b = make_run([item(i, correct=(i % 2 == 0 or i == 1)) for i in range(1, 7)])
    run_b["settings"] = dict(run_b["settings"], k=4)
    text = "\n".join(compare_runs.compare_lines(run_a, run_b, "right", ("a.json", "b.json")))
    assert "A: a.json" in text and "B: b.json" in text and "Settings that differ: k (5 -> 4)" in text
    assert "B fixed 1: 00001" in text and "B broke 0: none" in text and "B minus A: +16.7 points" in text
    assert "rough: very few questions" in text and "%" not in text


def test_compare_lines_warn_when_more_than_one_setting_changed_or_nothing_is_shared():
    run_a, run_b = make_run([item(1, correct=True)]), make_run([item(2, correct=True)], split="practice")
    run_b["settings"] = dict(run_b["settings"], k=3, filter="company")
    text = "\n".join(compare_runs.compare_lines(run_a, run_b, "right"))
    assert "More than one setting changed" in text and "different splits" in text and "nothing to compare" in text


def test_a_range_that_includes_zero_is_called_not_clearly_different():
    run_a = make_run([item(i, correct=(i <= 10)) for i in range(1, 21)])
    run_b = make_run([item(i, correct=(i <= 11)) for i in range(1, 21)])
    text = "\n".join(compare_runs.compare_lines(run_a, run_b, "right"))
    assert "not clearly different" in text


def test_compare_main_reads_two_files(tmp_path):
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    run_eval.save_run(make_run([item(1, found=False), item(2, found=True)]), first)
    run_eval.save_run(make_run([item(1, found=True), item(2, found=True)]), second)
    _, out = printed(compare_runs.main, [str(first), str(second), "--metric", "found_page"])
    assert "Metric: found_page" in out and "B fixed 1" in out


# ====================================================================================
# the judge check (2.7)
# ====================================================================================

def many_cases():
    return [make_case(i, split="practice") for i in range(1, 13)] + [make_case(i, split="test") for i in range(101, 131)]


def test_pick_ids_is_fixed_and_in_file_order():
    ids = cal.pick_ids(many_cases())
    assert len(ids) == 20 and ids == cal.pick_ids(many_cases())
    assert ids[:10] == [f"financebench_id_{i:05d}" for i in range(1, 11)]
    test_ids = ids[10:]
    assert test_ids == sorted(test_ids) and len(set(test_ids)) == 10 and all(100 < int(i[-5:]) < 131 for i in test_ids)
    assert len(cal.pick_ids(many_cases()[:14])) == 12  # 10 practice and the 2 test cases there are


def test_hand_marks_template_is_made_once_and_marks_are_read(tmp_path):
    path = tmp_path / "marks.csv"
    ids = ["a", "b", "c", "d"]
    assert cal.write_marks_template(ids, path) is True and cal.write_marks_template(ids, path) is False
    assert path.read_text(encoding="utf-8").splitlines()[:2] == ["id,mark,note", "a,,"]
    path.write_text("id,mark,note\na,PASS,\nb,fail,too low\nc,yes,\nd,maybe,\ne,\n,0,\n", encoding="utf-8")
    marks, missing = cal.read_marks(path)
    assert marks == {"a": True, "b": False, "c": True} and missing == ["d", "e"]
    with pytest.raises(cases_module.CaseError, match="--pick"):
        cal.read_marks(tmp_path / "none.csv")


def test_the_delivered_hand_marks_file_matches_the_real_questions():
    real_cases = ROOT / "data" / "cases" / "test_cases.jsonl"
    delivered = ROOT / "data" / "calibration" / "hand_marks.csv"
    if not real_cases.exists() or not delivered.exists():
        pytest.skip("the real question file or hand_marks.csv is not here")
    with delivered.open(encoding="utf-8-sig", newline="") as handle:
        ids = [row["id"] for row in csv.DictReader(handle)]
    assert len(ids) == 20
    assert ids == cal.pick_ids(cases_module.load_cases(path=real_cases))


def test_calibration_stats():
    pairs = [(True, True), (True, False), (False, False), (False, True)]
    assert cal.calibration_stats(pairs) == {"n": 4, "agreement": 50, "catch": 50, "false_alarm": 50, "your_fails": 2}
    stats = cal.calibration_stats([(True, True), (True, True)])
    assert stats["catch"] is None and stats["agreement"] == 100 and stats["false_alarm"] == 0


def prepare_pick(monkeypatch):
    monkeypatch.setattr(cal, "load_cases", many_cases)
    asked = []

    def answer_fn(question, k=None):
        asked.append((question, k))
        return make_answer(question)

    return answer_fn, asked


def test_pick_makes_the_answers_the_reading_page_and_the_marks_file(monkeypatch):
    answer_fn, asked = prepare_pick(monkeypatch)
    _, out = printed(cal.main, ["--pick"], answer_fn=answer_fn)
    assert len(asked) == 20 and all(k is None for _, k in asked)
    assert "COLAB: no" in out and "CLAUDE COST: about" in out and "read_these.txt" in out
    saved = json.loads(cal.answers_path().read_text(encoding="utf-8"))
    assert len(saved["ids"]) == 20 and saved["answers"][saved["ids"][0]]["source"] == "practice" and saved["answers"][saved["ids"][-1]]["source"] == "test"
    reading = cal.reading_path().read_text(encoding="utf-8")
    assert reading.count("QUESTION:") == 20 and "REFERENCE ANSWER: $1,577 million" in reading and "evidence text of page 60" in reading
    assert "BOT ANSWER: Capex was 1,577 million dollars." in reading
    assert (cal.folder() / ".gitignore").read_text(encoding="utf-8").split() == ["answers.json", "read_these.txt"]
    rows = list(csv.DictReader(cal.marks_path().open(encoding="utf-8", newline="")))
    assert [row["id"] for row in rows] == saved["ids"] and all(row["mark"] == "" for row in rows)


def test_pick_refuses_to_replace_answers_unless_again_and_weak_uses_one_piece(monkeypatch):
    answer_fn, asked = prepare_pick(monkeypatch)
    printed(cal.main, ["--pick"], answer_fn=answer_fn)
    with pytest.raises(SystemExit, match="--again"):
        printed(cal.main, ["--pick"], answer_fn=answer_fn)
    asked.clear()
    printed(cal.main, ["--pick", "--again", "--weak", "3"], answer_fn=answer_fn)
    assert [k for _, k in asked].count(1) == 3 and [k for _, k in asked][:10] == [None] * 10
    saved = json.loads(cal.answers_path().read_text(encoding="utf-8"))
    assert [item["source"] for item in saved["answers"].values()].count("weak") == 3


def test_pick_stops_on_an_error_and_saves_nothing(monkeypatch):
    monkeypatch.setattr(cal, "load_cases", many_cases)

    def bad(question, k=None):
        raise RuntimeError("Claude did not answer")

    with pytest.raises(SystemExit, match="Nothing was saved"):
        printed(cal.main, ["--pick"], answer_fn=bad)
    assert not cal.answers_path().exists()


def write_marks(ids, fails):
    with cal.marks_path().open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "mark", "note"])
        for number, case_id in enumerate(ids):
            writer.writerow([case_id, "fail" if number < fails else "pass", ""])


def test_calibrate_needs_the_answers_and_all_the_marks(monkeypatch):
    monkeypatch.setattr(cal, "load_cases", many_cases)
    with pytest.raises(SystemExit, match="--pick"):
        printed(cal.main, [])
    answer_fn, _ = prepare_pick(monkeypatch)
    printed(cal.main, ["--pick"], answer_fn=answer_fn)
    with pytest.raises(SystemExit, match="20 of the 20 ids have no mark"):
        printed(cal.main, [], ask_fn=FakeAsk(YES))


def test_calibrate_prints_agreement_catch_rate_and_false_alarm_rate(monkeypatch):
    answer_fn, _ = prepare_pick(monkeypatch)
    printed(cal.main, ["--pick"], answer_fn=answer_fn)
    ids = json.loads(cal.answers_path().read_text(encoding="utf-8"))["ids"]
    write_marks(ids, fails=6)
    ask = FakeAsk(YES)  # the judge says yes to everything: it catches none of your 6 fails
    _, out = printed(cal.main, [], ask_fn=ask)
    assert "Your marks: 14 pass, 6 fail." in out and len(ask.calls) == 20
    row = next(line for line in out.splitlines() if line.startswith("claude"))
    assert row.split()[1:4] == ["70", "0", "0"] and "Agrees most with you: claude (70 out of 100)" in out
    assert "disagrees on" in out and "WARNING" not in out and "%" not in out
    _, again = printed(cal.main, [], ask_fn=ask)
    assert len(ask.calls) == 20 and "0.000 c" in again  # the second run costs nothing


def test_calibrate_warns_when_you_marked_too_few_fails(monkeypatch):
    answer_fn, _ = prepare_pick(monkeypatch)
    printed(cal.main, ["--pick"], answer_fn=answer_fn)
    write_marks(json.loads(cal.answers_path().read_text(encoding="utf-8"))["ids"], fails=2)
    _, out = printed(cal.main, [], ask_fn=FakeAsk(NO))
    assert "WARNING: fewer than 6 fails" in out and "--weak 5 --again" in out


def test_calibrate_all_compares_each_model_that_has_a_key(monkeypatch):
    answer_fn, _ = prepare_pick(monkeypatch)
    printed(cal.main, ["--pick"], answer_fn=answer_fn)
    write_marks(json.loads(cal.answers_path().read_text(encoding="utf-8"))["ids"], fails=6)
    monkeypatch.setattr(cal, "available_models", lambda: (["claude", "groq"], ["gemini"]))
    monkeypatch.setattr(config, "GROQ_API_KEY", "k")
    post = lambda url, headers, body, service: {"choices": [{"message": {"content": NO}}], "usage": {}}  # noqa: E731
    _, out = printed(cal.main, ["--judge", "all"], ask_fn=FakeAsk(YES), post_fn=post)
    assert "gemini: no key in .env, skipped." in out
    assert any(line.startswith("claude") for line in out.splitlines()) and any(line.startswith("groq") for line in out.splitlines())
    groq = next(line for line in out.splitlines() if line.startswith("groq"))
    assert groq.split()[1:4] == ["30", "100", "100"]  # a judge that says no to everything catches all fails, and wrongly fails all passes


def test_calibrate_reports_a_judge_that_could_not_finish(monkeypatch):
    answer_fn, _ = prepare_pick(monkeypatch)
    printed(cal.main, ["--pick"], answer_fn=answer_fn)
    write_marks(json.loads(cal.answers_path().read_text(encoding="utf-8"))["ids"], fails=6)
    monkeypatch.setattr(config, "GROQ_API_KEY", None)
    _, out = printed(cal.main, ["--judge", "groq"], post_fn=lambda *a: {})
    assert "could not finish: Missing GROQ_API_KEY" in out


def test_available_models_follow_the_keys(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(config, "GEMINI_API_KEY", None)
    monkeypatch.setattr(config, "GROQ_API_KEY", "k")
    assert cal.available_models() == (["claude", "groq"], ["gemini"])


# ====================================================================================
# the experiments (2.8)
# ====================================================================================

def test_tokenize_keeps_numbers_whole_and_drops_stopwords():
    assert experiments.tokenize("The capex was $1,577.5 million in FY2018") == ["capex", "1,577.5", "million", "fy2018"]


def test_bm25_ranks_the_matching_piece_first_and_honours_the_allowed_set():
    texts = ["revenue grew in europe", "capital expenditures were 1,577 million", "net income rose in europe", "capital markets and europe"]
    index = experiments.Bm25(texts)
    scores = index.scores("capital expenditures")
    assert max(scores, key=scores.get) == 1 and set(scores) == {1, 3}
    assert index.scores("capital expenditures", allowed={0, 2, 3}).keys() == {3}
    assert index.scores("nothing here") == {} and index.scores("") == {}
    rare, common = index.scores("expenditures")[1], index.scores("europe")[0]
    assert rare > common  # a rare word counts more than a common one


def payloads():
    return [
        {"doc": DOC, "company": "3M", "year": 2018, "period": "FY", "type": "10K", "page": 60, "piece": 1, "text": "3M capital expenditures were 1,577 million", "source": "real", "access": "public"},
        {"doc": DOC, "company": "3M", "year": 2018, "period": "FY", "type": "10K", "page": 61, "piece": 1, "text": "3M dividends paid per share", "source": "real", "access": "public"},
        {"doc": "Adobe_2020_10K", "company": "Adobe", "year": 2020, "period": "FY", "type": "10K", "page": 9, "piece": 1, "text": "Adobe capital expenditures were 442 million", "source": "real", "access": "public"},
    ]


def test_the_piece_store_filters_and_returns_hits():
    store = experiments.PieceStore(payloads())
    assert store.allowed() is None and store.allowed("3M") == {0, 1} and store.allowed(["3M", "Adobe"], 2020) == {2}
    hits = store.search("capital expenditures 1,577", 5)
    assert [(hit.doc, hit.page) for hit in hits][0] == (DOC, 60) and all(isinstance(hit, Hit) and hit.score > 0 for hit in hits)
    only_adobe = store.search("capital expenditures", 5, company="Adobe")
    assert [hit.doc for hit in only_adobe] == ["Adobe_2020_10K"]
    assert store.search("zzzz", 5) == [] and len(store.search("capital", 1)) == 1


def test_fuse_rewards_pieces_that_both_rankings_like():
    a = [make_hit(page=1), make_hit(page=2), make_hit(page=3)]
    b = [make_hit(page=3), make_hit(page=4), make_hit(page=1)]
    fused = experiments.fuse([a, b], 3)
    assert [hit.page for hit in fused][:2] == [1, 3]  # in both lists
    assert len(fused) == 3 and fused[0].score == pytest.approx(1 / 61 + 1 / 63)
    assert experiments.fuse([[]], 5) == [] and len(experiments.fuse([a], 2)) == 2


def make_collection(count):
    client = QdrantClient(":memory:")
    client.create_collection("c", vectors_config=VectorParams(size=2, distance=Distance.COSINE))
    client.upsert("c", [PointStruct(id=i + 1, vector=[1.0, 0.0], payload=dict(payloads()[i % 3], piece=i)) for i in range(count)])
    return client


def test_load_payloads_downloads_once_and_then_uses_the_cache(monkeypatch):
    client = make_collection(3)
    messages = []
    first = experiments.load_payloads(client, "c", say=messages.append)
    assert len(first) == 3 and len(messages) == 1 and "Downloading the 3 pieces" in messages[0]
    assert (Path(config.DATA_DIR) / "cache" / "pieces_c.jsonl").exists()
    monkeypatch.setattr(client, "scroll", lambda **kwargs: (_ for _ in ()).throw(AssertionError("no second download")))
    assert experiments.load_payloads(client, "c", say=messages.append) == first and len(messages) == 1
    client.upsert("c", [PointStruct(id=9, vector=[1.0, 0.0], payload=payloads()[0])])  # the index changed
    monkeypatch.undo()
    assert len(experiments.load_payloads(make_collection(4), "c", say=messages.append)) == 4


def hit_table():
    """Question text -> the hits a fake search gives: the right page is first for case 1, third for case 2, absent for case 3."""
    return {
        make_case(1).question: [make_hit(page=60), make_hit(page=2)],
        make_case(2).question: [make_hit(page=3), make_hit(page=4), make_hit(page=60), make_hit(page=6)],
        make_case(3).question: [make_hit(page=7)],
    }


def fake_search_factory(table):
    calls = []

    def search(question, k=None, mode=None, collection=None, company=None, year=None):
        calls.append({"question": question, "k": k, "mode": mode, "collection": collection, "company": company, "year": year})
        return table.get(question, [])[:k]

    search.calls = calls
    return search


def three_cases():
    return [make_case(1), make_case(2), make_case(3)]


def test_score_summarize_flips_and_the_result_screen():
    ctx = experiments.Context(search_fn=fake_search_factory(hit_table()))
    variants = [experiments.Variant("top 2", lambda case: ctx.search(case.question, 8), 2), experiments.Variant("top 4", lambda case: ctx.search(case.question, 8), 4)]
    cases = three_cases()
    marks = experiments.score(cases, variants)
    summary, several = experiments.summarize(cases, marks)
    assert summary["top 2"]["hit"] == 1 and summary["top 4"]["hit"] == 2 and several == 0
    assert summary["top 4"]["mrr"] == pytest.approx((1 + 1 / 3 + 0) / 3) and summary["top 4"]["first"] == 1
    assert experiments.flips(cases, marks, "top 2", "top 4") == ([cases[1].id], [])
    text = "\n".join(experiments.result_lines(2, cases, variants, marks))
    assert "Experiment 2" in text and "top 2 -> top 4: found 1 more, lost 0   gained: 00002" in text
    assert "Most right pages found: top 4 (2 of 3)" in text and "%" not in text


def test_experiment_two_says_when_top_four_is_as_good_as_top_five():
    ctx = experiments.Context(search_fn=fake_search_factory(hit_table()))
    cases = [make_case(1), make_case(3)]
    variants = [experiments.Variant("top 4", lambda case: ctx.search(case.question, 8), 4), experiments.Variant("top 5", lambda case: ctx.search(case.question, 8), 5)]
    text = "\n".join(experiments.result_lines(2, cases, variants, experiments.score(cases, variants)))
    assert "Top 4 scores as well as top 5: use top 4" in text


def test_run_one_experiment_two_searches_once_per_question(monkeypatch):
    search_fn = fake_search_factory(hit_table())
    ctx = experiments.Context(search_fn=search_fn)
    lines = []
    assert experiments.run_one(2, three_cases(), ctx, say=lines.append) is True
    assert len(search_fn.calls) == 3 and all(call["k"] == 8 for call in search_fn.calls)
    out = "\n".join(lines)
    assert "COLAB: no | MODEL: unchanged | CLAUDE COST: 0 cents" in out and "top 3" in out and "top 8" in out
    log = (Path(config.RESULTS_DIR) / "experiments.md").read_text(encoding="utf-8")
    assert "### Experiment 2: Pieces returned (top k)" in log and "| top 4 |" in log and "Chosen:" in log
    saved = json.loads((Path(config.RESULTS_DIR) / "exp2_top_k.json").read_text(encoding="utf-8"))
    assert saved["questions"] == 3 and saved["variants"]["top 4"]["places"][make_case(2).id] == 3
    assert "What was the capex" not in json.dumps(saved) and "What was the capex" not in log
    experiments.run_one(2, three_cases(), experiments.Context(search_fn=search_fn), say=lines.append)
    assert (Path(config.RESULTS_DIR) / "experiments.md").read_text(encoding="utf-8").count("### Experiment 2") == 2


def test_experiment_four_runs_each_filter_mode():
    search_fn = fake_search_factory(hit_table())
    ctx = experiments.Context(search_fn=search_fn)
    experiments.run_one(4, three_cases(), ctx, say=lambda text: None)
    assert [call["mode"] for call in search_fn.calls[::3]] == ["off", "company", "company+year"]


def test_experiment_one_mixes_keyword_search_into_vector_search(monkeypatch):
    docs = payloads()
    store = experiments.PieceStore(docs)
    case = make_case(1, answer="$1,577 million")
    table = {case.question: [make_hit(page=99, piece=5), make_hit(page=98, piece=5)]}  # the vector search misses the right page
    ctx = experiments.Context(search_fn=fake_search_factory(table), store=store)
    monkeypatch.setattr(case.__class__, "question", case.question, raising=False)
    variants = experiments.variants_1(ctx)
    assert [variant.name for variant in variants] == ["vector only", "keyword only", "both mixed"]
    keyword_hits = variants[1].hits(case)
    assert keyword_hits == [] or keyword_hits[0].doc  # a question without matching words may find nothing
    own = make_case(1)
    object.__setattr__(own, "question", "3M capital expenditures 1,577 million")
    table[own.question] = [make_hit(page=99, piece=5)]
    marks = experiments.score([own], variants)
    assert not marks["vector only"][0]["hit"] and marks["keyword only"][0]["hit"] and marks["both mixed"][0]["hit"]


def test_the_filter_for_keyword_search_comes_from_the_question(monkeypatch):
    ctx = experiments.Context(search_fn=fake_search_factory({}))
    assert ctx.filters("What was the capex of 3M in 2018?") == (None, None)  # FILTER_MODE is off
    monkeypatch.setattr(config, "FILTER_MODE", "company")
    company, year = ctx.filters("What was the capex of 3M in 2018?")
    assert experiments._listed(company) == {"3M"} and year is None


def test_experiment_three_needs_colab_until_the_plain_index_exists():
    search_fn = fake_search_factory(hit_table())
    lines = []
    ctx = experiments.Context(search_fn=search_fn, exists_fn=lambda name: False)
    assert experiments.run_one(3, three_cases(), ctx, say=lines.append) is False
    assert "NEEDS COLAB" in "\n".join(lines) and "filings_v1_plain" in "\n".join(lines) and search_fn.calls == []
    assert not (Path(config.RESULTS_DIR) / "experiments.md").exists()
    ctx = experiments.Context(search_fn=search_fn, exists_fn=lambda name: name == "filings_v1_plain")
    assert experiments.run_one(3, three_cases(), ctx, say=lambda text: None) is True
    assert {call["collection"] for call in search_fn.calls} == {"filings_v1", "filings_v1_plain"}


def test_experiment_five_costs_cents_and_needs_spend_and_remembers_its_rewrites():
    search_fn = fake_search_factory(hit_table())
    ask = FakeAsk("3M capital expenditures purchases of property plant and equipment", cents=0.004)
    lines = []
    ctx = experiments.Context(search_fn=search_fn, ask_fn=ask)
    assert experiments.run_one(5, three_cases(), ctx, spend=False, say=lines.append) is False
    assert ask.calls == [] and "about" in lines[1] and "Add --spend" in "\n".join(lines)
    assert experiments.run_one(5, three_cases(), ctx, spend=True, say=lines.append) is True
    assert len(ask.calls) == 3 and ctx.rewrite_cents == pytest.approx(0.012)
    assert "<question>" in ask.calls[0][1] and ask.calls[0][2] == 120
    rewrite_calls = [call for call in search_fn.calls if call["question"].startswith("3M capital")]
    assert rewrite_calls and all(call["mode"] == "off" for call in rewrite_calls)
    assert "The rewrites cost 0.012 cents" in "\n".join(lines)
    again = experiments.Context(search_fn=search_fn, ask_fn=ask)
    experiments.run_one(5, three_cases(), again, spend=True, say=lambda text: None)
    assert len(ask.calls) == 3 and again.rewrite_cents == 0  # the second run came from the cache


def test_usable_cases_never_include_the_holdout(monkeypatch):
    everything = [make_case(1, split="practice"), make_case(2, split="test"), make_case(3, split="holdout")]
    monkeypatch.setattr(experiments, "load_cases", lambda: everything)
    assert [case.split for case in experiments.usable_cases()] == ["practice", "test"]
    assert len(experiments.usable_cases(first=1)) == 1


def test_experiments_main_lists_runs_and_refuses_unknown_numbers(monkeypatch):
    _, out = printed(experiments.main, ["--list"])
    assert "COLAB: YES, once" in out and "filings_v1_plain" in out and "needs --spend" in out and "judge" in out.lower()
    _, same = printed(experiments.main, [])
    assert same == out
    monkeypatch.setattr(experiments, "load_cases", lambda: three_cases())
    _, run = printed(experiments.main, ["2", "--first", "2", "6", "7"], search_fn=fake_search_factory(hit_table()))
    assert "2 questions" in run and "Experiment 2" in run and "not built yet" in run and "calibrate_judge --judge all" in run
    with pytest.raises(SystemExit):
        printed(experiments.main, ["9"])


def test_a_missing_year_index_gets_a_clear_instruction(monkeypatch):
    monkeypatch.setattr(experiments, "load_cases", lambda: three_cases())

    def search_fn(question, **kwargs):
        raise RuntimeError("Qdrant search failed. Index required but not found for year")

    with pytest.raises(SystemExit, match="add_year_index.py --collection filings_v1"):
        printed(experiments.main, ["4"], search_fn=search_fn)
