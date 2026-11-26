"""Tests for the fixes of 25 November (apply_fixes.py): fix reading, fix numbers, fix neighbors, the rewrite test and the reranker test.

No network, no key, no model, no Qdrant server: a fake Claude, a fake search and a fake reranker model stand in.
Run from the project folder:
    python -m pytest -q
"""
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from bot import answer as bot_answer  # noqa: E402
from bot import llm, lookup, prompts  # noqa: E402
from evals import code_checks as cc  # noqa: E402
from evals import experiments, judge, rerank_test, report, rewrite_test, run_eval  # noqa: E402
from evals.cases import Case, Evidence  # noqa: E402
from evals.judge import JsonlCache  # noqa: E402
from retrieval.search import Hit  # noqa: E402

DOC = "3M_2018_10K"
QUESTIONS = ["What was the capex of 3M in 2018?", "What was the revenue of 3M in 2018?", "What was the net income of 3M in 2018?"]


@pytest.fixture(autouse=True)
def clean_state(tmp_path, monkeypatch):
    """Every test gets its own empty folders, the filter off and every switch at its old setting."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "FILTER_MODE", "off")
    monkeypatch.setattr(config, "QUERY_REWRITE", False)
    monkeypatch.setattr(config, "QUERY_HINTS", True)
    monkeypatch.setattr(config, "QUERY_COUNT", 1)
    monkeypatch.setattr(config, "PAGE_CONTEXT", False)
    monkeypatch.setattr(config, "PAGE_CONTEXT_PAGES", 3)
    monkeypatch.setattr(config, "REWRITE_CACHE", None)
    monkeypatch.setattr(config, "NEIGHBOR_PAGES", 0)
    monkeypatch.setattr(config, "QUERY_CLEAN", False)
    monkeypatch.setattr(config, "REWRITE_PROMPT", "rewrite_v2")
    lookup._caches.clear()


def printed(function, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = function(*args, **kwargs)
    return result, buffer.getvalue()


# ---------- fakes ----------

def make_hit(page=60, piece=1, text="Capex was 1,577 million.", score=0.5, doc=DOC):
    return Hit(score=score, doc=doc, company="3M", year=2018, period="FY", type="10K", page=page, piece=piece, text=text, source="real", access="public")


def hits_on(*pages):
    return [make_hit(page=page, text=f"piece on page {page}") for page in pages]


def make_case(number=1, pages=(60,), question=None, answer="$1,577 million"):
    return Case(
        id=f"financebench_id_{number:05d}", split="test", company="3M", doc=DOC, doc_type="10K", question_type="metrics-generated",
        reasoning_type="Numerical reasoning", question=question or f"What was the capex in case {number}?", answer=answer, justification="",
        evidence=tuple(Evidence(page=page, text=f"evidence text of page {page}") for page in pages),
    )


def make_answer(question="q"):
    return bot_answer.Answer(
        question=question, answer="Capex was 1,577 million dollars.", citations=[{"doc": DOC, "page": 60}], found=True,
        hits=[make_hit(page=60)], tokens_in=2000, tokens_out=100, cost_cents=0.03, seconds=0.5, search_seconds=0.1,
        calls=1, format_ok=True, prompt="bot_v3", model="claude-test",
    )


class FakeAsk:
    """Stands in for llm.ask: gives the texts one by one, then repeats the last one."""

    def __init__(self, *texts, cents=0.01):
        self.texts, self.cents, self.calls = list(texts), cents, []

    def __call__(self, system, user, max_tokens=500):
        self.calls.append((system, user, max_tokens))
        text = self.texts.pop(0) if len(self.texts) > 1 else self.texts[0]
        return llm.Reply(text, 700, 20, self.cents, 0.1, "claude-test", "end_turn", 1)


class LineAsk:
    """Gives a search line for a question (v2 or v3 request), and a new order of pieces for a rerank request."""

    def __init__(self, v2=None, v3=None, orders=None, cents=0.01):
        self.v2, self.v3, self.orders, self.cents, self.calls = v2 or {}, v3 or {}, orders or {}, cents, []

    def __call__(self, system, user, max_tokens=500):
        self.calls.append((system, user, max_tokens))
        question = user.split("<question>\n", 1)[1].split("\n</question>", 1)[0]
        if system == rerank_test.CLAUDE_SYSTEM:
            text = self.orders[question]
        elif "filing would contain" in system:
            text = self.v3[question]
        else:
            text = self.v2.get(question, question)
        return llm.Reply(text, 700, 20, self.cents, 0.1, "claude-test", "end_turn", 1)

    def rerank_calls(self):
        return sum(call[0] == rerank_test.CLAUDE_SYSTEM for call in self.calls)


def fake_fetch(pages, calls=None):
    def fetch(doc, page):
        if calls is not None:
            calls.append((doc, page))
        return [make_hit(page=page, piece=0, text=pages[page], score=1.0, doc=doc)] if page in pages else []

    return fetch


def search_for(table):
    def search(query, **options):
        return list(table.get(query, []))[: options.get("k") or 5]

    return search


# ---------- fix reading: the prompt ----------

def test_bot_v3_is_the_default_and_the_older_prompts_stay():
    assert prompts.DEFAULT_BOT_PROMPT == "bot_v3"
    assert {"bot_v1", "bot_v2", "bot_v3"} <= set(prompts.BOT_PROMPTS)
    system = prompts.BOT_PROMPTS["bot_v3"].system
    for phrase in ("Use only the pieces", "exact item", "table heading", "under 60 words", "found to false", '"working"'):
        assert phrase in system
    assert prompts.BOT_PROMPTS["bot_v3"].piece == prompts.BOT_PROMPTS["bot_v2"].piece


def test_a_reply_with_working_notes_is_read_like_any_other():
    reply = '{"working": "7,838 - 442 = 7,396", "answer": "Free cash flow was 7,396 million dollars.", "citations": [{"doc": "ADOBE_2022_10K", "page": 3}], "found": true}'
    hit = make_hit(doc="ADOBE_2022_10K", page=3, text="Adobe | 2022 | 10K | page 3\nfree cash flow")
    ask = FakeAsk(reply)
    result = bot_answer.answer("What was free cash flow?", search_fn=lambda question, **options: [hit], ask_fn=ask)
    assert result.answer == "Free cash flow was 7,396 million dollars." and result.found is True and result.format_ok is True
    assert result.prompt == "bot_v3" and '"working"' in ask.calls[0][0]


# ---------- fix reading: the judge slip ----------

def test_the_judge_reads_the_last_verdict_when_it_corrects_itself():
    text = '{"verdict": "yes", "reason": "same number"}\n\nWait, the answer says 5 and the reference says 6.\n{"verdict": "no", "reason": "5 is not 6"}'
    assert judge.read_verdict(text) == (False, "5 is not 6")
    flipped = '{"verdict": "no", "reason": "looks different"}\nOn a second look the facts match.\n{"verdict": "yes", "reason": "the same facts"}'
    assert judge.read_verdict(flipped) == (True, "the same facts")
    assert judge.read_verdict(YES_ONE) == (True, "same facts")


YES_ONE = '{"reason": "same facts", "verdict": "yes"}'


def test_a_broken_reply_still_gives_its_last_verdict_word():
    broken = '{"verdict": "yes", "reason": "he said "x""}\nNo wait.\n{"verdict": "no", "reason": "she said "y" instead"}'
    passed, reason = judge.read_verdict(broken)
    assert passed is False and "y" in reason
    assert judge.read_verdict('{"reason": "the "net" figure", "verdict": "yes"}')[0] is True
    with pytest.raises(judge.JudgeError):
        judge.read_verdict("I cannot tell.")


def test_judge_rules_version_2_change_only_the_reply_format():
    assert judge.CURRENT == {"correct": "correct_v2", "supported": "supported_v2"}
    for kind, old, new in (("correct", "correct_v1", "correct_v2"), ("supported", "supported_v1", "supported_v2")):
        before, after = judge.RULES[kind][old], judge.RULES[kind][new]
        assert after.user == before.user and after.kind == before.kind == kind
        assert before.system.endswith(judge.VERDICT_FORMAT) and after.system.endswith(judge.VERDICT_FORMAT_V2)
        assert before.system.removesuffix(judge.VERDICT_FORMAT) == after.system.removesuffix(judge.VERDICT_FORMAT_V2)  # the rule text is the same word for word
    assert judge.VERDICT_FORMAT_V2.index('"reason"') < judge.VERDICT_FORMAT_V2.index('"verdict"')


def test_version_2_is_judged_again_but_a_repeat_is_free():
    ask = FakeAsk('{"reason": "same facts", "verdict": "yes"}')
    old = judge.judge_correct("q", "r", "a", model="claude", ask_fn=ask, version="correct_v1")
    new = judge.judge_correct("q", "r", "a", model="claude", ask_fn=ask, version="correct_v2")
    again = judge.judge_correct("q", "r", "a", model="claude", ask_fn=ask)
    assert len(ask.calls) == 2 and old.version == "correct_v1" and new.version == "correct_v2" and again.cached and again.version == "correct_v2"
    assert ask.calls[0][0] != ask.calls[1][0]


# ---------- fix numbers ----------

def test_a_whole_number_written_with_cents_counts_as_whole():
    assert cc._decimals("3215.00") == 0 and cc._decimals("3,215.00") == 0 and cc._decimals("100.0") == 0
    assert cc._decimals("12.50") == 2 and cc._decimals("0.80") == 2 and cc._decimals("99.00") == 2 and cc._decimals("42") == 0
    assert cc._decimals("100.0", percent=True) == 1
    reference = "$3215.00"
    assert cc.number_match(reference, "Free cash flow was 3,215.4 million dollars.").status == cc.PASS
    assert cc.number_match(reference, "Free cash flow was 3,100 million dollars.").status == cc.FAIL


def test_a_number_the_question_states_is_not_asked_for():
    question = "Are there any product categories that represent more than 20% of Boeing's revenue for FY2022?"
    reference = "Yes. Boeing has categories that represent more than 20% of revenue for FY2022. Commercial Airplanes is 39%, Global Services is 26%."
    answer = "Yes. Commercial Airplanes is 38.8% of revenue and Global Services is 26.4%."
    assert cc.number_match(reference, answer).status == cc.FAIL  # the old call: the 20% is asked for
    assert cc.number_match(reference, answer, question).status == cc.PASS
    assert cc.number_match(reference, "Yes. Commercial Airplanes is 45% of revenue and Global Services is 26.4%.", question).status == cc.FAIL
    only_the_threshold = "Yes, more than 20%."
    assert cc.number_match(only_the_threshold, "Yes.", question).status == cc.SKIP


def test_answer_numbers_keeps_its_old_call_and_can_skip_the_question_numbers():
    assert [item.text for item in cc.answer_numbers("36%. The answer assumes FY2023 is the 12 months to June.")] == ["36%"]
    asked = cc.key_numbers("Is it above 20%?")
    assert [item.text for item in cc.answer_numbers("More than 20%. The share is 39%.", skip=asked)] == ["39%"]


def test_a_change_given_as_one_number_is_found_from_the_two_numbers():
    reference = "No. For AMCOR there has been a slight decline in gross margins by 0.8%."
    assert cc.number_match(reference, "Gross margin declined from 19.4% in FY2022 to 18.5% in FY2023.").status == cc.PASS
    assert cc.number_match(reference, "Gross margin fell from 22.0% in FY2022 to 18.5% in FY2023.").status == cc.FAIL
    assert cc.number_match("Gross margin for AMCOR was 0.8%.", "Gross margin declined from 19.4% to 18.5%.").status == cc.FAIL  # no change word: no difference rule
    assert cc.number_match("Debt fell by $229 million.", "Debt was $3,480 million in 2022 and $3,251 million in 2023.").status == cc.PASS
    assert cc.number_match("Revenue rose by 5 million.", "Revenue was 100 million and 90 million.").status == cc.FAIL


def test_a_plain_wrong_number_still_fails():
    assert cc.number_match("The ratio was 2.77.", "The ratio was 3.01.").status == cc.FAIL
    assert cc.number_match("The ratio was 2.77.", "The ratio was 2.77 and sales were 5.").status == cc.PASS


def test_run_checks_gives_the_question_to_the_number_check():
    case = make_case(question="Is it more than 20% of revenue?", answer="Yes. More than 20% of revenue. Commercial Airplanes is 39%.")
    record = {"answer": "Yes. Commercial Airplanes is 38.8% of revenue.", "citations": [{"doc": DOC, "page": 60}], "found": True,
              "pieces": [{"doc": DOC, "page": 60, "piece": 1, "score": 0.5}], "format_ok": True, "format_note": ""}
    assert cc.run_checks(case, record)["number_match"].status == cc.PASS


# ---------- the rewrite request and the clean-up of its lines ----------

def test_the_new_switches_exist_and_name_known_prompts():
    assert hasattr(config, "NEIGHBOR_PAGES") and hasattr(config, "QUERY_CLEAN") and hasattr(config, "REWRITE_PROMPT")
    assert lookup.REWRITE_PROMPTS == ("rewrite_v2", "rewrite_v3")


def test_rewrite_v2_keeps_the_cache_key_it_always_had(tmp_path):
    question = QUESTIONS[0]
    old_key = hashlib.sha256("\n".join(("rewrite_v2", config.CLAUDE_MODEL, lookup.rewrite_system(1, True), question)).encode("utf-8")).hexdigest()
    cache = JsonlCache(tmp_path / "rewrites.jsonl")
    cache.put(old_key, {"queries": ["3M capital expenditures 2018"]})
    ask = FakeAsk("must not be asked")
    made = lookup.make_queries(question, count=1, hints=True, ask_fn=ask, cache=cache, prompt="rewrite_v2", clean=False)
    default = lookup.make_queries(question, count=1, hints=True, ask_fn=ask, cache=cache)
    assert made.queries == default.queries == ["3M capital expenditures 2018"] and ask.calls == []


def test_each_rewrite_request_keeps_its_own_cache_entry(tmp_path):
    cache = JsonlCache(tmp_path / "rewrites.jsonl")
    ask = FakeAsk("line for v2", "line for v3")
    first = lookup.make_queries(QUESTIONS[0], 1, True, ask_fn=ask, cache=cache, prompt="rewrite_v2", clean=False)
    second = lookup.make_queries(QUESTIONS[0], 1, True, ask_fn=ask, cache=cache, prompt="rewrite_v3", clean=False)
    again = lookup.make_queries(QUESTIONS[0], 1, True, ask_fn=ask, cache=cache, prompt="rewrite_v3", clean=False)
    assert (first.queries, second.queries, again.queries) == (["line for v2"], ["line for v3"], ["line for v3"])
    assert len(ask.calls) == 2 and ask.calls[0][0] != ask.calls[1][0] and "filing would contain" in ask.calls[1][0]
    assert lookup.rewrite_system(1, True) == lookup.rewrite_system(1, True, "rewrite_v2")
    with pytest.raises(ValueError):
        lookup.rewrite_system(1, True, "rewrite_v9")


def test_clean_query_takes_out_labels_markdown_and_a_last_word_search():
    assert lookup.clean_query("**Query:** 3M capital expenditures 2018 search") == "3M capital expenditures 2018"
    assert lookup.clean_query("Search query - Adobe revenue fiscal 2022") == "Adobe revenue fiscal 2022"
    assert lookup.clean_query('"3M capex 2018"') == "3M capex 2018"
    assert lookup.clean_query("Alphabet revenue from Google Search") == "Alphabet revenue from Google Search"  # a real item name keeps its last word
    assert lookup.clean_query("", "What was the capex?") == "What was the capex?"
    assert lookup.clean_query("##", "x" * 500) == "x" * lookup.MAX_QUERY_CHARS


def test_clean_queries_cleans_deeply_only_when_asked(monkeypatch):
    bullet = chr(0x2022)
    text = f"{bullet} Query: Adobe revenue 2022 search\n- adobe revenue 2022\n3. Cash flow statement Adobe 2022"
    assert lookup.clean_queries(text, "q", 3, deep=False) == ["Query: Adobe revenue 2022 search", "adobe revenue 2022", "Cash flow statement Adobe 2022"]
    assert lookup.clean_queries(text, "q", 3, deep=True) == ["Adobe revenue 2022", "Cash flow statement Adobe 2022"]
    assert lookup.clean_queries(text, "q", 3) == lookup.clean_queries(text, "q", 3, deep=False)
    monkeypatch.setattr(config, "QUERY_CLEAN", True)
    assert lookup.clean_queries(text, "q", 3) == lookup.clean_queries(text, "q", 3, deep=True)


def test_stored_lines_are_cleaned_only_when_asked(tmp_path):
    question = QUESTIONS[0]
    key = hashlib.sha256("\n".join(("rewrite_v2", config.CLAUDE_MODEL, lookup.rewrite_system(1, True), question)).encode("utf-8")).hexdigest()
    cache = JsonlCache(tmp_path / "rewrites.jsonl")
    cache.put(key, {"queries": ["Query: 3M capex 2018 search"]})
    ask = FakeAsk("never asked")
    assert lookup.make_queries(question, 1, True, ask_fn=ask, cache=cache, clean=False).queries == ["Query: 3M capex 2018 search"]
    assert lookup.make_queries(question, 1, True, ask_fn=ask, cache=cache, clean=True).queries == ["3M capex 2018"] and ask.calls == []


# ---------- fix neighbors ----------

def test_top_pages_lists_the_first_different_pages():
    hits = hits_on(73, 73, 80, 90, 80)
    assert lookup.top_pages(hits, 2) == [(DOC, 73), (DOC, 80)]
    assert lookup.top_pages(hits, 9) == [(DOC, 73), (DOC, 80), (DOC, 90)]
    assert lookup.top_pages([], 3) == []


def test_neighbors_are_the_pages_before_and_after_nearest_first():
    pages = {number: f"whole page {number}" for number in (1, 2, 72, 73, 74, 79, 80, 81)}
    best = hits_on(73, 80, 1)
    whole, count, _ = lookup.expand_pages(best, fake_fetch(pages), 3)
    more, added, problem = lookup.add_neighbors(whole, lookup.top_pages(best, 3), fake_fetch(pages), 1)
    assert [hit.page for hit in more] == [73, 80, 1, 72, 74, 79, 81, 2] and count == 3 and added == 5 and problem == ""
    assert [hit.text for hit in more][3] == "whole page 72" and more[3].piece == 0
    wider, added_two, _ = lookup.add_neighbors(whole, lookup.top_pages(best, 3), fake_fetch({**pages, 71: "whole page 71", 75: "whole page 75"}), 2)
    assert [hit.page for hit in wider][3:8] == [72, 74, 79, 81, 2] and {71, 75} <= {hit.page for hit in wider} and added_two == 7


def test_neighbors_skip_pages_already_in_the_list_and_pages_that_do_not_exist():
    pages = {73: "p73", 74: "p74"}
    whole, _, _ = lookup.expand_pages(hits_on(73, 74), fake_fetch(pages), 2)
    calls = []
    more, added, _ = lookup.add_neighbors(whole, [(DOC, 73), (DOC, 74)], fake_fetch(pages, calls), 1)
    assert added == 0 and [hit.page for hit in more] == [73, 74] and sorted(calls) == [(DOC, 72), (DOC, 75)]


def test_a_neighbor_page_that_cannot_be_read_is_reported_and_skipped():
    def fetch(doc, page):
        if page == 74:
            raise RuntimeError("index down")
        return [make_hit(page=page, piece=0, text="p", doc=doc)]

    more, added, problem = lookup.add_neighbors(hits_on(73), [(DOC, 73)], fetch, 1)
    assert [hit.page for hit in more] == [73, 72] and added == 1 and "RuntimeError: index down" in problem


def test_no_neighbors_when_the_number_is_zero():
    hits = hits_on(73)
    assert lookup.add_neighbors(hits, [(DOC, 73)], fake_fetch({72: "x"}), 0) == (hits, 0, "")


def find_with(monkeypatch, neighbors, pages, calls=None):
    monkeypatch.setattr(config, "PAGE_CONTEXT", True)
    monkeypatch.setattr(config, "PAGE_CONTEXT_PAGES", 2)
    search_fn = search_for({QUESTIONS[0]: hits_on(73, 80)})
    return lookup.find(QUESTIONS[0], search_fn=search_fn, fetch_fn=fake_fetch(pages, calls), neighbors=neighbors)


def test_find_adds_neighbor_pages_only_when_asked(monkeypatch):
    pages = {number: f"whole page {number}" for number in (72, 73, 74, 79, 80, 81)}
    off_calls = []
    off = find_with(monkeypatch, None, pages, off_calls)  # NEIGHBOR_PAGES is 0
    assert [hit.page for hit in off] == [73, 80] and sorted(off_calls) == [(DOC, 73), (DOC, 80)] and "neighbor_pages" not in off.events[0]
    on = find_with(monkeypatch, 1, pages)
    assert [hit.page for hit in on] == [73, 80, 72, 74, 79, 81]
    assert on.events[0]["neighbor_pages"] == 4 and on.events[0]["whole_pages"] == 2
    monkeypatch.setattr(config, "NEIGHBOR_PAGES", 1)  # the switch in config.py does the same as the argument
    assert [hit.page for hit in find_with(monkeypatch, None, pages)] == [73, 80, 72, 74, 79, 81]


def test_no_neighbors_when_no_whole_page_could_be_read(monkeypatch):
    calls = []
    found = find_with(monkeypatch, 1, {}, calls)
    assert [hit.page for hit in found] == [73, 80] and "neighbor_pages" not in found.events[0] and sorted(calls) == [(DOC, 73), (DOC, 80)]


# ---------- run_eval: --neighbors, --ids and the price guess ----------

def test_the_price_guess_counts_neighbors_for_plain_runs_only():
    plain = run_eval.estimate_cents(75, "claude", "plain", rewrite=True)
    assert run_eval.estimate_cents(75, "claude", "plain", True, 0) == plain
    near = run_eval.estimate_cents(75, "claude", "plain", True, 1)
    assert near[1] > plain[1] and near[2] > plain[2] and plain[0] < near[0] < plain[0] * 3
    assert run_eval.estimate_cents(75, "claude", "agent", True, 1) == run_eval.estimate_cents(75, "claude", "agent", True, 0)
    assert run_eval.estimate_cents(75, "gemini", "plain", True, 1)[2] == 0.0


def test_run_name_marks_a_run_of_chosen_questions():
    assert run_eval.run_name("fixes", "test", None) == "fixes_test.json"
    assert run_eval.run_name("fixes", "test", 5) == "fixes_test_first5.json"
    assert run_eval.run_name("near", "test", None, 7) == "near_test_ids7.json"


def test_ids_picks_the_questions_and_neighbors_reach_the_run(monkeypatch):
    cases = [make_case(number) for number in range(1, 6)]
    monkeypatch.setattr(run_eval, "load_cases", lambda split=None: list(cases))
    monkeypatch.setattr(config, "PAGE_CONTEXT", True)
    seen = []

    def answer_fn(question):
        seen.append((question, config.NEIGHBOR_PAGES))
        return make_answer(question)

    _, out = printed(run_eval.main, ["--split", "test", "--tag", "near", "--no-judge", "--ids", "00004,00002", "--neighbors", "on"], answer_fn=answer_fn)
    assert [question for question, _ in seen] == [cases[3].question, cases[1].question]
    assert all(near == 1 for _, near in seen) and config.NEIGHBOR_PAGES == 0  # on for the run, put back afterwards
    assert "neighbor pages 1" in out and "%" not in out
    run = json.loads((report.runs_dir() / "near_test_ids2.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in run["cases"]] == [cases[3].id, cases[1].id] and run["settings"]["neighbors"] == 1


def test_an_unknown_id_stops_before_any_question_is_asked(monkeypatch):
    monkeypatch.setattr(run_eval, "load_cases", lambda split=None: [make_case(1), make_case(2)])

    def boom(question):
        raise AssertionError("nothing may be asked")

    with pytest.raises(SystemExit) as stop:
        printed(run_eval.main, ["--split", "test", "--tag", "x", "--no-judge", "--ids", "00001,00099"], answer_fn=boom)
    assert "00099" in str(stop.value) and "test split" in str(stop.value)


def test_a_normal_run_has_no_neighbors_and_keeps_its_old_file_name(monkeypatch):
    monkeypatch.setattr(run_eval, "load_cases", lambda split=None: [make_case(1), make_case(2)])
    _, out = printed(run_eval.main, ["--split", "test", "--tag", "plain", "--no-judge"], answer_fn=lambda question: make_answer(question))
    run = json.loads((report.runs_dir() / "plain_test.json").read_text(encoding="utf-8"))
    assert "neighbor pages 0" in out and "neighbors" not in run["settings"] and len(run["cases"]) == 2


# ---------- the rewrite test ----------

def rewrite_world(monkeypatch):
    cases = [make_case(1, (60,), QUESTIONS[0]), make_case(2, (70,), QUESTIONS[1]), make_case(3, (80,), QUESTIONS[2])]
    monkeypatch.setattr(experiments, "usable_cases", lambda first=None: cases[:first] if first else cases)
    clean_line = "3M capital expenditures 2018"
    v3_lines = {QUESTIONS[0]: "3M cash flows purchases of property plant and equipment 2018", QUESTIONS[1]: "3M income statement net sales 2018",
                QUESTIONS[2]: "3M income statement net income attributable to 3M 2018"}
    table = {
        QUESTIONS[0]: hits_on(5), QUESTIONS[1]: hits_on(70), QUESTIONS[2]: [],  # the question as typed
        clean_line: hits_on(60), "3M revenue 2018": hits_on(70), "3M net income 2018": hits_on(7),  # what the rewrite the bot uses now finds
        v3_lines[QUESTIONS[0]]: hits_on(60), v3_lines[QUESTIONS[1]]: hits_on(70), v3_lines[QUESTIONS[2]]: hits_on(80),
    }
    v2_lines = {QUESTIONS[0]: "Query: " + clean_line + " search", QUESTIONS[1]: "3M revenue 2018", QUESTIONS[2]: "3M net income 2018"}
    return LineAsk(v2=v2_lines, v3=v3_lines), search_for(table)


def test_the_rewrite_test_is_free_by_default_and_changes_nothing_in_config(monkeypatch):
    ask, search = rewrite_world(monkeypatch)
    _, out = printed(rewrite_test.main, [], search_fn=search, ask_fn=ask)
    assert "COLAB: no" in out and "CLAUDE COST: 0 cents" in out and "rewrite_test --new" in out
    assert "Noise in the rewrites the bot uses now: 1 of 3 lines" in out and "%" not in out
    assert "found 1 more, lost 0" in out and len(ask.calls) == 3  # one rewrite per question, none of the new request
    assert all("filing would contain" not in call[0] for call in ask.calls)
    saved = json.loads((Path(config.RESULTS_DIR) / "rewrite_test.json").read_text(encoding="utf-8"))
    assert saved["questions"] == 3 and len(saved["variants"]) == 3
    assert "### Rewrite test" in (Path(config.RESULTS_DIR) / "experiments.md").read_text(encoding="utf-8")
    assert config.REWRITE_PROMPT == "rewrite_v2" and config.QUERY_CLEAN is False
    printed(rewrite_test.main, [], search_fn=search, ask_fn=ask)
    assert len(ask.calls) == 3  # the rewrites are in the cache now


def test_the_rewrite_test_with_new_asks_for_the_new_request_and_says_when_it_wins(monkeypatch):
    ask, search = rewrite_world(monkeypatch)
    _, out = printed(rewrite_test.main, ["--new"], search_fn=search, ask_fn=ask)
    assert "CLAUDE COST: about" in out and "The new line wins" in out and "rewrite_v3" in out and "%" not in out
    assert len(ask.calls) == 6 and sum("filing would contain" in call[0] for call in ask.calls) == 3
    saved = json.loads((Path(config.RESULTS_DIR) / "rewrite_test.json").read_text(encoding="utf-8"))
    assert len(saved["variants"]) == 4 and config.REWRITE_PROMPT == "rewrite_v2"


def test_the_rewrite_test_stops_when_it_would_cost_too_much(monkeypatch):
    ask, search = rewrite_world(monkeypatch)
    monkeypatch.setattr(rewrite_test, "SPEND_FREE_CENTS", 0.0001)
    with pytest.raises(SystemExit) as stop:
        printed(rewrite_test.main, ["--new"], search_fn=search, ask_fn=ask)
    assert "--spend" in str(stop.value) and ask.calls == []
    monkeypatch.setattr(rewrite_test, "MAX_CENTS", 0.0001)
    with pytest.raises(SystemExit) as stop:
        printed(rewrite_test.main, ["--new", "--spend"], search_fn=search, ask_fn=ask)
    assert "more than" in str(stop.value) and ask.calls == []


# ---------- the reranker test ----------

class FakeModel:
    """Stands in for the cross-encoder: a piece that says 'right' gets the best score."""

    def predict(self, pairs):
        return [1.0 if "right" in text else 0.0 for _, text in pairs]


def rerank_world(monkeypatch):
    cases = [make_case(1, (60,), QUESTIONS[0]), make_case(2, (70,), QUESTIONS[1]), make_case(3, (80,), QUESTIONS[2])]
    monkeypatch.setattr(experiments, "usable_cases", lambda first=None: cases[:first] if first else cases)
    seventh = [make_hit(page=number, text="other piece") for number in (1, 2, 3, 4, 5, 6)] + [make_hit(page=60, text="right piece with the capex")]
    table = {QUESTIONS[0]: seventh, QUESTIONS[1]: hits_on(70, 2, 3), QUESTIONS[2]: hits_on(9, 8, 7)}
    ask = LineAsk(orders={QUESTIONS[0]: "7, 1, 2, 3, 4, 5, 6", QUESTIONS[1]: "1, 2, 3", QUESTIONS[2]: "1, 2, 3"})
    return ask, search_for(table)


def test_parse_order_completes_and_cleans_the_order():
    assert rerank_test.parse_order("3, 1, 2", 3) == [2, 0, 1]
    assert rerank_test.parse_order("2,2,9,0", 4) == [1, 0, 2, 3]  # a repeat and numbers out of range are dropped, the missing ones follow
    assert rerank_test.parse_order("no numbers here", 3) == [0, 1, 2]


def test_the_claude_request_numbers_the_pieces_and_a_longer_request_costs_more():
    user = rerank_test.claude_user("What was the capex?", hits_on(1, 2))
    assert user.startswith("<question>\nWhat was the capex?\n</question>") and f"[1] {DOC}, page 1:" in user and "[2] " in user
    one = rerank_test.estimate_claude_cents([user])
    assert 0 < one < rerank_test.estimate_claude_cents([user] * 20)


def test_without_a_backend_only_the_free_part_runs_and_nothing_is_downloaded(monkeypatch):
    ask, search = rerank_world(monkeypatch)
    _, out = printed(rerank_test.main, [], search_fn=search, ask_fn=ask)
    assert "COLAB: no" in out and "CLAUDE COST: 0 cents" in out and "%" not in out
    assert "The most a reranker can add here: 1 questions (1 of 3 now, 2 of 3 when the right page is anywhere in the best 20)." in out
    assert ask.rerank_calls() == 0
    saved = json.loads((Path(config.RESULTS_DIR) / "rerank_test.json").read_text(encoding="utf-8"))
    assert saved["questions"] == 3 and len(saved["variants"]) == 2


def test_the_local_model_is_not_downloaded_without_permission(monkeypatch):
    ask, search = rerank_world(monkeypatch)
    loaded = []
    _, out = printed(rerank_test.main, ["--backend", "local"], search_fn=search, ask_fn=ask, cached_fn=lambda name: False, model_fn=lambda name: loaded.append(name))
    assert loaded == [] and "Nothing was downloaded" in out and "--allow-download" in out and "Showing the free part only" in out
    assert "reranked" not in out


def test_the_local_reranker_can_lift_the_right_page_into_the_best_five(monkeypatch):
    ask, search = rerank_world(monkeypatch)
    loaded = []
    _, out = printed(rerank_test.main, ["--backend", "local", "--allow-download"], search_fn=search, ask_fn=ask, cached_fn=lambda name: False,
                     model_fn=lambda name: loaded.append(name) or FakeModel())
    assert loaded == [rerank_test.DEFAULT_MODEL] and "Downloading" in out and "reranked: best 5 of the top 20" in out
    assert "This reranker added 1." in out and "at least 3 more and loses none" in out and "%" not in out


def test_the_claude_reranker_costs_once_and_is_remembered(monkeypatch):
    ask, search = rerank_world(monkeypatch)
    _, out = printed(rerank_test.main, ["--backend", "claude"], search_fn=search, ask_fn=ask)
    assert "Claude reranking: 3 questions x 10 pieces" in out and "This reranker added 1." in out and ask.rerank_calls() == 3
    assert (Path(config.DATA_DIR) / "cache" / "rerank_cache.jsonl").exists()
    calls = len(ask.calls)
    printed(rerank_test.main, ["--backend", "claude"], search_fn=search, ask_fn=ask)
    assert len(ask.calls) == calls  # the orders and the search lines come from the cache


def test_the_claude_reranker_needs_spend_when_it_costs_more_than_a_cent_and_never_more_than_the_limit(monkeypatch):
    ask, search = rerank_world(monkeypatch)
    monkeypatch.setattr(rerank_test, "SPEND_FREE_CENTS", 0.0001)
    _, out = printed(rerank_test.main, ["--backend", "claude"], search_fn=search, ask_fn=ask)
    assert "Add --spend to run it" in out and ask.rerank_calls() == 0 and "reranked" not in out
    monkeypatch.setattr(rerank_test, "MAX_CENTS", 0.0001)
    with pytest.raises(SystemExit) as stop:
        printed(rerank_test.main, ["--backend", "claude", "--spend"], search_fn=search, ask_fn=ask)
    assert "more than" in str(stop.value) and ask.rerank_calls() == 0
