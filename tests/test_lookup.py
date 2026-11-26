"""Tests for step 5.1: bot/lookup.py (query rewrite, finance word hints, several queries, whole pages), the citation top-up in bot/answer.py,
the honest "not found" in the citation check, the agent and switch flags of run_eval, and experiment 8.

No network, no key, no model, no Qdrant server: a fake Claude and a fake search stand in.
Run from the project folder:
    python -m pytest -q
"""
import contextlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import add_page_index  # noqa: E402
import config  # noqa: E402
from bot import answer as bot_answer  # noqa: E402
from bot import llm, lookup, prompts  # noqa: E402
from evals import code_checks as cc  # noqa: E402
from evals import experiments, report, run_eval  # noqa: E402
from evals.cases import Case, Evidence  # noqa: E402
from evals.judge import JsonlCache  # noqa: E402
from retrieval.search import Hit  # noqa: E402

DOC = "ACTIVISIONBLIZZARD_2019_10K"


@pytest.fixture(autouse=True)
def clean_state(tmp_path, monkeypatch):
    """Every test gets its own empty folders, the filter off and the lookup switches as in config.py."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "FILTER_MODE", "off")
    monkeypatch.setattr(config, "QUERY_REWRITE", True)
    monkeypatch.setattr(config, "QUERY_HINTS", True)
    monkeypatch.setattr(config, "QUERY_COUNT", 1)
    monkeypatch.setattr(config, "PAGE_CONTEXT", True)
    monkeypatch.setattr(config, "PAGE_CONTEXT_PAGES", 3)
    monkeypatch.setattr(config, "REWRITE_CACHE", None)
    lookup._caches.clear()


def printed(function, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = function(*args, **kwargs)
    return result, buffer.getvalue()


# ---------- fakes ----------

def make_hit(page=73, piece=1, text="text", score=0.5, doc=DOC):
    return Hit(score=score, doc=doc, company="Activision Blizzard", year=2019, period="FY", type="10K", page=page, piece=piece, text=text, source="real", access="public")


class FakeAsk:
    """Stands in for llm.ask: gives the texts one by one, then repeats the last one."""

    def __init__(self, *texts, cents=0.01):
        self.texts, self.cents, self.calls = list(texts), cents, []

    def __call__(self, system, user, max_tokens=500):
        self.calls.append((system, user, max_tokens))
        text = self.texts.pop(0) if len(self.texts) > 1 else self.texts[0]
        return llm.Reply(text, 700, 20, self.cents, 0.1, "claude-test", "end_turn", 1)


class FakeSearch:
    """Stands in for retrieval.search.search: the hits of a query come from a table."""

    def __init__(self, table):
        self.table, self.calls = table, []

    def __call__(self, query, **options):
        self.calls.append((query, options))
        return list(self.table.get(query, []))[: options.get("k") or 5]


def fake_fetch(pages):
    def fetch(doc, page):
        return [make_hit(page=page, piece=0, text=pages[page], score=1.0, doc=doc)] if page in pages else []

    return fetch


def make_case(number=1, pages=(60,)):
    return Case(
        id=f"financebench_id_{number:05d}", split="test", company="3M", doc="3M_2018_10K", doc_type="10K", question_type="metrics-generated",
        reasoning_type="Numerical reasoning", question=f"What was the capex in case {number}?", answer="$1,577 million", justification="",
        evidence=tuple(Evidence(page=page, text=f"evidence text of page {page}") for page in pages),
    )


def make_answer(question="q"):
    return bot_answer.Answer(
        question=question, answer="Capex was 1,577 million dollars.", citations=[{"doc": "3M_2018_10K", "page": 60}], found=True,
        hits=[make_hit(page=60, doc="3M_2018_10K")], tokens_in=2000, tokens_out=100, cost_cents=0.03, seconds=0.5, search_seconds=0.1,
        calls=1, format_ok=True, prompt="bot_v2", model="claude-test",
    )


# ---------- the queries ----------

def test_clean_queries_removes_bullets_numbers_quotes_and_repeats():
    text = '1. "Activision capital expenditures 2019"\n- activision capital expenditures 2019\n\u2022 total net revenues 2019\n\nextra line'
    assert lookup.clean_queries(text, "q", 3) == ["Activision capital expenditures 2019", "total net revenues 2019", "extra line"]
    assert lookup.clean_queries(text, "q", 1) == ["Activision capital expenditures 2019"]
    assert lookup.clean_queries("  \n ", "the question", 2) == ["the question"]
    assert len(lookup.clean_queries("x" * 1000, "q", 1)[0]) == lookup.MAX_QUERY_CHARS


def test_the_request_holds_the_word_hints_and_the_reply_rule():
    one = lookup.rewrite_system(1, True)
    assert "capex = capital expenditures" in one and "Reply with the query only" in one and "never an order" in one
    many = lookup.rewrite_system(3, True)
    assert "1 to 3 lines" in many and "Reply with the query only" not in many
    assert "capex" not in lookup.rewrite_system(1, False) and "Word hints" not in lookup.rewrite_system(2, False)


def test_make_queries_asks_once_and_remembers(tmp_path):
    cache = JsonlCache(tmp_path / "rewrites.jsonl")
    ask = FakeAsk("Activision capital expenditures purchases of property plant equipment 2019", cents=0.004)
    made = lookup.make_queries("capex of Activision in 2019?", ask_fn=ask, cache=cache)
    assert made.queries == ["Activision capital expenditures purchases of property plant equipment 2019"]
    assert made.cents == 0.004 and made.tokens_in == 700 and made.tokens_out == 20 and made.problem == ""
    assert "<question>\ncapex of Activision in 2019?\n</question>" in ask.calls[0][1] and ask.calls[0][2] == 120
    again = lookup.make_queries("capex of Activision in 2019?", ask_fn=ask, cache=cache)
    assert again.queries == made.queries and again.cents == 0 and len(ask.calls) == 1
    other = lookup.make_queries("capex of Activision in 2019?", hints=False, ask_fn=ask, cache=cache)  # another request, another cache entry
    assert len(ask.calls) == 2 and other.cents == 0.004


def test_make_queries_can_write_several_queries():
    ask = FakeAsk("Activision 2019 capital expenditures cash flow\n2. Activision 2019 total net revenues income statement\nthird line\nfourth line")
    made = lookup.make_queries("capex as a share of revenue?", count=3, ask_fn=ask)
    assert made.queries == ["Activision 2019 capital expenditures cash flow", "Activision 2019 total net revenues income statement", "third line"]
    assert ask.calls[0][2] == 200 and "1 to 3 lines" in ask.calls[0][0]


def test_a_failed_rewrite_falls_back_to_the_question_but_the_fuse_stops_everything():
    def broken(system, user, max_tokens=500):
        raise RuntimeError("Claude did not answer")

    made = lookup.make_queries("What is capex?", ask_fn=broken)
    assert made.queries == ["What is capex?"] and "RuntimeError" in made.problem and made.cents == 0

    def stopped(system, user, max_tokens=500):
        raise llm.BudgetStop("Spend fuse")

    with pytest.raises(llm.BudgetStop):
        lookup.make_queries("What is capex?", ask_fn=stopped)


# ---------- merging and whole pages ----------

def test_merge_hits_keeps_the_best_of_the_first_search_and_adds_the_best_of_the_others():
    first = [make_hit(page=page) for page in (1, 2, 3, 4, 5)]
    second = [make_hit(page=page) for page in (10, 11)]
    third = [make_hit(page=20)]
    assert [hit.page for hit in lookup.merge_hits([first, second, third], 5)] == [1, 2, 3, 10, 20]
    assert [hit.page for hit in lookup.merge_hits([first, second], 5)] == [1, 2, 3, 10, 11]
    assert lookup.merge_hits([first], 3) == first[:3] and lookup.merge_hits([[], []], 5) == []
    repeat = [make_hit(page=1), make_hit(page=11)]  # page 1 is already in the list: the next best piece is taken
    assert [hit.page for hit in lookup.merge_hits([first, repeat], 5)] == [1, 2, 3, 11, 4]
    assert len(lookup.merge_hits([first, second, third], 4)) == 4


def test_expand_pages_sends_the_whole_page_in_place_of_the_first_hit():
    hits = [
        make_hit(page=73, piece=1, score=0.8, text="a"), make_hit(page=70, piece=1, score=0.7, text="b"), make_hit(page=73, piece=2, score=0.6, text="c"),
        make_hit(page=37, piece=1, score=0.5, text="d"), make_hit(page=99, piece=1, score=0.4, text="e"),
    ]
    fetch = fake_fetch({73: "WHOLE 73", 70: "WHOLE 70", 37: "WHOLE 37"})
    out, count, problem = lookup.expand_pages(hits, fetch, 2)
    assert [(hit.page, hit.text) for hit in out] == [(73, "WHOLE 73"), (70, "WHOLE 70"), (37, "d"), (99, "e")]
    assert count == 2 and problem == "" and out[0].score == 0.8 and out[0].piece == 0


def test_a_page_that_cannot_be_read_stays_as_the_piece_and_a_long_page_is_cut():
    hits = [make_hit(page=73, text="piece"), make_hit(page=70, text="other")]

    def broken(doc, page):
        raise RuntimeError("Qdrant is down")

    out, count, problem = lookup.expand_pages(hits, broken, 3)
    assert (out, count) == (hits, 0) and problem == "RuntimeError: Qdrant is down"
    assert lookup.expand_pages(hits, fake_fetch({}), 3) == (hits, 0, "")
    out, count, problem = lookup.expand_pages(hits[:1], fake_fetch({73: "x" * 50}), 1, max_chars=10)
    assert count == 1 and out[0].text == "x" * 10 + "\n[the page is cut here]"


# ---------- find ----------

def test_find_with_both_switches_off_is_the_plain_search_of_the_question(monkeypatch):
    monkeypatch.setattr(config, "QUERY_REWRITE", False)
    monkeypatch.setattr(config, "PAGE_CONTEXT", False)
    search, ask = FakeSearch({"What is capex?": [make_hit(page=1), make_hit(page=2)]}), FakeAsk("never used")
    found = lookup.find("What is capex?", search_fn=search, ask_fn=ask)
    assert [hit.page for hit in found] == [1, 2] and ask.calls == [] and found.events == () and found.cost_cents == 0
    assert search.calls == [("What is capex?", {"k": 5, "company": None, "access": None, "collection": None, "year": None, "mode": "off"})]


def test_find_rewrites_keeps_the_filter_of_the_question_and_sends_whole_pages(monkeypatch):
    monkeypatch.setattr(config, "FILTER_MODE", "company+year")
    monkeypatch.setattr(lookup.reader, "read_question", lambda question, mode: SimpleNamespace(company_filter="Activision Blizzard", year_filter=2019))
    rewritten = "Activision Blizzard 2019 capital expenditures"
    search = FakeSearch({rewritten: [make_hit(page=73, text="part one", score=0.8), make_hit(page=70, text="income", score=0.7)]})
    ask = FakeAsk(rewritten, cents=0.005)
    found = lookup.find("What is the FY2019 capex of Activision Blizzard?", ask_fn=ask, search_fn=search, fetch_fn=fake_fetch({73: "WHOLE PAGE 73", 70: "WHOLE PAGE 70"}))
    query, options = search.calls[0]
    assert query == rewritten and options["company"] == "Activision Blizzard" and options["year"] == 2019 and options["mode"] == "off"
    assert [hit.text for hit in found] == ["WHOLE PAGE 73", "WHOLE PAGE 70"]
    assert found.cost_cents == 0.005 and found.tokens_in == 700 and found.tokens_out == 20 and found.queries == [rewritten]
    assert found.events == [{"stage": "lookup", "queries": [rewritten], "whole_pages": 2}]


def test_find_reads_whole_pages_from_the_index_and_respects_the_role_levels(monkeypatch):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    def payload(page, piece, text, access="public"):
        return {"doc": DOC, "company": "Activision Blizzard", "year": 2019, "period": "FY", "type": "10K", "page": page, "piece": piece, "text": text, "source": "real", "access": access}

    client = QdrantClient(":memory:")
    client.create_collection("c", vectors_config=VectorParams(size=2, distance=Distance.COSINE))
    client.upsert("c", [
        PointStruct(id=1, vector=[1.0, 0.0], payload=payload(73, 1, "Activision | 2019 | 10K | page 73\nOperating activities | 1,831")),
        PointStruct(id=2, vector=[1.0, 0.0], payload=payload(73, 2, "Activision | 2019 | 10K | page 73\nCapital expenditures | (116)")),
        PointStruct(id=3, vector=[1.0, 0.0], payload=payload(80, 1, "restricted page", access="hr")),
    ])
    monkeypatch.setattr(config, "QUERY_REWRITE", False)
    search = FakeSearch({"capex?": [make_hit(page=73, piece=1, text="Operating activities | 1,831"), make_hit(page=80, piece=1, text="restricted page")]})
    found = lookup.find("capex?", search_fn=search, fetch_fn=lookup.page_reader(["public"], "c", client=client))
    assert found[0].text.count("page 73") == 1 and "Operating activities" in found[0].text and "Capital expenditures | (116)" in found[0].text
    assert found[1].text == "restricted page" and found.events[0]["whole_pages"] == 1  # a page above the role is never read whole


def test_find_searches_every_query_and_merges(monkeypatch):
    monkeypatch.setattr(config, "QUERY_COUNT", 3)
    monkeypatch.setattr(config, "PAGE_CONTEXT", False)
    table = {
        "capital expenditures cash flow": [make_hit(page=73), make_hit(page=72), make_hit(page=71)],
        "total net revenues income statement": [make_hit(page=70), make_hit(page=69)],
    }
    search, ask = FakeSearch(table), FakeAsk("capital expenditures cash flow\ntotal net revenues income statement")
    found = lookup.find("capex as a share of revenue?", search_fn=search, ask_fn=ask)
    assert [call[0] for call in search.calls] == list(table) and [hit.page for hit in found] == [73, 72, 71, 70, 69]
    assert found.queries == list(table) and found.events[0]["whole_pages"] == 0


def test_find_says_so_when_a_page_cannot_be_read(monkeypatch):
    monkeypatch.setattr(config, "QUERY_REWRITE", False)

    def no_index(doc, page):
        raise RuntimeError("Index required but not found for page")

    found = lookup.find("capex?", search_fn=FakeSearch({"capex?": [make_hit(page=73)]}), fetch_fn=no_index)
    assert [hit.page for hit in found] == [73] and found.events[0]["whole_pages"] == 0 and "Index required" in found.events[0]["page_problem"]


def test_find_uses_the_question_when_the_rewrite_fails_and_says_so(monkeypatch):
    monkeypatch.setattr(config, "PAGE_CONTEXT", False)

    def broken(system, user, max_tokens=500):
        raise RuntimeError("Claude did not answer")

    search = FakeSearch({"What is capex?": [make_hit(page=1)]})
    found = lookup.find("What is capex?", search_fn=search, ask_fn=broken)
    assert [hit.page for hit in found] == [1] and "RuntimeError" in found.events[0]["problem"] and found.cost_cents == 0
    with pytest.raises(ValueError):
        lookup.find("   ", search_fn=search, ask_fn=broken)


def test_find_remembers_rewrites_in_the_cache_file_of_the_config(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "REWRITE_CACHE", tmp_path / "cache" / "rewrite_cache.jsonl")
    monkeypatch.setattr(config, "PAGE_CONTEXT", False)
    ask, search = FakeAsk("Activision capital expenditures 2019"), FakeSearch({"Activision capital expenditures 2019": [make_hit()]})
    first = lookup.find("capex?", search_fn=search, ask_fn=ask)
    second = lookup.find("capex?", search_fn=search, ask_fn=ask)
    assert len(ask.calls) == 1 and first.cost_cents == 0.01 and second.cost_cents == 0 and (tmp_path / "cache" / "rewrite_cache.jsonl").exists()


def test_the_command_by_hand_prints_the_queries_and_the_price(monkeypatch):
    monkeypatch.setattr(llm, "ask", FakeAsk("Activision 2019 capital expenditures", cents=0.0123))
    _, out = printed(lookup.main, ["capex of Activision in 2019?"])
    assert "COLAB: no" in out and "CLAUDE COST: 0.0123 cents" in out and "1. Activision 2019 capital expenditures" in out


# ---------- the one time page index ----------

class FakeIndexClient:
    def __init__(self, schema):
        self.schema, self.made = schema, []

    def get_collection(self, name):
        return SimpleNamespace(payload_schema=self.schema, points_count=11125)

    def create_payload_index(self, name, field, kind):
        self.made.append((name, field, str(kind)))
        self.schema = {**self.schema, field: "made"}


def test_the_page_index_is_made_once():
    client = FakeIndexClient({"company": "keyword", "doc": "keyword", "year": "integer"})
    assert add_page_index.add_page_index(client, "filings_v1") == "made"
    assert client.made == [("filings_v1", "page", "integer")]
    assert add_page_index.add_page_index(client, "filings_v1") == "already there" and len(client.made) == 1


def test_add_page_index_main_reports_the_state(monkeypatch, capsys):
    import retrieval.search as search_module

    monkeypatch.setattr(search_module, "get_client", lambda: FakeIndexClient({"company": "keyword"}))
    add_page_index.main(["--collection", "filings_v2"])
    out = capsys.readouterr().out
    assert "COLAB: no | MODEL: unchanged | CLAUDE COST: 0 cents" in out and "Index on 'page': made." in out and "company, page" in out


# ---------- the answer ----------

def good_reply(answer="Capital expenditures were 116 million dollars in 2019.", citations=((DOC, 73),), found=True):
    return json.dumps({"answer": answer, "citations": [{"doc": doc, "page": page} for doc, page in citations], "found": found})


def test_answer_adds_the_price_of_the_rewrite_and_the_lookup_events():
    looked = lookup.Found([make_hit(page=73, text="Capital expenditures | (116) | (131) | (155)")])
    looked.cost_cents, looked.tokens_in, looked.tokens_out = 0.004, 700, 20
    looked.events = [{"stage": "lookup", "queries": ["q"], "whole_pages": 1}]
    result = bot_answer.answer("q", search_fn=lambda question, **options: looked, ask_fn=FakeAsk(good_reply(), cents=0.01))
    assert result.cost_cents == pytest.approx(0.014) and (result.tokens_in, result.tokens_out) == (1400, 40) and result.calls == 1
    assert result.events == looked.events and result.citations == [{"doc": DOC, "page": 73}] and result.prompt == prompts.DEFAULT_BOT_PROMPT


def test_a_plain_search_function_still_works_and_adds_nothing():
    result = bot_answer.answer("q", search_fn=lambda question, **options: [make_hit()], ask_fn=FakeAsk(good_reply(), cents=0.01))
    assert result.cost_cents == pytest.approx(0.01) and result.events == [] and result.hits[0].page == 73


def test_a_found_answer_without_citations_cites_the_sent_pages_that_hold_its_numbers():
    hits = [make_hit(page=73, text="Capital expenditures | (116) | (131) | (155)\nTotal net revenues | 6,489 | 7,500 | 7,017"), make_hit(page=5, text="Nothing here 42")]
    text = "Capital expenditures were 116 million dollars against revenue of 6,489 million dollars."
    cites, events = bot_answer.add_missing_citations(text, [], True, hits)
    assert cites == [{"doc": DOC, "page": 73}] and events == [{"stage": "citation", "added": [f"{DOC} p73"]}]
    assert bot_answer.add_missing_citations(text, [{"doc": DOC, "page": 5}], True, hits) == ([{"doc": DOC, "page": 5}], [])  # Claude's own citation is never changed
    assert bot_answer.add_missing_citations(text, [], False, hits) == ([], [])  # not found: nothing to cite
    cites, events = bot_answer.add_missing_citations("Revenue was 999 dollars.", [], True, hits)
    assert cites == [] and events == [{"stage": "citation", "unsupported": True}]


def test_answer_tops_up_the_citation_of_a_found_answer_and_records_it():
    hits = [make_hit(page=73, text="Capital expenditures | (116) | (131) | (155)")]
    result = bot_answer.answer("q", search_fn=lambda question, **options: hits, ask_fn=FakeAsk(good_reply(citations=()), cents=0.01))
    assert result.citations == [{"doc": DOC, "page": 73}] and result.events == [{"stage": "citation", "added": [f"{DOC} p73"]}]
    assert cc.check_citation_valid(result.to_dict()).status == cc.PASS


def test_an_honest_not_found_is_not_a_citation_failure():
    case = SimpleNamespace(pages=[(DOC, 73)])
    not_found = {"citations": [], "found": False}
    assert cc.check_citation_hit(case, not_found).status == cc.SKIP
    assert cc.check_citation_hit(case, {"citations": [], "found": True}).status == cc.FAIL
    assert cc.check_citation_hit(case, {"citations": [{"doc": DOC, "page": 5}], "found": True}).status == cc.FAIL
    assert cc.check_citation_hit(case, {"citations": [{"doc": DOC, "page": 73}], "found": True}).status == cc.PASS


def test_the_default_prompt_asks_for_under_60_words_and_cites_calculated_numbers():
    assert prompts.DEFAULT_BOT_PROMPT == "bot_v3" and "bot_v1" in prompts.BOT_PROMPTS and "bot_v2" in prompts.BOT_PROMPTS
    system = prompts.BOT_PROMPTS["bot_v2"].system
    assert "under 60 words" in system and "whole page" in system and "cite the pages of the numbers you used" in system


# ---------- run_eval: the flags ----------

def test_the_price_guess_counts_the_rewrite_and_the_agent():
    off = run_eval.estimate_cents(10, "claude", "plain", rewrite=False)
    on = run_eval.estimate_cents(10, "claude", "plain", rewrite=True)
    agent = run_eval.estimate_cents(10, "claude", "agent", rewrite=True)
    assert on[1] > off[1] and agent[1] == pytest.approx(off[1] * run_eval.AGENT_FACTOR) and on[2] == off[2] == agent[2]


def test_run_eval_flags_set_the_switches_for_the_run_and_put_them_back(monkeypatch):
    monkeypatch.setattr(run_eval, "load_cases", lambda split=None: [make_case(1)])
    seen = {}

    def answer_fn(question):
        seen.update(rewrite=config.QUERY_REWRITE, pages=config.PAGE_CONTEXT, cache=config.REWRITE_CACHE)
        return make_answer(question)

    _, out = printed(run_eval.main, ["--split", "test", "--tag", "flags", "--no-judge", "--rewrite", "off", "--page-context", "off"], answer_fn=answer_fn)
    assert seen["rewrite"] is False and seen["pages"] is False and str(seen["cache"]).endswith("rewrite_cache.jsonl")
    assert config.QUERY_REWRITE is True and config.PAGE_CONTEXT is True and config.REWRITE_CACHE is None
    assert "query rewrite off | whole pages off" in out
    run = json.loads((report.runs_dir() / "flags_test.json").read_text(encoding="utf-8"))
    assert run["settings"]["rewrite"] is False and run["settings"]["page_context"] is False and run["settings"]["mode"] == "plain"
    assert run["settings"]["prompt"] == prompts.DEFAULT_BOT_PROMPT


def test_the_lookup_note_counts_whole_pages_and_names_the_problem():
    def item(events):
        return {"answer": {"events": events}}

    ok = {"stage": "lookup", "queries": ["q"], "whole_pages": 2}
    bad = {"stage": "lookup", "queries": ["q"], "whole_pages": 0, "page_problem": "RuntimeError: Index required but not found for page"}
    assert run_eval.lookup_note([item([ok]), item([bad]), {"answer": None}]) == "Lookup: a whole page was read for 1 of 2 questions. Problems: RuntimeError: Index required but not found for page"
    assert run_eval.lookup_note([item([]), item([{"stage": "injection"}])]) == ""


def test_run_eval_agent_mode_runs_the_agent(monkeypatch):
    from bot import agent

    monkeypatch.setattr(run_eval, "load_cases", lambda split=None: [make_case(1)])
    calls = []

    def fake_agent(question, **options):
        calls.append((question, options))
        return make_answer(question)

    monkeypatch.setattr(agent, "run_agent", fake_agent)
    _, out = printed(run_eval.main, ["--split", "test", "--tag", "agentrun", "--no-judge", "--mode", "agent"])
    assert len(calls) == 1 and "agent mode" in out
    run = json.loads((report.runs_dir() / "agentrun_test.json").read_text(encoding="utf-8"))
    assert run["settings"]["mode"] == "agent"


# ---------- experiment 8 ----------

def eight_setup():
    cases = [make_case(1), make_case(2)]

    def hit(page):
        return make_hit(page=page, doc=cases[0].doc)

    table = {
        cases[0].question: [hit(2)], cases[1].question: [hit(3)],  # no rewrite: both miss
        "plain rewrite": [hit(1)],  # the first rewrite: both miss
        "hinted capex query": [hit(60)],  # with word hints: both hit
        "capital expenditures cash flow": [hit(60)], "revenue income statement": [hit(61)],
    }
    calls = []

    def ask(system, user, max_tokens=500):
        calls.append(system)
        if "1 to 3 lines" in system:
            text = "capital expenditures cash flow\nrevenue income statement"
        elif "Word hints" in system:
            text = "hinted capex query"
        else:
            text = "plain rewrite"
        return llm.Reply(text, 100, 10, 0.004, 0.1, "claude-test", "end_turn", 1)

    def search(question, k=None, mode=None, collection=None, company=None, year=None):
        return table.get(question, [])[:k]

    return cases, ask, search, calls


def test_experiment_eight_scores_all_four_ways_and_remembers_the_queries():
    cases, ask, search, calls = eight_setup()
    lines = []
    ctx = experiments.Context(search_fn=search, ask_fn=ask)
    assert experiments.run_one(8, cases, ctx, spend=False, say=lines.append) is False
    assert calls == [] and "Add --spend" in "\n".join(lines)
    assert experiments.run_one(8, cases, ctx, spend=True, say=lines.append) is True
    saved = json.loads((Path(config.RESULTS_DIR) / "exp8_queries.json").read_text(encoding="utf-8"))
    hits = {name: item["summary"]["hit"] for name, item in saved["variants"].items()}
    assert hits == {"no rewrite": 0, "rewrite by Claude": 0, "rewrite + word hints": 2, "up to 3 queries + word hints": 2}
    assert len(calls) == 6 and ctx.rewrite_cents == pytest.approx(0.024)
    again = experiments.Context(search_fn=search, ask_fn=ask)
    experiments.run_one(8, cases, again, spend=True, say=lambda text: None)
    assert len(calls) == 6 and again.rewrite_cents == 0  # the second run came from the cache
    assert "Experiment 8" in (Path(config.RESULTS_DIR) / "experiments.md").read_text(encoding="utf-8")


def test_experiment_eight_is_listed_and_numbers_up_to_eight_are_accepted():
    _, out = printed(experiments.main, ["--list"])
    assert "8  Query words" in out and out.index("7  Judge model") < out.index("8  Query words")
    with pytest.raises(SystemExit):
        printed(experiments.main, ["9"])
