"""Tests for steps 2.1 and 2.2: the Claude wrapper (retry and spend fuse), the prompts and the bot.

No network and no key: a fake Claude client and a fake search stand in. Run from the project folder:
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

import config  # noqa: E402
from bot import answer as bot_answer  # noqa: E402
from bot import llm, prompts  # noqa: E402
from retrieval.search import Hit  # noqa: E402


@pytest.fixture(autouse=True)
def clean_state(tmp_path, monkeypatch):
    """Every test gets its own empty ledger folder."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)


def printed(function, *args, **kwargs):
    """Run a function and return (its result, everything it printed)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = function(*args, **kwargs)
    return result, buffer.getvalue()


# ---------- fakes ----------

class StatusError(Exception):
    """Looks like an error of the Claude library: it has a status_code."""

    def __init__(self, status, message="error"):
        super().__init__(message)
        self.status_code = status


class APIConnectionError(Exception):
    """Same class name as the library's lost connection error. It has no status_code."""


def message(text, tokens_in=1000, tokens_out=100, stop="end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=tokens_in, output_tokens=tokens_out),
        stop_reason=stop,
    )


class FakeClient:
    """Gives the replies one by one. An item that is an exception is raised instead."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []
        self.messages = self

    def create(self, **arguments):
        self.calls.append(arguments)
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def write_ledger(cents):
    data = {"cents": cents, "calls": 5, "tokens_in": 1, "tokens_out": 1}
    (config.DATA_DIR / "ledger.json").write_text(json.dumps(data), encoding="utf-8")


# ---------- llm.ask ----------

def test_ask_returns_text_tokens_and_cost():
    client = FakeClient(message("OK", 1000, 100))
    reply = llm.ask("rules", "question", max_tokens=50, client=client)
    assert reply.text == "OK"
    assert (reply.tokens_in, reply.tokens_out) == (1000, 100)
    expected = (1000 * config.PRICE_IN_CENTS_PER_MTOK + 100 * config.PRICE_OUT_CENTS_PER_MTOK) / 1_000_000
    assert reply.cost_cents == pytest.approx(expected)
    assert reply.tries == 1 and reply.model == config.CLAUDE_MODEL and reply.stop_reason == "end_turn"


def test_ask_sends_model_system_and_user_and_nothing_else():
    client = FakeClient(message("OK"))
    llm.ask("the rules", "the question", max_tokens=77, client=client)
    # No temperature: the current Claude library has no such setting and would refuse the call.
    assert client.calls == [{
        "model": config.CLAUDE_MODEL,
        "max_tokens": 77, "thinking": {"type": "disabled"},
        "system": "the rules",
        "messages": [{"role": "user", "content": "the question"}],
    }]


def test_ask_joins_text_blocks_and_skips_other_blocks():
    reply_message = message("")
    reply_message.content = [
        SimpleNamespace(type="text", text="A"),
        SimpleNamespace(type="tool_use"),
        SimpleNamespace(type="text", text="B"),
    ]
    assert llm.ask("s", "u", client=FakeClient(reply_message)).text == "AB"


def test_the_ledger_adds_up_every_call():
    client = FakeClient(message("a", 1000, 100), message("b", 2000, 200))
    llm.ask("s", "u", client=client)
    llm.ask("s", "u", client=client)
    ledger = llm.read_ledger()
    assert (ledger["calls"], ledger["tokens_in"], ledger["tokens_out"]) == (2, 3000, 300)
    assert ledger["cents"] == pytest.approx(llm.cost_cents(3000, 300))
    assert llm.spent_cents() == pytest.approx(ledger["cents"])


def test_a_missing_ledger_means_nothing_is_spent():
    assert llm.read_ledger() == {"cents": 0, "calls": 0, "tokens_in": 0, "tokens_out": 0}


# ---------- the spend fuse ----------

def test_the_fuse_refuses_a_call_at_the_limit():
    write_ledger(config.FUSE_CENTS)
    client = FakeClient(message("OK"))
    with pytest.raises(llm.BudgetStop, match="Spend fuse"):
        llm.ask("s", "u", client=client)
    assert client.calls == []  # Claude was never called


def test_the_fuse_lets_a_call_through_just_below_the_limit():
    write_ledger(config.FUSE_CENTS - 0.01)
    assert llm.ask("s", "u", client=FakeClient(message("OK"))).text == "OK"


def test_a_damaged_ledger_stops_the_calls():
    (config.DATA_DIR / "ledger.json").write_text("this is not json", encoding="utf-8")
    client = FakeClient(message("OK"))
    with pytest.raises(llm.BudgetStop, match="damaged"):
        llm.ask("s", "u", client=client)
    assert client.calls == []


# ---------- retries and errors ----------

def test_a_rate_limit_and_an_overload_are_retried_with_waits():
    waits = []
    client = FakeClient(StatusError(429), StatusError(529), message("OK"))
    reply = llm.ask("s", "u", client=client, sleep=waits.append)
    assert reply.text == "OK" and reply.tries == 3
    assert waits == [2, 5]
    assert llm.read_ledger()["calls"] == 1  # the failed tries cost nothing


def test_a_lost_connection_is_retried():
    client = FakeClient(APIConnectionError("lost"), message("OK"))
    reply = llm.ask("s", "u", client=client, sleep=lambda seconds: None)
    assert reply.text == "OK" and reply.tries == 2


def test_it_gives_up_after_four_tries():
    client = FakeClient(*[StatusError(529)] * 4)
    with pytest.raises(RuntimeError, match="4 tries"):
        llm.ask("s", "u", client=client, sleep=lambda seconds: None)
    assert len(client.calls) == 4
    assert llm.read_ledger()["calls"] == 0


def test_a_bad_key_is_not_retried_and_the_message_is_plain():
    client = FakeClient(StatusError(401))
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        llm.ask("s", "u", client=client, sleep=lambda seconds: None)
    assert len(client.calls) == 1


def test_an_unknown_model_gives_a_plain_message():
    with pytest.raises(RuntimeError, match="CLAUDE_MODEL"):
        llm.ask("s", "u", client=FakeClient(StatusError(404)))


def test_a_bad_request_gives_a_plain_message_and_is_not_retried():
    client = FakeClient(StatusError(400, "max_tokens is too large"))
    with pytest.raises(RuntimeError, match="400"):
        llm.ask("s", "u", client=client)
    assert len(client.calls) == 1


# ---------- llm.main ----------

def test_main_makes_one_call_and_prints_the_cost_in_cents(monkeypatch):
    monkeypatch.setattr(llm, "get_client", lambda: FakeClient(message("OK", 14, 4)))
    _, out = printed(llm.main, [])
    assert "Claude says: OK" in out
    assert "cost 0.0" in out and "cents" in out
    assert "Ledger:" in out and "in 1 calls" in out


def test_main_spent_makes_no_call(monkeypatch):
    monkeypatch.setattr(llm, "get_client", lambda: pytest.fail("no call expected"))
    _, out = printed(llm.main, ["--spent"])
    assert f"Ledger: 0.0000 of {config.FUSE_CENTS} cents used in 0 calls" in out


def test_main_says_so_when_the_fuse_is_hit():
    write_ledger(config.FUSE_CENTS)
    with pytest.raises(SystemExit, match="Spend fuse"):
        llm.main([])


# ---------- prompts ----------

def test_prompt_versions_have_names_and_fill_in():
    assert prompts.DEFAULT_BOT_PROMPT in prompts.BOT_PROMPTS
    for name, version in prompts.BOT_PROMPTS.items():
        assert version.name == name
    version = prompts.BOT_PROMPTS[prompts.DEFAULT_BOT_PROMPT]
    piece = version.piece.format(doc="A_2020_10K", page=3, text="text with {braces}")
    assert piece.startswith('<piece doc="A_2020_10K" page="3">') and "{braces}" in piece
    assert "<question>\nWhy?\n</question>" in version.user.format(pieces=piece, question="Why?")


def test_the_system_prompt_holds_the_agreed_rules():
    system = prompts.BOT_PROMPTS["bot_v1"].system
    for words in ("only the pieces", "never orders", "found", "citations", "JSON", "investment advice", "unit and period"):
        assert words in system


# ---------- reading the reply ----------

GOOD = json.dumps({"answer": "100 million dollars in fiscal year 2022", "citations": [{"doc": "ADOBE_2022_10K", "page": 3}], "found": True})


def test_parse_reply_reads_plain_fenced_and_wrapped_json():
    data = {"a": 1}
    assert bot_answer.parse_reply('{"a": 1}') == data
    assert bot_answer.parse_reply('```json\n{"a": 1}\n```') == data
    assert bot_answer.parse_reply('Here you go: {"a": 1} I hope this helps.') == data
    assert bot_answer.parse_reply("no json here") is None
    assert bot_answer.parse_reply("[1, 2]") is None
    assert bot_answer.parse_reply('{"a": ') is None  # cut off


def test_read_fields_accepts_the_agreed_shape():
    fields = bot_answer.read_fields(json.loads(GOOD))
    assert fields == ("100 million dollars in fiscal year 2022", [{"doc": "ADOBE_2022_10K", "page": 3}], True, "")


def test_read_fields_repairs_and_notes_small_mistakes():
    data = {"answer": 5, "citations": [{"doc": "A", "page": "7"}, {"doc": "B"}, "junk"], "found": "true"}
    text, citations, found, note = bot_answer.read_fields(data)
    assert text == "5" and found is True
    assert citations == [{"doc": "A", "page": 7}]
    for part in ("answer is not text", "found is not true or false", "written as text", "not a doc and a page"):
        assert part in note


def test_read_fields_notes_missing_keys():
    text, citations, found, note = bot_answer.read_fields({"answer": "x"})
    assert (text, citations, found) == ("x", [], False)
    assert "citations is missing" in note and "found is missing" in note


# ---------- the bot ----------

def make_hit(doc="ADOBE_2022_10K", page=3, text="Adobe | 2022 | 10K | page 3\nrevenue was 100", score=0.9, piece=1):
    return Hit(score, doc, "Adobe", 2022, "FY", "10K", page, piece, text, "real", "public")


def fake_search(hits):
    calls = []

    def search_fn(question, k=None, company=None, access=None, collection=None):
        calls.append({"question": question, "k": k, "company": company, "access": access, "collection": collection})
        return hits

    search_fn.calls = calls
    return search_fn


def fake_ask(*texts, tokens_in=1000, tokens_out=100):
    """An ask function that gives the texts one by one and remembers what it was asked."""
    asked = []

    def ask_fn(system, user, max_tokens=500):
        asked.append({"system": system, "user": user, "max_tokens": max_tokens})
        cents = llm.cost_cents(tokens_in, tokens_out)
        return llm.Reply(texts[len(asked) - 1], tokens_in, tokens_out, cents, 0.5, "fake_model", "end_turn", 1)

    ask_fn.asked = asked
    return ask_fn


def test_answer_asks_claude_with_tagged_pieces_and_reads_the_reply():
    hits = [make_hit(), make_hit("AES_2022_10K", 9, "AES | 2022 | 10K | page 9\n{curly} text", 0.5)]
    ask_fn = fake_ask(GOOD)
    result = bot_answer.answer("What was revenue?", search_fn=fake_search(hits), ask_fn=ask_fn)
    asked = ask_fn.asked[0]
    assert asked["system"] == prompts.BOT_PROMPTS[prompts.DEFAULT_BOT_PROMPT].system
    assert asked["max_tokens"] == bot_answer.BOT_MAX_TOKENS
    assert '<piece doc="ADOBE_2022_10K" page="3">' in asked["user"]
    assert '<piece doc="AES_2022_10K" page="9">' in asked["user"] and "{curly}" in asked["user"]
    assert "<question>\nWhat was revenue?\n</question>" in asked["user"]
    assert result.answer.startswith("100 million dollars") and result.citations == [{"doc": "ADOBE_2022_10K", "page": 3}]
    assert result.found is True and result.format_ok and result.format_note == ""
    assert result.calls == 1 and (result.tokens_in, result.tokens_out) == (1000, 100)
    assert result.cost_cents == pytest.approx(llm.cost_cents(1000, 100))
    assert result.hits == hits and result.prompt == prompts.DEFAULT_BOT_PROMPT and result.seconds >= 0 and result.skipped_reason == ""


def test_answer_passes_the_search_options_on():
    search_fn = fake_search([make_hit()])
    bot_answer.answer("q", k=3, company="Adobe", access=["public"], collection="other", search_fn=search_fn, ask_fn=fake_ask(GOOD))
    assert search_fn.calls == [{"question": "q", "k": 3, "company": "Adobe", "access": ["public"], "collection": "other"}]


def test_answer_retries_once_after_a_reply_without_json():
    ask_fn = fake_ask("Sorry, I think the answer is 100.", GOOD)
    result = bot_answer.answer("q", search_fn=fake_search([make_hit()]), ask_fn=ask_fn)
    assert result.calls == 2 and result.format_ok and result.found
    assert ask_fn.asked[1]["user"] == ask_fn.asked[0]["user"] + prompts.BOT_PROMPTS["bot_v1"].retry
    assert ask_fn.asked[1]["max_tokens"] == 2 * bot_answer.BOT_MAX_TOKENS
    assert (result.tokens_in, result.tokens_out) == (2000, 200)
    assert result.cost_cents == pytest.approx(2 * llm.cost_cents(1000, 100))


def test_answer_after_two_bad_replies_says_so():
    result = bot_answer.answer("q", search_fn=fake_search([make_hit()]), ask_fn=fake_ask("not json", "still not json"))
    assert result.calls == 2 and not result.format_ok
    assert result.format_note == "no JSON object in the reply"
    assert result.answer == "" and result.found is False and result.raw == "still not json"


def test_answer_says_when_the_reply_was_cut_short():
    def ask_fn(system, user, max_tokens=500):
        return llm.Reply("", 1000, max_tokens, 0.01, 0.5, "fake_model", "max_tokens", 1)

    result = bot_answer.answer("q", search_fn=fake_search([make_hit()]), ask_fn=ask_fn)
    assert result.calls == 2 and not result.format_ok
    assert result.format_note == "no JSON object in the reply (the reply was cut at max_tokens)"


def test_answer_keeps_a_reply_with_the_wrong_shape():
    reply = json.dumps({"answer": "x", "citations": []})
    result = bot_answer.answer("q", search_fn=fake_search([make_hit()]), ask_fn=fake_ask(reply))
    assert result.calls == 1 and not result.format_ok
    assert "found is missing" in result.format_note and result.answer == "x" and result.found is False


def test_a_weak_search_does_not_call_claude():
    ask_fn = fake_ask(GOOD)
    result = bot_answer.answer("q", min_score=0.95, search_fn=fake_search([make_hit(score=0.9)]), ask_fn=ask_fn)
    assert ask_fn.asked == [] and result.calls == 0 and result.cost_cents == 0
    assert result.found is False and result.answer == "Not in the documents." and result.format_ok
    assert "below" in result.skipped_reason


def test_a_strong_search_does_call_claude():
    ask_fn = fake_ask(GOOD)
    result = bot_answer.answer("q", min_score=0.5, search_fn=fake_search([make_hit(score=0.9)]), ask_fn=ask_fn)
    assert result.calls == 1 and result.skipped_reason == ""


def test_no_pieces_does_not_call_claude():
    ask_fn = fake_ask(GOOD)
    result = bot_answer.answer("q", search_fn=fake_search([]), ask_fn=ask_fn)
    assert ask_fn.asked == [] and result.skipped_reason == "search found no pieces" and result.found is False


def test_the_fuse_stops_the_bot_before_any_call():
    write_ledger(config.FUSE_CENTS)
    with pytest.raises(llm.BudgetStop):
        bot_answer.answer("q", search_fn=fake_search([make_hit()]), ask_fn=llm.ask)


def test_to_dict_is_plain_data():
    result = bot_answer.answer("q", search_fn=fake_search([make_hit()]), ask_fn=fake_ask(GOOD))
    plain = result.to_dict()
    json.dumps(plain)  # must not fail
    assert plain["pieces"] == [{"doc": "ADOBE_2022_10K", "page": 3, "piece": 1, "score": 0.9}]
    assert plain["found"] is True and plain["cost_cents"] > 0 and plain["calls"] == 1
    assert "revenue was 100" in result.to_dict(include_text=True)["pieces"][0]["text"]


# ---------- checking it by hand ----------

def case_line(case_id, split, page, answer):
    """One line of a cases file, as a dict."""
    return {
        "id": case_id, "split": split, "company": "Adobe", "doc": "ADOBE_2022_10K", "doc_type": "10K",
        "question_type": "domain relevant", "reasoning_type": "Information extraction",
        "question": f"Question {case_id}?", "answer": answer, "justification": "",
        "evidence": [{"page": page, "text": "evidence text"}],
    }


@pytest.fixture
def cases_file(tmp_path, monkeypatch):
    """A tiny cases file: two practice cases and one test case. clean_state already points DATA_DIR at tmp_path."""
    (tmp_path / "pdf_list.txt").write_text("ADOBE_2022_10K\n", encoding="utf-8")
    monkeypatch.setattr(config, "PDF_LIST", tmp_path / "pdf_list.txt")
    lines = [
        case_line("id_1", "practice", 3, "$100 million"),
        case_line("id_2", "practice", 1, "$20 million"),
        case_line("id_3", "test", 5, "$7 million"),
    ]
    (tmp_path / "cases").mkdir()
    (tmp_path / "cases" / "test_cases.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")


def test_practice_run_uses_only_the_practice_cases(cases_file):
    _, out = printed(bot_answer.run_practice, 10, fake_search([make_hit(page=3)]), fake_ask(GOOD, GOOD))
    assert "[1/2] id_1" in out and "[2/2] id_2" in out
    assert "id_3" not in out and "Answered 2 of 2." in out


def test_practice_run_says_how_to_make_the_file_when_it_is_missing():
    with pytest.raises(SystemExit, match="evals.build_cases"):
        printed(bot_answer.main, ["--practice"], search_fn=fake_search([make_hit()]), ask_fn=fake_ask(GOOD))


def test_practice_run_prints_each_answer_next_to_its_reference(cases_file):
    (cited, total), out = printed(bot_answer.run_practice, 10, fake_search([make_hit(page=3)]), fake_ask(GOOD, GOOD))
    assert (cited, total) == (1, 2)  # question 1 has its evidence on page 3, question 2 does not
    assert "Reference: $100 million" in out and "Reference: $20 million" in out
    assert "Evidence:  ADOBE_2022_10K p3" in out and "Evidence:  ADOBE_2022_10K p1" in out
    assert "Bot:       100 million dollars in fiscal year 2022" in out
    assert "cited an evidence page: yes" in out and "cited an evidence page: no" in out
    assert "Answered 2 of 2. Cited an evidence page: 1 of 2. Said found true: 2 of 2." in out
    assert "Ledger:" in out and "write 3 lines" in out


def test_practice_run_goes_on_after_one_error(cases_file):
    calls = []

    def ask_fn(system, user, max_tokens=500):
        calls.append(user)
        if len(calls) == 1:
            raise RuntimeError("Claude did not answer")
        return llm.Reply(GOOD, 1000, 100, 0.01, 0.5, "fake_model", "end_turn", 1)

    _, out = printed(bot_answer.run_practice, 10, fake_search([make_hit()]), ask_fn)
    assert "ERROR:     Claude did not answer" in out
    assert "Answered 1 of 2." in out


def test_practice_run_stops_when_the_fuse_is_hit(cases_file):
    def ask_fn(system, user, max_tokens=500):
        raise llm.BudgetStop("Spend fuse: 90.00 cents are used and the limit is 90. No Claude call was made.")

    with pytest.raises(SystemExit, match="Spend fuse"):
        printed(bot_answer.main, ["--practice"], search_fn=fake_search([make_hit()]), ask_fn=ask_fn)


def test_main_answers_one_question():
    _, out = printed(bot_answer.main, ["What was revenue?"], search_fn=fake_search([make_hit()]), ask_fn=fake_ask(GOOD))
    assert "Pieces:    ADOBE_2022_10K p3 (0.900)" in out
    assert "Answer:    100 million dollars in fiscal year 2022" in out
    assert "Cited:     ADOBE_2022_10K p3" in out
    assert "Found: true | format: ok" in out and "Cost:" in out


def test_main_needs_a_question_or_practice():
    with pytest.raises(SystemExit):
        bot_answer.main([])
