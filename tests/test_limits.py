"""Tests for step 4.6, the limits of the public demo: daily spend, total spend, visitor, question length and agent steps.

No network, no key, no model: a local log folder and a fake Claude stand in. Run from the project folder:
    python -m pytest -q
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

import config
from bot import agent, llm
from bot.answer import Answer
from monitor import flow, store
from security import guards

NOW = datetime(2026, 11, 23, 12, 0, tzinfo=timezone.utc)


def spend(cents, when=NOW, source="app"):
    store.get_client().insert(store.REQUESTS, {"source": source, "cost_cents": cents, "created_at": store.iso(when), "question": "q", "answer": "a"})


def limit(asked=0):
    return store.check_limits(asked, now=NOW)


def test_a_day_with_room_left_is_allowed():
    spend(1.8)  # 1.8 and the 0.15 reserve are 1.95: under 2 cents
    assert limit().ok


def test_the_daily_limit_stops_a_question_that_could_cross_it():
    spend(1.9)  # 1.9 and the reserve are 2.05: over 2 cents
    result = limit()
    assert not result.ok and result.code == "daily" and "tomorrow" in result.text


def test_yesterdays_spend_does_not_count_against_today():
    spend(1.9, when=NOW - timedelta(days=1))
    assert limit().ok


def test_the_total_limit_closes_the_demo_for_good():
    for day in range(1, 6):
        spend(2.9, when=NOW - timedelta(days=day))  # 14.5 in all, plus the reserve is 14.65: still allowed
    assert limit().ok
    spend(0.4, when=NOW - timedelta(days=7))  # 14.9 in all, plus the reserve is 15.05
    result = limit()
    assert not result.ok and result.code == "total"


def test_only_the_public_app_spends_the_demo_money():
    spend(14.0, source="eval")
    assert limit().ok


def test_the_visitor_limit_is_10_questions():
    assert config.DEMO_SESSION_QUESTIONS == 10
    assert limit(asked=9).ok
    result = limit(asked=10)
    assert not result.ok and result.code == "session"


class Broken:
    """A log that is switched off: it can neither be read nor written."""

    def select(self, *args, **kwargs):
        raise ConnectionError("down")

    def insert(self, *args, **kwargs):
        raise ConnectionError("down")


def test_when_the_log_is_off_the_demo_still_answers_and_counts_its_own_spend():
    assert store.check_limits(0, client=Broken(), now=datetime.now(timezone.utc)).ok  # nothing spent yet: it answers
    answer = Answer("q", "a", [], False, [], cost_cents=1.95, role="guest")
    assert store.log_request(answer, client=Broken()) is None  # the line is lost, the spend is not
    result = store.check_limits(0, client=Broken(), now=datetime.now(timezone.utc))
    assert not result.ok and result.code == "daily"
    assert store.spend_cents(client=Broken()) == 1.95


def test_eval_runs_do_not_count_in_the_servers_own_spend():
    store.log_request(Answer("q", "a", [], False, [], cost_cents=5.0), source="eval", client=Broken())
    assert store.spend_cents(client=Broken()) == 0


def test_a_refused_question_never_reaches_the_bot():
    spend(1.95)
    called = []
    outcome = flow.ask("What was revenue?", ask_fn=lambda *args, **kwargs: called.append(1), now=NOW)
    assert outcome.result is None and outcome.limit.code == "daily" and not called and outcome.request_id is None


def test_the_question_limit_is_500_characters():
    assert config.MAX_QUESTION_CHARS == 500 and guards.MAX_QUESTION_CHARS == 500
    assert guards.check_input("x" * 500, "guest").ok
    stopped = guards.check_input("x" * 501, "guest")
    assert not stopped.ok and "500" in stopped.reason


def test_the_agent_stops_after_4_tool_steps():
    assert agent.MAX_STEPS == 4
    asked = []

    def ask_fn(system, user, max_tokens=500):
        asked.append(user)
        text = json.dumps({"tool": "calculate", "args": {"expression": f"{len(asked)}+1"}})  # it never stops calling tools
        return llm.Reply(text, 500, 50, llm.cost_cents(500, 50), 0.1, "fake", "end_turn", 1)

    result = agent.run_agent("q", role="guest", ask_fn=ask_fn, search_fn=lambda *a, **k: [], page_fn=lambda *a: [])
    steps = [event for event in result.events if event.get("stage") == "agent"]
    stop = [event for event in result.events if event.get("stage") == "agent_stop"][0]
    assert len(steps) == 4 and stop["steps"] == 4 and len(asked) <= agent.MAX_CALLS


def test_the_agent_stops_at_12000_input_tokens():
    assert agent.MAX_TOKENS_IN == 12000


def test_the_limits_are_the_planned_numbers():
    assert (config.DEMO_DAILY_CENTS, config.DEMO_TOTAL_CENTS) == (2.0, 15.0)
    assert 0 < config.DEMO_DAILY_CENTS < config.DEMO_TOTAL_CENTS < config.BUDGET_CENTS  # the demo is a small part of the whole budget
    assert config.DEMO_NOTICE.startswith("Public SEC filings. Not investment advice.")
