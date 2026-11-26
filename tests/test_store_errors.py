"""Step 4.1 follow-up: a failed log call says what Supabase said, and the check command tries a write."""
import logging
from types import SimpleNamespace as NS

import pytest

from bot.answer import Answer
from monitor import flow, store

DENIED = {"code": "42501", "message": "permission denied for table requests",
          "hint": "Grant the required privileges to the current role with: GRANT INSERT ON public.requests TO anon;"}


def failing_http(status, body=None):
    """A fake HTTP library whose every reply is an error. body None means the reply is not JSON."""
    def raise_error():
        raise RuntimeError("bad status")

    def say():
        if body is None:
            raise ValueError("not JSON")
        return body

    class Http:
        def request(self, method, url, **kwargs):
            return NS(status_code=status, json=say, raise_for_status=raise_error)

    return Http()


class Client:
    """A store that can always be read. The write gives back saved, or fails with reason."""

    def __init__(self, reason=None, saved=None):
        self.reason, self.saved = reason, saved

    def select(self, table, filters=None, order=None, limit=None, columns="*"):
        return []

    def insert(self, table, record):
        if self.reason:
            raise store.StoreError(self.reason)
        return self.saved


def test_a_refused_call_says_the_status_and_what_supabase_said_and_never_the_key():
    client = store.SupabaseClient("https://x.supabase.co", "SECRETKEY123", http=failing_http(403, DENIED))
    with pytest.raises(store.StoreError) as caught:
        client.insert(store.REQUESTS, {"a": 1})
    text = store.describe(caught.value)
    assert text.startswith("HTTP 403: permission denied for table requests") and "GRANT INSERT" in text
    assert "SECRETKEY123" not in text and "SECRETKEY123" not in repr(caught.value)


def test_a_reply_that_is_not_json_still_gives_the_status():
    client = store.SupabaseClient("https://x.supabase.co", "k", http=failing_http(502))
    with pytest.raises(store.StoreError, match="^HTTP 502$"):
        client.select(store.REQUESTS)


def test_a_long_message_is_cut_and_other_failures_are_named_by_their_kind_only():
    client = store.SupabaseClient("https://x.supabase.co", "k", http=failing_http(400, {"message": "x" * 1000}))
    with pytest.raises(store.StoreError) as caught:
        client.update(store.REQUESTS, 1, {"thumb": 1})
    assert len(str(caught.value)) < 330
    assert store.describe(ConnectionError("secret host details")) == "ConnectionError"
    assert store.describe(store.StoreError("HTTP 401: nope")) == "HTTP 401: nope"


def test_a_log_that_cannot_be_written_gives_none_and_the_warning_says_why(caplog):
    store.set_client(store.SupabaseClient("https://x.supabase.co", "k", http=failing_http(403, DENIED)))
    result = Answer("What was revenue?", "It was 1.", [], True, [], format_ok=True, role="guest")
    with caplog.at_level(logging.WARNING, logger="monitor.store"):
        assert store.log_request(result) is None
    assert "the log is not reachable (HTTP 403: permission denied for table requests" in caplog.text


def test_the_write_check_saves_a_test_record_that_nothing_counts():
    assert store.write_check() == ""
    assert [item["source"] for item in store.get_client().select(store.REQUESTS)] == ["check"]
    assert store.requests_between() == [] and store.spend_cents() == 0


def test_the_write_check_gives_the_reason_or_says_nothing_came_back():
    assert store.write_check(Client("HTTP 403: permission denied for table requests")) == "HTTP 403: permission denied for table requests"
    assert "nothing came back" in store.write_check(Client(saved=None))
    assert store.write_check(Client(saved={"id": 5})) == ""


def test_the_check_command_fails_loudly_when_the_log_can_be_read_but_not_written(capsys):
    store.set_client(Client("HTTP 403: permission denied for table requests"))
    assert store.main(["--check"]) == 1
    out = capsys.readouterr().out
    assert "Write test FAILED (HTTP 403: permission denied for table requests)" in out and "not saved until this is fixed" in out
    assert "The next question would be allowed." in out  # the demo itself keeps answering


def test_the_check_command_says_the_log_can_be_written(capsys):
    store.set_client(Client(saved={"id": 1}))
    assert store.main(["--check"]) == 0
    assert "Write test: one test record was saved" in capsys.readouterr().out


def test_the_check_command_says_what_supabase_said_when_the_log_cannot_be_read(capsys):
    class Unreadable(Client):
        def select(self, *args, **kwargs):
            raise store.StoreError("HTTP 401: Invalid API key")

    store.set_client(Unreadable())
    assert store.main(["--check"]) == 1
    assert "The store cannot be reached (HTTP 401: Invalid API key)" in capsys.readouterr().out


def test_the_price_of_a_plain_question_is_not_below_what_the_practice_questions_cost():
    measured = [0.0423, 0.0404, 0.1312, 0.1537, 0.0471, 0.1256, 0.0488, 0.1311, 0.0508, 0.0543]  # the 10 practice questions, plain mode
    assert flow.CENTS["plain"] >= sum(measured) / len(measured)
    assert flow.CENTS["agent"] > flow.CENTS["plain"]
