"""Step 4.1: the request log and the demo limits.

    python -m monitor.store --check                           which store is used, can it be reached, what the demo has spent
    python -m monitor.store --push results\\runs\\base_test.json   save one eval run in the eval_runs table

Where the records go: Supabase when SUPABASE_URL and SUPABASE_KEY are set, otherwise two local files in data\\logs
(kept out of Git), so the app and the dashboard also work on the laptop with no keys. Plain HTTPS with httpx: no extra install.

A logging failure never crashes the app: log_request, log_run, set_thumb and the read helpers catch their own errors and
give back a safe value (None, False or an empty list). When the log cannot be read, the demo limits fall back to this
server's own count of what it has spent since it started (TALLY), so the demo keeps answering and still stays inside its limits.
"""
import argparse
import json
import logging
import re
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

log = logging.getLogger(__name__)

REQUESTS, RUNS = "requests", "eval_runs"
TIMEOUT = 6.0  # seconds to wait for Supabase
PAGE = 1000  # Supabase gives at most this many records per call
KEEP_ANSWER = 4000  # characters of an answer kept in the log
KEEP_STEP_TEXT = 300  # characters of any text inside one agent step
KEEP_STEPS = 12
OPS = {"eq", "neq", "gt", "gte", "lt", "lte"}
TALLY = []  # (time, cents) of every public answer this server has paid for: the fallback when the log cannot be read


# ---------- time ----------

def now_utc():
    return datetime.now(timezone.utc)


def iso(moment):
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse_time(text):
    """A time from Supabase or from a local file as an aware UTC datetime. Python 3.10 needs exactly 6 digits after the dot."""
    text = str(text).strip().replace("Z", "+00:00")
    text = re.sub(r"\.(\d+)", lambda found: "." + found.group(1)[:6].ljust(6, "0"), text, count=1)
    moment = datetime.fromisoformat(text)
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def day_start(now=None):
    return (now or now_utc()).astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


class StoreError(Exception):
    """Supabase answered with an error. The text is the HTTP status and what Supabase said (never our key)."""


def describe(err):
    """A short plain phrase for a failed call: what Supabase said, or only the kind of failure when it did not answer."""
    return str(err) if isinstance(err, StoreError) else type(err).__name__


def _reason(reply):
    """The HTTP status of a failed reply and the message Supabase sent with it."""
    try:
        body = reply.json()
        said = " ".join(str(body[name]) for name in ("message", "hint") if body.get(name))[:300]
    except Exception:  # noqa: BLE001 - the reply was not JSON
        said = ""
    return f"HTTP {getattr(reply, 'status_code', '?')}" + (f": {said}" if said else "")


def write_check(client=None):
    """Save one test record (source check: nothing counts it) to see that the log can be written. Gives an empty text when it can, otherwise the reason."""
    try:
        saved = (client or get_client()).insert(REQUESTS, {"source": "check", "question": "write test"})
    except Exception as err:  # noqa: BLE001
        return describe(err)
    return "" if (saved or {}).get("id") is not None else "the test record was sent but nothing came back"


# ---------- the two stores: the same three calls ----------

class SupabaseClient:
    """Supabase over HTTPS (PostgREST). filters is a list of (column, operator, value), for example ("cost_cents", "gt", 0)."""

    def __init__(self, url, key, http=None, timeout=TIMEOUT):
        self.base = url.rstrip("/") + "/rest/v1/"
        self.headers = {"apikey": key, "Authorization": "Bearer " + key, "Content-Type": "application/json"}
        self.http, self.timeout = http, timeout

    def _call(self, method, table, headers=None, **kwargs):
        if self.http is None:
            import httpx

            self.http = httpx
        reply = self.http.request(method, self.base + table, headers={**self.headers, **(headers or {})}, timeout=self.timeout, **kwargs)
        try:
            reply.raise_for_status()
        except Exception as err:  # noqa: BLE001
            raise StoreError(_reason(reply)) from err
        return reply

    def insert(self, table, record):
        data = self._call("POST", table, headers={"Prefer": "return=representation"}, json=record).json()
        return data[0] if isinstance(data, list) and data else None

    def select(self, table, filters=None, order=None, limit=None, columns="*"):
        params = [("select", columns)] + [(column, f"{op}.{value}") for column, op, value in (filters or [])]
        if order:
            params.append(("order", order))
        if limit:
            params.append(("limit", str(limit)))
        return self._call("GET", table, params=params).json()

    def update(self, table, record_id, values):
        self._call("PATCH", table, params={"id": f"eq.{record_id}"}, json=values)
        return True


class FileClient:
    """The same calls on local files, one JSON-lines file per table. For the laptop and for tests."""

    def __init__(self, folder):
        self.folder, self.lock = Path(folder), threading.Lock()

    def _path(self, table):
        return self.folder / f"{table}.jsonl"

    def _read(self, table):
        path = self._path(table)
        if not path.exists():
            return []
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                continue  # a half written line is skipped, not fatal
        return records

    def _write(self, table, records):
        self.folder.mkdir(parents=True, exist_ok=True)
        text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
        self._path(table).write_text(text, encoding="utf-8")

    def insert(self, table, record):
        with self.lock:
            records = self._read(table)
            saved = {"id": max([item.get("id", 0) for item in records], default=0) + 1, "created_at": now_utc().isoformat(timespec="microseconds"), **record}
            self.folder.mkdir(parents=True, exist_ok=True)
            with self._path(table).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(saved, ensure_ascii=False) + "\n")
        return saved

    def select(self, table, filters=None, order=None, limit=None, columns="*"):
        records = self._read(table)
        for column, op, value in filters or []:
            records = [item for item in records if _matches(item.get(column), op, value, column)]
        if order:
            name, _, direction = order.partition(".")
            records.sort(key=lambda item: (_sortable(item.get(name), name), item.get("id", 0)), reverse=(direction == "desc"))
        if limit:
            records = records[: int(limit)]
        if columns != "*":
            keep = columns.split(",")
            records = [{name: item.get(name) for name in keep} for item in records]
        return records

    def update(self, table, record_id, values):
        with self.lock:
            records, changed = self._read(table), False
            for item in records:
                if item.get("id") == record_id:
                    item.update(values)
                    changed = True
            if changed:
                self._write(table, records)
        return changed


def _sortable(value, column):
    if value is None:
        return 0
    return parse_time(value).timestamp() if column == "created_at" else value


def _matches(have, op, want, column):
    if op not in OPS:
        raise ValueError(f"unknown filter operator {op!r}")
    if have is None:
        return False
    if column == "created_at":
        have, want = parse_time(have), parse_time(want)
    elif isinstance(have, (int, float)) and not isinstance(have, bool):
        have, want = float(have), float(want)
    else:
        have, want = str(have), str(want)
    return {"eq": have == want, "neq": have != want, "gt": have > want, "gte": have >= want, "lt": have < want, "lte": have <= want}[op]


# ---------- which store ----------

_client = None


def get_client():
    """Supabase when both keys are set, otherwise local files in data\\logs."""
    global _client
    if _client is None:
        if config.SUPABASE_URL and config.SUPABASE_KEY:
            _client = SupabaseClient(config.SUPABASE_URL, config.SUPABASE_KEY)
        else:
            _client = FileClient(Path(config.DATA_DIR) / "logs")
    return _client


def set_client(client):
    """Tests use this. None means: choose again from the keys."""
    global _client
    _client = client


def store_name(client=None):
    return "Supabase" if isinstance(client or get_client(), SupabaseClient) else "local files (data\\logs)"


# ---------- what is logged ----------

def _clip(text, size):
    text = "" if text is None else str(text)
    return text if len(text) <= size else text[:size] + " ..."


def _trim(value):
    if isinstance(value, str):
        return _clip(value, KEEP_STEP_TEXT)
    if isinstance(value, dict):
        return {str(key): _trim(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_trim(item) for item in value[:KEEP_STEPS]]
    return value


def _hide_personal_data(text):
    """The question is logged with personal data hidden (the same hiding the pieces get). Never fails."""
    try:
        from security import pii

        return pii.mask_text(text, use_presidio=False)[0]
    except Exception:  # noqa: BLE001 - a logging helper must not stop the answer
        return text


def request_record(result, mode="plain", session="", source="app"):
    """One log record from a finished Answer. The question is the one that went through the input guard, the answer the
    one that came out of the output guard. The raw reply from Claude is never kept."""
    events = result.events or []
    answered = not result.blocked
    return {
        "source": source,
        "session": session or "",
        "role": result.role or "guest",
        "mode": mode,
        "question": _clip(_hide_personal_data(result.question), getattr(config, "MAX_QUESTION_CHARS", 500)),
        "answer": _clip(result.answer, KEEP_ANSWER),
        "citations": _trim(result.citations or []),
        "steps": _trim([event for event in events if event.get("stage") == "agent"]),
        "found": bool(result.found),
        "tokens_in": int(result.tokens_in or 0),
        "tokens_out": int(result.tokens_out or 0),
        "cost_cents": round(float(result.cost_cents or 0.0), 5),
        "seconds": round(float(result.seconds or 0.0), 3),
        "blocked": result.blocked or "",
        "pii_hidden": any(event.get("stage") == "pii" for event in events),
        "injection": any(event.get("stage") == "injection" for event in events),
        "no_citation": bool(answered and result.found and not result.citations),
    }


def run_record(run):
    """One eval_runs record from a saved run file (the dict that run_eval writes)."""
    totals = run.get("totals", {})
    return {
        "tag": str(run.get("tag", "?")),
        "split": run.get("split"),
        "partial": bool(run.get("partial")),
        "questions": int(totals.get("questions", len(run.get("cases", [])))),
        "settings": run.get("settings", {}),
        "scores": run.get("scores", {}),
        "cost_cents": round(float(totals.get("cost_cents", 0.0)), 5),
    }


def _safe(action, fallback, client=None):
    """Run action(client); on any failure give back fallback and say only what kind of failure it was."""
    try:
        return action(client or get_client())
    except Exception as err:  # noqa: BLE001 - logging never crashes the app
        log.warning("the log is not reachable (%s)", describe(err))
        return fallback


# ---------- writing ----------

def reset_tally():
    TALLY.clear()


def log_request(result, mode="plain", session="", source="app", client=None):
    """Save one request. Returns its id, or None when it could not be saved. Never raises."""
    if source == "app" and getattr(result, "cost_cents", 0):
        TALLY.append((now_utc(), float(result.cost_cents)))

    def save(store):
        saved = store.insert(REQUESTS, request_record(result, mode, session, source))
        return (saved or {}).get("id")

    return _safe(save, None, client)


def log_run(run, client=None):
    """Save the scores of one eval run. Returns its id, or None. Never raises."""
    return _safe(lambda store: (store.insert(RUNS, run_record(run)) or {}).get("id"), None, client)


def set_thumb(request_id, thumb, client=None):
    """1 for a good answer, -1 for a bad one. Returns True when it was saved."""
    if thumb not in (1, -1) or request_id is None:
        return False
    return bool(_safe(lambda store: store.update(REQUESTS, request_id, {"thumb": thumb}), False, client))


def set_review(request_id, failure_type, note="", client=None):
    """Your weekly review: a failure type and a note on one request. Returns True when it was saved."""
    values = {"failure_type": _clip(failure_type, 80), "review_note": _clip(note, 500)}
    return bool(_safe(lambda store: store.update(REQUESTS, request_id, values), False, client))


# ---------- reading ----------

def requests_between(start=None, end=None, source="app", limit=PAGE, client=None):
    """Request records from start up to end, newest first. An empty list when the log cannot be read."""
    filters = [("source", "eq", source)] if source else []
    if start:
        filters.append(("created_at", "gte", iso(start)))
    if end:
        filters.append(("created_at", "lt", iso(end)))
    return _safe(lambda store: store.select(REQUESTS, filters, order="created_at.desc", limit=limit), [], client)


def get_request(request_id, client=None):
    """One request record by id, or None."""
    found = _safe(lambda store: store.select(REQUESTS, [("id", "eq", request_id)], limit=1), [], client)
    return found[0] if found else None


def recent_runs(limit=20, client=None):
    return _safe(lambda store: store.select(RUNS, order="created_at.desc", limit=limit), [], client)


# ---------- the demo limits ----------

def spend_cents(since=None, client=None):
    """What the public demo has spent on Claude (cents) since a time, or in all. From the log; when the log cannot be read,
    from this server's own count."""
    filters = [("source", "eq", "app"), ("cost_cents", "gt", 0)]
    if since:
        filters.append(("created_at", "gte", iso(since)))
    try:
        records = (client or get_client()).select(REQUESTS, filters, columns="cost_cents", limit=PAGE)
    except Exception as err:  # noqa: BLE001
        log.warning("the spend could not be read from the log (%s): using this server's own count", describe(err))
        return sum(cents for moment, cents in TALLY if since is None or moment >= since)
    return sum(float(item.get("cost_cents") or 0.0) for item in records)


@dataclass
class Limit:
    ok: bool
    code: str = ""  # empty when ok, otherwise session, daily or total
    text: str = ""  # plain words for the visitor


def check_limits(session_questions=0, client=None, now=None):
    """Is one more question allowed? The visit first (free), then the two money limits read from the log."""
    if session_questions >= config.DEMO_SESSION_QUESTIONS:
        return Limit(False, "session", f"This visit has used its {config.DEMO_SESSION_QUESTIONS} questions. Please come back later.")
    today, total = spend_cents(day_start(now), client), spend_cents(None, client)
    reserve = config.DEMO_RESERVE_CENTS
    if total + reserve > config.DEMO_TOTAL_CENTS:
        return Limit(False, "total", "The demo has used all of the money set aside for it. It is closed.")
    if today + reserve > config.DEMO_DAILY_CENTS:
        return Limit(False, "daily", "The demo has used its money for today. It opens again tomorrow (UTC).")
    return Limit(True)


# ---------- the command ----------

def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Check the request log, or save one eval run in it (step 4.1).")
    parser.add_argument("--check", action="store_true", help="say which store is used and what the demo has spent")
    parser.add_argument("--push", metavar="RUN_FILE", help="save the scores of one saved run in the eval_runs table")
    args = parser.parse_args(argv)
    print("COLAB: no | MODEL: unchanged | CLAUDE COST: 0 cents")
    if args.push:
        from evals import report

        run = report.load_run(args.push)
        saved = log_run(run)
        print(f"Saved as eval run {saved} in {store_name()}." if saved is not None else f"Could not save it in {store_name()} (see the warning above).")
        return 0 if saved is not None else 1
    print(f"Store: {store_name()}")
    if isinstance(get_client(), FileClient):
        print("SUPABASE_URL and SUPABASE_KEY are not set, so records go to local files. Nothing is sent online.")
    try:
        get_client().select(REQUESTS, limit=1)
    except Exception as err:  # noqa: BLE001
        print(f"The store cannot be reached ({describe(err)}). The demo would keep answering and count its own spend until it is back.")
        return 1
    problem = write_check()
    print(f"Write test FAILED ({problem}). Questions are answered but not saved until this is fixed." if problem else "Write test: one test record was saved, so the log can be written.")
    today, total = spend_cents(day_start()), spend_cents()
    print(f"Demo spend today: {today:.3f} of {config.DEMO_DAILY_CENTS} cents. In all: {total:.3f} of {config.DEMO_TOTAL_CENTS} cents.")
    print(f"Questions in the last 24 hours: {len(requests_between(now_utc() - timedelta(days=1)))}")
    limit = check_limits(0)
    print("The next question would be allowed." if limit.ok else f"The next question would be refused: {limit.text}")
    return 1 if problem else 0


if __name__ == "__main__":
    sys.exit(main())
