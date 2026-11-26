"""Step 4.3: daily numbers, alerts and the weekly review. Free: no Claude call.

    python -m monitor.metrics                      today's numbers and the alerts that fire
    python -m monitor.metrics --week               pick the weekly sample (10 random requests plus every thumbs down)
                                                   and write it to data\\logs\\weekly_review.md
    python -m monitor.metrics --review 12 "search missed" "the right page was in FY2021"
                                                   write your failure type and note on request 12
    python -m monitor.metrics --to-case 12         add request 12 as a candidate question in data\\cases\\new_failures.jsonl

The alerts only print (here and on the dashboard). They send no email. The rules are starting guesses: change them
after you have seen a real week.
"""
import argparse
import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from monitor import store  # noqa: E402

WINDOW = 50  # "the last 50 answers"
MIN_SAMPLE = 10  # fewer answers than this are too few to raise an alarm about
ALERT_SPEND = 80  # out of 100 of the daily limit
ALERT_NO_CITATION = 10  # out of 100 of the last 50 answers
ALERT_SLOW_SECONDS = 20
ALERT_THUMBS_DOWN = 20  # out of 100 of the last 50 rated answers
BLOCK_FACTOR, BLOCK_MIN = 2, 5  # blocked today more than double the usual, and at least BLOCK_MIN
USUAL_DAYS = 7
WEEK_SAMPLE = 10
RULES = (
    f"Spend today is above {ALERT_SPEND} out of 100 of the daily limit.",
    f"More than {ALERT_NO_CITATION} out of 100 of the last {WINDOW} answers have no citation.",
    f"The slowest 1 in 20 answers takes more than {ALERT_SLOW_SECONDS} seconds.",
    f"Thumbs down are more than {ALERT_THUMBS_DOWN} out of 100 of the last {WINDOW} rated answers.",
    f"Blocked messages today are more than double the usual count (and at least {BLOCK_MIN}).",
)


def when(record):
    return store.parse_time(record["created_at"])


def out_of_100(part, whole):
    return round(100 * part / whole) if whole else 0


def slowest_one_in_20(values):
    """The time that 19 answers out of 20 beat (the 95th place). 0.0 when there is nothing."""
    ordered = sorted(values)
    return ordered[max(math.ceil(0.95 * len(ordered)) - 1, 0)] if ordered else 0.0


def daily_numbers(records, now=None):
    """The numbers of one day (UTC) and of the last WINDOW answers, from a list of request records."""
    now = now or store.now_utc()
    start = store.day_start(now)
    ordered = sorted(records, key=when, reverse=True)
    today = [item for item in ordered if when(item) >= start]
    answers = [item for item in ordered if not item.get("blocked")][:WINDOW]
    rated = [item for item in answers if item.get("thumb") in (1, -1)]
    past = [item for item in ordered if start - timedelta(days=USUAL_DAYS) <= when(item) < start]
    past_days = {when(item).date() for item in past}
    blocked_today = sum(bool(item.get("blocked")) for item in today)
    return {
        "date": start.date().isoformat(),
        "requests_today": len(today),
        "blocked_today": blocked_today,
        "block_rate": out_of_100(blocked_today, len(today)),
        "usual_blocked": sum(bool(item.get("blocked")) for item in past) / max(len(past_days), 1),
        "spend_today_cents": sum(float(item.get("cost_cents") or 0.0) for item in today),
        "answers": len(answers),
        "no_citation": out_of_100(sum(bool(item.get("no_citation")) for item in answers), len(answers)),
        "slowest_seconds": slowest_one_in_20([float(item.get("seconds") or 0.0) for item in answers]),
        "average_seconds": sum(float(item.get("seconds") or 0.0) for item in answers) / max(len(answers), 1),
        "average_cents": sum(float(item.get("cost_cents") or 0.0) for item in answers) / max(len(answers), 1),
        "rated": len(rated),
        "thumbs_down": out_of_100(sum(item.get("thumb") == -1 for item in rated), len(rated)),
    }


@dataclass
class Alert:
    name: str
    text: str


def alerts(numbers, daily_limit=None):
    """The alert rules that fire for these numbers, as a list of Alert. An empty list means all quiet."""
    limit = config.DEMO_DAILY_CENTS if daily_limit is None else daily_limit
    found = []
    spent = numbers["spend_today_cents"]
    if limit and 100 * spent / limit > ALERT_SPEND:
        found.append(Alert("spend", f"Spend today is {round(100 * spent / limit)} out of 100 of the daily limit ({spent:.3f} of {limit} cents)."))
    if numbers["answers"] >= MIN_SAMPLE and numbers["no_citation"] > ALERT_NO_CITATION:
        found.append(Alert("no citation", f"{numbers['no_citation']} out of 100 of the last {numbers['answers']} answers have no citation (alert above {ALERT_NO_CITATION})."))
    if numbers["answers"] >= MIN_SAMPLE and numbers["slowest_seconds"] > ALERT_SLOW_SECONDS:
        found.append(Alert("slow", f"The slowest 1 in 20 answers takes {numbers['slowest_seconds']:.1f} seconds (alert above {ALERT_SLOW_SECONDS})."))
    if numbers["rated"] >= MIN_SAMPLE and numbers["thumbs_down"] > ALERT_THUMBS_DOWN:
        found.append(Alert("thumbs down", f"{numbers['thumbs_down']} out of 100 of the last {numbers['rated']} rated answers got a thumbs down (alert above {ALERT_THUMBS_DOWN})."))
    blocked, usual = numbers["blocked_today"], numbers["usual_blocked"]
    if blocked >= BLOCK_MIN and blocked > BLOCK_FACTOR * usual:
        found.append(Alert("blocked", f"{blocked} messages were blocked today; the usual is {usual:.1f} a day."))
    return found


def number_lines(numbers):
    return [
        f"TODAY (UTC, {numbers['date']})",
        f"  requests {numbers['requests_today']:4d}   blocked {numbers['blocked_today']} ({numbers['block_rate']} out of 100)   usual blocked a day {numbers['usual_blocked']:.1f}",
        f"  spend {numbers['spend_today_cents']:.3f} of {config.DEMO_DAILY_CENTS} cents",
        f"LAST {WINDOW} ANSWERS ({numbers['answers']} answered)",
        f"  no citation {numbers['no_citation']} out of 100   slowest 1 in 20 takes {numbers['slowest_seconds']:.1f} s   average {numbers['average_seconds']:.1f} s and {numbers['average_cents']:.4f} cents",
        f"  rated {numbers['rated']}   thumbs down {numbers['thumbs_down']} out of 100 of the rated",
    ]


# ---------- the weekly review ----------

def weekly_sample(records, now=None, count=WEEK_SAMPLE):
    """Every thumbs down of the last 7 days plus `count` random requests. The same week gives the same sample."""
    now = now or store.now_utc()
    week = [item for item in records if when(item) >= now - timedelta(days=7)]
    down = [item for item in week if item.get("thumb") == -1]
    rest = [item for item in week if item.get("thumb") != -1]
    year, number, _ = now.isocalendar()
    picked = random.Random(f"{year}-{number}").sample(rest, min(count, len(rest)))
    sample = [dict(item, why="thumbs down") for item in down] + [dict(item, why="random") for item in picked]
    return sorted(sample, key=when)


def review_lines(sample):
    lines = [f"# Weekly review: {len(sample)} requests", "",
             "Read each one. Write a failure type with:  python -m monitor.metrics --review ID \"type\" \"note\"",
             "Turn a real failure into a candidate test question with:  python -m monitor.metrics --to-case ID", ""]
    for item in sample:
        cited = ", ".join(f"{c.get('doc')} p{c.get('page')}" for c in item.get("citations") or []) or "nothing"
        flags = [name for name in ("blocked", "no_citation", "pii_hidden", "injection") if item.get(name)]
        lines += [
            f"## Request {item.get('id')}: {item['created_at'][:16]}, {item.get('role')}, {item.get('mode')} ({item['why']})",
            f"Question: {item.get('question')}",
            f"Answer: {str(item.get('answer'))[:700]}",
            f"Cited: {cited}. Flags: {', '.join(flags) or 'none'}. {float(item.get('seconds') or 0):.1f} s, {float(item.get('cost_cents') or 0):.4f} cents.",
            f"Your failure type: {item.get('failure_type') or '(not written yet)'}", "",
        ]
    return lines


def candidate_case(record):
    """A request as a candidate test question. The reference answer and the evidence page are for you to fill in."""
    return {
        "id": f"request_{record.get('id')}", "question": record.get("question"), "role": record.get("role"),
        "seen": record.get("created_at"), "bot_answer": record.get("answer"),
        "failure_type": record.get("failure_type") or "", "note": record.get("review_note") or "",
        "status": "needs a reference answer and the evidence page from the filing",
    }


def add_case(record, path=None):
    """Append the candidate to data\\cases\\new_failures.jsonl. Returns (path, True), or (path, False) when it is already there."""
    path = Path(path or Path(config.DATA_DIR) / "cases" / "new_failures.jsonl")
    case = candidate_case(record)
    if path.exists() and any(f'"id": "{case["id"]}"' in line for line in path.read_text(encoding="utf-8").splitlines()):
        return path, False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    return path, True


# ---------- the command ----------

def main(argv=None, client=None, now=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Daily numbers, alerts and the weekly review (step 4.3).")
    parser.add_argument("--week", action="store_true", help="pick the weekly review sample and write it to data\\logs\\weekly_review.md")
    parser.add_argument("--review", nargs="+", metavar=("ID", "TYPE"), help="ID, then the failure type, then an optional note")
    parser.add_argument("--to-case", type=int, metavar="ID", help="add one request as a candidate test question")
    args = parser.parse_args(argv)
    print("COLAB: no | MODEL: unchanged | CLAUDE COST: 0 cents")
    now = now or store.now_utc()

    if args.review:
        if len(args.review) < 2 or not args.review[0].isdigit():
            sys.exit('Use:  --review 12 "search missed" "an optional note"')
        saved = store.set_review(int(args.review[0]), args.review[1], " ".join(args.review[2:]), client)
        print("Saved." if saved else "Could not save it (see the warning above).")
        return 0 if saved else 1
    if args.to_case is not None:
        record = store.get_request(args.to_case, client)
        if not record:
            sys.exit(f"Request {args.to_case} was not found in {store.store_name(client)}.")
        path, added = add_case(record)
        print(f"{'Added to' if added else 'Already in'} {path}. Fill in the reference answer and the evidence page. Read it before you commit: it holds a visitor's words.")
        return 0

    if args.week:
        sample = weekly_sample(store.requests_between(now - timedelta(days=7), client=client), now)
        path = Path(config.DATA_DIR) / "logs" / "weekly_review.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(review_lines(sample)) + "\n", encoding="utf-8")
        print(f"{len(sample)} requests picked ({sum(item['why'] == 'thumbs down' for item in sample)} with a thumbs down). Read them in {path}")
        return 0

    records = store.requests_between(now - timedelta(days=USUAL_DAYS + 1), client=client)
    numbers = daily_numbers(records, now)
    print(f"Store: {store.store_name(client)}\n")
    print("\n".join(number_lines(numbers)))
    fired = alerts(numbers)
    print("\nALERTS" if fired else "\nALERTS: none fire.")
    for alert in fired:
        print(f"  ALERT {alert.name}: {alert.text}")
    print("\nThe rules:\n" + "\n".join(f"  {rule}" for rule in RULES))
    return 1 if fired else 0


if __name__ == "__main__":
    sys.exit(main())
