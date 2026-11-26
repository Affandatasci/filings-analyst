"""One question through the whole monitored path: limits, guards, trace, log. The app and the command line both use it.

    python -m monitor.flow "What drove the change in operating margin for 3M in FY2022?" --role analyst --mode plain
    python -m monitor.flow --practice 10        the 10 practice questions, logged like visitors (about 0.85 cent, plain mode)

COSTS CLAUDE CENTS: about 0.085 cent a question in plain mode (measured on the 10 practice questions), about 0.3 in agent mode
(a guess: not measured yet). The demo limits apply here too
(config.py: DEMO_DAILY_CENTS and DEMO_TOTAL_CENTS), and so does the spend fuse.
"""
import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from monitor import store, tracing  # noqa: E402

log = logging.getLogger(__name__)
CENTS = {"plain": 0.085, "agent": 0.3}  # a rough price of one question (plain: measured on the 10 practice questions)


@dataclass
class Outcome:
    result: object  # the Answer, or None when a limit said no
    limit: store.Limit
    request_id: object = None  # the id of the log record, or None when it could not be saved


def failed_answer(question, role, err):
    """An Answer that stands in for a crash, so the visitor sees one plain sentence and the log still gets a record."""
    from bot import llm
    from bot.answer import Answer

    log.warning("a question failed (%s)", type(err).__name__)
    text = "The project's Claude budget is used up, so the demo is paused." if isinstance(err, llm.BudgetStop) else "Something went wrong on our side. Please try again later."
    return Answer(question, text, [], False, [], format_ok=True, skipped_reason="error", role=role, blocked="error:" + type(err).__name__)


def ask(question, role="guest", mode="plain", session="", asked=0, ask_fn=None, client=None, now=None, tag="", source="app"):
    """Check the limits, answer through the guards (traced), log the result. Never raises. asked = questions in this visit so far."""
    limit = store.check_limits(asked, client=client, now=now)
    if not limit.ok:
        return Outcome(None, limit)
    try:
        if ask_fn is None:
            from security import guards

            ask_fn = guards.guarded_answer
        with tracing.trace(role=role, mode=mode, question=question, source=source, tag=tag) as span:
            result = ask_fn(question, role=role, mode=mode)
            span.finish(result)
    except Exception as err:  # noqa: BLE001 - a crash becomes one plain sentence and a log record
        result = failed_answer(question, role, err)
    return Outcome(result, limit, store.log_request(result, mode=mode, session=session, source=source, client=client))


def main(argv=None, ask_fn=None, client=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Ask through the limits, the guards and the log (step 4.1).")
    parser.add_argument("question", nargs="?", help="one question")
    parser.add_argument("--practice", type=int, metavar="N", help="the first N practice questions instead of one question")
    parser.add_argument("--role", default="guest", help="guest (default), analyst or hr")
    parser.add_argument("--mode", default="plain", choices=("plain", "agent"))
    args = parser.parse_args(argv)
    if not args.question and not args.practice:
        sys.exit('Give a question, or --practice 10.   Example:  python -m monitor.flow "What was 3M revenue in FY2018?"')
    if args.practice:
        from evals.cases import load_cases

        questions = [case.question for case in load_cases("practice")][: max(args.practice, 1)]
    else:
        questions = [args.question]
    print(f"COLAB: no | MODEL: unchanged | CLAUDE COST: about {len(questions) * CENTS[args.mode]:.2f} cents ({len(questions)} question{'s' if len(questions) != 1 else ''}, {args.mode} mode)")
    print(f"Log: {store.store_name(client)}\n")
    saved = 0
    for number, question in enumerate(questions, 1):
        outcome = ask(question, args.role, args.mode, session="cli", ask_fn=ask_fn, client=client)
        if outcome.result is None:
            print(f"{number}. STOPPED by a limit: {outcome.limit.text}")
            break
        saved += outcome.request_id is not None
        result = outcome.result
        print(f"{number}. {result.answer[:110]!r}  ({result.cost_cents:.4f} cents, {result.seconds:.1f} s, "
              f"{'blocked: ' + result.blocked if result.blocked else 'logged as ' + str(outcome.request_id)})")
    tracing.flush()
    print(f"\n{saved} of {len(questions)} questions were saved in the log.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
