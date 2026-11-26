"""Step 4.2: LangSmith traces. One trace per request: the search, the Claude calls, the tool steps and the checks.

    with tracing.trace(role="guest", mode="plain", question=q) as span:
        result = guarded_answer(q)
        span.finish(result)

The public app sends 1 trace in 5 (TRACE_ONE_IN in config.py), so it stays far under the free 5,000 traces a month.
Eval runs send every one: wrap the answer function with tracing.traced(answer_fn, tag="base").
Tags: role, mode, source (app or eval) and run tag. Nothing here ever stops an answer: with no LANGSMITH_API_KEY, no
langsmith library, or any error on the way, trace() does nothing and the answer goes on.
"""
import contextlib
import importlib.util
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

log = logging.getLogger(__name__)
PROJECT = "filings-analyst"


def enabled():
    if not config.LANGSMITH_API_KEY:
        return False
    try:
        return importlib.util.find_spec("langsmith") is not None
    except (ImportError, ValueError):
        return False


def wanted(source="app", sampler=random.random):
    """Eval runs trace everything. The public app traces 1 request in TRACE_ONE_IN."""
    if not enabled():
        return False
    return source != "app" or sampler() < 1 / max(int(config.TRACE_ONE_IN), 1)


class Span:
    """What trace() gives you. finish(result) writes the outcome on the trace; without a trace it does nothing."""

    def __init__(self, run=None):
        self.run = run

    def finish(self, result):
        if self.run is None:
            return
        try:
            self.run.add_outputs({
                "answer": result.answer, "citations": result.citations, "found": result.found, "blocked": result.blocked,
                "cost_cents": round(result.cost_cents, 5), "seconds": round(result.seconds, 3), "steps": len([e for e in result.events if e.get("stage") == "agent"]),
                "pieces": [f"{hit.doc} p{hit.page}" for hit in result.hits],
            })
        except Exception as err:  # noqa: BLE001
            log.warning("a trace could not be completed (%s)", type(err).__name__)


@contextlib.contextmanager
def trace(role="guest", mode="plain", question="", source="app", tag="", sampler=random.random):
    """Open a trace when this request is picked. Always yields a Span. An error inside the block still reaches the caller."""
    with contextlib.ExitStack() as stack:
        run = None
        if wanted(source, sampler):
            try:
                import langsmith

                stack.enter_context(langsmith.tracing_context(enabled=True))
                tags = [f"role:{role}", f"mode:{mode}", f"source:{source}"] + ([f"run:{tag}"] if tag else [])
                run = stack.enter_context(langsmith.trace(
                    "ask", run_type="chain", inputs={"question": question, "role": role, "mode": mode},
                    tags=tags, metadata={"role": role, "mode": mode, "source": source, "run": tag}, project_name=PROJECT,
                ))
            except Exception as err:  # noqa: BLE001 - no trace is better than no answer
                log.warning("tracing is off for this request (%s)", type(err).__name__)
                run = None
        yield Span(run)


def traced(answer_fn, tag="", source="eval"):
    """Wrap an answer function (question -> Answer) so each call is traced. For eval runs."""
    def wrapper(question, *args, **kwargs):
        with trace(question=question, source=source, tag=tag, role=getattr(config, "EVAL_ROLE", "guest"), mode="plain") as span:
            result = answer_fn(question, *args, **kwargs)
            span.finish(result)
            return result

    return wrapper


def flush():
    """Send what is still waiting. Call it at the end of a short command; the web app does not need it."""
    if not enabled():
        return
    try:
        import langsmith

        langsmith.Client().flush()
    except Exception as err:  # noqa: BLE001
        log.warning("the traces could not be flushed (%s)", type(err).__name__)
