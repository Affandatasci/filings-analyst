"""Step 4.5: the second page. Scores against the pass lines, scores by slice, the attack table, live numbers, recent failures.

It reads results\\ (the two score files and the attack table that you commit) and the request log. It makes no Claude call.
    streamlit run app\\dashboard.py          alone, or as the second page of app\\streamlit_app.py

What visitors typed is listed only when SHOW_QUESTIONS is set to 1 (your laptop). The public page shows counts and reasons.
"""
import json
import sys
from datetime import timedelta
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from evals import report  # noqa: E402
from monitor import metrics, store  # noqa: E402
from scripts import check_scores  # noqa: E402

SHOWN_FAILURES = 20


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def score_table(latest, baseline, lines):
    """The score table as a list of dicts, with the same verdicts as the push check. Plain words for each score."""
    words = dict(report.METRICS)
    _, table = check_scores.check(latest, baseline, lines)
    verdict = {"ok": "ok", "UNDER THE LINE": "under the pass line", "DROPPED": "dropped more than 5 points"}
    return [{"Score": words.get(name, name), "Baseline": base, "Latest": now, "Pass line": line, "Result": verdict.get(result, result)} for name, base, now, line, result in table]


def slice_table(latest, field):
    groups = (latest.get("slices") or {}).get(field, {})
    table = [{"Group": name, "Questions": item["n"], "Right (out of 100)": item["right"], "Page found (out of 100)": item["found_page"],
             "Note": "few questions, rough" if item["n"] < report.MIN_SLICE else ""} for name, item in groups.items()]
    return sorted(table, key=lambda entry: (-entry["Questions"], entry["Group"]))


def newest_attack_table(folder):
    files = sorted(Path(folder).glob("attack_table_*.txt"), key=lambda path: path.stat().st_mtime)
    return (files[-1].name, files[-1].read_text(encoding="utf-8")) if files else None


def failure_table(records, show_questions=False, limit=SHOWN_FAILURES):
    """Blocked, uncited and thumbs-down requests, newest first. The question text only when show_questions is True."""
    table = []
    for item in sorted(records, key=metrics.when, reverse=True):
        reasons = [name for name, flag in (("blocked: " + str(item.get("blocked")), item.get("blocked")), ("no citation", item.get("no_citation")), ("thumbs down", item.get("thumb") == -1)) if flag]
        if reasons:
            table.append({"Time": item["created_at"][:16].replace("T", " "), "Role": item.get("role"), "Mode": item.get("mode"), "Why": ", ".join(reasons),
                         "Question": str(item.get("question"))[:120] if show_questions else "(hidden on the public page)"})
    return table[:limit]


def page():
    results = Path(config.RESULTS_DIR)
    st.title("Scores and monitor")
    st.caption("Every number comes from a saved file or from the request log. With 75 questions a score can move by about 11 points out of 100 for no reason at all.")

    st.header("Scores against the pass lines")
    baseline, latest = read_json(results / "baseline_scores.json"), read_json(results / "latest_scores.json")
    if not baseline:
        st.info("There is no baseline yet. It is saved in step 2.8 with: python scripts\\check_scores.py --baseline results\\runs\\base_test.json")
    else:
        latest = latest or baseline
        if not config.PASS_LINES:
            st.warning("The pass lines are not set yet (step 2.9), so only the drop rule is shown.")
        st.dataframe(score_table(latest, baseline, dict(config.PASS_LINES)), hide_index=True)
        problems, _ = check_scores.check(latest, baseline, dict(config.PASS_LINES))
        st.caption(f"Baseline: run {baseline.get('tag')} ({baseline.get('questions')} questions). Latest: run {latest.get('tag')} on the {latest.get('split')} questions ({latest.get('questions')} questions).")
        for problem in problems:
            st.error(problem)
        st.header("Scores by slice")
        fields = [(field, label) for field, label in report.SLICES if (latest.get("slices") or {}).get(field)]
        if fields:
            field = st.selectbox("Cut by", [name for name, _ in fields], format_func=dict(fields).get)
            st.dataframe(slice_table(latest, field), hide_index=True)

    st.header("The attack table")
    table = newest_attack_table(results)
    if table:
        st.code(table[1], language=None)
        st.caption(f"File: {table[0]}. The 40 attack messages are mine. They find holes; they do not prove safety.")
    else:
        st.info("No attack table yet. Run the attacks in step 3.7 and commit results\\attack_table_<tag>.txt.")
    blocks = read_json(results / "false_blocks.json")
    if blocks:
        st.write(f"Real questions blocked by mistake: {blocks.get('blocked')} of {blocks.get('questions')} ({blocks.get('per_100')} out of every 100), run {blocks.get('run')}.")

    st.header("Live numbers")
    now = store.now_utc()
    records = store.requests_between(now - timedelta(days=metrics.USUAL_DAYS + 1))
    if not records:
        st.info(f"No requests in the log yet ({store.store_name()}).")
    else:
        numbers = metrics.daily_numbers(records, now)
        columns = st.columns(5)
        columns[0].metric("Requests today", numbers["requests_today"])
        columns[1].metric("Spend today (cents)", f"{numbers['spend_today_cents']:.3f}")
        columns[2].metric("No citation, out of 100", numbers["no_citation"])
        columns[3].metric("Slowest 1 in 20 (seconds)", f"{numbers['slowest_seconds']:.1f}")
        columns[4].metric("Thumbs down, out of 100", numbers["thumbs_down"])
        fired = metrics.alerts(numbers)
        for alert in fired:
            st.warning(alert.text)
        if not fired:
            st.success("No alert fires.")
        with st.expander("The alert rules"):
            st.write("\n".join(f"- {rule}" for rule in metrics.RULES))
        st.subheader("Recent failures")
        failures = failure_table(records, config.SHOW_QUESTIONS)
        if failures:
            st.dataframe(failures, hide_index=True)
        else:
            st.write("No blocked, uncited or thumbs-down request in the last 8 days.")

    runs = store.recent_runs(10)
    if runs:
        st.subheader("History of eval runs")
        st.dataframe([{"Date": item["created_at"][:10], "Run": item.get("tag"), "Split": item.get("split"), "Questions": item.get("questions"),
                       "Partial": bool(item.get("partial")), "Right (out of 100)": (item.get("scores", {}).get("correct") or item.get("scores", {}).get("number_match") or {}).get("points"),
                       "Cost (cents)": item.get("cost_cents")} for item in runs], hide_index=True)


if __name__ == "__main__":  # a Streamlit page runs as __main__; an import (the tests) does not start the page
    page()
