"""Step 2.8: two saved runs side by side, on the questions they share. Free: no Claude call.

    python -m evals.compare_runs results\\runs\\base_test.json results\\runs\\top4_test.json
    python -m evals.compare_runs A.json B.json --metric found_page

With 75 questions one score can move by about 11 points out of 100 for no reason at all. So this does not only compare
the totals: it lists the questions that flipped, and it gives a range for the difference (B minus A) from 1,000
resamples of the questions. If the range includes 0, the two runs are not clearly different.

Metrics: right (default: the judge's correct, else the number match), correct, number_match, found_page, supported,
citation_valid, citation_hit, json_shape, length.
"""
import argparse
import random
import sys
from pathlib import Path

from evals import report

RESAMPLES = 1000
SEED = 2026  # the same two runs always give the same range
CHOICES = ("right",) + tuple(name for name, _ in report.METRICS)
SHOW = 12


def value_of(case, name):
    return report.is_right(case) if name == "right" else report.metric(case, name)


def paired(run_a, run_b, name):
    """(ids, a values, b values) for the questions that both runs measured. Values are 0 or 1."""
    a = {case["id"]: value_of(case, name) for case in run_a["cases"]}
    b = {case["id"]: value_of(case, name) for case in run_b["cases"]}
    ids = [case_id for case_id in a if case_id in b and a[case_id] is not None and b[case_id] is not None]
    return ids, [int(a[i]) for i in ids], [int(b[i]) for i in ids]


def bootstrap_difference(a, b, resamples=RESAMPLES, seed=SEED):
    """(difference B minus A, low, high) in points out of 100. The range holds 95 of 100 resamples."""
    n = len(a)
    if n == 0:
        return 0.0, 0.0, 0.0
    diffs = [y - x for x, y in zip(a, b)]
    point = 100.0 * sum(diffs) / n
    if n < 2:
        return point, point, point
    rng = random.Random(seed)
    totals = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) for _ in range(resamples))
    low, high = totals[int(0.025 * resamples)], totals[int(0.975 * resamples) - 1]
    return point, 100.0 * low / n, 100.0 * high / n


def flips(ids, a, b):
    """(ids that B gets right and A did not, ids that A got right and B does not)."""
    return [i for i, x, y in zip(ids, a, b) if y > x], [i for i, x, y in zip(ids, a, b) if x > y]


def short_id(case_id):
    return case_id.removeprefix("financebench_id_")


def listed(ids):
    shown = ", ".join(short_id(i) for i in ids[:SHOW])
    return shown + (f" and {len(ids) - SHOW} more" if len(ids) > SHOW else "") if ids else "none"


def describe(label, run, path):
    settings = run.get("settings", {})
    return (
        f"{label}: {Path(path).name} ({len(run['cases'])} questions, filter {settings.get('filter', '?')}, pieces {settings.get('k', '?')}, "
        f"collection {settings.get('collection', '?')}, prompt {settings.get('prompt', '?')}, judge {settings.get('judge', '?')})"
    )


def changed_settings(run_a, run_b):
    a, b = run_a.get("settings", {}), run_b.get("settings", {})
    return [key for key in sorted(set(a) | set(b)) if a.get(key) != b.get(key)]


def compare_lines(run_a, run_b, name, paths=("A", "B")):
    lines = [describe("A", run_a, paths[0]), describe("B", run_b, paths[1])]
    if run_a.get("split") != run_b.get("split"):
        lines.append(f"These are different splits ({run_a.get('split')} and {run_b.get('split')}): only the questions in both are compared.")
    changed = changed_settings(run_a, run_b)
    lines.append("Settings that differ: " + (", ".join(f"{key} ({run_a.get('settings', {}).get(key)} -> {run_b.get('settings', {}).get(key)})" for key in changed) or "none"))
    if len(changed) > 1:
        lines.append("More than one setting changed, so a difference cannot be blamed on one of them. Change one setting at a time.")
    ids, a, b = paired(run_a, run_b, name)
    lines.append("")
    if not ids:
        lines.append(f"No question was measured for '{name}' in both runs, so there is nothing to compare.")
        return lines
    low_note = " (rough: very few questions)" if len(ids) < 20 else ""
    lines.append(f"Metric: {name}. {len(ids)} questions are in both runs{low_note}.")
    lines.append(f"  A: {sum(a)} of {len(ids)} ok ({round(100 * sum(a) / len(ids))} out of 100)")
    lines.append(f"  B: {sum(b)} of {len(ids)} ok ({round(100 * sum(b) / len(ids))} out of 100)")
    point, low, high = bootstrap_difference(a, b)
    lines.append(f"  B minus A: {point:+.1f} points out of 100, range {low:+.1f} to {high:+.1f} ({RESAMPLES:,} resamples)")
    if low <= 0 <= high:
        lines.append("  The range includes 0: the two runs are not clearly different.")
    else:
        lines.append(f"  The range does not include 0: {'B' if point > 0 else 'A'} is better on these questions.")
    gained, lost = flips(ids, a, b)
    lines.append(f"Flipped: {len(gained) + len(lost)} questions")
    lines.append(f"  B fixed {len(gained)}: {listed(gained)}")
    lines.append(f"  B broke {len(lost)}: {listed(lost)}")
    for label, run in (("A", run_a), ("B", run_b)):
        totals = run.get("totals", {})
        count = max(totals.get("questions", 0) - totals.get("errors", 0), 1)
        lines.append(f"  {label}: cost {totals.get('cost_cents', 0.0):.3f} cents ({totals.get('cost_cents', 0.0) / count:.4f} per question), {totals.get('seconds', 0.0) / count:.1f} s per question")
    return lines


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Compare two saved runs question by question (step 2.8).")
    parser.add_argument("run_a", help="the first run file (the one you compare against)")
    parser.add_argument("run_b", help="the second run file (the change)")
    parser.add_argument("--metric", choices=CHOICES, default="right", help="what to compare (default: right)")
    args = parser.parse_args(argv)
    run_a, run_b = report.load_run(args.run_a), report.load_run(args.run_b)
    print("\n".join(compare_lines(run_a, run_b, args.metric, (args.run_a, args.run_b))))


if __name__ == "__main__":
    main()
