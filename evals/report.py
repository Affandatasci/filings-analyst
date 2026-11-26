"""Step 2.6: the one page report of a saved run. Free: no Claude call.

    python -m evals.report                                   the newest run in results\\runs
    python -m evals.report results\\runs\\base_test.json      one run

Scores are in plain words: how many questions fail out of every 100, with the range that this many
questions can really support (about plus or minus 11 for 75 questions: say so, do not hide it).
The report also cuts the scores by company, document type, question type and reasoning type, shows where the
wrong answers break (search or bot), counts your failure types, and shows cost and the pass lines.

Your failure notes (step 2.6) go in results\\failure_notes.md, one line each, with the question id and the type in brackets:
    - 00012 [search missed] the evidence page ranked ninth
Use ids, not question text. The report counts the types.
"""
import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import config

SLICES = (("company", "company"), ("doc_type", "document type"), ("question_type", "question type"), ("reasoning_type", "reasoning type"))
METRICS = (  # name, plain words for "this is true"
    ("correct", "answer is correct (judge)"),
    ("number_match", "numbers match the reference"),
    ("found_page", "right page among the pieces"),
    ("supported", "every claim is in the pieces (judge)"),
    ("citation_valid", "citations are real"),
    ("citation_hit", "a citation is an evidence page"),
    ("json_shape", "reply in the agreed format"),
    ("length", "answer is short"),
)
JUDGE_METRICS = ("correct", "supported")
MIN_SLICE = 4  # smaller groups are listed but not called weak
NOTE_LINE = re.compile(r"^\s*[-*]?\s*(?:financebench_id_)?(\d+)\s*\[([^\]]+)\]")


def runs_dir():
    return Path(config.RESULTS_DIR) / "runs"


# ---------- reading one case result ----------

def metric(case, name):
    """True, False, or None when the metric was not measured for this case (skipped, no judge, or an error)."""
    if case.get("error"):
        return None
    if name in JUDGE_METRICS:
        verdict = (case.get("judge") or {}).get(name)
        return None if verdict is None else bool(verdict["passed"])
    mark = (case.get("marks") or {}).get(name)
    if mark is None or mark["status"] == "skip":
        return None
    return mark["status"] == "pass"


def is_right(case):
    """Is the answer right? The judge's word when there is one, otherwise the number match."""
    verdict = metric(case, "correct")
    return verdict if verdict is not None else metric(case, "number_match")


def compute_scores(cases):
    """{metric: {"good": n, "of": n, "points": out of 100}} for every metric that was measured at least once."""
    scores = {}
    for name, _ in METRICS:
        values = [metric(case, name) for case in cases]
        values = [value for value in values if value is not None]
        if values:
            scores[name] = {"good": sum(values), "of": len(values), "points": round(100 * sum(values) / len(values), 1)}
    return scores


def wilson(good, n, z=1.96):
    """A 95 in 100 range for a share, as (low, high) between 0 and 1."""
    if n == 0:
        return 0.0, 0.0
    p = good / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# ---------- the report ----------

def read_notes(path=None):
    """Counts of your failure types from results\\failure_notes.md. Returns (Counter of types, number of notes)."""
    path = Path(path) if path else Path(config.RESULTS_DIR) / "failure_notes.md"
    types = Counter()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            match = NOTE_LINE.match(line)
            if match:
                types[match.group(2).strip().lower()] += 1
    return types, sum(types.values())


def where_it_breaks(cases):
    """Counts for the first two marks read together: did search find the page, and is the answer right?"""
    table = Counter()
    for case in cases:
        found, right = metric(case, "found_page"), is_right(case)
        if found is None or right is None:
            continue
        table[("yes" if found else "no", "yes" if right else "no")] += 1
    return table


def out_of_100(good, n):
    return round(100 * good / n) if n else 0


def build_report(run, notes=None, pass_lines=None):
    """The report as a list of lines. run is a loaded result file."""
    cases = run.get("cases", [])
    scored = [case for case in cases if not case.get("error")]
    errors = len(cases) - len(scored)
    scores = compute_scores(cases)
    settings = run.get("settings", {})
    lines = [f"REPORT {run.get('tag', '?')} / {run.get('split', '?')}: {len(scored)} questions scored ({run.get('created', '')[:16]})"]
    lines.append(
        "Settings: filter {filter} | pieces {k} | collection {collection} | prompt {prompt} | bot {claude_model} | judge {judge}".format(
            **{key: settings.get(key, "?") for key in ("filter", "k", "collection", "prompt", "claude_model", "judge")}
        )
    )
    if run.get("partial"):
        lines.append(f"NOT A FULL RUN: only {len(cases)} of the questions of this split were run, so these scores are rough.")
    if run.get("stopped"):
        lines.append("STOPPED: " + run["stopped"])
    if errors:
        lines.append(f"{errors} question{'s' if errors != 1 else ''} had an error and {'are' if errors != 1 else 'is'} not scored (see the file).")
    judge_errors = run.get("totals", {}).get("judge_errors", 0)
    if judge_errors:
        lines.append(f"{judge_errors} answer{'s' if judge_errors != 1 else ''} could not be judged (see judge_error in the file).")
    lines.append("")
    lines.append("SCORES (how many fail out of every 100 questions; the range is what this many questions can support)")
    for name, words in METRICS:
        if name not in scores:
            continue
        good, n = scores[name]["good"], scores[name]["of"]
        low, high = wilson(good, n)
        lines.append(f"  {words:42} {out_of_100(n - good, n):3d} fail   (range {round(100 * (1 - high))} to {round(100 * (1 - low))})   {good} of {n} ok")
    if "correct" not in scores and "number_match" in scores:
        lines.append("  (no judge in this run, so 'right' means the numbers match)")
    lines.append("")

    table = where_it_breaks(cases)
    total = sum(table.values())
    if total:
        lines.append("WHERE THE ANSWERS BREAK (search found the page x answer right)")
        meaning = {
            ("yes", "yes"): "works",
            ("yes", "no"): "bot misread the page: fix the prompt, table reading, number rules",
            ("no", "yes"): "lucky, or the answer is on another page: check the evidence by hand",
            ("no", "no"): "search failed: fix pieces, filter, settings",
        }
        for key, words in meaning.items():
            lines.append(f"  page found {key[0]:3} / answer right {key[1]:3}: {table[key]:3d} of {total}   {words}")
        lines.append("")

    lines.append("BY SLICE (answer right, and right page found, out of every 100; groups under %d are rough)" % MIN_SLICE)
    for field, label in SLICES:
        groups = defaultdict(list)
        for case in scored:
            groups[case.get(field) or "(none)"].append(case)
        lines.append(f"  by {label}:")
        for value, group in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
            right = [is_right(case) for case in group if is_right(case) is not None]
            found = [metric(case, "found_page") for case in group if metric(case, "found_page") is not None]
            right_text = f"{out_of_100(sum(right), len(right)):3d}" if right else "  -"
            found_text = f"{out_of_100(sum(found), len(found)):3d}" if found else "  -"
            rough = "   (rough)" if len(group) < MIN_SLICE else ""
            lines.append(f"    {str(value)[:34]:34} {len(group):3d} questions   right {right_text}   page found {found_text}{rough}")
    lines.append("")

    types, count = notes if notes is not None else read_notes()
    lines.append("YOUR FAILURE TYPES (results\\failure_notes.md)")
    if count:
        for name, number in types.most_common(8):
            lines.append(f"  {number:3d}  {name}")
    else:
        lines.append("  none yet. Read every wrong answer and write one line each, for example:  - 00012 [search missed] note")
    lines.append("")

    totals = run.get("totals", {})
    lines.append(
        "COST: {total:.3f} cents in all (bot {bot:.3f}, judge {judge:.3f}); {per:.4f} cents and {sec:.1f} s per question".format(
            total=totals.get("cost_cents", 0.0), bot=totals.get("bot_cents", 0.0), judge=totals.get("judge_cents", 0.0),
            per=totals.get("cost_cents", 0.0) / max(len(scored), 1), sec=totals.get("seconds", 0.0) / max(len(scored), 1),
        )
    )
    pass_lines = pass_lines if pass_lines is not None else getattr(config, "PASS_LINES", None)
    lines.append("")
    lines.append("PASS LINES (a score in points out of 100 that must be reached)")
    if not pass_lines:
        lines.append("  not set yet. They are set in step 2.9, at the baseline minus 5 points, never at a number you hope for.")
    else:
        for name, line in pass_lines.items():
            if name in scores:
                points = scores[name]["points"]
                lines.append(f"  {name:16} {points:5.1f}  line {line}  {'ok' if points >= line else 'BELOW'}")
            else:
                lines.append(f"  {name:16} not measured in this run")
    return lines


# ---------- files ----------

def load_run(path):
    path = Path(path)
    try:
        run = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise SystemExit(f"Could not read {path}: {err}")
    if "cases" not in run:
        raise SystemExit(f"{path} is not a run file (it has no 'cases').")
    return run


def newest_run():
    files = sorted(runs_dir().glob("*.json"), key=lambda item: item.stat().st_mtime) if runs_dir().exists() else []
    if not files:
        raise SystemExit("There is no run yet. Make one with:  python -m evals.run_eval --split practice --tag base")
    return files[-1]


def save_report(run, lines):
    path = Path(config.RESULTS_DIR) / f"report_{run.get('tag', 'run')}_{run.get('split', '')}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="The one page report of a saved run (step 2.6).")
    parser.add_argument("file", nargs="?", help="a run file (default: the newest in results\\runs)")
    args = parser.parse_args(argv)
    path = Path(args.file) if args.file else newest_run()
    run = load_run(path)
    lines = build_report(run)
    print("\n".join(lines))
    print(f"\nSaved to {save_report(run, lines)}")


if __name__ == "__main__":
    main()
