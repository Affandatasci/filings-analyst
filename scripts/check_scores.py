"""Step 4.6: the push check. Free: it reads two small score files, never calls Claude and needs no key.

    python scripts\\check_scores.py                  check results\\latest_scores.json against results\\baseline_scores.json and the pass lines
    python scripts\\check_scores.py --baseline results\\runs\\base_test.json   (step 2.8) save one FULL test run as results\\baseline_scores.json
    python scripts\\check_scores.py --latest results\\runs\\final_test.json    save a run as results\\latest_scores.json, the file the check reads
    python scripts\\check_scores.py --show-lines     print PASS_LINES (the baseline minus 5 points) to paste into config.py (step 2.9)

The check fails (exit code 1) when the latest run is partial, is not the same split as the baseline, has a score under its
pass line in config.py, or has a score more than 5 points under the baseline. The two score files hold numbers only (no
question text), so they can be pushed. The run files in results\\runs stay on your laptop.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from evals import report  # noqa: E402

MAX_DROP = 5.0  # points out of 100
LINE_GAP = 5.0  # a pass line is the baseline minus this
WORDS = dict(report.METRICS)
SETTING_NAMES = ("filter", "k", "collection", "prompt", "claude_model", "embed_model", "judge")


def points(values):
    values = [value for value in values if value is not None]
    return round(100 * sum(values) / len(values), 1) if values else None


def score_file(run):
    """The small, pushable summary of one run file: scores, the same scores by slice, and what was run."""
    cases = run.get("cases", [])
    scored = [case for case in cases if not case.get("error")]
    slices = {}
    for field, _ in report.SLICES:
        groups = defaultdict(list)
        for case in scored:
            groups[str(case.get(field) or "(none)")].append(case)
        slices[field] = {
            value: {"n": len(group), "right": points([report.is_right(case) for case in group]), "found_page": points([report.metric(case, "found_page") for case in group])}
            for value, group in sorted(groups.items())
        }
    settings = run.get("settings", {})
    return {
        "version": 1, "tag": run.get("tag"), "split": run.get("split"), "created": run.get("created"),
        "partial": bool(run.get("partial") or run.get("stopped")),
        "questions": len(scored), "errors": len(cases) - len(scored),
        "cost_cents": run.get("totals", {}).get("cost_cents", 0.0),
        "settings": {name: settings.get(name) for name in SETTING_NAMES},
        "scores": report.compute_scores(cases), "slices": slices,
    }


def read_scores(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise SystemExit(f"Could not read {path}: {err}")
    if "scores" not in data:
        raise SystemExit(f"{path} is not a score file (it has no 'scores').")
    return data


def pass_lines_from(baseline):
    return {name: round(item["points"] - LINE_GAP, 1) for name, item in baseline["scores"].items()}


def check(latest, baseline, lines):
    """(problems, table). problems is a list of plain sentences, empty when the check passes."""
    problems, table = [], []
    if latest.get("partial"):
        problems.append(f"The latest run ({latest.get('tag')}) is a partial run: only some of the questions were run, so its scores are rough.")
    if latest.get("split") != baseline.get("split"):
        problems.append(f"The latest run is on the {latest.get('split')} questions and the baseline is on the {baseline.get('split')} questions. They cannot be compared.")
    for name in sorted(set(baseline["scores"]) | set(lines)):
        base, now, line = baseline["scores"].get(name), latest["scores"].get(name), lines.get(name)
        if now is None:
            problems.append(f"{name} was not measured in the latest run.")
            continue
        verdict = "ok"
        if line is not None and now["points"] < line:
            problems.append(f"{name} is {now['points']}, under its pass line {line}.")
            verdict = "UNDER THE LINE"
        if base is not None and base["points"] - now["points"] > MAX_DROP:
            problems.append(f"{name} fell from {base['points']} to {now['points']}, more than {MAX_DROP:g} points under the baseline.")
            verdict = "DROPPED"
        table.append((name, base["points"] if base else None, now["points"], line, verdict))
    return problems, table


def show(value):
    return f"{value:6.1f}" if value is not None else "     -"


def main(argv=None, results=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="The push check for scores (step 4.6).")
    parser.add_argument("--baseline", metavar="RUN_FILE", help="save one full test run as results\\baseline_scores.json")
    parser.add_argument("--latest", metavar="RUN_FILE", help="save a run as results\\latest_scores.json")
    parser.add_argument("--show-lines", action="store_true", help="print PASS_LINES for config.py (the baseline minus 5 points)")
    args = parser.parse_args(argv)
    print("COLAB: no | MODEL: unchanged | CLAUDE COST: 0 cents")
    folder = Path(results or config.RESULTS_DIR)
    base_path, latest_path = folder / "baseline_scores.json", folder / "latest_scores.json"

    for flag, path in (("baseline", base_path), ("latest", latest_path)):
        if getattr(args, flag):
            summary = score_file(report.load_run(getattr(args, flag)))
            if flag == "baseline" and (summary["partial"] or summary["split"] != "test"):
                sys.exit("The baseline must be a full run on the test questions (no --limit, and the run must not have stopped early).")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"Saved {path} ({summary['questions']} questions, run {summary['tag']}, {summary['split']}).")
            if summary["partial"]:
                print("WARNING: this is a partial run. The push check will fail until you save a full run as latest.")
    if args.baseline or args.latest:
        return 0

    if not base_path.exists():
        if config.PASS_LINES:
            print("FAIL: PASS_LINES is set in config.py but results\\baseline_scores.json does not exist. Do step 2.8 first.")
            return 1
        print("Nothing to check yet: there is no results\\baseline_scores.json (step 2.8).")
        return 0
    baseline = read_scores(base_path)
    if args.show_lines:
        print("Paste this over  PASS_LINES = {}  in config.py (delete a line if you do not want that score to gate a push):\n")
        print("PASS_LINES = {\n" + "".join(f'    "{name}": {value},\n' for name, value in pass_lines_from(baseline).items()) + "}")
        return 0
    if latest_path.exists():
        latest = read_scores(latest_path)
    else:
        latest = baseline
        print("There is no latest_scores.json yet, so the baseline is checked against its own pass lines.")

    lines = dict(config.PASS_LINES)
    problems, table = check(latest, baseline, lines)
    print(f"Baseline: run {baseline.get('tag')} on {baseline.get('split')} ({baseline.get('questions')} questions). Latest: run {latest.get('tag')} ({latest.get('questions')} questions).")
    if not lines:
        print("The pass lines are not set yet (step 2.9), so only the drop rule is checked.")
    print(f"\n{'score':44}{'baseline':>9}{'latest':>8}{'line':>8}")
    for name, base, now, line, verdict in table:
        print(f"{WORDS.get(name, name)[:43]:44}{show(base)}{show(now)}{show(line)}   {verdict}")
    if latest.get("errors"):
        print(f"\nNote: {latest['errors']} question(s) of the latest run had an error and were not scored.")
    if problems:
        print("\nFAIL")
        for problem in problems:
            print("  " + problem)
        return 1
    print("\nOK: no score is under its line and none dropped more than 5 points.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
