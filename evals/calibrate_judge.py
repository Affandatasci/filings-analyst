"""Step 2.7: check the judge against your own 20 marks. Experiment 7 (the judge model) is the same command with --judge all.

    python -m evals.calibrate_judge --pick           1. the bot answers 20 questions: the 10 practice and 10 test ones (picked with a fixed seed)
                                                     It writes data\\calibration\\read_these.txt. Read it, and write pass or fail for each id
                                                     in data\\calibration\\hand_marks.csv. These marks are your own work: nobody can do them for you.
    python -m evals.calibrate_judge                  2. the judge marks the same 20 answers. It prints agreement, catch rate and false alarm rate
    python -m evals.calibrate_judge --judge all      the same for every judge model that has a key in .env

COSTS CLAUDE CENTS: step 1 is about 0.7 cents (20 bot answers). Step 2 is about 0.15 cents with the claude judge, and nothing with
gemini or groq. A second run of step 2 costs nothing (the judge keeps its answers). No Colab, no model change.

pass means: the bot's answer is right (it matches the reference answer; small rounding is fine).
At least 6 of the 20 should be fail by your reading. If the bot is too good, add answers from a weak setting (top 1):
    python -m evals.calibrate_judge --pick --weak 5 --again        (then your marks must be done again)
Aim for agreement 85 out of 100 and catch rate 80 out of 100. With 20 answers these numbers are rough. Rewrite a judge rule at
most 3 times, then freeze its version name. The questions and answers stay in data\\calibration, which is kept out of Git.
"""
import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import config
from bot import answer as bot_answer
from bot import llm
from evals import judge as judges
from evals import run_eval
from evals.cases import CaseError, load_cases

PICK_SEED = 2026
PRACTICE_COUNT = 10
TEST_COUNT = 10
MIN_FAILS = 6
AIM_AGREEMENT, AIM_CATCH = 85, 80
PASS_WORDS = ("pass", "p", "yes", "y", "1", "true", "right", "ok", "correct")
FAIL_WORDS = ("fail", "f", "no", "n", "0", "false", "wrong", "incorrect")
EVIDENCE_CHARS = 900


def folder():
    return Path(config.DATA_DIR) / "calibration"


def marks_path():
    return folder() / "hand_marks.csv"


def answers_path():
    return folder() / "answers.json"


def reading_path():
    return folder() / "read_these.txt"


# ---------- which 20 answers ----------

def pick_ids(cases):
    """The 10 practice ids and 10 test ids (a fixed seed), in file order. The same file always gives the same 20."""
    practice = [case.id for case in cases if case.split == "practice"][:PRACTICE_COUNT]
    test = [case.id for case in cases if case.split == "test"]
    chosen = sorted(random.Random(PICK_SEED).sample(range(len(test)), min(TEST_COUNT, len(test))))
    return practice + [test[i] for i in chosen]


def write_marks_template(ids, path=None):
    """Make hand_marks.csv with the 20 ids and empty marks, unless it exists. Returns True when it was made."""
    path = Path(path) if path else marks_path()
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "mark", "note"])
        for case_id in ids:
            writer.writerow([case_id, "", ""])
    return True


def read_marks(path=None):
    """Your marks. Returns ({id: True for pass, False for fail}, [ids without a usable mark])."""
    path = Path(path) if path else marks_path()
    if not path.exists():
        raise CaseError(f"{path} is missing. Run:  python -m evals.calibrate_judge --pick")
    marks, missing = {}, []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            case_id = (row.get("id") or "").strip()
            if not case_id:
                continue
            word = (row.get("mark") or "").strip().lower()
            if word in PASS_WORDS:
                marks[case_id] = True
            elif word in FAIL_WORDS:
                marks[case_id] = False
            else:
                missing.append(case_id)
    return marks, missing


def calibration_stats(pairs):
    """pairs: (your mark, the judge's mark), True for pass. Returns agreement, catch rate and false alarm rate in points out of 100."""
    def share(top, bottom):
        return round(100 * top / bottom) if bottom else None

    your_fails = [(mine, judged) for mine, judged in pairs if not mine]
    your_passes = [(mine, judged) for mine, judged in pairs if mine]
    return {
        "n": len(pairs),
        "agreement": share(sum(mine == judged for mine, judged in pairs), len(pairs)),
        "catch": share(sum(not judged for _, judged in your_fails), len(your_fails)),  # your fails that the judge also fails
        "false_alarm": share(sum(not judged for _, judged in your_passes), len(your_passes)),  # your passes that the judge wrongly fails
        "your_fails": len(your_fails),
    }


# ---------- step 1: make the 20 answers ----------

def short(text, width=300):
    return " ".join(str(text).split())[:width]


def reading_text(cases, answers):
    """The page to read: for each answer the question, the reference, the evidence and the bot's answer."""
    parts = []
    for number, case in enumerate(cases, start=1):
        item = answers[case.id]
        cited = ", ".join(f"{c['doc']} p{c['page']}" for c in item["citations"]) or "nothing"
        evidence = "\n".join(f"  [page {e.page}] {' '.join(e.text.split())[:EVIDENCE_CHARS]}" for e in case.evidence)
        parts.append(
            f"=== {number} of {len(cases)}: {case.id.removeprefix('financebench_id_')} ({item['source']}) {case.doc} ===\n"
            f"QUESTION: {case.question}\n"
            f"REFERENCE ANSWER: {case.answer}\n"
            f"EVIDENCE:\n{evidence}\n"
            f"BOT ANSWER: {item['answer'] or '(no answer)'}\n"
            f"BOT CITED: {cited}   FOUND: {str(item['found']).lower()}\n"
            f"YOUR MARK: pass or fail for {case.id} in {marks_path()}\n"
        )
    return "\n".join(parts)


def pick(args, answer_fn=None):
    try:
        cases = load_cases()
    except CaseError as err:
        sys.exit(str(err))
    ids = pick_ids(cases)
    by_id = {case.id: case for case in cases}
    if answers_path().exists() and not args.again:
        sys.exit(f"{answers_path()} exists: your marks belong to those answers. Add --again to make new answers (then do the marks again).")
    weak = set([i for i in ids if by_id[i].split == "test"][: args.weak or 0])
    total, _, _ = run_eval.estimate_cents(len(ids), None)
    print(f"{len(ids)} answers to make ({len(weak)} with the weak setting, 1 piece).")
    print(f"COLAB: no | MODEL: unchanged | CLAUDE COST: about {total:.2f} cents (20 bot answers, no judge)")
    print(f"Ledger: {llm.spent_cents():.3f} of {config.FUSE_CENTS} cents used.\n")
    answer_fn = answer_fn or (lambda question, k=None: bot_answer.answer(question, k=k))
    answers = {}
    for number, case_id in enumerate(ids, start=1):
        case, k = by_id[case_id], (1 if case_id in weak else None)
        try:
            record = answer_fn(case.question, k=k).to_dict(include_text=True)
        except llm.BudgetStop as err:
            sys.exit(str(err))
        except Exception as err:  # noqa: BLE001
            sys.exit(f"Question {case_id} failed ({type(err).__name__}: {short(err, 200)}). Nothing was saved. Fix that and run --pick again.")
        record.pop("question", None)
        record.pop("raw", None)
        source = "weak" if case_id in weak else case.split
        answers[case_id] = {
            "source": source, "k": k or config.TOP_K, "answer": record["answer"], "citations": record["citations"],
            "found": record["found"], "pieces": record["pieces"], "cost_cents": record["cost_cents"],
        }
        print(f"[{number}/{len(ids)}] {case_id.removeprefix('financebench_id_')} {source}: {short(record['answer'], 70) or '(no answer)'}")
    run_eval.ensure_private_folder(folder())
    (folder() / ".gitignore").write_text("answers.json\nread_these.txt\n", encoding="utf-8")
    temp = answers_path().with_suffix(".tmp")
    temp.write_text(json.dumps({"seed": PICK_SEED, "ids": ids, "answers": answers}, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, answers_path())
    reading_path().write_text(reading_text([by_id[i] for i in ids], answers), encoding="utf-8")
    made = write_marks_template(ids)
    print(f"\nNow read:  {reading_path()}")
    print(f"Then write pass or fail for each of the {len(ids)} ids in:  {marks_path()}  ({'just made' if made else 'it already existed: check that its ids are these'})")
    if not made:
        try:
            old = set(read_marks()[0]) | set(read_marks()[1])
        except CaseError:
            old = set()
        if old != set(ids):
            print("WARNING: the ids in hand_marks.csv are not the ids of these answers. Rename the file and run --pick again, or fix the ids.")
    print(f"At least {MIN_FAILS} should be fail. Then run:  python -m evals.calibrate_judge")


# ---------- step 2: the judge marks them too ----------

def available_models():
    """The judge models that have what they need: claude needs ANTHROPIC_API_KEY, gemini and groq their keys."""
    keys = {"claude": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}
    return [model for model in judges.MODELS if getattr(config, keys[model], None)], [model for model in judges.MODELS if not getattr(config, keys[model], None)]


def judge_all(models, by_id, answers, ids, ask_fn=None, post_fn=None):
    """Judge the 20 answers with each model. Returns {model: {"verdicts": {id: Verdict}, "cents": float, "error": str}}."""
    results = {}
    for model in models:
        verdicts, cents, error = {}, 0.0, ""
        for case_id in ids:
            case = by_id[case_id]
            try:
                verdict = judges.judge_correct(case.question, case.answer, answers[case_id]["answer"] or "(no answer)", model=model, ask_fn=ask_fn, post_fn=post_fn)
            except (judges.JudgeError, llm.BudgetStop, RuntimeError) as err:
                error = str(err)
                break
            verdicts[case_id] = verdict
            cents += verdict.cost_cents
        results[model] = {"verdicts": verdicts, "cents": cents, "error": error}
    return results


def calibrate(args, ask_fn=None, post_fn=None):
    if not answers_path().exists():
        sys.exit(f"{answers_path()} is missing. Run first:  python -m evals.calibrate_judge --pick")
    try:
        saved = json.loads(answers_path().read_text(encoding="utf-8"))
        marks, missing = read_marks()
        by_id = {case.id: case for case in load_cases()}
    except (ValueError, CaseError) as err:
        sys.exit(f"Could not read the files: {err}")
    ids = saved["ids"]
    answers = saved["answers"]
    unmarked = [i for i in ids if i not in marks]
    if unmarked:
        sys.exit(
            f"{len(unmarked)} of the {len(ids)} ids have no mark in {marks_path()}: " + ", ".join(i.removeprefix("financebench_id_") for i in unmarked[:10])
            + ("" if not missing else f". Write pass or fail in the mark column (not: {', '.join(m.removeprefix('financebench_id_') for m in missing[:5])}).")
        )
    fails = sum(not marks[i] for i in ids)
    print(f"Your marks: {len(ids) - fails} pass, {fails} fail.")
    if fails < MIN_FAILS:
        print(f"WARNING: fewer than {MIN_FAILS} fails. The catch rate cannot be trusted. Add weak answers:  python -m evals.calibrate_judge --pick --weak {MIN_FAILS - fails + 1} --again")

    if args.judge == "all":
        models, skipped = available_models()
        for model in skipped:
            print(f"{model}: no key in .env, skipped.")
    else:
        models = [args.judge or judges.default_model()]
    if not models:
        sys.exit("No judge model has a key. Add ANTHROPIC_API_KEY, GEMINI_API_KEY or GROQ_API_KEY to .env.")
    per_call = llm.cost_cents(*run_eval.JUDGE_CORRECT_TOKENS)
    claude_part = per_call * len(ids) if "claude" in models else 0.0
    print(f"COLAB: no | MODEL: unchanged | CLAUDE COST: about {claude_part:.2f} cents ({'claude judge' if claude_part else 'free judge only'}); a second run costs nothing")
    print(f"Ledger: {llm.spent_cents():.3f} of {config.FUSE_CENTS} cents used.\n")

    results = judge_all(models, by_id, answers, ids, ask_fn, post_fn)
    print(f"{'judge':8} {'agreement':>10} {'catch rate':>11} {'false alarm':>12} {'cost':>9}   rules")
    best = None
    for model, outcome in results.items():
        if outcome["error"]:
            print(f"{model:8} could not finish: {outcome['error']}")
            continue
        stats = calibration_stats([(marks[i], outcome["verdicts"][i].passed) for i in ids])
        show = lambda value: "-" if value is None else str(value)  # noqa: E731
        print(f"{model:8} {show(stats['agreement']):>10} {show(stats['catch']):>11} {show(stats['false_alarm']):>12} {outcome['cents']:>7.3f} c   {judges.CURRENT['correct']}")
        if best is None or (stats["agreement"] or 0) > best[1]:
            best = (model, stats["agreement"] or 0)
    print("(all scores are out of 100: agreement = same mark as you; catch rate = your fails that the judge also fails; false alarm = your passes that the judge fails)")
    if best:
        print(f"Agrees most with you: {best[0]} ({best[1]} out of 100). Aim: agreement {AIM_AGREEMENT}, catch rate {AIM_CATCH}. With 20 answers these numbers are rough.")
    for model, outcome in results.items():
        wrong = [i for i in ids if i in outcome["verdicts"] and outcome["verdicts"][i].passed != marks[i]]
        for case_id in wrong[:10]:
            verdict = outcome["verdicts"][case_id]
            print(f"  {model} disagrees on {case_id.removeprefix('financebench_id_')}: you {'pass' if marks[case_id] else 'fail'}, judge {'pass' if verdict.passed else 'fail'}: {short(verdict.reason, 110)}")
    print("If the judge is weak, rewrite its rule as a new version (correct_v2) in evals\\judge.py: at most 3 rewrites, then freeze it.")
    print(f"Ledger: {llm.spent_cents():.3f} of {config.FUSE_CENTS} cents used.")


def main(argv=None, answer_fn=None, ask_fn=None, post_fn=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Check the judge against your own 20 marks (step 2.7).")
    parser.add_argument("--pick", action="store_true", help="step 1: make the 20 answers to read")
    parser.add_argument("--weak", type=int, metavar="N", help="with --pick: answer N of the test questions with 1 piece, to get more wrong answers")
    parser.add_argument("--again", action="store_true", help="with --pick: make new answers over the old ones")
    parser.add_argument("--judge", choices=judges.MODELS + ("all",), help="step 2: judge model (default: JUDGE_MODEL in config.py, else claude)")
    args = parser.parse_args(argv)
    if args.pick:
        pick(args, answer_fn)
    else:
        calibrate(args, ask_fn, post_fn)


if __name__ == "__main__":
    main()
