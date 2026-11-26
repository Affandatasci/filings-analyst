"""Step 2.6: run one split of questions through the bot, check and judge every answer, save one result file.

    python -m evals.run_eval --split practice --tag base               the 10 practice questions (under 1 cent: it runs at once)
    python -m evals.run_eval --split test --tag base --limit 10        the first 10 test questions
    python -m evals.run_eval --split test --tag base --spend           all 75 test questions (it prints the cost first)
    python -m evals.run_eval --split test --tag base --dry-run         only says what would run and what it would cost

COSTS CLAUDE CENTS. The bot answers each question (1 call) and the judge marks it (2 calls). A run that would cost
more than 1 cent only starts when you add --spend. The spend fuse in config.py (90 cents) still stops everything.
A free judge (--judge gemini or --judge groq) takes the judge part of the cost away.
No Colab, no model change: this only asks the questions of the index you already have.

Other options: --filter off|company|company+year   --k N (pieces sent to the bot)   --collection NAME   --prompt NAME
               --judge claude|gemini|groq   --no-judge (code checks only)
               --neighbors off|on|2   also send the pages before and after the whole pages (about 2.6 times the text with on; config.NEIGHBOR_PAGES)
               --ids 01928,00735      only these questions of the split: a cheap test of one change (saved as <tag>_<split>_ids<N>.json)
The holdout is opened once, in the last run: --split holdout needs --final, and a second time needs --again.

Saves results\\runs\\<tag>_<split>.json (a --limit run adds _first<N>; that folder is kept out of Git: it holds the
text of the pieces) and results\\report_<tag>_<split>.txt. The file holds no question text and no reference answer.
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import config
from bot import answer as bot_answer
from bot import llm, lookup, prompts
from evals import code_checks, judge as judges, report
from evals.cases import SPLITS, CaseError, load_cases
from monitor import store, tracing
from retrieval import reader

SPEND_FREE_CENTS = 1.0  # a run that costs more than this needs --spend
AGENT_FACTOR = 1.2  # an agent question costs about this many plain ones (measured on 5 practice questions: 0.070 against 0.058 cent)
MAX_ERRORS_IN_A_ROW = 3  # then something is wrong (no key, Qdrant down): stop instead of failing 75 times
BOT_TOKENS = (6500, 300)  # a typical bot call: the rules and the pieces in, a short JSON out (measured on the 10 practice questions)
JUDGE_CORRECT_TOKENS = (700, 60)
JUDGE_SUPPORTED_TOKENS = (6700, 60)  # this call carries the pieces too
NEIGHBOR_TEXT = 1.6  # the pages before and after (neighbors=1) add this many times the text of one question (measured: 2.6 times in all)

def estimate_cents(count, judge_model="claude", mode="plain", rewrite=None, neighbors=0):
    """A rough price from the prices in config.py. Returns (total, bot part, judge part) in cents.

    neighbors: pages before and after each whole page (config.NEIGHBOR_PAGES). The pieces in the bot request and in the judge request get that much bigger.
    """
    more = 1.0 + NEIGHBOR_TEXT * max(int(neighbors or 0), 0) if mode != "agent" else 1.0
    bot = count * llm.cost_cents(BOT_TOKENS[0] * more, BOT_TOKENS[1])
    if mode == "agent":
        bot *= AGENT_FACTOR
    elif config.QUERY_REWRITE if rewrite is None else rewrite:
        bot += count * llm.cost_cents(*lookup.REWRITE_TOKENS)  # the rewrite of each question
    judge = count * (llm.cost_cents(*JUDGE_CORRECT_TOKENS) + llm.cost_cents(JUDGE_SUPPORTED_TOKENS[0] * more, JUDGE_SUPPORTED_TOKENS[1])) if judge_model == "claude" else 0.0
    return bot + judge, bot, judge


def short_id(case_id):
    return case_id.removeprefix("financebench_id_")


def lookup_note(items):
    """One line on what bot/lookup.py did, from the events of the answers. Empty when it did nothing (agent mode, or both switches off)."""
    events = [event for item in items if item["answer"] for event in item["answer"].get("events", []) if event.get("stage") == "lookup"]
    if not events:
        return ""
    pages = sum(event.get("whole_pages", 0) > 0 for event in events)
    problems = sorted({event[key] for event in events for key in ("problem", "page_problem") if event.get(key)})
    note = f"Lookup: a whole page was read for {pages} of {len(events)} questions."
    neighbors = sum(event.get("neighbor_pages", 0) > 0 for event in events)
    if neighbors:
        note += f" The pages next to them were added for {neighbors}."
    return note + (" Problems: " + "; ".join(problems[:2]) if problems else "")


def run_name(tag, split, limit, ids=0):
    return f"{tag}_{split}" + (f"_first{limit}" if limit else "") + (f"_ids{ids}" if ids else "") + ".json"


def ensure_private_folder(folder):
    """The folder gets its own .gitignore, so its files never reach Git."""
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / ".gitignore"
    if not marker.exists():
        marker.write_text("*\n!.gitignore\n", encoding="utf-8")


def save_run(run, path):
    ensure_private_folder(path.parent)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(run, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)  # the file is never half written


# ---------- the loop ----------

def case_line(number, total, item):
    sign = {"pass": "ok", "fail": "NO", "skip": "--"}
    marks, verdicts = item["marks"], item["judge"] or {}
    parts = [
        f"page {sign[marks['found_page']['status']]}",
        f"numbers {sign[marks['number_match']['status']]}",
        f"cite {sign[marks['citation_valid']['status']]}/{sign[marks['citation_hit']['status']]}",
        f"format {sign[marks['json_shape']['status']]}",
    ]
    for name in ("correct", "supported"):
        if name in verdicts:
            parts.append(f"{name} {'yes' if verdicts[name]['passed'] else 'NO'}")
    cents = item["answer"]["cost_cents"] + sum(verdict["cost_cents"] for verdict in verdicts.values())
    if item.get("judge_error"):
        parts.append("JUDGE ERROR")
    return f"[{number}/{total}] {short_id(item['id'])} {item['doc'][:24]:24} " + " | ".join(parts) + f" | {cents:.4f} cents"


def run_cases(cases, answer_fn, judge_fn=None, say=print):
    """Answer, check and judge every case. Returns (items, stopped reason, seconds).

    One bad question does not stop the run: its error is saved in its item. The spend fuse stops the run,
    and so do 3 errors one after another. What was done is kept.
    """
    items, in_a_row, stopped, started = [], 0, "", time.time()
    for number, case in enumerate(cases, start=1):
        item = {
            "id": case.id, "split": case.split, "company": case.company, "doc": case.doc, "doc_type": case.doc_type,
            "question_type": case.question_type, "reasoning_type": case.reasoning_type,
            "evidence": [[doc, page] for doc, page in case.pages],
            "error": "", "judge_error": "", "answer": None, "marks": None, "judge": None,
        }
        try:
            record = answer_fn(case.question).to_dict(include_text=True)
            marks = code_checks.run_checks(case, record)
        except llm.BudgetStop as err:
            stopped = str(err)
            say(f"STOP: {stopped}")
            break
        except Exception as err:  # noqa: BLE001 - one bad question must not stop the others
            item["error"] = f"{type(err).__name__}: {str(err)[:200]}"
            items.append(item)
            say(f"[{number}/{len(cases)}] {short_id(case.id)} ERROR {item['error']}")
            in_a_row += 1
            if in_a_row >= MAX_ERRORS_IN_A_ROW:
                stopped = f"{MAX_ERRORS_IN_A_ROW} errors one after another, the last one: {item['error']}"
                say(f"STOP: {stopped}")
                break
            continue
        in_a_row = 0
        record.pop("question", None)  # the question text is not saved
        record.pop("raw", None)  # neither is the raw reply
        item["answer"] = record
        item["marks"] = {name: mark.to_dict() for name, mark in marks.items()}
        fuse = False
        if judge_fn:
            try:
                item["judge"] = {name: verdict.to_dict() for name, verdict in judge_fn(case, record).items()}
            except llm.BudgetStop as err:
                stopped, fuse = str(err), True
            except Exception as err:  # noqa: BLE001 - the answer is kept even when the judge fails
                item["judge_error"] = f"{type(err).__name__}: {str(err)[:200]}"
        items.append(item)
        say(case_line(number, len(cases), item))
        if fuse:
            say(f"STOP: {stopped}")
            break
    return items, stopped, time.time() - started


def build_run(tag, split, limit, partial, stopped, settings, items, seconds):
    bot_cents = sum(item["answer"]["cost_cents"] for item in items if item["answer"])
    judge_cents = sum(v["cost_cents"] for item in items if item["judge"] for v in item["judge"].values())
    return {
        "version": 1, "tag": tag, "split": split, "limit": limit, "partial": partial, "stopped": stopped,
        "created": datetime.now().isoformat(timespec="seconds"),
        "settings": settings,
        "totals": {
            "questions": len(items),
            "errors": sum(bool(item["error"]) for item in items),
            "judge_errors": sum(bool(item["judge_error"]) for item in items),
            "cost_cents": round(bot_cents + judge_cents, 6), "bot_cents": round(bot_cents, 6), "judge_cents": round(judge_cents, 6),
            "seconds": round(sum(item["answer"]["seconds"] for item in items if item["answer"]), 2),
            "wall_seconds": round(seconds, 1),
        },
        "scores": report.compute_scores(items),
        "cases": items,
    }


# ---------- the command ----------

def main(argv=None, answer_fn=None, judge_fn=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Run a split through the bot, check and judge it, save the result (step 2.6).")
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--tag", required=True, help="a short name for this run, for example base or top4")
    parser.add_argument("--limit", type=int, metavar="N", help="only the first N questions of the split")
    parser.add_argument("--spend", action="store_true", help="allow a run that costs more than 1 cent")
    parser.add_argument("--dry-run", action="store_true", help="say what would run and what it costs, make no call")
    parser.add_argument("--filter", choices=reader.MODES, help="company filter for this run (default: FILTER_MODE in config.py)")
    parser.add_argument("--k", type=int, help="pieces sent to the bot (default: TOP_K in config.py)")
    parser.add_argument("--collection", help="index to search (default: COLLECTION in config.py)")
    parser.add_argument("--prompt", help="bot prompt version (default: DEFAULT_BOT_PROMPT)")
    parser.add_argument("--mode", choices=("plain", "agent"), default="plain", help="plain: search once, then answer (default). agent: Claude may search again, read a whole page and calculate (about 3 times the price)")
    parser.add_argument("--rewrite", choices=("on", "off"), help="query rewrite for this run (default: QUERY_REWRITE in config.py)")
    parser.add_argument("--page-context", choices=("on", "off"), help="whole pages for the best hits in this run (default: PAGE_CONTEXT in config.py)")
    parser.add_argument("--neighbors", choices=("off", "on", "2"), help="also send the pages before and after the whole pages: on = one page each side, about 2.6 times the text (default: NEIGHBOR_PAGES in config.py)")
    parser.add_argument("--ids", metavar="ID,ID", help="only these questions of the split, short ids such as 01928,00735: a cheap test of one change (saved as <tag>_<split>_ids<N>.json)")
    parser.add_argument("--judge", choices=judges.MODELS, help="judge model (default: JUDGE_MODEL in config.py, else claude)")
    parser.add_argument("--no-judge", action="store_true", help="code checks only: no judge calls")
    parser.add_argument("--final", action="store_true", help="needed to open the holdout")
    parser.add_argument("--again", action="store_true", help="open the holdout a second time")
    args = parser.parse_args(argv)

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        sys.exit("The tag may only hold letters, digits, - and _ (for example base or top4).")
    if args.limit is not None and args.limit < 1:
        sys.exit("--limit must be 1 or more.")
    if args.split == "holdout":
        if not args.final:
            sys.exit("The holdout is opened once, in the last run, to check that the settings did not just fit the test. Add --final when that time has come.")
        done = sorted(report.runs_dir().glob("*_holdout*.json")) if report.runs_dir().exists() else []
        if done and not args.again:
            sys.exit(f"The holdout was already opened ({done[0].name}). Opening it again makes it a second test set. Add --again only if you really mean it.")
    try:
        everything = load_cases(args.split)
    except CaseError as err:
        sys.exit(str(err))
    cases = everything[: args.limit] if args.limit else everything
    if args.ids:
        by_short = {short_id(item.id): item for item in everything}
        wanted = list(dict.fromkeys(piece.strip() for piece in args.ids.split(",") if piece.strip()))
        unknown = [piece for piece in wanted if piece not in by_short]
        if unknown:
            sys.exit(f"Not in the {args.split} split: {', '.join(unknown)}")
        cases = [by_short[piece] for piece in wanted]
        if args.limit:
            cases = cases[: args.limit]

    try:
        judge_model = None if args.no_judge else (args.judge or judges.default_model())
    except judges.JudgeError as err:
        sys.exit(str(err))
    use_rewrite = config.QUERY_REWRITE if args.rewrite is None else args.rewrite == "on"
    use_pages = config.PAGE_CONTEXT if args.page_context is None else args.page_context == "on"
    near = int(getattr(config, "NEIGHBOR_PAGES", 0)) if args.neighbors is None else {"off": 0, "on": 1, "2": 2}[args.neighbors]
    near = near if use_pages and args.mode == "plain" else 0  # the pages next to the whole pages need whole pages, and the agent does its own reading
    total, bot_part, judge_part = estimate_cents(len(cases), judge_model, args.mode, use_rewrite, near)
    print(f"{len(cases)} {args.split} questions, tag {args.tag}. Judge: {judge_model or 'none (code checks only)'}.")
    print(
        f"COLAB: no | MODEL: unchanged ({config.EMBED_MODEL}, {config.EMBED_DIM} numbers) | "
        f"CLAUDE COST: about {total:.2f} cents (bot {bot_part:.2f}, judge {judge_part:.2f}), rough"
    )
    if args.mode == "agent":
        print("Search: agent mode (Claude searches, reads whole pages and calculates by itself).")
    else:
        print(f"Search: query rewrite {'on' if use_rewrite else 'off'} | whole pages {'on' if use_pages else 'off'} | queries per question {config.QUERY_COUNT} | neighbor pages {near}")
    print(f"Ledger: {llm.spent_cents():.3f} of {config.FUSE_CENTS} cents used.\n")
    if args.dry_run:
        print("Dry run: no question was asked.")
        return
    if total > SPEND_FREE_CENTS and not args.spend:
        sys.exit(f"This run costs about {total:.1f} cents. Add --spend to run it, or --limit 10 for a small try.")

    changes = {"REWRITE_CACHE": Path(config.DATA_DIR) / "cache" / "rewrite_cache.jsonl", "QUERY_REWRITE": use_rewrite, "PAGE_CONTEXT": use_pages, "NEIGHBOR_PAGES": near}
    if args.filter:
        changes["FILTER_MODE"] = args.filter
    before = {name: getattr(config, name) for name in changes}
    for name, value in changes.items():
        setattr(config, name, value)
    try:
        settings = {
            "filter": config.FILTER_MODE, "k": args.k or config.TOP_K, "collection": args.collection or config.COLLECTION,
            "prompt": args.prompt or prompts.DEFAULT_BOT_PROMPT, "claude_model": config.CLAUDE_MODEL, "embed_model": config.EMBED_MODEL,
            "mode": args.mode, "rewrite": use_rewrite, "page_context": use_pages, "queries": config.QUERY_COUNT,
            "piece_tokens": config.PIECE_TOKENS, "page_offset": config.PAGE_OFFSET,
            "judge": judge_model or "none", "judge_rules": dict(judges.CURRENT) if judge_model else {},
        }
        if near:
            settings["neighbors"] = near
        if answer_fn is None and args.mode == "agent":
            from bot import agent as bot_agent

            answer_fn = lambda question: bot_agent.run_agent(question, k=args.k, collection=args.collection)  # noqa: E731
        answer_fn = answer_fn or (lambda question: bot_answer.answer(question, k=args.k, collection=args.collection, prompt=args.prompt))
        answer_fn = tracing.traced(answer_fn, tag=args.tag)  # step 4.2: every eval question is traced (does nothing without a LangSmith key)
        if judge_model and judge_fn is None:
            judge_fn = lambda case, record: judges.judge_record(case, record, model=judge_model)  # noqa: E731
        if not judge_model:
            judge_fn = None
        print("The first search loads the embedding model, please wait ...\n")
        items, stopped, seconds = run_cases(cases, answer_fn, judge_fn)
    finally:
        for name, value in before.items():
            setattr(config, name, value)

    partial = len(items) < len(everything)
    run = build_run(args.tag, args.split, args.limit, partial, stopped, settings, items, seconds)
    path = report.runs_dir() / run_name(args.tag, args.split, args.limit, len(cases) if args.ids else 0)

    save_run(run, path)
    store.log_run(run)  # step 4.1: the scores join the history in the eval_runs table (local files without Supabase keys)
    tracing.flush()  # send the traces that are still waiting
    lines = report.build_report(run)
    
    report_path = report.save_report(run, lines)
    print("\n" + "\n".join(lines))
    print(f"\nSaved: {path}\nReport: {report_path}")
    if lookup_note(items):
        print(lookup_note(items))
    print(f"Ledger: {llm.spent_cents():.3f} of {config.FUSE_CENTS} cents used.")
    print("Next: read every wrong answer, and write one line each in results\\failure_notes.md (see the report).")
    if stopped:
        sys.exit(1)


if __name__ == "__main__":
    main()
