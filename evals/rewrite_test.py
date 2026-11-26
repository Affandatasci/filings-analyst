"""Rewrite test: is there a better search line than the one the bot writes now?

    python -m evals.rewrite_test                  FREE. The question as typed, the rewrite the bot uses now, and the same rewrite with the noise taken out
    python -m evals.rewrite_test --new            also asks Claude for the new request (rewrite_v3: a line in the words of the filing): about 1 cent for the 85 questions

Scored on search alone, like experiments 5 and 8: is an evidence page among the best 5 pieces, and how high does it rank. It uses the 85 practice and test
questions and the same company and year filter as the bot. The holdout is never opened. Nothing in config.py changes: the bot keeps its rewrite until you
choose. Every rewrite is kept in data\\cache\\rewrite_cache.jsonl, the cache the bot and run_eval already use, so a rewrite you paid for here is free later.

If the new line wins, switch it on in config.py (both lines):
    REWRITE_PROMPT = "rewrite_v3"
    QUERY_CLEAN = True
Results are added to results\\experiments.md (write your choice there) and saved as results\\rewrite_test.json.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import config
from bot import llm, lookup
from evals import experiments
from evals.cases import CaseError
from evals.experiments import Variant
from evals.judge import JsonlCache

SPEND_FREE_CENTS = 1.0  # more than this needs --spend
MAX_CENTS = 3.0  # this test never spends more than this, whatever --spend says
SHOW = 3  # lines to read


def build_variants(ctx, cache, ask, new, spent):
    """The variants to score. spent collects the cents of every Claude call that is made (rewrites that are not in the cache yet)."""
    k = config.TOP_K
    lines = {}

    def line(case, prompt, clean):
        key = (case.id, prompt, clean)
        if key not in lines:
            made = lookup.make_queries(case.question, count=1, hints=True, ask_fn=ask, cache=cache, prompt=prompt, clean=clean)
            spent.append(made.cents)
            lines[key] = made.queries[0]
        return lines[key]

    def searched(prompt, clean):
        def hits(case):
            company, year = ctx.filters(case.question)  # the filter is read from the question, not from the rewrite
            return ctx.search(line(case, prompt, clean), k, mode="off", company=company, year=year)

        return hits

    variants = [
        Variant("question as typed (no rewrite)", lambda case: ctx.search(case.question, k), k),
        Variant("now: rewrite_v2 with word hints", searched("rewrite_v2", False), k),
        Variant("now, with the noise taken out", searched("rewrite_v2", True), k),
    ]
    if new:
        variants.append(Variant("new: rewrite_v3 line, noise taken out", searched("rewrite_v3", True), k))
    return variants, line


def result_lines(cases, variants, marks):
    summary, several = experiments.summarize(cases, marks)
    n = len(cases)
    lines = [f"Rewrite test: {n} questions (practice and test; the holdout is not used). Search only, filter {config.FILTER_MODE}.", ""]
    lines.append(f"  {'variant':42} {'right page found':18} {'MRR':>6}  first place")
    for variant in variants:
        item = summary[variant.name]
        lines.append(f"  {variant.name:42} {str(item['hit']) + ' of ' + str(n):18} {item['mrr']:6.3f}  {item['first']} of {n}")
    base = variants[1].name
    for variant in variants[2:]:
        gained, lost = experiments.flips(cases, marks, base, variant.name)
        lines.append(f"  {base} -> {variant.name}: found {len(gained)} more, lost {len(lost)}" + (f"   gained: {experiments.listed(gained)}" if gained else "") + (f"   lost: {experiments.listed(lost)}" if lost else ""))
    lines.append("")
    best = variants[-1]
    if len(variants) == 4:
        now, new = summary[variants[1].name], summary[best.name]
        if new["hit"] >= now["hit"] + 2 and new["mrr"] >= now["mrr"]:
            lines.append("The new line wins. To use it, set REWRITE_PROMPT = \"rewrite_v3\" and QUERY_CLEAN = True in config.py. Its rewrites are already in the cache.")
        else:
            lines.append("The new line does not win clearly (it needs 2 more questions found and no lower MRR). No change in config.py: the bot keeps rewrite_v2.")
    else:
        lines.append("To compare the new request (rewrite_v3), run:  python -m evals.rewrite_test --new")
    lines.append(f"A change of 1 or 2 questions out of {n} can be luck: read the questions that flipped before you choose.")
    return lines


def save_outputs(cases, variants, marks, extra=""):
    """Add a section to results\\experiments.md and save results\\rewrite_test.json. Returns both paths."""
    folder = Path(config.RESULTS_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    summary, several = experiments.summarize(cases, marks)
    n = len(cases)
    lines = [f"### Rewrite test ({datetime.now().strftime('%Y-%m-%d %H:%M')})", f"{n} questions, search only, filter {config.FILTER_MODE}. {extra}".strip(), "",
             "| variant | right page found | MRR | first place |", "|---|---|---|---|"]
    for variant in variants:
        item = summary[variant.name]
        lines.append(f"| {variant.name} | {item['hit']} of {n} | {item['mrr']:.3f} | {item['first']} of {n} |")
    lines += ["", "Chosen: (write your choice and the reason here)", ""]
    log = folder / "experiments.md"
    head = "" if log.exists() else "# Experiments\n\nEvery setting tried, one at a time. Experiments 1 to 5 are scored on search alone.\n\n"
    with log.open("a", encoding="utf-8") as handle:
        handle.write(head + "\n".join(lines) + "\n")
    data = {
        "test": "rewrite", "created": datetime.now().isoformat(timespec="seconds"), "questions": n, "filter": config.FILTER_MODE,
        "variants": {variant.name: {"k": variant.k, "summary": summary[variant.name], "places": {case.id: marks[variant.name][i]["place"] for i, case in enumerate(cases)}} for variant in variants},
    }
    path = folder / "rewrite_test.json"
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return log, path


def main(argv=None, search_fn=None, ask_fn=None, store=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Is there a better search line than the one the bot writes now? Search only (build rewrite).")
    parser.add_argument("--new", action="store_true", help="also ask Claude for the new request (rewrite_v3): about 1 cent")
    parser.add_argument("--first", type=int, metavar="N", help="only the first N of the 85 questions (a quick try)")
    parser.add_argument("--collection", help="index to test (default: COLLECTION in config.py)")
    parser.add_argument("--spend", action="store_true", help="allow this test when it costs more than 1 cent")
    args = parser.parse_args(argv)
    try:
        cases = experiments.usable_cases(args.first)
    except CaseError as err:
        sys.exit(str(err))
    guess = len(cases) * llm.cost_cents(*lookup.REWRITE_TOKENS) if args.new else 0.0
    cost = f"about {guess:.2f} cents with --new (nothing for rewrites already in the cache)" if args.new else "0 cents (rewrites that are not in the cache yet would cost about 0.01 cent each)"
    print(f"COLAB: no | MODEL: unchanged ({config.EMBED_MODEL}, {config.EMBED_DIM} numbers) | CLAUDE COST: {cost}")
    if guess > MAX_CENTS:
        sys.exit(f"That is more than the {MAX_CENTS} cents this test allows. Use --first 40.")
    if guess > SPEND_FREE_CENTS and not args.spend:
        sys.exit(f"This test costs about {guess:.1f} cents. Add --spend to run it, or --first 40 for a small try.")
    print(f"{len(cases)} questions. Ledger: {llm.spent_cents():.3f} of {config.FUSE_CENTS} cents used. The first search loads the embedding model, please wait ...\n")
    ctx = experiments.Context(args.collection, search_fn, store, ask_fn)
    cache = JsonlCache(Path(config.DATA_DIR) / "cache" / "rewrite_cache.jsonl")
    spent = []
    try:
        variants, line = build_variants(ctx, cache, ask_fn, args.new, spent)
        marks = experiments.score(cases, variants)
    except llm.BudgetStop as err:
        sys.exit(str(err))
    print("\n".join(result_lines(cases, variants, marks)))
    noisy = sum(line(case, "rewrite_v2", False) != line(case, "rewrite_v2", True) for case in cases)
    print(f"\nNoise in the rewrites the bot uses now: {noisy} of {len(cases)} lines (a label, markdown or a last word 'search').")
    if args.new:
        changed = [case for case in cases if line(case, "rewrite_v2", True) != line(case, "rewrite_v3", True)][:SHOW]
        print("Lines to read (now -> new):")
        for case in changed:
            print(f"  {' '.join(case.question.split())[:80]}\n    now: {line(case, 'rewrite_v2', True)[:150]}\n    new: {line(case, 'rewrite_v3', True)[:150]}")
    extra = f"The new rewrites cost {sum(spent):.3f} cents." if sum(spent) else ""
    log, path = save_outputs(cases, variants, marks, extra)
    print(f"\nAdded to {log} (write your choice there) and saved {path}. Ledger: {llm.spent_cents():.3f} of {config.FUSE_CENTS} cents used.")


if __name__ == "__main__":
    main()
