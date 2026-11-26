"""Turn 6 (experiment 4): does reading the company and the year out of the question help search?

From the project folder:
    python scripts\\compare_filters.py                 the reader check, then search in the 3 modes on the 85 questions
    python scripts\\compare_filters.py --reader-only   only the reader check: no model, no Qdrant, free, a few seconds
    python scripts\\compare_filters.py --first 10      only the first 10 of the 85 questions (a quick try)
    python scripts\\compare_filters.py --collection filings_v1_plain

The three modes: off (no filter), company, company+year. Search only: no Claude call, no cost.
The 18 holdout questions are never used here. They are opened once, in the last run.
"""
import argparse
import functools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from evals.cases import CaseError, load_cases  # noqa: E402
from ingest.names import parse_name  # noqa: E402
from retrieval import reader  # noqa: E402

MODES = reader.MODES  # off, company, company+year
SHOW = 10  # how many question numbers to list on one line


def usable_cases(first=None):
    """The practice and test cases (85). The holdout is left out on purpose."""
    cases = [case for case in load_cases() if case.split != "holdout"]
    return cases[:first] if first else cases


def short_id(case_id):
    return case_id.removeprefix("financebench_id_")


def listed(ids):
    """A short list of ids for one line, for example '00012, 00031 and 3 more'."""
    shown = ", ".join(short_id(i) for i in ids[:SHOW])
    return shown + (f" and {len(ids) - SHOW} more" if len(ids) > SHOW else "")


# ---------- part 1: how well does the reader read the questions? (no model, no Qdrant) ----------

def reader_check(cases, filings=None):
    """Count what the reader does with each question. Returns a dict of counts."""
    filings = reader.known_filings() if filings is None else tuple(filings)
    stats = {
        "cases": len(cases),
        "named": {"none": 0, "one": 0, "several": 0},
        "true_named": 0,  # the question names the company of its own filing
        "wrong_filter": [],  # the filter would hide the right company: must stay empty
        "allowed": {mode: 0 for mode in MODES},  # the right filing is still searched
        "sizes": {mode: [] for mode in MODES},  # how many filings are searched
        "lost": [],  # (case id, doc, mode): the right filing was filtered away
        "years": {"none": 0, "one": 0, "several": 0},
    }
    for case in cases:
        true = parse_name(case.doc)
        companies = reader.companies_named(case.question)
        stats["named"]["none" if not companies else "one" if len(companies) == 1 else "several"] += 1
        if companies and true.company in companies:
            stats["true_named"] += 1
        if companies and true.company not in companies:
            stats["wrong_filter"].append(case.id)
        years = reader.years_named(case.question)
        stats["years"]["none" if not years else "one" if len(years) == 1 else "several"] += 1
        for mode in MODES:
            reading = reader.read_question(case.question, mode=mode, filings=filings)
            allowed = [
                d
                for d in filings
                if (reading.company_filter is None or d.company in reading.company_filter)
                and (reading.year_filter is None or d.year in reading.year_filter)
            ]
            stats["sizes"][mode].append(len(allowed))
            if any(d.name == case.doc for d in allowed):
                stats["allowed"][mode] += 1
            else:
                stats["lost"].append((case.id, case.doc, mode))
    stats["filings"] = len(filings)
    return stats


def print_reader_check(stats):
    n, total = stats["cases"], stats["filings"]
    named = stats["named"]
    print(f"Reader check on {n} questions (the holdout is not used)")
    print(f"  question names no company: {named['none']}   one company: {named['one']}   several: {named['several']}")
    print(f"  the company of the right filing is named: {stats['true_named']} of {n}")
    print(f"  a filter that would hide the right company: {len(stats['wrong_filter'])}   (must be 0)")
    print(f"  years named: none {stats['years']['none']}   one {stats['years']['one']}   several {stats['years']['several']}")
    print(f"  {'mode':14} {'right filing still searched':28} {'filings searched, on average':30} {'only one filing searched'}")
    for mode in MODES:
        sizes = stats["sizes"][mode]
        average = sum(sizes) / len(sizes) if sizes else 0
        right, searched, single = f"{stats['allowed'][mode]} of {n}", f"{average:.1f} of {total}", f"{sum(s == 1 for s in sizes)} of {n}"
        print(f"  {mode:14} {right:28} {searched:30} {single}")
    for case_id, doc, mode in stats["lost"][:SHOW]:
        print(f"  LOST: {short_id(case_id)} {doc} filtered away in mode {mode}")
    for case_id in stats["wrong_filter"][:SHOW]:
        print(f"  WRONG COMPANY: {short_id(case_id)}")
    print()


# ---------- part 2: does search find the evidence page more often? ----------

def first_place(hits, pages):
    """(place of the first hit that is an evidence page, or None; how many different evidence pages are in the hits)."""
    wanted = set(pages)
    place = next((n for n, hit in enumerate(hits, start=1) if (hit.doc, hit.page) in wanted), None)
    return place, len(wanted & {(hit.doc, hit.page) for hit in hits})


def run_modes(cases, search_fn, modes=MODES, collection=None):
    """Search every case in every mode. Returns {mode: [(place, evidence pages found), ...]} in case order."""
    results = {}
    for mode in modes:
        marks = []
        for case in cases:
            hits = search_fn(case.question, mode=mode, collection=collection)
            marks.append(first_place(hits, case.pages))
        results[mode] = marks
    return results


def summarize(cases, results):
    """Counts for each mode. Returns ({mode: counts}, number of questions that need more than one page)."""
    several = [i for i, case in enumerate(cases) if len(set(case.pages)) > 1]
    summary = {}
    for mode, marks in results.items():
        summary[mode] = {
            "top5": sum(place is not None for place, _ in marks),
            "first": sum(place == 1 for place, _ in marks),
            "all_pages": sum(marks[i][1] == len(set(cases[i].pages)) for i in several),
        }
    return summary, len(several)


def flips(cases, results, before, after):
    """Ids of the questions that 'after' finds and 'before' missed (gained), and the other way round (lost)."""
    gained, lost = [], []
    for case, was, now in zip(cases, results[before], results[after]):
        if was[0] is None and now[0] is not None:
            gained.append(case.id)
        if was[0] is not None and now[0] is None:
            lost.append(case.id)
    return gained, lost


def print_search_report(cases, results):
    summary, several = summarize(cases, results)
    n = len(cases)
    print(f"Search on {n} questions (the holdout is not used). Top {config.TOP_K} pieces for each question.")
    print(f"  {'mode':14} {'evidence page in top 5':24} {'in first place':16} {'all pages found (questions with several pages)'}")
    for mode, counts in summary.items():
        top5, first, every = f"{counts['top5']} of {n}", f"{counts['first']} of {n}", f"{counts['all_pages']} of {several}"
        print(f"  {mode:14} {top5:24} {first:16} {every}")
    print()
    by_id = {case.id: case for case in cases}
    for before, after in (("off", "company"), ("company", "company+year")):
        if before not in results or after not in results:
            continue
        gained, lost = flips(cases, results, before, after)
        print(f"{before} -> {after}: found now {len(gained)} more, lost {len(lost)}")
        if gained:
            print(f"  gained: {listed(gained)}")
        for case_id in lost[:SHOW]:
            case = by_id[case_id]
            reading = reader.read_question(case.question, mode=after)
            print(f"  lost: {short_id(case_id)} right filing {case.doc}; the filter was company {reading.company_filter}, year {reading.year_filter}")
    best = max(summary, key=lambda mode: (summary[mode]["top5"], -MODES.index(mode)))
    print()
    print(f"Most evidence pages in the top {config.TOP_K}: {best} ({summary[best]['top5']} of {n}); on a tie the simpler mode wins.")
    print("To make the bot use a mode, set FILTER_MODE in config.py to it. Send me this screen first.")


def cached_embedder():
    """The question embedder, remembering answers: each question is embedded once for all 3 modes."""
    from ingest.embed import embed_texts

    memory = {}

    def embed(texts):
        key = tuple(texts)
        if key not in memory:
            memory[key] = embed_texts(list(texts))
        return memory[key]

    return embed


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Does reading the company and year out of the question help search? (turn 6)")
    parser.add_argument("--reader-only", action="store_true", help="only the reader check (no model, no Qdrant)")
    parser.add_argument("--first", type=int, metavar="N", help="only the first N of the 85 questions")
    parser.add_argument("--collection", help="index to test (default from config.py)")
    args = parser.parse_args(argv)

    try:
        cases = usable_cases(args.first)
    except CaseError as err:
        sys.exit(str(err))
    print_reader_check(reader_check(cases))
    if args.reader_only:
        return

    from retrieval.search import search

    search_fn = functools.partial(search, embed=cached_embedder())
    print(f"Searching {len(cases)} questions in {len(MODES)} modes. The first search loads the model, please wait ...\n")
    started = time.time()
    try:
        results = run_modes(cases, search_fn, collection=args.collection)
    except RuntimeError as err:
        message = str(err)
        if "index" in message.lower() and "year" in message.lower():
            print(message)
            sys.exit("The year filter needs an index on the year field. Run this once:  python scripts\\add_year_index.py")
        raise
    print_search_report(cases, results)
    print(f"{len(cases) * len(MODES)} searches in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
