"""Compare Qdrant collections on the 85 practice and test questions. Search only. FREE: the search lines come from the rewrite cache.

    python -m evals.compare_collections filings_v1 filings_v2_noheader filings_v2_large

For each collection it runs the search the bot runs (the rewrite of the question, the company and year filter, the best 5 pieces) and says:
  best 5       the right page is among the best 5 pieces (the measure of experiments 3 to 8)
  bot reads    the right page is among the first 3 different pages of those pieces, the pages the bot reads whole (a closer picture of the answer)
  MRR, first   how high the right page ranks
The model of each collection is found by itself from the size of its vectors (384 numbers = small, 1024 numbers = large).
The holdout is never opened. Claude is never called: a question whose rewrite is not in data\\cache\\rewrite_cache.jsonl stops the run.
The large model (about 1.3 GB) is downloaded only when you add --allow-download.
"""
import argparse
import functools
import json
import sys
from datetime import datetime
from pathlib import Path

import config
from bot import llm, lookup
from evals import experiments
from evals.cases import CaseError
from evals.code_checks import search_marks
from evals.judge import JsonlCache
from ingest.embed import model_on_disk

BY_SIZE = {dim: profile for profile, (_model, dim) in config.EMBED_PROFILES.items()}


class NotCached(RuntimeError):
    """A search line is not in the rewrite cache (making it would cost a little, so this tool does not)."""


def cached_lines(cases, cache, prompt=None, clean=None):
    """{case id: the search line the bot would use}, read from the rewrite cache only."""

    def refuse(system, user, max_tokens=120):
        raise llm.BudgetStop("not in the cache")

    lines, missing = {}, []
    for case in cases:
        try:
            lines[case.id] = lookup.make_queries(case.question, count=1, hints=True, ask_fn=refuse, cache=cache, prompt=prompt, clean=clean).queries[0]
        except llm.BudgetStop:
            missing.append(case.id)
    if missing:
        raise NotCached(
            f"{len(missing)} of {len(cases)} search lines are not in the rewrite cache yet ({experiments.listed(missing)}). "
            "Make them once (about 1 cent):  python -m evals.rewrite_test --new --spend"
        )
    return lines


def profile_of(client, name):
    """small or large, from the number of numbers in the vectors of the collection."""
    size = getattr(client.get_collection(name).config.params.vectors, "size", None)
    if size not in BY_SIZE:
        known = ", ".join(f"{dim} numbers = {profile}" for dim, profile in BY_SIZE.items())
        raise ValueError(f"The collection {name} has vectors of {size} numbers. This project knows: {known}.")
    return BY_SIZE[size]


def score_collection(name, cases, lines, client=None, search_fn=None, profile=None, k=None):
    """Search every case in one collection and count. Returns a dict (see the keys at the end).

    client: a Qdrant client (the profile and the piece count are read from it). search_fn: for tests. lines: {case id: search line}.
    """
    k = k or config.TOP_K
    pieces = None
    if client is not None:
        profile = profile or profile_of(client, name)
        pieces = client.count(collection_name=name, exact=True).count
        if search_fn is None:
            from retrieval import search as search_module

            search_fn = functools.partial(search_module.search, client=client)
    if profile:
        config.use_profile(profile)
    ctx = experiments.Context(name, search_fn=search_fn)
    found5, found3, places = [], [], {}
    for case in cases:
        company, year = ctx.filters(case.question)
        hits = ctx.search(lines[case.id], k, mode="off", company=company, year=year)[:k]
        found5.append(search_marks(case.pages, [(hit.doc, hit.page) for hit in hits]))
        found3.append(search_marks(case.pages, lookup.top_pages(hits, config.PAGE_CONTEXT_PAGES)))
        places[case.id] = found5[-1]["place"]
    n = len(cases)
    return {
        "collection": name,
        "profile": profile or config.EMBED_PROFILE,
        "pieces": pieces,
        "n": n,
        "best5": sum(item["hit"] for item in found5),
        "read3": sum(item["hit"] for item in found3),
        "mrr": round(sum(item["mrr"] for item in found5) / n, 4) if n else 0.0,
        "first": sum(item["place"] == 1 for item in found5),
        "places": places,
    }


def rank_key(item):
    """Better collections sort first: more right pages read by the bot, then more in the best 5, then a higher MRR."""
    return (item["read3"], item["best5"], item["mrr"])


def table_lines(items):
    """The result table as a list of lines, best collection first."""
    ordered = sorted(items, key=rank_key, reverse=True)
    width = max([len(item["collection"]) for item in ordered] + [10])
    lines = [f"  {'collection':{width}} {'model':6} {'pieces':>7}  {'best 5':>9}  {'bot reads':>9}  {'MRR':>6}  {'first':>5}"]
    for item in ordered:
        n = item["n"]
        pieces = f"{item['pieces']:,}" if item.get("pieces") else "?"
        lines.append(
            f"  {item['collection']:{width}} {item['profile']:6} {pieces:>7}  {str(item['best5']) + ' of ' + str(n):>9}  {str(item['read3']) + ' of ' + str(n):>9}  {item['mrr']:6.3f}  {item['first']:>5}"
        )
    return lines


def main(argv=None, client=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Compare Qdrant collections on the 85 practice and test questions (search only, free).")
    parser.add_argument("collections", nargs="+", help="the collections to compare, for example filings_v1 filings_v2_large")
    parser.add_argument("--prompt", choices=lookup.REWRITE_PROMPTS, help="rewrite request (default: REWRITE_PROMPT in config.py)")
    parser.add_argument("--clean", choices=("on", "off"), help="take the noise out of the search lines (default: QUERY_CLEAN in config.py)")
    parser.add_argument("--first", type=int, metavar="N", help="only the first N of the 85 questions (a quick try)")
    parser.add_argument("--allow-download", action="store_true", help="allow the download of the large model (about 1.3 GB) when it is not on this computer")
    args = parser.parse_args(argv)
    try:
        cases = experiments.usable_cases(args.first)
    except CaseError as err:
        sys.exit(str(err))
    print("COLAB: no | MODEL: small or large, found from each collection | CLAUDE COST: 0 cents (the search lines come from the rewrite cache)")
    if client is None:
        from retrieval.search import get_client

        client = get_client()
    cache = JsonlCache(Path(config.DATA_DIR) / "cache" / "rewrite_cache.jsonl")
    try:
        lines = cached_lines(cases, cache, args.prompt, None if args.clean is None else args.clean == "on")
    except NotCached as err:
        sys.exit(str(err))
    results = []
    for name in args.collections:
        try:
            profile = profile_of(client, name)
        except Exception as err:  # noqa: BLE001 - say which collection and why
            sys.exit(f"Cannot read the collection {name}: {type(err).__name__}: {str(err)[:200]}")
        model = config.EMBED_PROFILES[profile][0]
        if profile == "large" and not args.allow_download and not model_on_disk(model):
            sys.exit(f"{name} was made with the large model ({model}, about 1.3 GB), which is not on this computer yet. Add --allow-download to download it once.")
        print(f"Searching {name} with the {profile} model ...", flush=True)
        results.append(score_collection(name, cases, lines, client=client, profile=profile))
    print(f"\n{len(cases)} questions (practice and test; the holdout is not used). Search only, filter {config.FILTER_MODE}.\n")
    print("\n".join(table_lines(results)))
    base = results[0]
    print(f"\nCompared with {base['collection']} (the first one you named): change in questions where the bot reads the right page")
    for item in results[1:]:
        print(f"  {item['collection']}: {item['read3'] - base['read3']:+d}   (best 5: {item['best5'] - base['best5']:+d})")
    print(f"A change of 1 or 2 questions out of {len(cases)} can be luck. 3 or more is worth a look.")
    folder = Path(config.RESULTS_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "compare_collections.json"
    path.write_text(json.dumps({"created": datetime.now().isoformat(timespec="seconds"), "questions": len(cases), "results": results}, indent=1), encoding="utf-8")
    print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
