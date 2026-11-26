"""Step 1.5: check that search finds the evidence page.

From the project folder:
    python scripts\\check_index.py              the first 10 questions: top 3 pieces, and is the evidence page in the top 5?
    python scripts\\check_index.py --pages      page number test: does the question file count pages from 0 or from 1? (needs only the PDFs)
    python scripts\\check_index.py --first 103  a hit rate over more questions, one line each
    python scripts\\check_index.py --collection filings_v1_plain   test another index (experiment 3)
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from ingest.read_pdf import read_pdf  # noqa: E402

PASS_AT = 4  # the plan: at least 4 of the first 10 questions find the evidence page in the top 5
PAGE_TEST_QUESTIONS = 5  # questions used by --pages
MIN_SHARE = 0.3  # a page counts as found when it holds at least this part of the evidence text


def load_questions():
    """The questions that point to our 37 PDFs, in file order. The first 10 are the practice questions."""
    if not config.QUESTIONS_FILE.exists():
        sys.exit(f"{config.QUESTIONS_FILE} is missing. Run:  python scripts\\download_pdfs.py")
    names = {line.strip() for line in config.PDF_LIST.read_text(encoding="utf-8").splitlines() if line.strip()}
    questions = []
    for line in config.QUESTIONS_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if item["doc_name"] in names:
                questions.append(item)
    return questions


def evidence_pages(question):
    """The evidence of a question as (doc, page) pairs, with our page numbers, which start at 1."""
    return [(item["doc_name"], item["evidence_page_num"] + config.PAGE_OFFSET) for item in question["evidence"]]


# ---------- page number test ----------

def word_groups(text, size=4):
    """All groups of 4 words that follow each other in the text. Used to match two texts from different PDF readers."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


def page_groups(pages):
    """The word groups of every page of one PDF: {page number: set of groups}."""
    return {page.number: word_groups(page.text) for page in pages}


def find_page(evidence_text, groups):
    """The page that holds the evidence text. Returns (page number, share), or (None, share) if no page is close.
    The share is the part of the evidence text found on that page, from 0 to 1."""
    wanted = word_groups(evidence_text)
    best_number, best_share = None, 0.0
    for number, found in groups.items():
        share = len(wanted & found) / len(wanted) if wanted else 0.0
        if share > best_share:
            best_number, best_share = number, share
    return (best_number if best_share >= MIN_SHARE else None), best_share


def page_test(count=PAGE_TEST_QUESTIONS):
    """Find where the evidence text really sits in the PDF and compare it with the page number in the question file."""
    offsets, cache = [], {}
    for question in load_questions()[:count]:
        item = question["evidence"][0]
        doc, file_page = item["doc_name"], item["evidence_page_num"]
        if doc not in cache:
            path = config.PDF_DIR / f"{doc}.pdf"
            if not path.exists():
                sys.exit(f"Missing PDF: {path}")
            cache[doc] = page_groups(read_pdf(path))
        number, share = find_page(item["evidence_text"], cache[doc])
        start = f"{question['financebench_id']}  {doc:28} the file says page {file_page:3}"
        if number is None:
            print(f"{start}   text not found on any page")
            continue
        offsets.append(number - file_page)
        print(f"{start}   text is on PDF page {number:3}   difference {number - file_page:+d}")
    print()
    if len(offsets) < 3 or len(set(offsets)) != 1:
        print("No clear answer. Send me the lines above.")
        return 1
    offset = offsets[0]
    print(f"All {len(offsets)} agree: the PDF page is the file's page number {offset:+d}.")
    if config.PAGE_OFFSET == offset:
        print(f"PAGE_OFFSET in config.py is {offset}. Good.")
        return 0
    print(f"Set PAGE_OFFSET = {offset} in config.py (line 37). It is {config.PAGE_OFFSET} now.")
    return 1


# ---------- search check ----------

def evidence_found(question, hits):
    """(place of the first evidence page in the hits, or None; how many different evidence pages are in the hits)."""
    wanted = set(evidence_pages(question))
    place = next((n for n, hit in enumerate(hits, start=1) if (hit.doc, hit.page) in wanted), None)
    return place, len(wanted & {(hit.doc, hit.page) for hit in hits})


def preview(hit, width=80):
    """The start of a piece, without its header line."""
    lines = hit.text.splitlines()
    if len(lines) > 1 and lines[0].endswith(f", page {hit.page}"):
        lines = lines[1:]
    return " ".join(" ".join(lines).split())[:width]


def check(first, collection=None, search_fn=None):
    """Run the first questions through search. Returns (questions where the evidence page is in the top 5, questions run)."""
    if search_fn is None:
        from retrieval.search import search as search_fn  # loads the model on the first call
    questions = load_questions()[:first]
    detail = len(questions) <= 10
    print(f"Searching with {len(questions)} questions. The first search loads the model, please wait ...\n")
    started, found = time.time(), 0
    for number, question in enumerate(questions, start=1):
        hits = search_fn(question["question"], collection=collection)
        place, pages_found = evidence_found(question, hits)
        found += place is not None
        wanted = evidence_pages(question)
        if detail:
            print(f"[{number}/{len(questions)}] {question['financebench_id']}  {question['doc_name']}")
            print("Q: " + " ".join(question["question"].split())[:150])
            print("Evidence: " + ", ".join(f"{doc} p{page}" for doc, page in wanted))
            for rank, hit in enumerate(hits[:3], start=1):
                print(f"  {rank}  {hit.score:.3f}  {hit.doc} p{hit.page}  {preview(hit)}")
            if place is None:
                print(f"-> MISS: no evidence page in the top {len(hits)}\n")
            else:
                more = f" ({pages_found} of {len(wanted)} evidence pages)" if len(wanted) > 1 else ""
                print(f"-> HIT: an evidence page is at place {place}{more}\n")
        else:
            verdict = f"HIT {place}" if place else "MISS "
            print(f"[{number:3}/{len(questions)}] {verdict}  {question['doc_name']:30} {' '.join(question['question'].split())[:60]}")
    print()
    print(f"Evidence page in the top {config.TOP_K}: {found} of {len(questions)}  ({100 * found / max(len(questions), 1):.0f} out of 100)")
    print(f"{len(questions)} searches in {time.time() - started:.0f} s")
    return found, len(questions)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Check that search finds the evidence page (step 1.5).")
    parser.add_argument("--pages", action="store_true", help="page number test (needs only the PDFs)")
    parser.add_argument("--first", type=int, default=10, metavar="N", help="how many questions to run (default 10)")
    parser.add_argument("--collection", help="index to test (default from config.py)")
    args = parser.parse_args(argv)

    if args.pages:
        sys.exit(page_test())
    found, total = check(args.first, args.collection)
    if total == 10:
        if found >= PASS_AT:
            print(f"PASS: the plan needs {PASS_AT} or more.")
            return
        print(f"FEWER THAN {PASS_AT}. Do these in order, and tune nothing else yet:")
        print("  1. python scripts\\check_index.py --pages     (is PAGE_OFFSET right?)")
        print("  2. the header line: python -m ingest.build_index --no-header, then")
        print("     python scripts\\check_index.py --collection filings_v1_plain")
        sys.exit(1)


if __name__ == "__main__":
    main()
