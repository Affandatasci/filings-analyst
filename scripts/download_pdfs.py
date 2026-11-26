"""Step 1.1: make sure the 37 PDFs and the question file are on disk, then check them.

Looks in the PDF folder set in config.py. Downloads only what is missing.
Run from the project folder:  python scripts\\download_pdfs.py
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf  # noqa: E402

import config  # noqa: E402

BASE = "https://raw.githubusercontent.com/patronus-ai/financebench/main"
EXPECTED_PDFS = 37
EXPECTED_PAGES = 4802
EXPECTED_QUESTIONS = 103


def fetch(url, target):
    """Download one file. Nothing is written if the download fails."""
    request = urllib.request.Request(url, headers={"User-Agent": "filings-analyst"})
    with urllib.request.urlopen(request, timeout=120) as reply:
        data = reply.read()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def count_pages(path):
    with pymupdf.open(path) as doc:
        return doc.page_count


def main():
    if not config.PDF_LIST.exists():
        sys.exit(f"{config.PDF_LIST} is missing. Make data/pdf_list.txt first.")
    names = [line.strip() for line in config.PDF_LIST.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"{len(names)} names in {config.PDF_LIST.name}")
    print(f"PDF folder: {config.PDF_DIR}")
    config.PDF_DIR.mkdir(parents=True, exist_ok=True)

    problems = []
    ok = 0
    total_pages = 0
    total_bytes = 0
    for name in names:
        path = config.PDF_DIR / f"{name}.pdf"
        note = "already there"
        if not path.exists():
            try:
                fetch(f"{BASE}/pdfs/{name}.pdf", path)
                note = "downloaded"
            except (urllib.error.URLError, OSError) as err:
                problems.append(f"{name}: not on disk and the download failed ({err})")
                print(f"MISSING  {name}")
                continue
        try:
            pages = count_pages(path)
        except Exception as err:  # a broken or half written file
            problems.append(f"{name}: the file will not open ({err}). Delete it and run again.")
            print(f"BROKEN   {name}")
            continue
        size = path.stat().st_size
        ok += 1
        total_pages += pages
        total_bytes += size
        print(f"OK  {name:42} {pages:4} pages {size / 1e6:6.1f} MB  {note}")

    if not config.QUESTIONS_FILE.exists():
        try:
            fetch(f"{BASE}/data/financebench_open_source.jsonl", config.QUESTIONS_FILE)
            print("Question file downloaded.")
        except (urllib.error.URLError, OSError) as err:
            problems.append(f"question file: download failed ({err})")
    total_questions = 0
    matching = 0
    if config.QUESTIONS_FILE.exists():
        wanted = set(names)
        for line in config.QUESTIONS_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                total_questions += 1
                if json.loads(line)["doc_name"] in wanted:
                    matching += 1

    print()
    print(f"PDFs ok: {ok} of {len(names)} | pages: {total_pages:,} | size: {total_bytes / 1e6:.0f} MB")
    print(f"Questions in the file: {total_questions} | pointing to these PDFs: {matching}")
    if problems:
        print("\nProblems:")
        for problem in problems:
            print(" -", problem)
        sys.exit(1)
    if (ok, total_pages, matching) != (EXPECTED_PDFS, EXPECTED_PAGES, EXPECTED_QUESTIONS):
        print(
            f"\nThe counts differ from the plan ({EXPECTED_PDFS} PDFs, {EXPECTED_PAGES:,} pages, "
            f"{EXPECTED_QUESTIONS} questions). Send me the numbers above."
        )
        sys.exit(1)
    print("\nAll good: 37 PDFs, 4,802 pages, 103 questions.")


if __name__ == "__main__":
    main()
