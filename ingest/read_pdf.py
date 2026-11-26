"""Step 1.2: read the pages of a PDF.

Every page becomes a list of text lines. A table line stays on one line, with " | "
between the columns, for example:

    Total assets | $36,500 | $37,987

PyMuPDF has a table finder. We tested it on a real balance sheet: it cut words apart
("Totalassets") and took 3 seconds a page. So the lines are rebuilt from the word
positions instead. That is clean and takes about 15 seconds for all 37 PDFs.

Check it from the project folder:
    python -m ingest.read_pdf                    read all PDFs, print counts
    python -m ingest.read_pdf --look             print 3 table pages to look at by eye
    python -m ingest.read_pdf --show NAME PAGE   print one page
    python -m ingest.read_pdf --find NAME "text" list the pages that contain the text
"""
import argparse
import sys
import time
from dataclasses import dataclass

import pymupdf

import config
from ingest.names import DocInfo, parse_name

COLUMN_GAP = 10.0  # points. A bigger gap between two words starts a new column.
MIN_CHARS = 100  # a page with fewer letters and digits than this is marked no_text
NOISE_LINES = {"table of contents"}  # a link line at the top of many pages
SIGNS = {"$", "(", "$(", "($", "€", "£"}  # stick to the word after them
ENDINGS = {")", "%", ")%", "%)", "*"}  # stick to the word before them
BULLETS = {"·", "•", "●", "■", "▪", "○", "◦"}

# Three pages to look at by eye (step 1.2): balance sheet, income statement, segment table.
LOOK_PAGES = [
    ("3M_2018_10K", 58, "balance sheet"),
    ("3M_2018_10K", 56, "income statement"),
    ("3M_2018_10K", 126, "segment table"),
]


@dataclass
class Page:
    number: int  # starts at 1
    lines: list
    no_text: bool  # fewer than MIN_CHARS letters and digits

    @property
    def text(self):
        return "\n".join(self.lines)


@dataclass
class Document:
    info: DocInfo
    path: object  # a Path
    source: str = "real"  # real or fake (step 3.1 adds the fake files)
    access: str = "public"


def is_table_line(line):
    """A line with columns and at least one number."""
    return " | " in line and any(char.isdigit() for char in line)


def _join_line(words):
    """Join the words of one line. A big gap becomes ' | '."""
    words = sorted(words, key=lambda word: word[0])
    text = words[0][4]
    previous = words[0]
    for word in words[1:]:
        piece = word[4]
        if previous[4] in SIGNS or piece in ENDINGS:
            separator = ""
        elif previous[4] in BULLETS:
            separator = " "
        elif piece == "$" or word[0] - previous[2] > COLUMN_GAP:
            separator = " | "
        else:
            separator = " "
        text += separator + piece
        previous = word
    return text


def _page_lines(page):
    """Group the words of a page into lines by their height on the page."""
    words = [w for w in page.get_text("words") if w[4].strip()]
    words.sort(key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    bands = []  # each band is [average middle height, words]
    for word in words:
        middle = (word[1] + word[3]) / 2
        height = word[3] - word[1]
        if bands and abs(bands[-1][0] - middle) <= max(2.0, 0.45 * height):
            band = bands[-1]
            band[1].append(word)
            band[0] += (middle - band[0]) / len(band[1])
        else:
            bands.append([middle, [word]])
    lines = []
    for _, band_words in bands:
        line = _join_line(band_words).replace(" ", " ").strip()
        if line and line.lower() not in NOISE_LINES:
            lines.append(line)
    return lines


def read_pdf(path, only_pages=None):
    """Read a PDF. Returns a list of Page. only_pages is an optional set of page numbers."""
    pages = []
    with pymupdf.open(path) as doc:
        for number, page in enumerate(doc, start=1):
            if only_pages is not None and number not in only_pages:
                continue
            lines = _page_lines(page)
            letters = sum(1 for line in lines for char in line if char.isalnum())
            pages.append(Page(number, lines, letters < MIN_CHARS))
    return pages


def list_documents(only=None):
    """The documents to index: the real PDFs, then the 20 fake files of step 3.1 (when scripts\\make_fake_pdfs.py has made them)."""
    if not config.PDF_LIST.exists():
        sys.exit(f"{config.PDF_LIST} is missing. Make data/pdf_list.txt first (step 1.1).")
    from security.fakes import fake_documents

    names = [line.strip() for line in config.PDF_LIST.read_text(encoding="utf-8").splitlines() if line.strip()]
    fakes = fake_documents()
    if only:
        wanted = {name.lower() for name in only}
        known = {name.lower() for name in names} | {document.info.name.lower() for document in fakes}
        if wanted - known:
            sys.exit(f"Name not found in data/pdf_list.txt or among the fake files: {sorted(wanted - known)}")
        names = [name for name in names if name.lower() in wanted]
        fakes = [document for document in fakes if document.info.name.lower() in wanted]
    documents = []
    for name in names:
        path = config.PDF_DIR / f"{name}.pdf"
        if not path.exists():
            sys.exit(f"Missing PDF: {path}\nRun:  python scripts\\download_pdfs.py")
        documents.append(Document(parse_name(name), path))
    return documents + fakes


def _show(name, number):
    path = config.PDF_DIR / f"{name}.pdf"
    if not path.exists():
        sys.exit(f"Missing PDF: {path}")
    pages = read_pdf(path, only_pages={number})
    if not pages:
        sys.exit(f"{name} has no page {number}.")
    print(f"----- {name}, page {number}, {len(pages[0].lines)} lines, no_text={pages[0].no_text} -----")
    print(pages[0].text)


def _find(name, needle):
    path = config.PDF_DIR / f"{name}.pdf"
    if not path.exists():
        sys.exit(f"Missing PDF: {path}")
    found = 0
    for page in read_pdf(path):
        for line in page.lines:
            if needle.lower() in line.lower():
                print(f"page {page.number}: {line[:110]}")
                found += 1
                break
    print(f"{found} pages contain '{needle}'.")


def _read_all():
    started = time.time()
    total_pages = total_blank = 0
    documents = list_documents()
    for document in documents:
        pages = read_pdf(document.path)
        blank = sum(page.no_text for page in pages)
        table_lines = sum(is_table_line(line) for page in pages for line in page.lines)
        total_pages += len(pages)
        total_blank += blank
        print(f"OK  {document.info.name:42} {len(pages):4} pages  no_text {blank:3}  table lines {table_lines:6,}")
    seconds = time.time() - started
    print()
    print(f"Read {len(documents)} PDFs: {total_pages:,} pages, {total_blank} no_text "
          f"({100 * total_blank / total_pages:.1f} per 100 pages), {seconds:.0f} s.")
    if total_pages != 4802:
        print("The plan says 4,802 pages. Run scripts\\download_pdfs.py and send me the numbers above.")
    print("Next: python -m ingest.read_pdf --look   (then write one line about how the tables came out)")


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Read the PDFs (step 1.2).")
    parser.add_argument("--look", action="store_true", help="print 3 table pages")
    parser.add_argument("--show", nargs=2, metavar=("NAME", "PAGE"), help="print one page")
    parser.add_argument("--find", nargs=2, metavar=("NAME", "TEXT"), help="list pages that contain TEXT")
    args = parser.parse_args(argv)
    if args.show:
        _show(args.show[0], int(args.show[1]))
    elif args.find:
        _find(args.find[0], args.find[1])
    elif args.look:
        for name, number, what in LOOK_PAGES:
            print(f"\n===== {what} =====")
            _show(name, number)
    else:
        _read_all()


if __name__ == "__main__":
    main()
