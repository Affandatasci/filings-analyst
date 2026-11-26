"""Step 1.3: cut pages into pieces.

The rules (from the plan):
  - target 400 tokens, overlap 50, hard stop 480 (counted with the bge tokenizer)
  - a piece never crosses a page, and a page that fits in 480 tokens stays whole
  - every piece starts with a header line: company, year, type, page
  - a long table is split by lines, and each part repeats the table header
    (the column line with the years, for example "(Millions) | 2018 | 2017")

Check it from the project folder:
    python -m ingest.chunker                     cut all PDFs, print the piece count and the largest piece
    python -m ingest.chunker --show NAME PAGE    print the pieces of one page
    python -m ingest.chunker --no-header         the same without the header line (experiment 3, later)
"""
import argparse
import re
import sys
import time
from dataclasses import dataclass

import config
from ingest.read_pdf import is_table_line, list_documents, read_pdf

OVERLAP_TOKENS = 50  # the last lines of a piece start the next piece
MAX_LINE_TOKENS = 150  # a longer line is split by words
MIN_TAIL_TOKENS = 60  # a last piece with fewer new tokens joins the piece before it
HEAD_MAX_TOKENS = 80  # the repeated table header must fit in this
HEAD_SEARCH_LINES = 400  # how far back to look for the column line
HEADER_CELL = re.compile(r"^(?:(?:[A-Za-z]+\.? \d{1,2}, |Q[1-4] ?|FY ?|Fiscal )?(?:199\d|20[0-3]\d)|Q[1-4])$")


@dataclass
class Piece:
    doc: str
    page: int
    piece: int  # 1, 2, 3 ... inside the page
    text: str
    tokens: int


def is_column_line(line):
    """A table line that names the columns, for example '(Millions) | 2018 | 2017'."""
    cells = [cell.strip() for cell in line.split(" | ")]
    return len(cells) >= 2 and sum(bool(HEADER_CELL.match(cell)) for cell in cells) >= 2


def _split_long_lines(lines, counts, count):
    """Split a very long line by words, so every line is at most MAX_LINE_TOKENS."""
    new_lines, new_counts = [], []
    for line, tokens in zip(lines, counts):
        if tokens <= MAX_LINE_TOKENS:
            new_lines.append(line)
            new_counts.append(tokens)
            continue
        words = []
        for word in line.split():
            words.extend(word[i : i + 60] for i in range(0, len(word), 60))
        current, used = [], 0
        for word, size in zip(words, count(words)):
            if current and used + size > MAX_LINE_TOKENS:
                new_lines.append(" ".join(current))
                new_counts.append(used)
                current, used = [], 0
            current.append(word)
            used += size
        if current:
            new_lines.append(" ".join(current))
            new_counts.append(used)
    return new_lines, new_counts


def _table_head(lines, counts, first):
    """Line numbers of the table header to repeat when a piece starts inside a table."""
    if sum(is_table_line(line) for line in lines[first : first + 8]) < 2:
        return ()
    if any(is_column_line(line) for line in lines[first : first + 4]):
        return ()  # this piece brings its own column line
    stop = max(-1, first - 1 - HEAD_SEARCH_LINES)
    for found in range(first - 1, stop, -1):
        if not is_column_line(lines[found]):
            continue
        between = lines[found + 1 : first]
        if sum(is_table_line(line) for line in between) * 2 < len(between):
            return ()  # too much other text in between: not the same table
        head = tuple(range(max(0, found - 2), found + 1))
        if sum(counts[i] for i in head) <= HEAD_MAX_TOKENS:
            return head
        return (found,) if counts[found] <= HEAD_MAX_TOKENS else ()
    return ()


def _plan(lines, counts, fixed, target, limit):
    """Decide which lines go into which piece.

    Returns a list of (first, start, end, head): lines first..end-1 are in the piece,
    lines first..start-1 are the overlap, head holds the repeated table header lines.
    """
    n = len(lines)
    if fixed + sum(counts) <= limit:
        return [(0, 0, n, ())]
    plans = []
    start = 0
    while start < n:
        first, head = start, ()
        if plans:
            used = 0
            while first > plans[-1][1] and used + counts[first - 1] <= OVERLAP_TOKENS:
                first -= 1
                used += counts[first]
            head = _table_head(lines, counts, first)
        room = target - fixed - sum(counts[i] for i in head) - sum(counts[first:start])
        end, used = start, 0
        while end < n and used + counts[end] <= room:
            used += counts[end]
            end += 1
        end = max(end, start + 1)  # always move forward
        plans.append((first, start, end, head))
        start = end
    if len(plans) > 1:  # a tiny last piece joins the one before it
        first, start, end, head = plans[-1]
        if sum(counts[start:end]) < MIN_TAIL_TOKENS:
            plans.pop()
            before = plans[-1]
            plans[-1] = (before[0], before[1], n, before[3])
    return plans


def _compose(header, lines, plan):
    first, _, end, head = plan
    parts = [header] if header else []
    parts.extend(lines[i] for i in head)
    parts.extend(lines[first:end])
    return "\n".join(parts)


def cut_page(info, page, count, with_header=True, piece_tokens=None):
    """Cut one page into pieces. count(list of texts) returns the token count of each text.

    piece_tokens: the size to aim for (default config.PIECE_TOKENS, 400; allowed 150 to the hard stop). The hard stop stays.
    A page that fits in 1.2 times that size stays whole (480 for the default size).
    """
    target, stop = int(piece_tokens or config.PIECE_TOKENS), config.PIECE_HARD_STOP
    if not 150 <= target <= stop:
        raise ValueError(f"piece_tokens must be between 150 and {stop}, not {target}.")
    whole = stop if not piece_tokens else min(stop, round(target * 1.2))
    header = info.header(page.number) if with_header else ""
    sizes = count(([header] if header else []) + page.lines)
    fixed = sizes[0] if header else 0
    lines, counts = _split_long_lines(page.lines, sizes[1:] if header else sizes, count)
    for attempt in range(6):  # the check below should always pass at the first try
        shrink = 0.9**attempt
        plans = _plan(lines, counts, fixed, int(target * shrink), int(whole * shrink))
        texts = [_compose(header, lines, plan) for plan in plans]
        real = count(texts)
        if max(real) <= config.PIECE_HARD_STOP:
            return [Piece(info.name, page.number, k, text, size) for k, (text, size) in enumerate(zip(texts, real), 1)]
    raise RuntimeError(f"{info.name} page {page.number}: a piece stays above {config.PIECE_HARD_STOP} tokens.")


def cut_document(info, pages, count, with_header=True, piece_tokens=None):
    """Cut all pages of one document. Pages marked no_text give no pieces."""
    pieces = []
    for page in pages:
        if not page.no_text:
            pieces.extend(cut_page(info, page, count, with_header, piece_tokens))
    return pieces


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Cut the PDFs into pieces (step 1.3).")
    parser.add_argument("--no-header", action="store_true", help="no header line at the top of each piece")
    parser.add_argument("--piece-tokens", type=int, metavar="N", help="aim for pieces of N tokens (default 400, allowed 150 to 480)")
    parser.add_argument("--only", nargs="+", metavar="NAME", help="only these documents")
    parser.add_argument("--show", nargs=2, metavar=("NAME", "PAGE"), help="print the pieces of one page")
    args = parser.parse_args(argv)

    from ingest.embed import count_tokens  # loads the bge tokenizer (small download the first time)

    with_header = not args.no_header
    if args.show:
        name, number = args.show[0], int(args.show[1])
        document = list_documents([name])[0]
        pages = read_pdf(document.path, only_pages={number})
        if not pages or pages[0].no_text:
            sys.exit(f"{name} page {number} is missing or has no text.")
        for piece in cut_page(document.info, pages[0], count_tokens, with_header, args.piece_tokens):
            print(f"\n----- piece {piece.piece}, {piece.tokens} tokens -----")
            print(piece.text)
        return

    started = time.time()
    documents = list_documents(args.only)
    total, largest, tokens_sum = 0, (0, ""), 0
    for document in documents:
        t0 = time.time()
        pages = read_pdf(document.path)
        pieces = cut_document(document.info, pages, count_tokens, with_header, args.piece_tokens)
        top = max(piece.tokens for piece in pieces)
        total += len(pieces)
        tokens_sum += sum(piece.tokens for piece in pieces)
        if top > largest[0]:
            largest = (top, document.info.name)
        print(f"OK  {document.info.name:42} {len(pages):4} pages {len(pieces):5} pieces  largest {top:3}  {time.time() - t0:4.1f} s")
    print()
    print(f"Pieces: {total:,}   (the plan says about 11,100, range 10,000 to 12,500)")
    print(f"Largest piece: {largest[0]} tokens in {largest[1]}   (hard stop {config.PIECE_HARD_STOP})")
    print(f"Average piece: {tokens_sum / total:.0f} tokens")
    print(f"Time: {time.time() - started:.0f} s")
    if largest[0] > config.PIECE_HARD_STOP:
        sys.exit("A piece is above the hard stop. Send me the lines above.")
    if len(documents) == 37 and not 10000 <= total <= 12500:
        print("The count is outside the plan range. Send me the numbers above.")


if __name__ == "__main__":
    main()
