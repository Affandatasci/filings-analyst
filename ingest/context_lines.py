"""Context lines: one short line for every page that says what the page is, written once by a small open language model.

Why: a piece of a table ("Net sales | 32,765 | 32,136") does not say which statement it comes from. A line such as
"Consolidated statement of income of 3M for 2018, in millions: net sales, cost of sales, operating income, net income"
put in front of every piece of that page, when the piece is embedded, helps the search to find the right page.
The stored text of the piece does not change, so the bot reads the same pages as before. Only the vector changes.

The model is Qwen2.5 1.5B Instruct (Apache 2.0, about 3 GB). It runs on the Colab GPU (about 20 to 30 minutes for all pages) and costs
nothing. The lines are kept in a file that is only added to (JSON lines), so a stopped run continues where it stopped.

From the project folder (a laptop without a GPU is too slow for all pages: use the Colab notebook):
    python -m ingest.context_lines --out data\\cache\\context_lines.jsonl --only 3M_2018_10K --limit 20 --allow-download     a quick try
    python -m ingest.context_lines --out data\\cache\\context_lines.jsonl --allow-download                                   all pages
Without --allow-download nothing is downloaded and the run only says what it would do.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PAGE_CHARS = 3500  # the model reads the first part of a page: its name and its first items are there
LINE_CHARS = 200  # a longer line is cut at a word
MIN_LINE_CHARS = 12  # a shorter reply is not a line
BATCH = 16  # pages in one model call

SYSTEM = (
    "You write one search label for a page of a company filing (10-K, 10-Q, 8-K or earnings release). "
    "Say in ONE line of at most 30 words what the page is: the name of the statement or section "
    "(for example consolidated statements of income, balance sheet, cash flows, segment note, debt note, legal proceedings, risk factors, management discussion), "
    "the period and the unit when they show (for example fiscal 2018, in millions), and the main items it lists. "
    "Use the words of the page. Reply with the line only: no quotes, no label, no explanation."
)
USER = "Filing: {header}\n\nPage text:\n{text}\n\nThe one line:"
QUOTES = " \"'" + "".join(chr(code) for code in (0x201C, 0x201D, 0x2018, 0x2019))  # plain and curly quotes
LABEL = re.compile(r"^(?:context|line|label|summary|description|answer|page)\s*[:\-]\s*", re.IGNORECASE)


def load_lines(path):
    """{(doc, page): line} from the JSON lines file. A missing file gives an empty answer. A half written last line is ignored."""
    found = {}
    path = Path(path)
    if not path.exists():
        return found
    for text in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(text)
            found[(item["doc"], int(item["page"]))] = str(item["line"])
        except (ValueError, KeyError, TypeError):
            continue
    return found


def append_lines(path, items):
    """Add (doc, page, line) items to the file. Only ever adds, so a stop in the middle loses nothing that was written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for doc, page, line in items:
            handle.write(json.dumps({"doc": doc, "page": int(page), "line": line}) + "\n")
        handle.flush()


def clean_line(text, limit=LINE_CHARS):
    """The first usable line of a reply: no markdown, no label such as 'Context:', no quotes, cut at a word. Empty when there is none."""
    for raw in str(text or "").splitlines():
        line = " ".join(re.sub(r"[`*#>]+", " ", raw).split()).strip(QUOTES)
        line = LABEL.sub("", line).strip(QUOTES)
        if len(line) >= MIN_LINE_CHARS:
            return line if len(line) <= limit else line[:limit].rsplit(" ", 1)[0]
    return ""


def title_line(lines, limit=LINE_CHARS):
    """A line made without the model: the first lines of the page that look like a title. Used when the model gives nothing usable."""
    picked = []
    for line in lines[:14]:
        text = " ".join(line.split())
        if sum(char.isalpha() for char in text) >= 4 and len(text) <= 120:
            picked.append(text)
        if len(picked) == 3:
            break
    return " - ".join(picked)[:limit]


def page_prompt(header, lines, max_chars=PAGE_CHARS):
    """(system, user) for one page: the first max_chars of its text, cut at a line."""
    kept, used = [], 0
    for line in lines:
        if kept and used + len(line) > max_chars:
            break
        kept.append(line[:max_chars])
        used += len(line) + 1
    return SYSTEM, USER.format(header=header, text="\n".join(kept))


def ask_safely(ask, prompts):
    """ask(prompts) for a list of prompts. When the GPU memory is full, the list is split in two and each half is asked again."""
    try:
        return list(ask(prompts))
    except Exception as err:  # noqa: BLE001 - only a memory problem is handled here
        if "out of memory" not in str(err).lower() or len(prompts) < 2:
            raise
        middle = len(prompts) // 2
        return ask_safely(ask, prompts[:middle]) + ask_safely(ask, prompts[middle:])


def generate_lines(documents, ask, path, batch=BATCH, max_chars=PAGE_CHARS, limit=None, progress=print, checkpoint=None, every=300):
    """Write a context line for every page with text that has none in the file yet. Returns the numbers.

    ask(list of (system, user)) -> list of replies, one for each prompt (hf_asker makes one for the GPU).
    The fake files are skipped (they get no line). Longest pages go first, so a memory problem shows at once.
    A page with nothing usable (the model gave nothing and the page has no title lines) gets an empty line in the file: it counts as tried,
    so it is not tried again, and its pieces are built without a context line.
    checkpoint(): called about every `every` seconds, for example to copy the file to Google Drive.
    """
    from ingest.read_pdf import read_pdf

    have = load_lines(path)
    todo = []
    for document in documents:
        if document.source == "fake":
            continue
        for page in read_pdf(document.path):
            if page.no_text or (document.info.name, page.number) in have:
                continue
            todo.append((document.info.name, page.number, page_prompt(document.info.header(page.number), page.lines, max_chars), title_line(page.lines)))
    if limit:
        todo = todo[:limit]
    todo.sort(key=lambda item: -len(item[2][1]))
    started = last = time.time()
    done, fallbacks, empty = 0, 0, 0
    progress(f"Context lines: {len(have):,} already in the file, {len(todo):,} pages to do.")
    for start in range(0, len(todo), batch):
        part = todo[start : start + batch]
        replies = ask_safely(ask, [item[2] for item in part])
        items = []
        for (doc, page, _, fallback), reply in zip(part, replies):
            line = clean_line(reply)
            if not line:
                line, fallbacks = fallback, fallbacks + 1
            empty += not line
            items.append((doc, page, line))
        append_lines(path, items)
        done += len(part)
        if (start // batch) % 10 == 0 or done == len(todo):
            spent = time.time() - started
            left = spent / done * (len(todo) - done)
            progress(f"  {done:,} of {len(todo):,} pages, {spent / 60:.1f} min used, about {left / 60:.1f} min to go")
        if checkpoint and time.time() - last >= every:
            checkpoint()
            last = time.time()
    if checkpoint:
        checkpoint()
    return {"made": done, "already": len(have), "fallbacks": fallbacks, "empty": empty, "total": len(have) + done, "seconds": time.time() - started}


def hf_asker(model_name=MODEL, max_new_tokens=60, device=None):
    """ask(list of (system, user)) -> list of replies, with a small model of Hugging Face on the GPU (or the CPU: slow). Loads the model now."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    kind = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=kind)
    except TypeError:  # an older transformers calls it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=kind)
    model.to(device).eval()

    def ask(prompts):
        texts = [
            tokenizer.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
            for system, user in prompts
        ]
        encoded = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tokenizer.pad_token_id
            )
        return tokenizer.batch_decode(out[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True)

    ask.model, ask.tokenizer = model, tokenizer
    return ask


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Write one context line for every page of the filings (needs a GPU for all pages).")
    parser.add_argument("--out", required=True, help="the JSON lines file to add the lines to")
    parser.add_argument("--only", nargs="+", metavar="NAME", help="only these documents")
    parser.add_argument("--limit", type=int, metavar="N", help="only the first N pages that still need a line")
    parser.add_argument("--model", default=MODEL, help=f"the model (default {MODEL})")
    parser.add_argument("--allow-download", action="store_true", help="allow the download of the model (about 3 GB) when it is not on this computer yet")
    args = parser.parse_args(argv)

    from ingest.read_pdf import list_documents

    documents = list_documents(args.only)
    if not args.allow_download:
        print(f"{len(documents)} documents. Nothing was started: this run downloads {args.model} (about 3 GB) the first time.")
        print("Add --allow-download to run it. On a laptop without a GPU use the Colab notebook instead.")
        return 0
    print(f"Loading {args.model} ...", end=" ", flush=True)
    ask = hf_asker(args.model)
    print("ok")
    try:
        result = generate_lines(documents, ask, args.out, limit=args.limit)
    except KeyboardInterrupt:
        sys.exit("\nStopped. Run the same command again and it continues.")
    print(f"Done: {result['made']:,} new lines, {result['total']:,} in {args.out} ({result['fallbacks']} used the page title because the model gave nothing).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
