"""Step 2.2: the bot. A question goes in. Search, the prompt, Claude and the parsed answer come out.

    from bot.answer import answer
    result = answer("What was the net income of 3M in FY2018?")
    print(result.answer, result.citations, result.found, result.cost_cents)

Check it from the project folder:
    python -m bot.answer "What was the net income of 3M in FY2018?"    one question
    python -m bot.answer --practice                                    the 10 practice questions, each next to its reference answer
    python -m bot.answer --practice --first 3                          only the first 3
The practice questions come from data\\cases\\test_cases.jsonl. Make that file first with:  python -m evals.build_cases
"""
import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
import config
from bot import llm, lookup, prompts
from evals import cases

BOT_MAX_TOKENS = 600  # the answer is short; a retry after a bad reply gets twice this
PRACTICE_COUNT = cases.EXPECTED_COUNTS["practice"]


@dataclass
class Answer:
    question: str
    answer: str  # the bot's text. Empty when no JSON could be read from the reply
    citations: list  # [{"doc": "3M_2018_10K", "page": 60}]
    found: bool  # False when the bot says the pieces do not hold the answer
    hits: list  # the pieces sent to Claude (Hit objects), best first
    tokens_in: int = 0
    tokens_out: int = 0
    cost_cents: float = 0.0
    seconds: float = 0.0  # search plus Claude
    search_seconds: float = 0.0
    calls: int = 0  # Claude calls: 0 (Claude was not called), 1, or 2 after a bad reply
    format_ok: bool = False  # JSON with answer, citations and found, each of the right type
    format_note: str = ""  # what was wrong. Empty when format_ok
    skipped_reason: str = ""  # why Claude was not called, if it was not
    raw: str = ""  # the last reply text from Claude
    prompt: str = ""  # prompt version name, for example bot_v1
    model: str = ""
    role: str = ""  # guest, analyst or hr: set when the guards ran 
    events: list = field(default_factory=list)  # what the guards and the agent did: pieces removed, text masked, steps taken
    blocked: str = ""  # why the answer was replaced, for example output:secret_word. Empty when nothing was blocked


    def to_dict(self, include_text=False):
        """Plain data for a results file. include_text adds the text of each piece (the judge needs it)."""
        pieces = []
        for hit in self.hits:
            item = {"doc": hit.doc, "page": hit.page, "piece": hit.piece, "score": round(hit.score, 4)}
            if include_text:
                item["text"] = hit.text
            pieces.append(item)
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": self.citations,
            "found": self.found,
            "pieces": pieces,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_cents": round(self.cost_cents, 6),
            "seconds": round(self.seconds, 3),
            "search_seconds": round(self.search_seconds, 3),
            "calls": self.calls,
            "format_ok": self.format_ok,
            "format_note": self.format_note,
            "skipped_reason": self.skipped_reason,
            "raw": self.raw,
             "prompt": self.prompt,
            "model": self.model,
            "role": self.role,
            "events": self.events,
            "blocked": self.blocked,
        }


# ---------- reading the reply ----------

def parse_reply(text):
    """Find the JSON object in a reply. Handles a plain object, a code fence and text around it. Returns a dict or None."""
    text = text.strip()
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1).strip())
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    return None


def read_fields(data):
    """Turn the parsed JSON into (answer, citations, found, note). The note is empty when the shape is exactly as agreed."""
    notes = [f"{key} is missing" for key in ("answer", "citations", "found") if key not in data]
    text = data.get("answer")
    if text is None:
        text = ""
    elif not isinstance(text, str):
        notes.append("answer is not text")
        text = str(text)
    found = data.get("found")
    if not isinstance(found, bool):
        if "found" in data:
            notes.append("found is not true or false")
        found = str(found).strip().lower() == "true"
    citations = []
    listed = data.get("citations")
    if listed is None:
        listed = []
    elif not isinstance(listed, list):
        notes.append("citations is not a list")
        listed = []
    for item in listed:
        doc = item.get("doc") if isinstance(item, dict) else None
        page = item.get("page") if isinstance(item, dict) else None
        if isinstance(page, str) and page.strip().isdigit():
            page = int(page)
            notes.append("a page number was written as text")
        if isinstance(doc, str) and isinstance(page, int) and not isinstance(page, bool):
            citations.append({"doc": doc, "page": page})
        else:
            notes.append("a citation is not a doc and a page")
    return text, citations, found, "; ".join(dict.fromkeys(notes))


# ---------- a found answer always cites a page ----------

def pages_with_numbers(text, hits, limit=3):
    """The pages (as citations) of the sent pieces whose text holds a number that the answer rests on. At most limit."""
    from evals import code_checks

    wanted = code_checks.key_numbers(text)
    pages = []
    if not wanted:
        return pages
    for hit in hits:
        item = {"doc": hit.doc, "page": hit.page}
        if item in pages:
            continue
        got = code_checks.parse_numbers(hit.text)
        if any(code_checks.numbers_agree(number, other) for number in wanted for other in got):
            pages.append(item)
        if len(pages) >= limit:
            break
    return pages


def add_missing_citations(text, citations, found, hits):
    """When the answer is marked found but cites nothing, cite the sent pages whose text holds a number of the answer.

    Returns (citations, events). The citations that Claude wrote are never changed: a page that was not sent is caught by the output
    guard (security/guards.py, the answer is blocked) and by the check citation_valid. Only pages that were sent are ever added.
    """
    if not found or citations:
        return citations, []
    added = pages_with_numbers(text, hits)
    if added:
        return added, [{"stage": "citation", "added": [f"{item['doc']} p{item['page']}" for item in added]}]
    return citations, [{"stage": "citation", "unsupported": True}]


# ---------- the bot ----------

def answer(question, k=None, company=None, access=None, collection=None, prompt=None, min_score=None, search_fn=None, ask_fn=None, prepare_fn=None):
    """Search, ask Claude, read the reply. Returns an Answer.

    The search is bot.lookup.find: the question is rewritten as a search query, and the best pages are sent whole (switches in config.py).
    A found answer without any citation gets the sent pages that hold its numbers (add_missing_citations).
    min_score: if the best piece scores lower than this, Claude is not called and the answer is "Not in the documents."
    It is off (None) until a good value is measured. search_fn and ask_fn are for tests.
    prepare_fn: step 3.4. A function that gets the pieces after the search and gives back (pieces, events): the instruction scan
    and the personal data hiding. security/guards.py passes one. Without it nothing changes.
    """
    started = time.perf_counter()
    version = prompts.BOT_PROMPTS[prompt or prompts.DEFAULT_BOT_PROMPT]
    looked = (search_fn or lookup.find)(question, k=k, company=company, access=access, collection=collection)
    hits, events = list(looked), []
    if prepare_fn is not None and hits:
        hits, events = prepare_fn(hits)
    events = list(events) + list(getattr(looked, "events", ()))
    result = Answer(question, "", [], False, hits, prompt=version.name, model=config.CLAUDE_MODEL, events=events)
    result.tokens_in += getattr(looked, "tokens_in", 0)  # the rewrite of the question is a Claude call too
    result.tokens_out += getattr(looked, "tokens_out", 0)
    result.cost_cents += getattr(looked, "cost_cents", 0.0)
    result.search_seconds = time.perf_counter() - started
    if not hits or (min_score is not None and hits[0].score < min_score):
        result.answer = "Not in the documents."
        result.format_ok = True
        result.skipped_reason = "search found no pieces" if not hits else f"best score {hits[0].score:.3f} is below {min_score}"
        result.seconds = time.perf_counter() - started
        return result

    pieces = "\n".join(version.piece.format(doc=hit.doc, page=hit.page, text=hit.text) for hit in hits)
    user = version.user.format(pieces=pieces, question=question)
    ask = ask_fn or llm.ask
    tokens = BOT_MAX_TOKENS
    data = None
    for attempt in (1, 2):  # the second try only happens when the first reply held no JSON object
        reply = ask(version.system, user if attempt == 1 else user + version.retry, max_tokens=tokens)
        result.calls += 1
        result.tokens_in += reply.tokens_in
        result.tokens_out += reply.tokens_out
        result.cost_cents += reply.cost_cents
        result.raw = reply.text
        data = parse_reply(reply.text)
        if data is not None:
            break
        tokens = BOT_MAX_TOKENS * 2  # a cut off reply is the usual reason, so give the retry more room
    if data is None:
        result.format_note = "no JSON object in the reply"
        if reply.stop_reason == "max_tokens":
            result.format_note += " (the reply was cut at max_tokens)"
    else:
        result.answer, result.citations, result.found, result.format_note = read_fields(data)
        result.format_ok = not result.format_note
        result.citations, notes = add_missing_citations(result.answer, result.citations, result.found, result.hits)
        result.events = list(result.events) + notes
    result.seconds = time.perf_counter() - started
    return result


# ---------- check it by hand ----------

def short(text, width=300):
    return " ".join(str(text).split())[:width]


def run_practice(count=PRACTICE_COUNT, search_fn=None, ask_fn=None):
    """Answer the practice questions and print each one next to its reference answer."""
    practice = cases.load_cases("practice")[:count]
    print(f"Answering {len(practice)} questions. The first search loads the model, please wait ...\n")
    started, cost, cited_evidence, said_found, failed = time.time(), 0.0, 0, 0, 0
    for number, case in enumerate(practice, start=1):
        print(f"[{number}/{len(practice)}] {case.id}  {case.doc}")
        print("Question:  " + short(case.question))
        print("Reference: " + short(case.answer))
        wanted = case.pages
        print("Evidence:  " + ", ".join(f"{doc} p{page}" for doc, page in wanted))
        try:
            result = answer(case.question, search_fn=search_fn, ask_fn=ask_fn)
        except llm.BudgetStop:
            raise
        except Exception as err:  # one bad question should not stop the other nine
            failed += 1
            print(f"ERROR:     {short(err)}\n")
            continue
        cited = [(item["doc"], item["page"]) for item in result.citations]
        hit_evidence = any(pair in wanted for pair in cited)
        cited_evidence += hit_evidence
        said_found += result.found
        cost += result.cost_cents
        print("Bot:       " + (short(result.answer) if result.format_ok or result.answer else "NO JSON. Raw reply: " + short(result.raw, 200)))
        print("Cited:     " + (", ".join(f"{doc} p{page}" for doc, page in cited) or "nothing"))
        print(f"Found: {str(result.found).lower()} | cited an evidence page: {'yes' if hit_evidence else 'no'} | format: {'ok' if result.format_ok else result.format_note}")
        print("Pieces:    " + ", ".join(f"{hit.doc} p{hit.page}" for hit in result.hits))
        print(f"Cost:      {result.cost_cents:.4f} cents | {result.calls} call{'s' if result.calls != 1 else ''} | {result.seconds:.1f} s\n")
    total = len(practice) - failed
    print(f"Answered {total} of {len(practice)}. Cited an evidence page: {cited_evidence} of {total}. Said found true: {said_found} of {total}.")
    print(f"Cost: {cost:.3f} cents in all ({cost / max(total, 1):.4f} per question). Time: {time.time() - started:.0f} s.")
    print(f"Ledger: {llm.spent_cents():.3f} of {config.FUSE_CENTS} cents used.")
    print("Now read each answer next to its reference, and write 3 lines on what you see.")
    return cited_evidence, total


def main(argv=None, search_fn=None, ask_fn=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Ask the bot a question (step 2.2).")
    parser.add_argument("question", nargs="?", help="one question")
    parser.add_argument("--practice", action="store_true", help="the practice questions, next to their reference answers")
    parser.add_argument("--first", type=int, default=PRACTICE_COUNT, help="how many practice questions (default 10)")
    args = parser.parse_args(argv)
    if not args.practice and not args.question:
        parser.error("give a question in quotes, or use --practice")
    try:
        if args.practice:
            run_practice(args.first, search_fn, ask_fn)
            return
        result = answer(args.question, search_fn=search_fn, ask_fn=ask_fn)
    except (llm.BudgetStop, cases.CaseError) as err:
        sys.exit(str(err))
    print("Pieces:    " + ", ".join(f"{hit.doc} p{hit.page} ({hit.score:.3f})" for hit in result.hits))
    print("Answer:    " + (short(result.answer, 600) or "NO JSON. Raw reply: " + short(result.raw, 300)))
    print("Cited:     " + (", ".join(f"{item['doc']} p{item['page']}" for item in result.citations) or "nothing"))
    print(f"Found: {str(result.found).lower()} | format: {'ok' if result.format_ok else result.format_note}")
    print(f"Cost:      {result.cost_cents:.4f} cents | {result.tokens_in} tokens in, {result.tokens_out} out | {result.calls} call(s) | {result.seconds:.1f} s")


if __name__ == "__main__":
    main()
