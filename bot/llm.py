"""Step 2.1: the only place that talks to Claude.

    from bot.llm import ask
    reply = ask("Answer in one short sentence.", "What is 2 + 2?", max_tokens=50)
    print(reply.text, reply.cost_cents)

Every call has a time limit and a retry for rate limit and overload errors. There is no
temperature setting: the current Claude library does not have one, so two runs can differ a little.
A ledger file (data\\ledger.json) adds up the cost of every call. When the total reaches
FUSE_CENTS in config.py, the next call is refused before it costs anything.

Check it from the project folder:
    python -m bot.llm                 one tiny call: prints the answer and the cost in cents
    python -m bot.llm "Say hello"     the same with your own text
    python -m bot.llm --spent         prints the ledger and makes no call
"""
import argparse
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from functools import lru_cache

import config

TIMEOUT_SECONDS = 60  # one call may take at most this long
TRIES = 4  # the first call and up to 3 more
WAITS = (2, 5, 10)  # seconds to wait before try 2, 3 and 4
LEDGER_FIELDS = ("cents", "calls", "tokens_in", "tokens_out")

_lock = threading.Lock()  # two threads (the web app) must not write the ledger at the same time


class BudgetStop(RuntimeError):
    """The spend fuse: no more Claude calls."""


@dataclass(frozen=True)
class Reply:
    text: str
    tokens_in: int
    tokens_out: int
    cost_cents: float
    seconds: float  # the whole ask(), including waits before a retry
    model: str
    stop_reason: str  # end_turn, or max_tokens when the reply was cut short
    tries: int  # 1 when the first call worked


# ---------- the ledger and the fuse ----------

def ledger_path():
    return config.DATA_DIR / "ledger.json"


def read_ledger():
    """What the ledger says so far: cents, calls, tokens_in, tokens_out. A missing file means nothing is spent."""
    path = ledger_path()
    if not path.exists():
        return {key: 0 for key in LEDGER_FIELDS}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {key: data[key] for key in LEDGER_FIELDS}
    except (OSError, ValueError, KeyError, TypeError) as err:
        raise BudgetStop(
            f"{path} is damaged ({type(err).__name__}), so the spend is unknown. "
            "Look at the usage page in the Claude console. Fix or delete the file only if the spend is below the fuse."
        ) from err


def spent_cents():
    return read_ledger()["cents"]


def check_fuse():
    spent = spent_cents()
    if spent >= config.FUSE_CENTS:
        raise BudgetStop(f"Spend fuse: {spent:.2f} cents are used and the limit is {config.FUSE_CENTS}. No Claude call was made.")


def _add_to_ledger(tokens_in, tokens_out, cents):
    with _lock:
        data = read_ledger()
        data["cents"] = round(data["cents"] + cents, 6)
        data["calls"] += 1
        data["tokens_in"] += tokens_in
        data["tokens_out"] += tokens_out
        path = ledger_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(temp, path)  # the file is never half written


# ---------- the call ----------

@lru_cache(maxsize=1)
def get_client():
    """One Claude client for the whole program. The SDK retries are off: ask() does its own."""
    import anthropic

    return anthropic.Anthropic(api_key=config.need("ANTHROPIC_API_KEY"), timeout=TIMEOUT_SECONDS, max_retries=0)


def cost_cents(tokens_in, tokens_out):
    return (tokens_in * config.PRICE_IN_CENTS_PER_MTOK + tokens_out * config.PRICE_OUT_CENTS_PER_MTOK) / 1_000_000


def _status(err):
    return getattr(err, "status_code", None)


def _retryable(err):
    """Rate limit (429), overload and server errors (5xx), a lost connection or a time out."""
    status = _status(err)
    if status is not None:
        return status == 429 or status >= 500
    return type(err).__name__ in ("APIConnectionError", "APITimeoutError")


def _explain(err):
    status = _status(err)
    if status == 401:
        return "Claude refused the key (401). Check ANTHROPIC_API_KEY in .env."
    if status == 403:
        return "Claude says this key may not do that (403). Check the key and its workspace in the Claude console."
    if status == 404:
        return f"Claude does not know the model {config.CLAUDE_MODEL} (404). Check CLAUDE_MODEL in config.py."
    if status == 400:
        return f"Claude refused the request (400): {str(err)[:300]}"
    if _retryable(err):
        return f"Claude did not answer after {TRIES} tries ({type(err).__name__}, status {status}). Try again in a few minutes."
    return f"{type(err).__name__}: {str(err)[:300]}"


def ask(system, user, max_tokens=500, client=None, sleep=time.sleep):
    """One Claude call. Returns a Reply. Raises BudgetStop when the fuse is reached, RuntimeError when Claude cannot answer."""
    check_fuse()
    client = client or get_client()
    started = time.perf_counter()
    tries = 0
    while True:
        tries += 1
        try:
            message = client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=max_tokens, thinking={"type": "disabled"},  # Haiku 5.5 thinks by default; the thinking used up max_tokens (empty or cut replies)
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            break
        except Exception as err:
            if _retryable(err) and tries < TRIES:
                sleep(WAITS[tries - 1])
                continue
            raise RuntimeError(_explain(err)) from err
    text = "".join(block.text for block in message.content if getattr(block, "type", "") == "text")
    tokens_in, tokens_out = message.usage.input_tokens, message.usage.output_tokens
    cents = cost_cents(tokens_in, tokens_out)
    _add_to_ledger(tokens_in, tokens_out, cents)
    return Reply(text, tokens_in, tokens_out, cents, time.perf_counter() - started, config.CLAUDE_MODEL, message.stop_reason or "", tries)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="One tiny Claude call, or the spend ledger (step 2.1).")
    parser.add_argument("text", nargs="?", default="Reply with the word OK.", help="what to ask")
    parser.add_argument("--spent", action="store_true", help="print the ledger and make no call")
    args = parser.parse_args(argv)
    try:
        if not args.spent:
            reply = ask("Answer in one short sentence.", args.text, max_tokens=100)
            print("Claude says:", " ".join(reply.text.split()))
            print(f"tokens in {reply.tokens_in}, out {reply.tokens_out} | cost {reply.cost_cents:.5f} cents | {reply.seconds:.1f} s | {reply.model} | stop: {reply.stop_reason}")
            if not reply.text.strip():
                print("The reply is empty. If the stop reason is max_tokens, the model used its tokens before it wrote text. Send me these lines.")
        ledger = read_ledger()
    except (BudgetStop, RuntimeError) as err:
        sys.exit(str(err))
    print(f"Ledger: {ledger['cents']:.4f} of {config.FUSE_CENTS} cents used in {ledger['calls']} calls ({ledger_path()})")


if __name__ == "__main__":
    main()
