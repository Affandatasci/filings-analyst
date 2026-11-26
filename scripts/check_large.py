"""Check an embedding model before you use it: does it load, does it give real numbers (no NaN), and are similar texts close?

    python scripts\\check_large.py                      says what it would do, downloads nothing
    python scripts\\check_large.py --allow-download     downloads the large model once (about 1.3 GB), embeds 4 short texts and checks them
    python scripts\\check_large.py --profile small      the same check for the small model (already on your computer)
It costs nothing (no Claude call). Colab is not needed: 4 short texts are fast on a laptop.
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

TEXTS = (
    "What was the net income of 3M in fiscal 2018?",
    "3M Company consolidated statement of income: net income attributable to 3M was 5,349 million dollars in 2018.",
    "The board approved a new contract with a vendor for office furniture and cleaning services.",
    "Cash flows from operating activities, capital expenditures and payments for dividends.",
)


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


def check_vectors(vectors, dim):
    """(ok, lines) for the vectors of TEXTS. The vectors have length 1, so the dot product is the cosine."""
    lines, ok = [], True

    def say(good, text):
        nonlocal ok
        ok = ok and good
        lines.append(("  ok   " if good else "  FAIL ") + text)

    say(len(vectors) == len(TEXTS) and all(len(v) == dim for v in vectors), f"{len(vectors)} vectors of {dim} numbers")
    say(all(math.isfinite(x) for v in vectors for x in v), "every number is a real number (no NaN, no infinity)")
    norms = [math.sqrt(cosine(v, v)) for v in vectors]
    say(all(abs(n - 1) < 0.01 for n in norms), "every vector has length 1 (" + ", ".join(f"{n:.3f}" for n in norms) + ")")
    near, far = cosine(vectors[0], vectors[1]), cosine(vectors[0], vectors[2])
    say(near > far, f"the question is closer to the income statement text ({near:.3f}) than to the furniture text ({far:.3f})")
    return ok, lines


def main(argv=None, embed=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Check an embedding model: it loads, gives real numbers, and similar texts are close.")
    parser.add_argument("--profile", choices=sorted(config.EMBED_PROFILES), default="large", help="the model to check (default large)")
    parser.add_argument("--allow-download", action="store_true", help="allow the download of the model when it is not on this computer")
    args = parser.parse_args(argv)
    model, dim = config.use_profile(args.profile)
    print(f"COLAB: no | MODEL: {model} ({dim} numbers) | CLAUDE COST: 0 cents")
    if embed is None:
        from ingest import embed as embedder

        if not embedder.model_on_disk(model) and not args.allow_download:
            print(f"{model} is not on this computer yet. Nothing was downloaded. Add --allow-download to download it once.")
            return 0
        print("Loading the model (the first time this downloads it) ...", flush=True)
        embed = embedder.embed_texts
    ok, lines = check_vectors(embed(list(TEXTS)), dim)
    print("\n".join(lines))
    print("\nThe model is fine. You can use it." if ok else "\nThe model gave a bad answer. Do not build an index with it on this computer; use Colab.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
