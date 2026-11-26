"""apply_upgrades.py - the upgrades of 26 November in one file. This is the only file you run on your computer.

COLAB: needed later (notebook filings_colab.ipynb) | MODEL: small bge 384 stays the default | CLAUDE COST: about 1 cent (the rewrite v3 check), 0 with --no-spend

Put it in the project folder (the one with config.py) and run it with the Python of the project:
    python apply_upgrades.py --check      says what it would change and changes nothing
    python apply_upgrades.py              applies everything, runs your tests before and after, puts every file back if a test breaks,
                                          then runs the rewrite v3 check (about 1 cent) and switches rewrite v3 on when it is not worse
    python apply_upgrades.py --undo       takes the changes out again (edits you made yourself later stay)
    --no-test skips the tests.  --no-spend skips the rewrite v3 check (0 cents).  --rewrite-check runs only that check.
    --root FOLDER is for a file that is not in the project folder.
Run twice, the second time says everything is already in place. The old version of every file is kept in .upgrades_backup\\<date and time>\\ .

What it does (nothing here changes filings_v1 or the small model):
  two models     config.py, ingest/embed.py    EMBED_PROFILE: small (the default, as today) or large (bge large, 1024 numbers). The embedder follows it
                                               and stops with a clear message if a model gives NaN numbers
  piece size     ingest/chunker.py             pieces can be aimed at 150 to 480 tokens (the default stays 400)
  new indexes    ingest/build_index.py         options --collection NAME with --profile, --piece-tokens, --context: always a NEW collection
  context lines  ingest/context_lines.py       new: one line for each page from a small open model (the notebook runs it on the Colab GPU)
  compare        evals/compare_collections.py  new: free search-only comparison of collections (the bot reads the right page for how many questions)
  lab            evals/lab.py                  new: what the notebook builds, how it saves to Drive and how it picks the winners
  model check    scripts/check_large.py        new: does a model load, give real numbers and put similar texts close
  rewrite v3     config.py                     switched on after a 1 cent check, only when it finds at least as many right pages as rewrite v2
  tests          tests/test_upgrades.py        new: 37 tests for all of this
  no new bot prompt (no bot_v4) and no neighbors: NEIGHBOR_PAGES stays 0, the bot prompt stays as it is.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HUNKS = [
    ('config.py',
     r'''
# ---- Models ----
CLAUDE_MODEL = "claude-haiku-5-5"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384

# ---- Index and search ----
COLLECTION = "filings_v1"''',
     r'''
# ---- Models ----
CLAUDE_MODEL = "claude-haiku-5-5"
# Two embedding models (build two models). small is what the bot uses today (collection filings_v1). large is made in Colab
# (notebooks\filings_colab.ipynb) and must also be used when a question is searched in a collection made with it.
EMBED_PROFILES = {"small": ("BAAI/bge-small-en-v1.5", 384), "large": ("BAAI/bge-large-en-v1.5", 1024)}
EMBED_PROFILE = (os.getenv("EMBED_PROFILE") or "small").strip().lower()  # small (default) or large; set it here or in .env
if EMBED_PROFILE not in EMBED_PROFILES:
    raise SystemExit(f"EMBED_PROFILE must be small or large, not '{EMBED_PROFILE}'. Check config.py and .env.")
EMBED_MODEL, EMBED_DIM = EMBED_PROFILES[EMBED_PROFILE]


def use_profile(name):
    """Switch the embedding model for the rest of this run: small (384 numbers) or large (1024 numbers). Call it before the first search or build."""
    global EMBED_PROFILE, EMBED_MODEL, EMBED_DIM
    name = str(name).strip().lower()
    if name not in EMBED_PROFILES:
        raise ValueError(f"Unknown profile '{name}'. Use small or large.")
    EMBED_PROFILE = name
    EMBED_MODEL, EMBED_DIM = EMBED_PROFILES[name]
    return EMBED_MODEL, EMBED_DIM


# ---- Index and search ----
COLLECTION = "filings_v1"'''),
    ('ingest/embed.py',
     r'''"""Step 1.4: the embedding model (bge small) and the token counter.

Both come from the same model name in config.py, so the token counts match
what the model sees. The first run downloads the model (about 130 MB).
The model runs on the GPU when the machine has one (Colab) and on the CPU
otherwise (a laptop). The vectors are the same either way.
Nothing is loaded until a function is called, so the tests stay fast.''',
     r'''"""Step 1.4: the embedding model (bge small or bge large) and the token counter.

Both come from the same model name in config.py (EMBED_PROFILE: small or large), so the token counts match
what the model sees. The first run downloads the model (small: about 130 MB, large: about 1.3 GB).
The model runs on the GPU when the machine has one (Colab) and on the CPU
otherwise (a laptop). The vectors are the same either way.
Nothing is loaded until a function is called, so the tests stay fast.'''),
    ('ingest/embed.py',
     r'''BATCH = 32  # texts per model call


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.EMBED_MODEL)
    tokenizer.model_max_length = 10**6  # we only count here, so no "longer than 512" warning
    return tokenizer


def count_tokens(texts):
    """The number of tokens of each text, without the two special start and end tokens."""
    tokenizer = _tokenizer()
    sizes = []
    for i in range(0, len(texts), 256):
        encoded = tokenizer(list(texts[i : i + 256]), add_special_tokens=False)["input_ids"]''',
     r'''BATCH = 32  # texts per model call


def _name(model):
    """The model to use: the one given, else the one of config.py (EMBED_PROFILE)."""
    return model or config.EMBED_MODEL


@lru_cache(maxsize=2)
def _tokenizer(name):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.model_max_length = 10**6  # we only count here, so no "longer than 512" warning
    return tokenizer


def count_tokens(texts, model=None):
    """The number of tokens of each text, without the two special start and end tokens."""
    tokenizer = _tokenizer(_name(model))
    sizes = []
    for i in range(0, len(texts), 256):
        encoded = tokenizer(list(texts[i : i + 256]), add_special_tokens=False)["input_ids"]'''),
    ('ingest/embed.py',
     r'''    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBED_MODEL, device=_device())


def embed_texts(texts):
    """One vector (384 numbers, length 1) for each text. Use it for pieces and for questions."""
    vectors = _model().encode(list(texts), batch_size=BATCH, normalize_embeddings=True, show_progress_bar=False)
    return [vector.tolist() for vector in vectors]
''',
     r'''    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=2)
def _model(name):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device=_device())


def model_on_disk(name=None):
    """True when the model is already downloaded on this computer (so nothing has to be downloaded to use it)."""
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(_name(name), "config.json"), str)
    except Exception:  # noqa: BLE001 - when in doubt the answer is no
        return False


def all_finite(vectors):
    """True when every number of every vector is a real number (no NaN, no infinity)."""
    import numpy as np

    return bool(np.isfinite(np.asarray(vectors, dtype="float64")).all())


def embed_texts(texts, model=None):
    """One vector (384 numbers for small, 1024 for large, length 1) for each text. Use it for pieces and for questions."""
    vectors = _model(_name(model)).encode(list(texts), batch_size=BATCH, normalize_embeddings=True, show_progress_bar=False)
    if not all_finite(vectors):
        raise RuntimeError(
            "The embedding model gave numbers that are not valid (NaN). Nothing was uploaded. "
            "Test the model with:  python scripts\\check_large.py --allow-download   or use EMBED_PROFILE = small."
        )
    return [vector.tolist() for vector in vectors]
'''),
    ('ingest/chunker.py',
     r'''    return "\n".join(parts)


def cut_page(info, page, count, with_header=True):
    """Cut one page into pieces. count(list of texts) returns the token count of each text."""
    header = info.header(page.number) if with_header else ""
    sizes = count(([header] if header else []) + page.lines)
    fixed = sizes[0] if header else 0
    lines, counts = _split_long_lines(page.lines, sizes[1:] if header else sizes, count)
    for attempt in range(6):  # the check below should always pass at the first try
        shrink = 0.9**attempt
        plans = _plan(lines, counts, fixed, int(config.PIECE_TOKENS * shrink), int(config.PIECE_HARD_STOP * shrink))
        texts = [_compose(header, lines, plan) for plan in plans]
        real = count(texts)
        if max(real) <= config.PIECE_HARD_STOP:''',
     r'''    return "\n".join(parts)


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
        if max(real) <= config.PIECE_HARD_STOP:'''),
    ('ingest/chunker.py',
     r'''    raise RuntimeError(f"{info.name} page {page.number}: a piece stays above {config.PIECE_HARD_STOP} tokens.")


def cut_document(info, pages, count, with_header=True):
    """Cut all pages of one document. Pages marked no_text give no pieces."""
    pieces = []
    for page in pages:
        if not page.no_text:
            pieces.extend(cut_page(info, page, count, with_header))
    return pieces

''',
     r'''    raise RuntimeError(f"{info.name} page {page.number}: a piece stays above {config.PIECE_HARD_STOP} tokens.")


def cut_document(info, pages, count, with_header=True, piece_tokens=None):
    """Cut all pages of one document. Pages marked no_text give no pieces."""
    pieces = []
    for page in pages:
        if not page.no_text:
            pieces.extend(cut_page(info, page, count, with_header, piece_tokens))
    return pieces

'''),
    ('ingest/chunker.py',
     r'''        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Cut the PDFs into pieces (step 1.3).")
    parser.add_argument("--no-header", action="store_true", help="no header line at the top of each piece")
    parser.add_argument("--only", nargs="+", metavar="NAME", help="only these documents")
    parser.add_argument("--show", nargs=2, metavar=("NAME", "PAGE"), help="print the pieces of one page")
    args = parser.parse_args(argv)''',
     r'''        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Cut the PDFs into pieces (step 1.3).")
    parser.add_argument("--no-header", action="store_true", help="no header line at the top of each piece")
    parser.add_argument("--piece-tokens", type=int, metavar="N", help="aim for pieces of N tokens (default 400, allowed 150 to 480)")
    parser.add_argument("--only", nargs="+", metavar="NAME", help="only these documents")
    parser.add_argument("--show", nargs=2, metavar=("NAME", "PAGE"), help="print the pieces of one page")
    args = parser.parse_args(argv)'''),
    ('ingest/chunker.py',
     r'''        pages = read_pdf(document.path, only_pages={number})
        if not pages or pages[0].no_text:
            sys.exit(f"{name} page {number} is missing or has no text.")
        for piece in cut_page(document.info, pages[0], count_tokens, with_header):
            print(f"\n----- piece {piece.piece}, {piece.tokens} tokens -----")
            print(piece.text)
        return''',
     r'''        pages = read_pdf(document.path, only_pages={number})
        if not pages or pages[0].no_text:
            sys.exit(f"{name} page {number} is missing or has no text.")
        for piece in cut_page(document.info, pages[0], count_tokens, with_header, args.piece_tokens):
            print(f"\n----- piece {piece.piece}, {piece.tokens} tokens -----")
            print(piece.text)
        return'''),
    ('ingest/chunker.py',
     r'''    for document in documents:
        t0 = time.time()
        pages = read_pdf(document.path)
        pieces = cut_document(document.info, pages, count_tokens, with_header)
        top = max(piece.tokens for piece in pieces)
        total += len(pieces)
        tokens_sum += sum(piece.tokens for piece in pieces)''',
     r'''    for document in documents:
        t0 = time.time()
        pages = read_pdf(document.path)
        pieces = cut_document(document.info, pages, count_tokens, with_header, args.piece_tokens)
        top = max(piece.tokens for piece in pieces)
        total += len(pieces)
        tokens_sum += sum(piece.tokens for piece in pieces)'''),
    ('ingest/build_index.py',
     r'''    python -m ingest.build_index                  build or update the collection
    python -m ingest.build_index --only NAME      one document first, as a quick test
    python -m ingest.build_index --no-header      a second index without the header line (experiment 3, later)

It is safe to run again. Every piece has a fixed id (made from doc, page and piece number)
and a fingerprint of its content. A second run skips pieces that are already there and the''',
     r'''    python -m ingest.build_index                  build or update the collection
    python -m ingest.build_index --only NAME      one document first, as a quick test
    python -m ingest.build_index --no-header      a second index without the header line (experiment 3, later)
    python -m ingest.build_index --collection NAME --profile large --piece-tokens 250 --context FILE
                                                  a NEW collection with the large model, another piece size or context lines
                                                  (filings_v1 is never touched by these options; the notebook does the same in Colab)

It is safe to run again. Every piece has a fixed id (made from doc, page and piece number)
and a fingerprint of its content. A second run skips pieces that are already there and the'''),
    ('ingest/build_index.py',
     r'''
UPLOAD_BATCH = 64  # pieces per upload
INDEXED_FIELDS = ("company", "doc", "access")


def piece_id(doc, page, piece):''',
     r'''
UPLOAD_BATCH = 64  # pieces per upload
INDEXED_FIELDS = ("company", "doc", "access")
NUMBER_FIELDS = ("page", "year")  # whole pages (page) and the year filter (year) look these up


def piece_id(doc, page, piece):'''),
    ('ingest/build_index.py',
     r'''    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"filings-analyst/{doc}/{page}/{piece}"))


def make_payload(document, piece):
    """The labels stored with each piece, plus a fingerprint (sha) of all of them."""
    info = document.info
    payload = {
        "doc": info.name,''',
     r'''    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"filings-analyst/{doc}/{page}/{piece}"))


def make_payload(document, piece, context=""):
    """The labels stored with each piece, plus a fingerprint (sha) of all of them. context: the context line of the page (only for the vector)."""
    info = document.info
    payload = {
        "doc": info.name,'''),
    ('ingest/build_index.py',
     r'''        "source": document.source,
        "access": document.access,
    }
    payload["sha"] = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return payload


def retry(action, tries=3):''',
     r'''        "source": document.source,
        "access": document.access,
    }
    if context:
        payload["context"] = context  # used when the piece is embedded; the stored text, and so what the bot reads, stays the same
    payload["sha"] = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return payload


def embed_text(payload):
    """What the model reads for one piece: the context line of its page (when there is one) and then the piece."""
    return payload["context"] + "\n" + payload["text"] if payload.get("context") else payload["text"]


def retry(action, tries=3):'''),
    ('ingest/build_index.py',
     r'''            time.sleep(5 * attempt)


def ensure_collection(client, name):
    """Make the collection (384 numbers, cosine) and the indexes on company, doc and access."""
    if name not in [c.name for c in client.get_collections().collections]:
        client.create_collection(name, vectors_config=VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE))
        print(f"Created collection {name} ({config.EMBED_DIM} numbers, cosine).")
    have = set((client.get_collection(name).payload_schema or {}).keys())
    for field in INDEXED_FIELDS:
        if field not in have:
            client.create_payload_index(name, field, PayloadSchemaType.KEYWORD)


def stored(client, collection, doc):''',
     r'''            time.sleep(5 * attempt)


def ensure_collection(client, name, dim=None, on_disk=False, numbers=False):
    """Make the collection (default config.EMBED_DIM numbers, cosine) and the indexes on company, doc and access.

    dim: numbers per vector. on_disk: keep the stored texts on disk (for a small Qdrant cluster with several collections).
    numbers: also index page and year (a new collection needs both: whole pages and the year filter use them).
    An existing collection must have the same number of numbers per vector, or the run stops with a plain message.
    """
    size = int(dim or config.EMBED_DIM)
    if name not in [c.name for c in client.get_collections().collections]:
        extra = {"on_disk_payload": True} if on_disk else {}
        client.create_collection(name, vectors_config=VectorParams(size=size, distance=Distance.COSINE), **extra)
        print(f"Created collection {name} ({size} numbers, cosine).")
    else:
        found = getattr(client.get_collection(name).config.params.vectors, "size", None)
        if found is not None and found != size:
            sys.exit(f"The collection {name} holds vectors of {found} numbers, but this run makes {size} numbers. Use another collection name, or the other model (EMBED_PROFILE).")
    have = set((client.get_collection(name).payload_schema or {}).keys())
    for field in INDEXED_FIELDS:
        if field not in have:
            client.create_payload_index(name, field, PayloadSchemaType.KEYWORD)
    for field in NUMBER_FIELDS if numbers else ():
        if field not in have:
            client.create_payload_index(name, field, PayloadSchemaType.INTEGER)


def stored(client, collection, doc):'''),
    ('ingest/build_index.py',
     r'''            return found


def build(documents, client, embed, count, collection, with_header=True):
    """Read, cut, embed and upload. embed(texts) gives vectors, count(texts) gives token counts."""
    started = time.time()
    ensure_collection(client, collection)
    totals = {"pieces": 0, "new": 0, "changed": 0, "same": 0, "removed": 0}
    for number, document in enumerate(documents, start=1):
        name = document.info.name
        pieces = cut_document(document.info, read_pdf(document.path), count, with_header)
        wanted = {piece_id(p.doc, p.page, p.piece): make_payload(document, p) for p in pieces}
        have = stored(client, collection, name)
        todo = [pid for pid, payload in wanted.items() if have.get(pid) != payload["sha"]]
        print(f"[{number:2}/{len(documents)}] {name:42} {len(wanted):5} pieces, embedding {len(todo):5} ...", end="", flush=True)
        for i in range(0, len(todo), UPLOAD_BATCH):
            ids = todo[i : i + UPLOAD_BATCH]
            vectors = embed([wanted[pid]["text"] for pid in ids])
            points = [PointStruct(id=pid, vector=vector, payload=wanted[pid]) for pid, vector in zip(ids, vectors)]
            retry(lambda points=points: client.upsert(collection_name=collection, points=points))
        stale = [pid for pid in have if pid not in wanted]''',
     r'''            return found


def build(documents, client, embed, count, collection, with_header=True, piece_tokens=None, context=None, on_disk=False, numbers=False, dim=None):
    """Read, cut, embed and upload. embed(texts) gives vectors, count(texts) gives token counts.

    piece_tokens: aim for this piece size (default config.PIECE_TOKENS). context: {(doc, page): one line}, put in front of each piece of that
    page when the piece is embedded. on_disk, numbers and dim: see ensure_collection.
    """
    started = time.time()
    ensure_collection(client, collection, dim, on_disk, numbers)
    lines = context or {}
    totals = {"pieces": 0, "new": 0, "changed": 0, "same": 0, "removed": 0}
    for number, document in enumerate(documents, start=1):
        name = document.info.name
        pieces = cut_document(document.info, read_pdf(document.path), count, with_header, piece_tokens)
        wanted = {piece_id(p.doc, p.page, p.piece): make_payload(document, p, lines.get((p.doc, p.page), "")) for p in pieces}
        have = stored(client, collection, name)
        todo = [pid for pid, payload in wanted.items() if have.get(pid) != payload["sha"]]
        print(f"[{number:2}/{len(documents)}] {name:42} {len(wanted):5} pieces, embedding {len(todo):5} ...", end="", flush=True)
        for i in range(0, len(todo), UPLOAD_BATCH):
            ids = todo[i : i + UPLOAD_BATCH]
            vectors = embed([embed_text(wanted[pid]) for pid in ids])
            points = [PointStruct(id=pid, vector=vector, payload=wanted[pid]) for pid, vector in zip(ids, vectors)]
            retry(lambda points=points: client.upsert(collection_name=collection, points=points))
        stale = [pid for pid in have if pid not in wanted]'''),
    ('ingest/build_index.py',
     r'''    parser.add_argument("--only", nargs="+", metavar="NAME", help="only these documents")
    parser.add_argument("--no-header", action="store_true", help="no header line (goes to a second collection)")
    parser.add_argument("--collection", help="collection name (default from config.py)")
    args = parser.parse_args(argv)

    with_header = not args.no_header
    collection = args.collection or (config.COLLECTION if with_header else config.COLLECTION + "_plain")
    documents = list_documents(args.only)
    client = QdrantClient(url=config.need("QDRANT_URL"), api_key=config.need("QDRANT_API_KEY"), timeout=60)
    try:''',
     r'''    parser.add_argument("--only", nargs="+", metavar="NAME", help="only these documents")
    parser.add_argument("--no-header", action="store_true", help="no header line (goes to a second collection)")
    parser.add_argument("--collection", help="collection name (default from config.py)")
    parser.add_argument("--profile", choices=sorted(config.EMBED_PROFILES), help="embedding model: small (384 numbers) or large (1024 numbers, about 1.3 GB). Needs --collection")
    parser.add_argument("--piece-tokens", type=int, metavar="N", help="aim for pieces of N tokens (default 400, allowed 150 to 480). Needs --collection")
    parser.add_argument("--context", metavar="FILE", help="context lines made by ingest\\context_lines.py. Needs --collection")
    args = parser.parse_args(argv)

    if (args.profile or args.piece_tokens or args.context) and not args.collection:
        sys.exit("--profile, --piece-tokens and --context make a different index, so give it a new name with --collection NAME. The collection of config.py stays as it is.")
    if args.profile:
        config.use_profile(args.profile)
    with_header = not args.no_header
    collection = args.collection or (config.COLLECTION if with_header else config.COLLECTION + "_plain")
    context = None
    if args.context:
        from ingest.context_lines import load_lines

        context = load_lines(args.context)
        if not context:
            sys.exit(f"{args.context} has no context lines. Make them first:  python -m ingest.context_lines --out {args.context} --allow-download")
    documents = list_documents(args.only)
    client = QdrantClient(url=config.need("QDRANT_URL"), api_key=config.need("QDRANT_API_KEY"), timeout=60)
    try:'''),
    ('ingest/build_index.py',
     r'''        sys.exit(f"Cannot reach Qdrant ({type(err).__name__}). Check QDRANT_URL and QDRANT_API_KEY in .env.")
    from ingest.embed import count_tokens, embed_texts  # loads torch, so only here

    print(f"Collection {collection}: {len(documents)} documents, header line {'on' if with_header else 'off'}.")
    print("Loading the tokenizer and the model. The first run downloads about 130 MB ...", end=" ", flush=True)
    count_tokens(["test"])
    embed_texts(["test"])
    print("ok\n")
    try:
        totals = build(documents, client, embed_texts, count_tokens, collection, with_header)
    except KeyboardInterrupt:
        sys.exit("\nStopped. Run the same command again and it continues.")
''',
     r'''        sys.exit(f"Cannot reach Qdrant ({type(err).__name__}). Check QDRANT_URL and QDRANT_API_KEY in .env.")
    from ingest.embed import count_tokens, embed_texts  # loads torch, so only here

    print(f"Collection {collection}: {len(documents)} documents, header line {'on' if with_header else 'off'}, model {config.EMBED_MODEL} ({config.EMBED_DIM} numbers)"
          + (f", pieces of {args.piece_tokens} tokens" if args.piece_tokens else "") + (f", {len(context):,} context lines" if context else "") + ".")
    print(f"Loading the tokenizer and the model. The first run downloads about {'130 MB' if config.EMBED_PROFILE == 'small' else '1.3 GB'} ...", end=" ", flush=True)
    count_tokens(["test"])
    embed_texts(["test"])
    print("ok\n")
    try:
        totals = build(documents, client, embed_texts, count_tokens, collection, with_header, args.piece_tokens, context, on_disk=bool(args.collection), numbers=True)
    except KeyboardInterrupt:
        sys.exit("\nStopped. Run the same command again and it continues.")
'''),
    ('.gitignore',
     r'''# Backups made by apply_fixes.py (python apply_fixes.py --undo puts everything back)
.fixes_backup/
''',
     r'''# Backups made by apply_fixes.py (python apply_fixes.py --undo puts everything back)
.fixes_backup/

# Backups made by apply_upgrades.py (python apply_upgrades.py --undo puts everything back)
.upgrades_backup/
'''),
]

NEW_FILES = {
    'ingest/context_lines.py': r'''"""Context lines: one short line for every page that says what the page is, written once by a small open language model.

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
''',
    'evals/compare_collections.py': r'''"""Compare Qdrant collections on the 85 practice and test questions. Search only. FREE: the search lines come from the rewrite cache.

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
''',
    'evals/lab.py': r'''"""The lab: what the Colab notebook builds and how it chooses. The same code runs on a laptop.

Five techniques. Each one makes a NEW Qdrant collection, so filings_v1 and the small model stay exactly as they are:
  1 header off        the header line ("3M FY2018 10-K, page 60") is not put in front of each piece
  2 large model       bge large, 1024 numbers instead of 384
  3 context lines     one line for each page, written by a small open model, put in front of each piece when it is embedded
  4 piece size        pieces aimed at 250 or at 480 tokens instead of 400
  5 winners combined  the winning settings together, once with the small model and once with the large model (two models)

A technique wins when the bot reads the right page for at least 2 more questions (of 85) than it does with filings_v1.
Everything is scored on search alone with the rewrites of the cache, so the lab costs no Claude money.
The work is kept in a small state file (and the context lines in their own file), so a stopped run continues where it stopped.
"""
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import config
from evals import compare_collections

PROTECTED = ("filings_v1", "filings_v1_plain")  # the lab never writes to these
PREFIX = "filings_v2_"  # every collection the lab makes starts with this
WIN_BY = 2  # more questions found than filings_v1 that make a technique a winner (1 can be luck)
SCORE_PROMPT, SCORE_CLEAN = "rewrite_v2", False  # the search lines of the baseline (66 of 85 in the saved experiments)


@dataclass(frozen=True)
class Spec:
    name: str  # the Qdrant collection
    profile: str  # small or large
    header: bool = True
    piece_tokens: int = 400
    context: bool = False
    technique: str = ""  # 1 to 5, or "now"
    title: str = ""

    def settings(self):
        return (self.profile, self.header, self.piece_tokens, self.context)


BASE = Spec("filings_v1", "small", True, 400, False, "now", "the index of today")
HEADER_OFF = Spec(PREFIX + "noheader", "small", False, 400, False, "1", "header line off")
LARGE = Spec(PREFIX + "large", "large", True, 400, False, "2", "large model (1024 numbers)")
CONTEXT = Spec(PREFIX + "context", "small", True, 400, True, "3", "context lines")
SIZE_SMALL = Spec(PREFIX + "size250", "small", True, 250, False, "4", "pieces of 250 tokens")
SIZE_BIG = Spec(PREFIX + "size480", "small", True, 480, False, "4", "pieces of 480 tokens")
KNOWN = (BASE, HEADER_OFF, LARGE, CONTEXT, SIZE_SMALL, SIZE_BIG)


def check_name(name):
    """The lab only writes to collections of its own. Never to the ones you already have."""
    if name in PROTECTED or name == config.COLLECTION or not name.startswith(PREFIX):
        raise ValueError(f"The lab does not write to '{name}'. Its collections start with {PREFIX} (filings_v1 stays as it is).")
    return name


# ---------- the work files, on this computer and on Google Drive ----------

class State:
    """What is built and what each collection scored, in one small JSON file."""

    def __init__(self, path):
        self.path = Path(path)
        self.data = {"built": {}, "scores": {}}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                self.data["built"].update(loaded.get("built", {}))
                self.data["scores"].update(loaded.get("scores", {}))
            except ValueError:
                pass  # a damaged file starts again: the collections themselves are still in Qdrant

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(self.path.name + ".tmp")
        temp.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        os.replace(temp, self.path)

    def size(self):
        return len(self.data["built"]) + len(self.data["scores"])


def amount(path):
    """How much work a file holds: lines of a JSON lines file, or built plus scored collections of the state file. 0 when it is missing."""
    path = Path(path)
    if not path.exists():
        return 0
    if path.suffix == ".json":
        return State(path).size()
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


class Workspace:
    """Work files on this computer (fast) with a copy on Google Drive (safe). save() copies to Drive, load() brings back from Drive."""

    FILES = ("state.json", "context_lines.jsonl")

    def __init__(self, local, remote):
        self.local, self.remote = Path(local), Path(remote)
        self.local.mkdir(parents=True, exist_ok=True)
        self.remote.mkdir(parents=True, exist_ok=True)

    def path(self, name):
        return self.local / name

    @staticmethod
    def _copy(source, target):
        temp = target.with_name(target.name + ".tmp")
        shutil.copyfile(source, temp)
        os.replace(temp, target)

    def save(self):
        """Copy what is here to Drive, when it holds more work than the copy on Drive. Returns the names copied."""
        done = []
        for name in self.FILES:
            if amount(self.local / name) and amount(self.local / name) >= amount(self.remote / name):
                self._copy(self.local / name, self.remote / name)
                done.append(name)
        return done

    def load(self):
        """Copy back from Drive what Drive holds more of. Returns the names copied."""
        done = []
        for name in self.FILES:
            if amount(self.remote / name) > amount(self.local / name):
                self._copy(self.remote / name, self.local / name)
                done.append(name)
        return done


# ---------- build and score ----------

def collection_exists(client, name):
    if hasattr(client, "collection_exists"):
        return bool(client.collection_exists(name))
    return name in [item.name for item in client.get_collections().collections]


def pieces_in(client, name):
    return client.count(collection_name=name, exact=True).count if collection_exists(client, name) else 0


def ensure_built(spec, state, client, documents, lines=None, embed=None, count=None, progress=print):
    """Build the collection of this spec, unless it is already complete. Safe to run again after a stop. Returns the number of pieces."""
    check_name(spec.name)
    known = state.data["built"].get(spec.name)
    now = pieces_in(client, spec.name)
    if known and now and now == known["pieces"] and known["settings"] == list(spec.settings()):
        progress(f"  {spec.name}: already built, {now:,} pieces. Nothing to do.")
        return now
    if spec.context and not lines:
        raise RuntimeError("This collection needs the context lines first (technique 3).")
    config.use_profile(spec.profile)
    if embed is None or count is None:
        from ingest.embed import count_tokens, embed_texts

        embed, count = embed or embed_texts, count or count_tokens
    from ingest import build_index

    piece_tokens = spec.piece_tokens if spec.piece_tokens != config.PIECE_TOKENS else None
    progress(f"  {spec.name}: {spec.title}, model {config.EMBED_MODEL}, {len(documents)} documents ...")
    totals = build_index.build(documents, client, embed, count, spec.name, with_header=spec.header, piece_tokens=piece_tokens,
                               context=lines if spec.context else None, on_disk=True, numbers=True)
    if totals["qdrant_these"] != totals["pieces"]:
        raise RuntimeError(f"{spec.name}: Qdrant holds {totals['qdrant_these']} pieces but {totals['pieces']} were cut. Run this cell again.")
    state.data["built"][spec.name] = {"settings": list(spec.settings()), "pieces": totals["pieces"], "seconds": round(totals["seconds"]), "time": time.strftime("%Y-%m-%d %H:%M")}
    state.data["scores"].pop(spec.name, None)  # a rebuilt collection must be scored again
    state.save()
    progress(f"  {spec.name}: {totals['pieces']:,} pieces in {totals['seconds'] / 60:.1f} min.")
    return totals["pieces"]


def questions():
    """(the 85 practice and test cases, {case id: search line}). The lines are read from the rewrite cache: no Claude call."""
    from evals import experiments
    from evals.judge import JsonlCache

    cases = experiments.usable_cases()
    cache = JsonlCache(Path(config.DATA_DIR) / "cache" / "rewrite_cache.jsonl")
    return cases, compare_collections.cached_lines(cases, cache, SCORE_PROMPT, SCORE_CLEAN)


def ensure_scored(name, state, client, cases, lines, search_fn=None, progress=print):
    """The score of one collection (kept in the state file). Scored again only when the number of pieces changed."""
    now = pieces_in(client, name) if client is not None else None
    known = state.data["scores"].get(name)
    if known and known.get("pieces") == now:
        return known
    item = compare_collections.score_collection(name, cases, lines, client=client, search_fn=search_fn)
    state.data["scores"][name] = item
    state.save()
    progress(f"  {name}: the bot reads the right page for {item['read3']} of {item['n']} questions (best 5: {item['best5']}).")
    return item


def verdict(scores, spec):
    """One plain line: how this technique did against filings_v1."""
    base, item = scores.get(BASE.name), scores.get(spec.name)
    if base is None or item is None:
        return f"{spec.title}: not scored yet."
    gain = item["read3"] - base["read3"]
    word = "WINS" if wins(item, base) else "Worse" if gain < 0 else "The same" if gain == 0 else "A little better, but one question can be luck"
    return (f"{spec.title}: the bot reads the right page for {item['read3']} of {item['n']} questions ({gain:+d} against filings_v1; "
            f"best 5: {item['best5'] - base['best5']:+d}). {word}.")


def show(state, names=None):
    """Print the table of the scored collections (best first)."""
    items = [state.data["scores"][name] for name in (names or state.data["scores"]) if name in state.data["scores"]]
    if items:
        print("\n".join(compare_collections.table_lines(items)))
    return items


# ---------- the choice ----------

def wins(item, base, by=WIN_BY):
    """True when this collection finds the right page for at least `by` more questions than the base (as the bot reads them)."""
    return item["read3"] >= base["read3"] + by


def decide(scores):
    """Which setting of each technique to keep. Returns (winners, lines).

    winners: {"header": bool, "profile": "small" or "large", "context": bool, "piece_tokens": int}. Only collections that were scored count.
    """
    base = scores.get(BASE.name)
    winners = {"header": True, "profile": "small", "context": False, "piece_tokens": BASE.piece_tokens}
    lines = []
    if base is None:
        return winners, ["filings_v1 has no score, so nothing can be compared."]
    lines.append(f"Today (filings_v1): the bot reads the right page for {base['read3']} of {base['n']} questions (best 5: {base['best5']}).")

    def judge(spec):
        item = scores.get(spec.name)
        if item is None:
            lines.append(f"  technique {spec.technique}, {spec.title}: not scored, so it is left out.")
            return None
        won = wins(item, base)
        lines.append(f"  technique {spec.technique}, {spec.title}: {item['read3']} of {item['n']} ({item['read3'] - base['read3']:+d}, best 5 {item['best5'] - base['best5']:+d}) -> {'WINS' if won else 'does not win'}")
        return item if won else None

    if judge(HEADER_OFF):
        winners["header"] = False
    if judge(LARGE):
        winners["profile"] = "large"
    if judge(CONTEXT):
        winners["context"] = True
    sizes = [(item, spec) for spec in (SIZE_SMALL, SIZE_BIG) for item in [judge(spec)] if item]
    if sizes:
        winners["piece_tokens"] = max(sizes, key=lambda pair: compare_collections.rank_key(pair[0]))[1].piece_tokens
    return winners, lines


def combined_specs(winners):
    """The two collections of technique 5: the winning settings with the small model and with the large model.

    Returns [(spec, same_as)]: same_as is the name of a collection that already has exactly these settings (then nothing is built), or None.
    """
    out = []
    for profile in ("small", "large"):
        spec = Spec(f"{PREFIX}best_{profile}", profile, winners["header"], winners["piece_tokens"], winners["context"], "5", f"winners combined, {profile} model")
        same = next((known.name for known in KNOWN if known.settings() == spec.settings()), None)
        out.append((spec, same))
    return out


def recommend(scores):
    """(the best collection or None, lines). None means: keep filings_v1."""
    base = scores.get(BASE.name)
    if not scores or base is None:
        return None, ["No scores yet."]
    best = max(scores.values(), key=compare_collections.rank_key)
    lines = []
    if best["collection"] == BASE.name or not wins(best, base):
        lines.append(f"Nothing beats filings_v1 by {WIN_BY} or more questions. Keep filings_v1 and the small model: no change in config.py.")
        return None, lines
    lines.append(f"Best: {best['collection']} ({best['profile']} model): {best['read3']} of {best['n']} questions, {best['read3'] - base['read3']:+d} against filings_v1 (best 5: {best['best5']}).")
    lines.append("To use it, change these lines in config.py (and run the tests):")
    lines.extend(config_lines(best))
    return best, lines


def other_model(scores, best):
    """(the best collection of the other model, lines). For when the large model is too big for the web page, or the other model is the one you want.

    The best collection of the model that `best` does not use, from everything that was scored (a single technique can be better than a combination).
    """
    others = [item for item in scores.values() if best and item["profile"] != best["profile"]]
    base = scores.get(BASE.name)
    if not others or base is None:
        return None, []
    item = max(others, key=compare_collections.rank_key)
    if item["collection"] == BASE.name:
        return item, [f"With the {item['profile']} model nothing beats filings_v1, so there is nothing to change for it."]
    lines = [f"The best collection with the {item['profile']} model: {item['collection']}: {item['read3']} of {item['n']} questions ({item['read3'] - base['read3']:+d} against filings_v1). To use it instead:"]
    return item, lines + config_lines(item)


def config_lines(item):
    lines = [f'    COLLECTION = "{item["collection"]}"', f'    EMBED_PROFILE = "{item["profile"]}"   # the question must be embedded with the same model']
    if item["profile"] == "large":
        lines.append("    # The large model needs about 1.3 GB of memory when a question is searched. A free Streamlit page may not hold it.")
    return lines
''',
    'scripts/check_large.py': r'''"""Check an embedding model before you use it: does it load, does it give real numbers (no NaN), and are similar texts close?

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
''',
    'tests/test_upgrades.py': r'''"""Tests for the upgrades of 26 November (apply_upgrades.py): two embedding models, a chosen piece size, context lines, the compare tool and the lab.

No network, no model, no key, no Qdrant server: an in memory Qdrant, a fake embedder and a fake language model stand in.
Run from the project folder:
    python -m pytest -q
"""
import json
import math
import os
import re
import sys
import zlib
from functools import partial
from pathlib import Path

import pymupdf
import pytest
from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import check_large  # noqa: E402
import config  # noqa: E402
from bot import llm, lookup  # noqa: E402
from evals import compare_collections, lab  # noqa: E402
from evals.cases import Case, Evidence  # noqa: E402
from evals.judge import JsonlCache  # noqa: E402
from ingest import build_index, chunker, context_lines, embed  # noqa: E402
from ingest.names import parse_name  # noqa: E402
from ingest.read_pdf import Document, Page  # noqa: E402
from retrieval.search import Hit, search  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes")
INFO = parse_name("3M_2018_10K")


@pytest.fixture(autouse=True)
def keep_the_profile(monkeypatch):
    """A test may switch the model. Every test starts and ends with the profile it found."""
    before = config.EMBED_PROFILE
    yield
    config.use_profile(before)


# ---------- helpers ----------

def words(texts):
    return [len(text.split()) for text in texts]  # one token per word


def page_of(lines, number=1):
    return Page(number, lines, False)


def text_lines(count, tag="x"):
    """Lines of exactly 10 words each."""
    return [f"{tag}{i} " + " ".join(["word"] * 9) for i in range(count)]


def make_pdf(path, pages):
    pdf = pymupdf.open()
    for items in pages:
        page = pdf.new_page()
        for x, y, text in items:
            page.insert_text((x, y), text, fontsize=10)
    pdf.save(str(path))
    pdf.close()


def prose_page(tag, lines=12):
    return [(72, 60 + 14 * i, f"{tag} line {i} has some plain words about the company results and plans") for i in range(lines)]


def make_document(folder, name, pages, source="real"):
    path = folder / f"{name}.pdf"
    make_pdf(path, pages)
    return Document(parse_name(name), path, source)


class FakeEmbedder:
    """Vectors of a chosen size. Remembers every text it was asked to embed."""

    def __init__(self, dim=None):
        self.dim, self.texts = dim or config.EMBED_DIM, []

    def __call__(self, texts):
        self.texts.extend(texts)
        return [[1.0 + len(text) % 7] * self.dim for text in texts]


def bag_of_words(dim):
    def embed_fn(texts):
        vectors = []
        for text in texts:
            vector = [0.0] * dim
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vector[zlib.crc32(word.encode()) % dim] += 1.0
            length = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / length for value in vector])
        return vectors

    return embed_fn


# ---------- two models ----------

def test_the_default_model_is_the_small_one():
    if os.getenv("EMBED_PROFILE"):
        pytest.skip("EMBED_PROFILE is set on this computer")
    assert config.EMBED_PROFILE == "small"
    assert (config.EMBED_MODEL, config.EMBED_DIM) == ("BAAI/bge-small-en-v1.5", 384)


def test_use_profile_switches_the_model_and_the_size_and_refuses_an_unknown_name():
    assert config.use_profile("large") == ("BAAI/bge-large-en-v1.5", 1024)
    assert (config.EMBED_PROFILE, config.EMBED_MODEL, config.EMBED_DIM) == ("large", "BAAI/bge-large-en-v1.5", 1024)
    assert config.use_profile(" Small ") == ("BAAI/bge-small-en-v1.5", 384)
    with pytest.raises(ValueError):
        config.use_profile("medium")


def test_the_embedder_uses_the_model_of_the_profile_or_the_one_it_is_given(monkeypatch):
    import numpy as np

    asked = []

    class Model:
        def __init__(self, name):
            self.name = name

        def encode(self, texts, **options):
            asked.append(self.name)
            return np.ones((len(texts), 4), dtype="float32")

    monkeypatch.setattr(embed, "_model", lambda name: Model(name))
    assert len(embed.embed_texts(["a", "b"])) == 2
    config.use_profile("large")
    embed.embed_texts(["a"])
    embed.embed_texts(["a"], model="some/other-model")
    assert asked == ["BAAI/bge-small-en-v1.5", "BAAI/bge-large-en-v1.5", "some/other-model"]


def test_numbers_that_are_not_real_stop_the_embedder(monkeypatch):
    import numpy as np

    class Broken:
        def encode(self, texts, **options):
            return np.full((len(texts), 4), np.nan, dtype="float32")

    monkeypatch.setattr(embed, "_model", lambda name: Broken())
    with pytest.raises(RuntimeError, match="NaN"):
        embed.embed_texts(["a"])
    assert embed.all_finite([[0.1, 0.2]]) and not embed.all_finite([[0.1, float("inf")]])


def test_the_token_counter_uses_the_model_of_the_profile(monkeypatch):
    names = []

    def tokenizer(name):
        names.append(name)
        return lambda texts, add_special_tokens=False: {"input_ids": [t.split() for t in texts]}

    monkeypatch.setattr(embed, "_tokenizer", tokenizer)
    config.use_profile("large")
    assert embed.count_tokens(["a b c", "d"]) == [3, 1]
    assert names == ["BAAI/bge-large-en-v1.5"]


# ---------- piece size ----------

def test_a_chosen_piece_size_changes_the_pieces_and_nothing_is_lost():
    lines = text_lines(120)
    base = chunker.cut_document(INFO, [page_of(lines)], words)
    small = chunker.cut_document(INFO, [page_of(lines)], words, piece_tokens=250)
    big = chunker.cut_document(INFO, [page_of(lines)], words, piece_tokens=480)
    assert len(small) > len(base) >= len(big)
    assert all(piece.tokens <= config.PIECE_HARD_STOP for piece in small + big)
    for pieces in (base, small, big):
        assert {line for piece in pieces for line in piece.text.splitlines()[1:]} == set(lines)
    assert [p.text for p in chunker.cut_document(INFO, [page_of(lines)], words, piece_tokens=400)] == [p.text for p in base]


def test_a_page_that_fits_one_and_a_fifth_times_the_size_stays_whole():
    assert len(chunker.cut_document(INFO, [page_of(text_lines(25))], words, piece_tokens=250)) == 1  # 255 tokens
    assert len(chunker.cut_document(INFO, [page_of(text_lines(32))], words, piece_tokens=250)) >= 2  # 325 tokens
    assert len(chunker.cut_document(INFO, [page_of(text_lines(32))], words)) == 1  # the default keeps a page up to 480 whole


def test_a_piece_size_outside_the_allowed_range_is_refused():
    for bad in (100, 481, 5000):
        with pytest.raises(ValueError, match="piece_tokens"):
            chunker.cut_document(INFO, [page_of(text_lines(40))], words, piece_tokens=bad)


# ---------- the index builder ----------

def test_a_context_line_is_embedded_but_the_stored_text_stays_the_same(tmp_path):
    document = make_document(tmp_path, "ADOBE_2022_10K", [prose_page("a", 40), prose_page("b")])
    client, embedder = QdrantClient(":memory:"), FakeEmbedder()
    lines = {("ADOBE_2022_10K", 1): "Income statement of Adobe for 2022 in millions", ("ADOBE_2022_10K", 2): "Notes to the statements"}
    first = build_index.build([document], client, embedder, words, "ctx", context=lines)
    points, _ = client.scroll("ctx", limit=100, with_payload=True)
    assert len(points) == first["pieces"] and first["pieces"] >= 3
    for point in points:
        payload = point.payload
        assert payload["context"] == lines[(payload["doc"], payload["page"])]
        assert payload["text"].startswith("Adobe FY2022 10-K, page ")  # what the bot reads is not changed
    assert len(embedder.texts) == first["pieces"] and all(text.startswith(tuple(lines.values())) for text in embedder.texts)

    again = build_index.build([document], client, embedder, words, "ctx", context=lines)
    assert again["new"] == 0 and again["changed"] == 0 and again["same"] == first["pieces"]
    lines[("ADOBE_2022_10K", 2)] = "A different line"
    third = build_index.build([document], client, embedder, words, "ctx", context=lines)
    assert third["new"] == 0 and third["changed"] == 1  # only the pieces of that page are embedded again


def test_without_context_the_payload_is_the_old_one(tmp_path):
    document = make_document(tmp_path, "ADOBE_2022_10K", [prose_page("a")])
    client = QdrantClient(":memory:")
    build_index.build([document], client, FakeEmbedder(), words, "plain")
    points, _ = client.scroll("plain", limit=10, with_payload=True)
    assert all("context" not in point.payload for point in points)
    piece = chunker.cut_document(document.info, [page_of(["x y z"], 1)], words)[0]
    assert build_index.make_payload(document, piece)["sha"] == build_index.make_payload(document, piece, "")["sha"]
    assert build_index.make_payload(document, piece)["sha"] != build_index.make_payload(document, piece, "a line")["sha"]


def test_a_piece_size_can_be_chosen_for_a_build(tmp_path):
    document = make_document(tmp_path, "ADOBE_2022_10K", [prose_page("a", 40), prose_page("b", 40)])
    client = QdrantClient(":memory:")
    default = build_index.build([document], client, FakeEmbedder(), words, "d400")
    small = build_index.build([document], client, FakeEmbedder(), words, "d250", piece_tokens=250)
    assert small["pieces"] > default["pieces"]


def test_a_new_collection_can_have_other_numbers_indexes_and_texts_on_disk(monkeypatch):
    client = QdrantClient(":memory:")
    made = []
    monkeypatch.setattr(client, "create_payload_index", lambda collection, field, kind: made.append((collection, field, str(kind))))  # in memory Qdrant keeps no indexes
    build_index.ensure_collection(client, "big", dim=1024, on_disk=True, numbers=True)
    assert client.get_collection("big").config.params.vectors.size == 1024
    assert [(field, "integer" in kind.lower()) for _, field, kind in made] == [("company", False), ("doc", False), ("access", False), ("page", True), ("year", True)]
    made.clear()
    build_index.ensure_collection(client, "plain_one", dim=384)  # the default: no number indexes
    assert [field for _, field, _ in made] == ["company", "doc", "access"]


def test_an_existing_collection_with_other_numbers_stops_the_run():
    client = QdrantClient(":memory:")
    build_index.ensure_collection(client, "small_one")
    with pytest.raises(SystemExit, match="384 numbers"):
        build_index.ensure_collection(client, "small_one", dim=1024)


def test_a_new_kind_of_index_needs_a_new_collection_name():
    for option in (["--piece-tokens", "250"], ["--profile", "large"], ["--context", "x.jsonl"]):
        with pytest.raises(SystemExit, match="--collection"):
            build_index.main(option)


# ---------- context lines ----------

def test_clean_line_takes_out_labels_markdown_and_quotes():
    assert context_lines.clean_line("Context: **Consolidated balance sheet of 3M for 2018, in millions**\nmore text") == "Consolidated balance sheet of 3M for 2018, in millions"
    assert context_lines.clean_line('\n\n"Line: Cash flow statement of the company in 2019"') == "Cash flow statement of the company in 2019"
    assert context_lines.clean_line("") == "" and context_lines.clean_line("ok") == "" and context_lines.clean_line(None) == ""
    assert len(context_lines.clean_line(" ".join(["word"] * 100))) <= context_lines.LINE_CHARS


def test_title_line_uses_the_first_lines_that_look_like_a_title():
    lines = ["3M Company and Subsidiaries", "12", "Consolidated Statement of Income", "(Millions, except per share amounts)", "Net sales | 32,765"]
    assert context_lines.title_line(lines) == "3M Company and Subsidiaries - Consolidated Statement of Income - (Millions, except per share amounts)"
    assert context_lines.title_line(["12", "34"]) == ""


def test_the_lines_file_is_only_added_to_and_a_half_written_line_is_ignored(tmp_path):
    path = tmp_path / "lines.jsonl"
    assert context_lines.load_lines(path) == {}
    context_lines.append_lines(path, [("A_2020_10K", 3, "first")])
    context_lines.append_lines(path, [("A_2020_10K", 4, "second")])
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"doc": "A_2020_10K", "pa')  # the run was stopped in the middle of a line
    assert context_lines.load_lines(path) == {("A_2020_10K", 3): "first", ("A_2020_10K", 4): "second"}


def test_generate_lines_writes_continues_and_falls_back_to_the_page_title(tmp_path):
    real = make_document(tmp_path, "ADOBE_2022_10K", [prose_page("a"), prose_page("b"), [(72, 72, "Hello")]])
    fake = make_document(tmp_path, "AES_2022_10K", [prose_page("c")], source="fake")
    seen = []

    def ask(prompts):
        seen.extend(prompts)
        return ["Income statement of Adobe for 2022: revenue and net income" if "a line 0" in user else "" for _, user in prompts]

    store = tmp_path / "lines.jsonl"
    result = context_lines.generate_lines([real, fake], ask, store, batch=1, progress=lambda *_: None)
    assert result["made"] == 2 and result["fallbacks"] == 1 and len(seen) == 2  # the fake file and the empty page 3 get no line
    lines = context_lines.load_lines(store)
    assert set(lines) == {("ADOBE_2022_10K", 1), ("ADOBE_2022_10K", 2)}
    assert lines[("ADOBE_2022_10K", 1)].startswith("Income statement of Adobe")
    assert lines[("ADOBE_2022_10K", 2)].startswith("b line 0 has some plain words")  # the title line, because the model gave nothing
    assert "Adobe FY2022 10-K, page 1" in seen[0][1] or "Adobe FY2022 10-K, page 1" in seen[1][1]
    again = context_lines.generate_lines([real, fake], ask, store, batch=1, progress=lambda *_: None)
    assert again["made"] == 0 and again["already"] == 2 and len(seen) == 2


def test_a_page_with_nothing_usable_gets_an_empty_line_and_is_not_tried_again(tmp_path):
    numbers = [(72, 60 + 14 * i, "1234567 | 8901234 | 5678901 | 2345678") for i in range(12)]  # enough digits to count as text, but no title line
    real = make_document(tmp_path, "ADOBE_2022_10K", [prose_page("a"), numbers])
    asked = []

    def ask(prompts):
        asked.extend(prompts)
        return ["" for _ in prompts]

    store = tmp_path / "lines.jsonl"
    result = context_lines.generate_lines([real], ask, store, batch=1, progress=lambda *_: None)
    assert result["made"] == 2 and result["fallbacks"] == 2 and result["empty"] == 1 and result["total"] == 2
    lines = context_lines.load_lines(store)
    assert set(lines) == {("ADOBE_2022_10K", 1), ("ADOBE_2022_10K", 2)} and lines[("ADOBE_2022_10K", 2)] == ""
    again = context_lines.generate_lines([real], ask, store, batch=1, progress=lambda *_: None)
    assert again["made"] == 0 and len(asked) == 2  # the empty page is not asked again
    built = FakeEmbedder()
    totals = build_index.build([real], QdrantClient(":memory:"), built, lambda texts: [50] * len(texts), "filings_v2_context", context=lines)
    assert totals["pieces"] >= 2 and not any(text.startswith("\n") for text in built.texts)  # no stray empty context line in front of a piece


def test_generate_lines_calls_the_checkpoint(tmp_path):
    real = make_document(tmp_path, "ADOBE_2022_10K", [prose_page("a"), prose_page("b")])
    calls = []
    context_lines.generate_lines([real], lambda prompts: ["A line about the page"] * len(prompts), tmp_path / "l.jsonl", batch=1,
                                 progress=lambda *_: None, checkpoint=lambda: calls.append(1), every=0)
    assert len(calls) >= 2


def test_a_full_gpu_memory_splits_the_batch_in_two():
    sizes = []

    def ask(prompts):
        sizes.append(len(prompts))
        if len(prompts) > 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
        return [f"reply {p}" for p in prompts]

    assert context_lines.ask_safely(ask, [1, 2, 3, 4]) == ["reply 1", "reply 2", "reply 3", "reply 4"]
    assert sizes == [4, 2, 2]
    def broken(prompts):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        context_lines.ask_safely(broken, [1, 2])


def test_the_context_tool_downloads_nothing_without_permission(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("ingest.read_pdf.list_documents", lambda only=None: [])
    assert context_lines.main(["--out", str(tmp_path / "x.jsonl")]) == 0
    assert "Nothing was started" in capsys.readouterr().out


# ---------- the compare tool ----------

DOCS = {
    "ADOBE_2022_10K": ["cloud subscription revenue grew because of new customers and renewals", "net income was higher because operating expenses grew more slowly"],
    "AES_2022_10K": ["wind and solar power plants produced electricity for the national grid", "coal plants were retired and replaced with natural gas and battery storage"],
}


def make_collection(tmp_path, client, name, dim, header=True):
    documents = [make_document(tmp_path, doc, [[(72, 72 + 14 * i, line) for i, line in enumerate(lines)]]) for doc, lines in DOCS.items()]
    build_index.build(documents, client, bag_of_words(dim), words, name, with_header=header, dim=dim)
    return documents


def make_case(number, doc, page, question):
    return Case(id=f"financebench_id_{number:05d}", split="test", company="X", doc=doc, doc_type="10K", question_type="x", reasoning_type="Information extraction",
                question=question, answer="a", justification="", evidence=(Evidence(page, "t"),))


CASES = [
    make_case(1, "ADOBE_2022_10K", 1, "How did cloud subscription revenue increase because of renewals"),
    make_case(2, "AES_2022_10K", 1, "Which power plants produced electricity for the grid"),
    make_case(3, "AES_2022_10K", 2, "Where did the sale of the water utility close"),  # not in the documents: a miss
]


@pytest.fixture
def no_filter(monkeypatch):
    monkeypatch.setattr(config, "FILTER_MODE", "off")


def test_the_profile_of_a_collection_comes_from_its_vector_size(tmp_path):
    client = QdrantClient(":memory:")
    make_collection(tmp_path, client, "c384", 384)
    make_collection(tmp_path, client, "c1024", 1024)
    assert compare_collections.profile_of(client, "c384") == "small"
    assert compare_collections.profile_of(client, "c1024") == "large"
    build_index.ensure_collection(client, "odd", dim=512)
    with pytest.raises(ValueError, match="512 numbers"):
        compare_collections.profile_of(client, "odd")


def test_a_collection_is_scored_on_the_best_5_and_on_the_pages_the_bot_reads(tmp_path, no_filter):
    client = QdrantClient(":memory:")
    make_collection(tmp_path, client, "c384", 384)
    lines = {case.id: case.question for case in CASES}
    search_fn = partial(search, client=client, embed=bag_of_words(384))
    item = compare_collections.score_collection("c384", CASES, lines, client=client, search_fn=search_fn)
    assert (item["n"], item["pieces"], item["profile"]) == (3, 2, "small")
    assert item["best5"] == 2 and item["read3"] == 2 and item["first"] == 2
    assert item["places"][CASES[2].id] is None and abs(item["mrr"] - 2 / 3) < 1e-3
    table = "\n".join(compare_collections.table_lines([item]))
    assert "c384" in table and "2 of 3" in table


def test_the_pages_the_bot_reads_are_fewer_than_the_best_5_pieces(tmp_path, no_filter, monkeypatch):
    # pieces of one page fill the best 5: the right page is the 4th page, so it is among the best 5 pieces but not among the 3 pages the bot reads
    hits = [Hit(1.0, "A_2020_10K", "A", 2020, "FY", "10K", page, 1, "t", "real", "public") for page in (1, 1, 2, 3, 4)]
    item = compare_collections.score_collection("any", [make_case(9, "A_2020_10K", 4, "q")], {"financebench_id_00009": "q"}, search_fn=lambda *a, **k: hits)
    assert item["best5"] == 1 and item["read3"] == 0


def test_search_lines_come_from_the_cache_and_a_missing_one_stops_without_asking_claude(tmp_path):
    cache = JsonlCache(tmp_path / "rewrite_cache.jsonl")

    def ask(system, user, max_tokens=120):
        return llm.Reply("net income Adobe 2022", 700, 20, 0.01, 0.1, "claude-test", "end_turn", 1)

    lookup.make_queries(CASES[0].question, count=1, hints=True, ask_fn=ask, cache=cache, prompt="rewrite_v2", clean=False)
    lookup.make_queries(CASES[1].question, count=1, hints=True, ask_fn=ask, cache=cache, prompt="rewrite_v2", clean=False)
    found = compare_collections.cached_lines(CASES[:2], cache, "rewrite_v2", False)
    assert set(found.values()) == {"net income Adobe 2022"}
    with pytest.raises(compare_collections.NotCached, match="1 of 3"):
        compare_collections.cached_lines(CASES, cache, "rewrite_v2", False)


# ---------- the lab ----------

def score(name, read3, best5=None, n=85, profile="small", mrr=0.5, pieces=100):
    return {"collection": name, "profile": profile, "pieces": pieces, "n": n, "best5": read3 if best5 is None else best5, "read3": read3, "mrr": mrr, "first": 10, "places": {}}


def test_the_lab_never_writes_to_the_collections_you_have():
    for name in ("filings_v1", "filings_v1_plain", "other"):
        with pytest.raises(ValueError):
            lab.check_name(name)
    assert lab.check_name("filings_v2_large") == "filings_v2_large"


def test_the_state_file_is_kept_and_a_damaged_one_starts_again(tmp_path):
    state = lab.State(tmp_path / "state.json")
    state.data["built"]["filings_v2_large"] = {"pieces": 5}
    state.save()
    assert lab.State(tmp_path / "state.json").data["built"] == {"filings_v2_large": {"pieces": 5}}
    (tmp_path / "state.json").write_text("{damaged", encoding="utf-8")
    assert lab.State(tmp_path / "state.json").size() == 0


def test_the_workspace_saves_to_drive_and_loads_back_and_never_replaces_more_work_with_less(tmp_path):
    first = lab.Workspace(tmp_path / "local1", tmp_path / "drive")
    context_lines.append_lines(first.path("context_lines.jsonl"), [("A_2020_10K", n, "line") for n in range(1, 6)])
    state = lab.State(first.path("state.json"))
    state.data["scores"]["filings_v1"] = score("filings_v1", 60)
    state.save()
    assert sorted(first.save()) == ["context_lines.jsonl", "state.json"]
    second = lab.Workspace(tmp_path / "local2", tmp_path / "drive")  # a new Colab session: nothing is here yet
    assert sorted(second.load()) == ["context_lines.jsonl", "state.json"]
    assert len(context_lines.load_lines(second.path("context_lines.jsonl"))) == 5 and lab.State(second.path("state.json")).size() == 1
    context_lines.append_lines(second.path("context_lines.jsonl"), [("A_2020_10K", 9, "x")])
    third = lab.Workspace(tmp_path / "local3", tmp_path / "drive")
    context_lines.append_lines(third.path("context_lines.jsonl"), [("A_2020_10K", 1, "only one line")])
    assert "context_lines.jsonl" not in third.save()  # Drive holds 5 lines, this one holds 1: Drive is kept
    assert len(context_lines.load_lines(tmp_path / "drive" / "context_lines.jsonl")) == 5
    assert sorted(second.save()) == ["context_lines.jsonl", "state.json"]  # 6 lines beat 5


def test_a_technique_wins_when_the_bot_reads_two_more_questions():
    scores = {"filings_v1": score("filings_v1", 60), "filings_v2_noheader": score("filings_v2_noheader", 62), "filings_v2_large": score("filings_v2_large", 61),
              "filings_v2_context": score("filings_v2_context", 66), "filings_v2_size250": score("filings_v2_size250", 58), "filings_v2_size480": score("filings_v2_size480", 63)}
    winners, lines = lab.decide(scores)
    assert winners == {"header": False, "profile": "small", "context": True, "piece_tokens": 480}
    text = "\n".join(lines)
    assert "technique 1, header line off" in text and "WINS" in text and "does not win" in text


def test_the_larger_gain_picks_the_piece_size_and_a_missing_score_is_left_out():
    scores = {"filings_v1": score("filings_v1", 60), "filings_v2_size250": score("filings_v2_size250", 65), "filings_v2_size480": score("filings_v2_size480", 63)}
    winners, lines = lab.decide(scores)
    assert winners["piece_tokens"] == 250 and winners["header"] is True and winners["profile"] == "small"
    assert any("not scored" in line for line in lines)
    assert lab.decide({})[0] == {"header": True, "profile": "small", "context": False, "piece_tokens": 400}


def test_the_combined_collections_reuse_a_collection_that_has_the_same_settings():
    nothing = {"header": True, "profile": "small", "context": False, "piece_tokens": 400}
    small, large = lab.combined_specs(nothing)
    assert small[1] == "filings_v1" and large[1] == "filings_v2_large"
    one = {"header": False, "profile": "small", "context": False, "piece_tokens": 400}
    small, large = lab.combined_specs(one)
    assert small[1] == "filings_v2_noheader" and large[1] is None
    assert (large[0].name, large[0].profile, large[0].header) == ("filings_v2_best_large", "large", False)
    both = {"header": False, "profile": "large", "context": True, "piece_tokens": 480}
    small, large = lab.combined_specs(both)
    assert small[1] is None and large[1] is None and small[0].context and small[0].piece_tokens == 480
    assert all(spec.name.startswith(lab.PREFIX) for spec, _ in (small, large))


def test_the_verdict_says_in_plain_words_how_a_technique_did():
    scores = {"filings_v1": score("filings_v1", 60), "filings_v2_large": score("filings_v2_large", 63), "filings_v2_context": score("filings_v2_context", 61),
              "filings_v2_noheader": score("filings_v2_noheader", 60), "filings_v2_size250": score("filings_v2_size250", 55)}
    assert lab.verdict(scores, lab.LARGE).endswith("WINS.") and "+3 against filings_v1" in lab.verdict(scores, lab.LARGE)
    assert "A little better" in lab.verdict(scores, lab.CONTEXT) and lab.verdict(scores, lab.HEADER_OFF).endswith("The same.")
    assert lab.verdict(scores, lab.SIZE_SMALL).endswith("Worse.") and "not scored" in lab.verdict(scores, lab.SIZE_BIG)


def test_the_other_model_is_the_best_collection_of_that_model_even_when_it_is_a_single_technique():
    scores = {"filings_v1": score("filings_v1", 60), "filings_v2_noheader": score("filings_v2_noheader", 64), "filings_v2_best_small": score("filings_v2_best_small", 62),
              "filings_v2_best_large": score("filings_v2_best_large", 66, profile="large")}
    best, _ = lab.recommend(scores)
    assert best["collection"] == "filings_v2_best_large"
    item, lines = lab.other_model(scores, best)
    assert item["collection"] == "filings_v2_noheader" and "+4 against filings_v1" in lines[0] and 'COLLECTION = "filings_v2_noheader"' in lines[1]
    only_base = {"filings_v1": score("filings_v1", 60), "filings_v2_best_large": score("filings_v2_best_large", 66, profile="large")}
    item, lines = lab.other_model(only_base, lab.recommend(only_base)[0])
    assert item["collection"] == "filings_v1" and "nothing beats filings_v1" in lines[0]
    assert lab.other_model(scores, None) == (None, [])


def test_the_recommendation_names_the_best_collection_or_says_to_keep_filings_v1():
    scores = {"filings_v1": score("filings_v1", 60), "filings_v2_large": score("filings_v2_large", 66, profile="large"), "filings_v2_context": score("filings_v2_context", 61)}
    best, lines = lab.recommend(scores)
    text = "\n".join(lines)
    assert best["collection"] == "filings_v2_large" and 'COLLECTION = "filings_v2_large"' in text and 'EMBED_PROFILE = "large"' in text and "1.3 GB" in text
    best, lines = lab.recommend({"filings_v1": score("filings_v1", 60), "filings_v2_context": score("filings_v2_context", 61)})
    assert best is None and "Keep filings_v1" in "\n".join(lines)
    assert lab.recommend({})[0] is None


def test_ensure_built_builds_once_and_the_large_collection_has_1024_numbers(tmp_path):
    document = make_document(tmp_path, "ADOBE_2022_10K", [prose_page("a", 40), prose_page("b")])
    client, state = QdrantClient(":memory:"), lab.State(tmp_path / "state.json")
    embedder = FakeEmbedder(1024)
    said = []
    count = lab.ensure_built(lab.LARGE, state, client, [document], embed=embedder, count=words, progress=said.append)
    assert count > 2 and client.get_collection("filings_v2_large").config.params.vectors.size == 1024 and config.EMBED_PROFILE == "large"
    assert state.data["built"]["filings_v2_large"]["pieces"] == count
    used = len(embedder.texts)
    assert lab.ensure_built(lab.LARGE, state, client, [document], embed=embedder, count=words, progress=said.append) == count
    assert len(embedder.texts) == used and any("already built" in line for line in said)
    with pytest.raises(RuntimeError, match="context lines"):
        lab.ensure_built(lab.CONTEXT, state, client, [document], embed=FakeEmbedder(), count=words, progress=said.append)
    with pytest.raises(ValueError):
        lab.ensure_built(lab.BASE, state, client, [document], embed=FakeEmbedder(), count=words, progress=said.append)


def test_ensure_built_uses_the_header_the_size_and_the_context_of_the_spec(tmp_path):
    document = make_document(tmp_path, "ADOBE_2022_10K", [prose_page("a", 40)])
    client, state = QdrantClient(":memory:"), lab.State(tmp_path / "state.json")
    lab.ensure_built(lab.HEADER_OFF, state, client, [document], embed=FakeEmbedder(), count=words, progress=lambda *_: None)
    points, _ = client.scroll("filings_v2_noheader", limit=50, with_payload=True)
    assert all(not point.payload["text"].startswith("Adobe FY2022") for point in points)
    lines = {("ADOBE_2022_10K", 1): "Adobe page one"}
    embedder = FakeEmbedder()
    lab.ensure_built(lab.CONTEXT, state, client, [document], lines=lines, embed=embedder, count=words, progress=lambda *_: None)
    assert all(text.startswith("Adobe page one") for text in embedder.texts)
    lab.ensure_built(lab.SIZE_SMALL, state, client, [document], embed=FakeEmbedder(), count=words, progress=lambda *_: None)
    assert client.count("filings_v2_size250").count > client.count("filings_v2_noheader").count


def test_ensure_scored_scores_again_only_when_the_pieces_changed(tmp_path, no_filter):
    client, state = QdrantClient(":memory:"), lab.State(tmp_path / "state.json")
    make_collection(tmp_path, client, "filings_v2_large", 384)
    lines = {case.id: case.question for case in CASES}
    calls = []

    def search_fn(question, **options):
        calls.append(question)
        return partial(search, client=client, embed=bag_of_words(384))(question, **options)

    first = lab.ensure_scored("filings_v2_large", state, client, CASES, lines, search_fn=search_fn, progress=lambda *_: None)
    used = len(calls)
    assert used >= 3 and lab.ensure_scored("filings_v2_large", state, client, CASES, lines, search_fn=search_fn, progress=lambda *_: None) == first
    assert len(calls) == used and state.data["scores"]["filings_v2_large"]["read3"] == first["read3"]


# ---------- the model check ----------

def test_the_model_check_accepts_good_vectors_and_refuses_bad_ones():
    good = [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [0.6, 0.8]]
    ok, lines = check_large.check_vectors(good, 2)
    assert ok and all(line.startswith("  ok") for line in lines)
    for broken in ([[1.0, 0.0], [float("nan"), 0.6], [0.0, 1.0], [0.6, 0.8]], [[1.0, 0.0], [0.0, 1.0], [0.8, 0.6], [0.6, 0.8]], [[2.0, 0.0], [0.8, 0.6], [0.0, 1.0], [0.6, 0.8]]):
        assert check_large.check_vectors(broken, 2)[0] is False


def test_the_model_check_runs_with_a_fake_model_and_downloads_nothing_by_itself(capsys, monkeypatch):
    def fake(texts):
        return [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [0.6, 0.8]]

    config.use_profile("small")
    assert check_large.main(["--profile", "small"], embed=lambda texts: [vector + [0.0] * 382 for vector in fake(texts)]) == 0
    assert "The model is fine" in capsys.readouterr().out
    monkeypatch.setattr(embed, "model_on_disk", lambda name=None: False)
    assert check_large.main(["--profile", "large"]) == 0
    assert "Nothing was downloaded" in capsys.readouterr().out
''',
}

ADOPT = [
    ('config.py',
     r'''REWRITE_PROMPT = "rewrite_v2"  # the rewrite request: rewrite_v2 (in use) or rewrite_v3 (a line in the words of the filing). Test it first: python -m evals.rewrite_test --new''',
     r'''REWRITE_PROMPT = "rewrite_v3"  # the rewrite request: rewrite_v3 (in use: apply_upgrades.py switched it on after its 1 cent check) or rewrite_v2 (the old request). Check again: python -m evals.rewrite_test --new'''),
    ('config.py',
     r'''QUERY_CLEAN = False  # build rewrite: take markdown, a label such as "Query:" and a last word "search" out of the rewritten query. Test it first: python -m evals.rewrite_test''',
     r'''QUERY_CLEAN = True  # build rewrite: take markdown, a label such as "Query:" and a last word "search" out of the rewritten query (on together with rewrite_v3)'''),
]

WHY = {
    "config.py": "two models: EMBED_PROFILE small (default) or large",
    "ingest/embed.py": "the embedder follows the profile and stops on NaN",
    "ingest/chunker.py": "a piece size can be chosen (default 400)",
    "ingest/build_index.py": "new collections: --profile, --piece-tokens, --context",
    ".gitignore": "keeps the backup folder out of Git",
    "ingest/context_lines.py": "new: context lines from a small open model",
    "evals/compare_collections.py": "new: free comparison of collections",
    "evals/lab.py": "new: what the Colab notebook builds and how it chooses",
    "scripts/check_large.py": "new: model check (loads, no NaN, sensible)",
    "tests/test_upgrades.py": "new: tests for all of this",
}
BACKUP = ".upgrades_backup"
MARKERS = ("config.py", "bot/lookup.py", "evals/rewrite_test.py", "ingest/build_index.py")
TEST_COMMAND = ["-m", "pytest", "-q", "-p", "no:cacheprovider"]
TEST_SECONDS = 900
CHECK_SECONDS = 1500
NOW_NAME = "now: rewrite_v2 with word hints"  # the two variants of evals/rewrite_test.py that decide
NEW_NAME = "new: rewrite_v3 line, noise taken out"


def read_text(path):
    raw = path.read_bytes().decode("utf-8")
    return raw.replace("\r\n", "\n"), "\r\n" in raw


def write_text(path, text, crlf=False):
    data = text.replace("\n", "\r\n") if crlf else text
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(data.encode("utf-8"))
    os.replace(temp, path)


def short(text, width=70):
    return " ".join(text.split())[:width]


def find_root(given):
    candidates = [Path(given)] if given else [Path(__file__).resolve().parent, Path.cwd()]
    for folder in candidates:
        if all((folder / marker).exists() for marker in MARKERS):
            return folder.resolve()
    return None


def plan(root):
    """What would happen. Returns (texts: {file: (before, after, crlf)}, steps: [(file, what)], problems: [text])."""
    texts, steps, problems = {}, [], []
    for rel in dict.fromkeys(item[0] for item in HUNKS):
        path = root / rel
        if not path.exists():
            if rel != ".gitignore":  # no .gitignore means no Git here: that is fine
                problems.append(f"{rel} is missing.")
            continue
        before, crlf = read_text(path)
        texts[rel] = [before, before, crlf]
    for number, (rel, old, new) in enumerate(HUNKS, start=1):
        if rel not in texts:
            continue
        text = texts[rel][1]
        if new in text:
            steps.append((rel, "already"))
            continue
        count = text.count(old)
        if count == 1:
            texts[rel][1] = text.replace(old, new, 1)
            steps.append((rel, "change"))
        elif count == 0:
            problems.append(f"{rel}: change {number} does not fit. The text it looks for is not in your file (\"{short(old)}\"). Did you edit that place?")
        else:
            problems.append(f"{rel}: change {number} is not clear: its text is in the file {count} times (\"{short(old)}\").")
    for rel, content in NEW_FILES.items():
        path = root / rel
        if not path.exists():
            steps.append((rel, "create"))
        elif read_text(path)[0] == content:
            steps.append((rel, "already"))
        else:
            problems.append(f"{rel} exists and is different from the one in this file. Move it away and run again.")
    return texts, steps, problems


def summary_lines(steps):
    per = {}
    for rel, what in steps:
        per.setdefault(rel, []).append(what)
    lines = []
    for rel in sorted(per):
        todo, done = per[rel].count("change") + per[rel].count("create"), per[rel].count("already")
        state = ("new file" if "create" in per[rel] else f"{todo} change{'s' if todo != 1 else ''} to make") if todo else "already in place"
        if todo and done:
            state += f", {done} already in place"
        lines.append(f"  {rel:30} {state:26} {WHY.get(rel, '')}")
    return lines


def run_tests(root):
    """(returncode, last line, set of failed test names, tail of the output)."""
    started = time.time()
    try:
        done = subprocess.run([sys.executable] + TEST_COMMAND, cwd=str(root), capture_output=True, text=True, timeout=TEST_SECONDS, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return 1, f"the tests did not finish in {TEST_SECONDS} seconds", {"(timeout)"}, ""
    lines = [line for line in done.stdout.splitlines() if line.strip()]
    failed = {line.split(" - ")[0].replace("FAILED ", "").strip() for line in lines if line.startswith(("FAILED ", "ERROR "))}
    last = lines[-1] if lines else "no output"
    return done.returncode, f"{last} ({time.time() - started:.0f} s)", failed, "\n".join(lines[-25:])


def back_up(folder, root, rels):
    """Copy each file to this run's own backup folder (one folder for each run, so an old copy never replaces edits you made later)."""
    for rel in rels:
        if (root / rel).exists():
            target = folder / (rel + ".bak")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / rel, target)


def restore(folder, root, changed, created):
    for rel in changed:
        source = folder / (rel + ".bak")
        if source.exists():
            shutil.copy2(source, root / rel)
        elif (root / rel).exists():
            (root / rel).unlink()  # the file did not exist before this run
    for rel in created:
        if (root / rel).exists():
            (root / rel).unlink()


def new_run_folder(root):
    return root / BACKUP / time.strftime("%Y%m%d_%H%M%S")


# ---------- rewrite v3: a 1 cent check, then on or off ----------

def decide_rewrite(now, new):
    """(switch it on?, reason) from the two summaries of evals/rewrite_test.py: {hit, mrr, first}. On when v3 is not worse on both numbers."""
    line = (f"rewrite_v2 (now): the right page in the best 5 for {now['hit']} questions, MRR {now['mrr']:.3f}.  "
            f"rewrite_v3: {new['hit']} questions, MRR {new['mrr']:.3f}.")
    if new["hit"] >= now["hit"] and new["mrr"] >= now["mrr"]:
        return True, line + "  rewrite_v3 is not worse, so it is switched on."
    return False, line + "  rewrite_v3 is worse on at least one number, so rewrite_v2 stays."


def adopt_plan(text):
    """Apply the ADOPT changes to the text of config.py. Returns (new text, problem). problem is empty when all fit."""
    for number, (rel, old, new) in enumerate(ADOPT, start=1):
        if new in text:
            continue
        if text.count(old) != 1:
            return text, f"the line of change {number} in {rel} is not as expected (\"{short(old, 50)}\"): you changed it. Switch rewrite_v3 on by hand."
        text = text.replace(old, new, 1)
    return text, ""


def rewrite_step(root, args, folder=None):
    """Run python -m evals.rewrite_test --new --spend (about 1 cent), read its result and switch rewrite v3 on when it is not worse. Returns 0 or 1."""
    config_path = root / "config.py"
    text, crlf = read_text(config_path)
    adopted, problem = adopt_plan(text)
    if adopted == text and not problem:
        print("Rewrite v3 is already switched on in config.py. Nothing to check.")
        return 0
    if problem:
        print("Rewrite v3 check not done: " + problem)
        return 1
    print("\nRewrite v3 check: python -m evals.rewrite_test --new --spend  (search only, about 1 cent, takes a few minutes) ...", flush=True)
    result = root / "results" / "rewrite_test.json"
    started = time.time()
    try:
        done = subprocess.run([sys.executable, "-m", "evals.rewrite_test", "--new", "--spend"], cwd=str(root), capture_output=True, text=True,
                              timeout=CHECK_SECONDS, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        print(f"  The check did not finish in {CHECK_SECONDS} seconds. rewrite_v2 stays. Try again later:  python apply_upgrades.py --rewrite-check")
        return 1
    tail = "\n".join(line for line in (done.stdout + done.stderr).splitlines() if line.strip()).splitlines()[-14:]
    fresh = result.exists() and result.stat().st_mtime >= started - 2
    if done.returncode != 0 or not fresh:
        print("  The check did not run. Its last lines:\n    " + "\n    ".join(tail))
        print("  rewrite_v2 stays. Fix the problem (keys in .env, Qdrant reachable) and run:  python apply_upgrades.py --rewrite-check")
        return 1
    try:
        variants = json.loads(result.read_text(encoding="utf-8"))["variants"]
        now, new = variants[NOW_NAME]["summary"], variants[NEW_NAME]["summary"]
    except (ValueError, KeyError):
        print("  The result file of the check is not as expected. rewrite_v2 stays. Its last lines:\n    " + "\n    ".join(tail))
        return 1
    on, reason = decide_rewrite(now, new)
    print("  " + reason)
    if not on:
        return 0
    folder = folder or new_run_folder(root)
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, folder / "config.py.before_rewrite_v3.bak")
    write_text(config_path, adopted, crlf)
    if not args.no_test:
        print("  Running the tests once more with rewrite_v3 on ...")
        after = run_tests(root)
        print("  after:  " + after[1])
        if after[0] != 0:
            write_text(config_path, text, crlf)
            print("  A test broke with rewrite_v3 on, so config.py was put back (rewrite_v2). The tests that broke:\n    " + "\n    ".join(sorted(after[2]) or ["(see above)"]))
            return 1
    print("  config.py now has REWRITE_PROMPT = rewrite_v3 and QUERY_CLEAN = True.")
    return 0


# ---------- apply and undo ----------

def apply(root, args):
    texts, steps, problems = plan(root)
    print("\n".join(summary_lines(steps)))
    if problems:
        print("\nNothing was changed, because:")
        for problem in problems:
            print("  - " + problem)
        return 1
    to_change = [rel for rel, (before, after, crlf) in texts.items() if before != after]
    to_create = [rel for rel, what in steps if what == "create"]
    if not to_change and not to_create:
        print("\nEverything is already in place. Nothing to change.")
        if args.check or args.no_spend:
            return 0
        return rewrite_step(root, args)
    if args.check:
        print("\nCheck only: nothing was changed. Run without --check to apply.")
        return 0
    before_tests = None
    if not args.no_test:
        print("\nRunning your tests first (about a minute) ...")
        before_tests = run_tests(root)
        print("  before: " + before_tests[1])
    folder = new_run_folder(root)
    existed = [rel for rel in to_change if (root / rel).exists()]
    back_up(folder, root, existed)
    for rel in to_change:
        write_text(root / rel, texts[rel][1], texts[rel][2])
    for rel in to_create:
        write_text(root / rel, NEW_FILES[rel])
    (folder / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").write_text(json.dumps({"changed": to_change, "created": to_create, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1), encoding="utf-8")
    print(f"\nChanged {len(to_change)} files and created {len(to_create)}. The old files are in {BACKUP}\\{folder.name}\\ .")
    if args.no_test:
        print("Tests skipped (--no-test). Run:  python -m pytest -q")
    else:
        print("Running the tests again ...")
        after = run_tests(root)
        print("  after:  " + after[1])
        worse = sorted(after[2] - before_tests[2])
        if worse or (after[0] != 0 and before_tests[0] == 0):
            restore(folder, root, to_change, to_create)
            print("\nA test broke, so every file was put back as it was. Nothing is changed. The tests that broke:")
            for name in worse or ["(see the end of the output below)"]:
                print("  - " + name)
            print("\n" + after[3])
            return 1
    code = None if args.no_spend else rewrite_step(root, args, folder)
    print("\nDone. Next steps:\n" + next_steps(code))
    return 0


def undo(root, args):
    problems, texts, removed = [], {}, []
    for rel in dict.fromkeys(item[0] for item in list(HUNKS) + list(ADOPT)):
        path = root / rel
        if not path.exists():
            continue
        text, crlf = read_text(path)
        texts[rel] = [text, text, crlf]
    for number, (rel, old, new) in reversed(list(enumerate(list(HUNKS) + list(ADOPT), start=1))):
        if rel not in texts:
            continue
        text = texts[rel][1]
        count = text.count(new)
        if count == 1:
            texts[rel][1] = text.replace(new, old, 1)
        elif count > 1:
            problems.append(f"{rel}: change {number} cannot be taken out, its text is in the file {count} times.")
    for rel, content in NEW_FILES.items():
        path = root / rel
        if path.exists():
            if read_text(path)[0] == content:
                removed.append(rel)
            else:
                print(f"  kept {rel}: you changed it after it was made.")
    if problems:
        print("Nothing was changed, because:")
        for problem in problems:
            print("  - " + problem)
        return 1
    changed = [rel for rel, (before, after, crlf) in texts.items() if before != after]
    for rel in changed:
        write_text(root / rel, texts[rel][1], texts[rel][2])
    for rel in removed:
        (root / rel).unlink()
    print(f"Taken out: {len(changed)} files changed back, {len(removed)} new files removed. Your own later edits stay.")
    print(f"The old copies of the files are still in {BACKUP}\\ if you want them. Delete that folder when you are happy.")
    print("Your Qdrant collections are not touched. Collections made by the notebook (names start with filings_v2_) stay until you delete them in the Qdrant page.")
    return 0


def next_steps(code):
    """The list of what to do next. code: what the rewrite v3 check gave (None when it was skipped)."""
    if code is None:
        first = "  1. Rewrite v3: skipped (--no-spend). Run the check when you want (about 1 cent):  python apply_upgrades.py --rewrite-check"
    elif code == 0:
        first = "  1. Rewrite v3: checked above (about 1 cent). To run that check again:  python apply_upgrades.py --rewrite-check"
    else:
        first = "  1. Rewrite v3: the check did not finish (see above). Run it again later:  python apply_upgrades.py --rewrite-check"
    return first + "\n" + NEXT_STEPS


NEXT_STEPS = """  2. Optional, on this computer, 0 cents: does the large model load and give real numbers (downloads about 1.3 GB once)
       python scripts\\check_large.py --allow-download
  3. COLAB (free T4 GPU), 0 cents: open filings_colab.ipynb in Colab and run it from the top. It builds the new Qdrant collections
     (filings_v2_...), saves to Google Drive every few cells, and ends with the winner and the lines to change in config.py.
  4. Back on this computer, free, 0 cents: compare the collections yourself. The notebook prints this line at its end with the names
     of the collections that exist, use those names:
       python -m evals.compare_collections filings_v1 filings_v2_best_small filings_v2_best_large --allow-download
  5. Optional, only if a new collection wins: one full run to confirm. COLAB: not needed | MODEL: as the winner | CLAUDE COST: about 10 cents
       python -m evals.run_eval --split test --tag v2 --collection NAME --spend           (add --judge gemini to cut the cost to about 5 cents)
     For a collection made with the large model also set EMBED_PROFILE=large first:  $env:EMBED_PROFILE = "large"
  Your own collection filings_v1 and the small model are not changed by any of these steps."""


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Apply the upgrades of 26 November: two models, piece size, context lines, compare tool, lab, rewrite v3 check.")
    parser.add_argument("--check", action="store_true", help="say what would change, change nothing")
    parser.add_argument("--undo", action="store_true", help="take the changes out again")
    parser.add_argument("--no-test", action="store_true", help="do not run the tests")
    parser.add_argument("--no-spend", action="store_true", help="skip the rewrite v3 check (it costs about 1 cent)")
    parser.add_argument("--rewrite-check", action="store_true", help="run only the rewrite v3 check (about 1 cent) and switch it on when it is not worse")
    parser.add_argument("--root", help="the project folder (default: the folder of this file)")
    args = parser.parse_args(argv)
    cost = "0 cents (this file only edits files)" if args.no_spend or args.undo or args.check else "about 1 cent (the rewrite v3 check)"
    print(f"COLAB: not needed for this file | MODEL: small bge 384 stays the default | CLAUDE COST: {cost}\n")
    root = find_root(args.root)
    if root is None:
        print("This does not look like the project folder (config.py, bot\\lookup.py, evals\\rewrite_test.py and ingest\\build_index.py are not all here).")
        print("Put apply_upgrades.py in the folder with config.py, or run it with  --root FOLDER .")
        return 1
    print(f"Project folder: {root}\n")
    if args.undo:
        return undo(root, args)
    if args.rewrite_check:
        return rewrite_step(root, args)
    return apply(root, args)


if __name__ == "__main__":
    sys.exit(main())
