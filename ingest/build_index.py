"""Step 1.4: embed every piece and upload it to Qdrant (collection filings_v1).

From the project folder:
    python -m ingest.build_index                  build or update the collection
    python -m ingest.build_index --only NAME      one document first, as a quick test
    python -m ingest.build_index --no-header      a second index without the header line (experiment 3, later)
    python -m ingest.build_index --collection NAME --profile large --piece-tokens 250 --context FILE
                                                  a NEW collection with the large model, another piece size or context lines
                                                  (filings_v1 is never touched by these options; the notebook does the same in Colab)

It is safe to run again. Every piece has a fixed id (made from doc, page and piece number)
and a fingerprint of its content. A second run skips pieces that are already there and the
same, updates the changed ones and removes pieces that no longer exist. If you stop it
with Ctrl+C, run it again and it continues.
"""
import argparse
import hashlib
import json
import sys
import time
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    VectorParams,
)

import config
from ingest.chunker import cut_document
from ingest.read_pdf import list_documents, read_pdf

UPLOAD_BATCH = 64  # pieces per upload
INDEXED_FIELDS = ("company", "doc", "access")
NUMBER_FIELDS = ("page", "year")  # whole pages (page) and the year filter (year) look these up


def piece_id(doc, page, piece):
    """The same doc, page and piece number always give the same id, so a second run updates."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"filings-analyst/{doc}/{page}/{piece}"))


def make_payload(document, piece, context=""):
    """The labels stored with each piece, plus a fingerprint (sha) of all of them. context: the context line of the page (only for the vector)."""
    info = document.info
    payload = {
        "doc": info.name,
        "company": info.company,
        "year": info.year,
        "period": info.period,
        "type": info.type,
        "page": piece.page,
        "piece": piece.piece,
        "text": piece.text,
        "source": document.source,
        "access": document.access,
    }
    if context:
        payload["context"] = context  # used when the piece is embedded; the stored text, and so what the bot reads, stays the same
    payload["sha"] = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return payload


def embed_text(payload):
    """What the model reads for one piece: the context line of its page (when there is one) and then the piece."""
    return payload["context"] + "\n" + payload["text"] if payload.get("context") else payload["text"]


def retry(action, tries=3):
    """Try a network call again if it fails. The build is safe to repeat."""
    for attempt in range(1, tries + 1):
        try:
            return action()
        except Exception as err:
            if attempt == tries:
                raise
            print(f"\n  network problem ({type(err).__name__}), trying again in {5 * attempt} s", flush=True)
            time.sleep(5 * attempt)


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


def stored(client, collection, doc):
    """What Qdrant already holds for one document: {id: sha}."""
    found, offset = {}, None
    only_doc = Filter(must=[FieldCondition(key="doc", match=MatchValue(value=doc))])
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=only_doc,
            limit=500,
            offset=offset,
            with_payload=["sha"],
            with_vectors=False,
        )
        for point in points:
            found[str(point.id)] = (point.payload or {}).get("sha")
        if offset is None:
            return found


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
        stale = [pid for pid in have if pid not in wanted]
        if stale:
            retry(lambda: client.delete(collection_name=collection, points_selector=PointIdsList(points=stale)))
        new = sum(pid not in have for pid in todo)
        for key, value in (("pieces", len(wanted)), ("new", new), ("changed", len(todo) - new),
                           ("same", len(wanted) - len(todo)), ("removed", len(stale))):
            totals[key] += value
        print(f" new {new}, changed {len(todo) - new}, removed {len(stale)}  ({time.time() - started:.0f} s in all)")
    names = [document.info.name for document in documents]
    these = Filter(must=[FieldCondition(key="doc", match=MatchAny(any=names))])
    totals["qdrant_these"] = client.count(collection_name=collection, count_filter=these, exact=True).count
    totals["qdrant_all"] = client.count(collection_name=collection, exact=True).count
    totals["seconds"] = time.time() - started
    return totals


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Embed the pieces and upload them to Qdrant (step 1.4).")
    parser.add_argument("--only", nargs="+", metavar="NAME", help="only these documents")
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
    try:
        client.get_collections()  # fail now, before the slow model load, if the URL or key is wrong
    except Exception as err:
        sys.exit(f"Cannot reach Qdrant ({type(err).__name__}). Check QDRANT_URL and QDRANT_API_KEY in .env.")
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

    minutes, seconds = divmod(int(totals["seconds"]), 60)
    print()
    print(f"Pieces cut:               {totals['pieces']:,}")
    print(f"New / changed / same:     {totals['new']:,} / {totals['changed']:,} / {totals['same']:,}")
    print(f"Removed:                  {totals['removed']:,}")
    print(f"Qdrant, these documents:  {totals['qdrant_these']:,}")
    print(f"Qdrant, whole collection: {totals['qdrant_all']:,}")
    print("Same count:               " + ("yes" if totals["qdrant_these"] == totals["pieces"] else "NO. Send me these lines."))
    print(f"Time:                     {minutes} min {seconds} s   (write this number down)")


if __name__ == "__main__":
    main()
