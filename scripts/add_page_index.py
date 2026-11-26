"""A one-time step. Index the page field of the Qdrant collection, so a whole page can be read.

The bot reads the whole page of the best hits (bot/lookup.py, PAGE_CONTEXT in config.py) and so does the read_page tool of the agent.
Both look a page up by doc and page. Qdrant refuses a lookup on a field that has no index, so the page field needs one.

From the project folder:
    python scripts\\add_page_index.py                        the main index (config.COLLECTION)
    python scripts\\add_page_index.py --collection filings_v1_plain

Safe to run twice. It changes no piece, calls no model and costs nothing.
Run it again after you build a new collection (for example filings_v2), because ingest/build_index.py indexes company, doc and
access, not page.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client.models import PayloadSchemaType  # noqa: E402

import config  # noqa: E402

FIELD = "page"


def add_page_index(client, collection):
    """Make the index if it is missing. Returns 'made' or 'already there'."""
    schema = client.get_collection(collection).payload_schema or {}
    if FIELD in schema:
        return "already there"
    client.create_payload_index(collection, FIELD, PayloadSchemaType.INTEGER)
    return "made"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Index the page field of the Qdrant collection.")
    parser.add_argument("--collection", help="index to change (default from config.py)")
    args = parser.parse_args(argv)
    collection = args.collection or config.COLLECTION

    from retrieval.search import get_client

    client = get_client()
    try:
        result = add_page_index(client, collection)
    except Exception as err:
        sys.exit(f"Could not change the collection '{collection}': {type(err).__name__}: {str(err)[:300]}")
    info = client.get_collection(collection)
    print("COLAB: no | MODEL: unchanged | CLAUDE COST: 0 cents")
    print(f"Collection {collection}: {info.points_count} pieces. Index on '{FIELD}': {result}.")
    print(f"Indexed fields now: {', '.join(sorted(info.payload_schema or {}))}")


if __name__ == "__main__":
    main()
