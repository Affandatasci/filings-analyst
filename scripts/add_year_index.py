"""Turn 6: a one-time step. Index the year field of the Qdrant collection, so the year filter is allowed and fast.

From the project folder:
    python scripts\\add_year_index.py                        the main index (config.COLLECTION)
    python scripts\\add_year_index.py --collection filings_v1_plain

Safe to run twice. It changes no piece, calls no model and costs nothing.
Run it again after you build a new collection (for example filings_v2), because
ingest/build_index.py indexes company, doc and access, but not year.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client.models import PayloadSchemaType  # noqa: E402

import config  # noqa: E402

FIELD = "year"


def add_year_index(client, collection):
    """Make the index if it is missing. Returns 'made' or 'already there'."""
    schema = client.get_collection(collection).payload_schema or {}
    if FIELD in schema:
        return "already there"
    client.create_payload_index(collection, FIELD, PayloadSchemaType.INTEGER)
    return "made"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Index the year field of the Qdrant collection (turn 6).")
    parser.add_argument("--collection", help="index to change (default from config.py)")
    args = parser.parse_args(argv)
    collection = args.collection or config.COLLECTION

    from retrieval.search import get_client

    client = get_client()
    try:
        result = add_year_index(client, collection)
    except Exception as err:
        sys.exit(f"Could not change the collection '{collection}': {type(err).__name__}: {str(err)[:300]}")
    info = client.get_collection(collection)
    print(f"Collection {collection}: {info.points_count} pieces. Index on '{FIELD}': {result}.")
    print(f"Indexed fields now: {', '.join(sorted(info.payload_schema or {}))}")


if __name__ == "__main__":
    main()
