"""Step 1.5: search the index.

    from retrieval.search import search
    hits = search("What was the net income of 3M in FY2018?")      # the top 5 pieces

The question is embedded with the same function and the same model as the pieces
(ingest/embed.py). Every hit is a plain Hit object, so the bot, the tests and the demo
never touch Qdrant types.

Turn 6 (experiment 4): with mode "company" or "company+year" (or config.FILTER_MODE) the company
and the year are read out of the question (retrieval/reader.py) and used as a filter.
The default is "off", so nothing changes until the experiment says otherwise.
"""
from dataclasses import dataclass
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

import config


@dataclass(frozen=True)
class Hit:
    score: float  # cosine similarity: higher is closer
    doc: str  # for example 3M_2018_10K
    company: str
    year: int
    period: str  # FY, Q1 to Q4, or empty for an 8K
    type: str  # 10K, 10Q, 8K or EARNINGS
    page: int  # starts at 1
    piece: int  # 1, 2, 3 ... inside the page
    text: str  # starts with the header line
    source: str  # real or fake
    access: str  # public, or a role label later

    @property
    def label(self):
        """How to point at this piece, for example '3M_2018_10K, page 60'."""
        return f"{self.doc}, page {self.page}"


@lru_cache(maxsize=1)
def get_client():
    """One Qdrant client for the whole program."""
    return QdrantClient(url=config.need("QDRANT_URL"), api_key=config.need("QDRANT_API_KEY"), timeout=60)


def _one_or_many(key, value):
    """One value matches exactly. A list or tuple of values matches any of them."""
    if isinstance(value, (list, tuple, set, frozenset)):
        values = sorted(value)
        if len(values) > 1:
            return FieldCondition(key=key, match=MatchAny(any=values))
        value = values[0]
    return FieldCondition(key=key, match=MatchValue(value=value))


def _make_filter(company, access, year=None):
    must = []
    if company:
        must.append(_one_or_many("company", company))
    if year:
        must.append(_one_or_many("year", year))
    if access:
        must.append(FieldCondition(key="access", match=MatchAny(any=list(access))))
    return Filter(must=must) if must else None


def _ask_qdrant(client, collection, vector, limit, query_filter):
    """Run the vector search with the method named in config.QDRANT_SEARCH. If that fails, try the other one."""
    methods = ["query_points", "search"]
    if config.QDRANT_SEARCH == "search":
        methods.reverse()
    problems = []
    for method in methods:
        if not hasattr(client, method):
            continue  # newer qdrant-client versions have no search()
        try:
            if method == "query_points":
                reply = client.query_points(
                    collection_name=collection, query=vector, query_filter=query_filter, limit=limit, with_payload=True
                )
                return reply.points
            return client.search(
                collection_name=collection, query_vector=vector, query_filter=query_filter, limit=limit, with_payload=True
            )
        except Exception as err:
            problems.append(f"{method}: {type(err).__name__}: {str(err)[:200]}")
    raise RuntimeError("Qdrant search failed.\n  " + ("\n  ".join(problems) or "this qdrant-client has no search method"))


def _to_hit(point):
    payload = point.payload or {}
    return Hit(
        score=float(point.score),
        doc=payload.get("doc", ""),
        company=payload.get("company", ""),
        year=int(payload.get("year", 0)),
        period=payload.get("period", ""),
        type=payload.get("type", ""),
        page=int(payload.get("page", 0)),
        piece=int(payload.get("piece", 0)),
        text=payload.get("text", ""),
        source=payload.get("source", ""),
        access=payload.get("access", ""),
    )


def search(question, k=None, company=None, access=None, collection=None, client=None, embed=None, year=None, mode=None):
    """Return the k best pieces for the question, best first.

    k           how many pieces (default config.TOP_K)
    company     only pieces of this company (experiment 4), for example "3M" or ("3M", "Adobe")
    year        only pieces of this filing year, for example 2018 or (2018, 2019)
    mode        off, company or company+year: read the company and year out of the question and
                filter by them (default config.FILTER_MODE). Ignored when company or year is given.
    access      only pieces whose access label is in this list, for example ["public"]
    collection  another index, for example "filings_v1_plain" (experiment 3)
    client, embed   for tests: a Qdrant client and an embedding function
    """
    if not question or not question.strip():
        raise ValueError("The question is empty.")
    mode = config.FILTER_MODE if mode is None else mode
    if company is None and year is None and mode != "off":
        from retrieval.reader import read_question  # the reader checks the mode name

        reading = read_question(question, mode=mode)
        company, year = reading.company_filter, reading.year_filter
    if embed is None:
        from ingest.embed import embed_texts as embed  # loads the model the first time
    vector = embed([question])[0]
    points = _ask_qdrant(
        client or get_client(),
        collection or config.COLLECTION,
        vector,
        config.TOP_K if k is None else k,
        _make_filter(company, access, year),
    )
    return [_to_hit(point) for point in points]
