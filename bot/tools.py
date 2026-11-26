"""Step 3.5: the 4 tools of the agent. All of them only read; none can change anything.

    search_filings(query, company, year)   the top 5 pieces for the role, with optional filters
    read_page(doc, page)                   the whole text of a page, if the role may read that file
    calculate(expression)                  safe arithmetic: + - * / and brackets. No eval
    list_documents(company)                which files exist for a company, to pick the right year

The role is given to Tools() by the app. The model can not pass a role: an unknown argument is refused.
Every piece a tool returns first goes through prepare_fn (the instruction scan and the personal data hiding of security/guards.py).
A tool never raises for a bad call: it returns a ToolResult with ok False and a plain error that the model reads and can fix.
"""
import ast
import operator
import re
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import config
from retrieval import reader
from security import access, fakes

TOOL_NAMES = ("search_filings", "read_page", "calculate", "list_documents")
TOOL_HELP = """- search_filings(query, company, year): the 5 best pieces for a query. company and year are optional.
- read_page(doc, page): the whole text of one page. Use it when a piece is cut off or a table needs a second look.
- calculate(expression): arithmetic with + - * / and brackets. Use it for every ratio or change, never work it out in your head.
- list_documents(company): which files exist for a company."""
ARGUMENTS = {  # name -> (required, optional)
    "search_filings": (("query",), ("company", "year")),
    "read_page": (("doc", "page"), ()),
    "calculate": (("expression",), ()),
    "list_documents": (("company",), ()),
}
PIECE = '<piece doc="{doc}" page="{page}">\n{text}\n</piece>'
MAX_PAGE_CHARS = 8000  # a longer page is cut, so one tool call can not eat the token limit
DOC_NAME = re.compile(r"^[A-Za-z0-9_.\-]{1,80}$")


@dataclass
class ToolResult:
    name: str
    args: dict
    ok: bool
    text: str  # what the model reads
    hits: list = field(default_factory=list)  # the pieces the model was given (Hit objects)
    error: str = ""
    refused: bool = False  # True when the role may not read the file that was asked for
    events: list = field(default_factory=list)  # what the guards did to the pieces

    def log(self):
        """A plain record of the step, for result.events."""
        return {"tool": self.name, "args": self.args, "ok": self.ok, "pieces": len(self.hits), "error": self.error, "refused": self.refused}


# ---------- checking the arguments ----------

def validate(name, args):
    """Returns (clean arguments, problem). The problem is empty when the call is valid."""
    if name not in TOOL_NAMES:
        return {}, f"unknown tool {name!r}. The tools are: {', '.join(TOOL_NAMES)}"
    if not isinstance(args, dict):
        return {}, "args must be a JSON object"
    required, optional = ARGUMENTS[name]
    extra = sorted(set(args) - set(required) - set(optional))
    if extra:
        return {}, f"unknown argument {extra[0]!r} for {name}"
    missing = [key for key in required if args.get(key) in (None, "")]
    if missing:
        return {}, f"{name} needs {missing[0]!r}"
    clean = {}
    for key in required + optional:
        value = args.get(key)
        if value in (None, ""):
            continue
        if key in ("year", "page"):
            if isinstance(value, str) and value.strip().isdigit():
                value = int(value)
            if isinstance(value, bool) or not isinstance(value, int):
                return {}, f"{key} must be a whole number"
            low, high = (1990, 2100) if key == "year" else (1, 5000)
            if not low <= value <= high:
                return {}, f"{key} must be between {low} and {high}"
        elif not isinstance(value, str):
            return {}, f"{key} must be text"
        else:
            value = value.strip()
            limit = {"query": 300, "expression": 200}.get(key, 80)
            if len(value) > limit:
                return {}, f"{key} is longer than {limit} characters"
            if key == "doc" and not DOC_NAME.match(value):
                return {}, "doc must be a file name such as 3M_2018_10K"
        clean[key] = value
    return clean, ""


# ---------- calculate: arithmetic without eval ----------

_OPERATIONS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def _evaluate(node, depth=0):
    if depth > 30:
        raise ValueError("the expression is too deep")
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _evaluate(node.operand, depth + 1)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATIONS:
        left, right = _evaluate(node.left, depth + 1), _evaluate(node.right, depth + 1)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ValueError("division by zero")
        return _OPERATIONS[type(node.op)](left, right)
    raise ValueError("only numbers, + - * / and brackets are allowed")


def calculate(expression):
    """The value of a plain arithmetic expression, as text. Raises ValueError with a plain reason when it can not be done."""
    text = str(expression).strip().replace("×", "*").replace("÷", "/").replace("$", "")
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)  # 1,577 means 1577
    if not text or len(text) > 200:
        raise ValueError("write the expression in 1 to 200 characters")
    if not re.fullmatch(r"[0-9+\-*/().\s]+", text):
        raise ValueError("only numbers, + - * / and brackets are allowed")
    try:
        value = _evaluate(ast.parse(text, mode="eval").body)
    except SyntaxError as err:
        raise ValueError("that is not a valid expression") from err
    if abs(value) >= 1e15:
        raise ValueError("the result is too large")
    return f"{value:.6f}".rstrip("0").rstrip(".") if value != int(value) else str(int(value))


# ---------- the files we hold, and one page ----------

def catalog():
    """Every file we hold: name, company, year, type, level and a short label. Real filings first, then the fake files."""
    entries = [
        {"name": info.name, "company": info.company, "year": info.year, "type": info.type, "level": "public", "label": info.label()}
        for info in reader.known_filings()
    ]
    entries += [{**item, "label": item["title"]} for item in fakes.catalog()]
    return entries


def fetch_page(doc, page, levels, client=None, collection=None):
    """The pieces of one page, joined into one Hit. Empty list when there is none (or the role may not read it)."""
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

    from retrieval import search as search_module

    must = [FieldCondition(key="doc", match=MatchValue(value=doc)), FieldCondition(key="page", match=MatchValue(value=page))]
    if levels is not None:
        must.append(FieldCondition(key="access", match=MatchAny(any=list(levels))))
    client = client or search_module.get_client()
    points, _ = client.scroll(collection_name=collection or config.COLLECTION, scroll_filter=Filter(must=must), limit=64, with_payload=True, with_vectors=False)
    hits = sorted((search_module._to_hit(SimpleNamespace(score=1.0, payload=point.payload)) for point in points), key=lambda hit: hit.piece)
    return join_page(hits)


def join_page(hits):
    """Several pieces of one page become one Hit. The header line is kept once."""
    if not hits:
        return []
    texts = [hits[0].text]
    for hit in hits[1:]:
        lines = hit.text.split("\n", 1)
        texts.append(lines[1] if len(lines) == 2 and lines[0].rstrip().endswith(f"page {hit.page}") else hit.text)
    return [replace(hits[0], piece=0, score=1.0, text="\n".join(texts))]


# ---------- the tools ----------

class Tools:
    """The tools for one question. role is None only for runs without guards."""

    def __init__(self, role=None, prepare_fn=None, search_fn=None, page_fn=None, catalog_fn=None, k=None, collection=None):
        self.role = access.check_role(role) if role is not None else None
        self.levels = access.allowed_levels(self.role) if self.role else None
        self.prepare_fn, self.search_fn, self.page_fn, self.catalog_fn = prepare_fn, search_fn, page_fn, catalog_fn
        self.k = k or config.TOP_K
        self.collection = collection  # None means the index of config.py

    def run(self, name, args):
        clean, problem = validate(name, args)
        if problem:
            return ToolResult(name, args if isinstance(args, dict) else {}, False, f"ERROR: {problem}", error=problem)
        try:
            return getattr(self, "_" + name)(clean)
        except ValueError as err:
            return ToolResult(name, clean, False, f"ERROR: {err}", error=str(err))

    def _prepare(self, hits):
        return self.prepare_fn(hits) if self.prepare_fn and hits else (hits, [])

    def _pieces_result(self, name, args, hits, empty):
        hits, events = self._prepare(hits)
        if not hits:
            return ToolResult(name, args, True, empty, events=events)
        return ToolResult(name, args, True, "\n".join(PIECE.format(doc=hit.doc, page=hit.page, text=hit.text) for hit in hits), hits=hits, events=events)

    def _search_filings(self, args):
        options = {"k": self.k, "access": self.levels}
        if self.collection:
            options["collection"] = self.collection
        if args.get("company"):
            options["company"] = reader.companies_named(args["company"]) or (args["company"],)
        if args.get("year"):
            options["year"] = args["year"]
        if self.search_fn is None:
            from retrieval.search import search as search_fn
        else:
            search_fn = self.search_fn
        return self._pieces_result("search_filings", args, search_fn(args["query"], **options), "No pieces found.")

    def _read_page(self, args):
        page_fn = self.page_fn or (lambda doc, page, levels: fetch_page(doc, page, levels, collection=self.collection))
        hits = page_fn(args["doc"], args["page"], self.levels)
        if self.levels is not None:
            hits = [hit for hit in hits if hit.access in self.levels]  # a second lock behind the search filter
        if not hits:
            level = fakes.level_of(args["doc"])
            refused = level is not None and self.levels is not None and level not in self.levels
            return ToolResult("read_page", args, False, "ERROR: no such page for your role.", error="no such page for your role", refused=refused)
        hits = [replace(hit, text=hit.text[:MAX_PAGE_CHARS] + ("\n[the page is cut here]" if len(hit.text) > MAX_PAGE_CHARS else "")) for hit in hits]
        return self._pieces_result("read_page", args, hits, "The page is empty.")

    def _calculate(self, args):
        return ToolResult("calculate", args, True, f"{args['expression']} = {calculate(args['expression'])}")

    def _list_documents(self, args):
        wanted = args["company"].strip().lower()
        named = {name.lower() for name in reader.companies_named(args["company"])}
        lines = []
        for item in (self.catalog_fn or catalog)():
            if self.levels is not None and item["level"] not in self.levels:
                continue
            company = item["company"].lower()
            if company in named or wanted in company or company in wanted:
                lines.append(f"{item['name']} ({item['label']})")
        return ToolResult("list_documents", args, True, "\n".join(lines[:40]) or "No files for that company.")
