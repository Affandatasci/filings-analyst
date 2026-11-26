"""All settings in one place. Keys come from .env and are never written in code."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# ---- Paths ----
DATA_DIR = ROOT / "data"
PDF_DIR = Path(os.getenv("PDF_DIR") or ROOT.parent / "agentic_rag" / "pdf")
PDF_LIST = DATA_DIR / "pdf_list.txt"
QUESTIONS_FILE = DATA_DIR / "financebench_open_source.jsonl"
RESULTS_DIR = ROOT / "results"

# ---- Keys (the Streamlit Secrets box first, then .env; never printed) ----
def _secret(name):
    """A key from the Streamlit Secrets box when the app runs there, otherwise from .env or the environment."""
    import sys

    streamlit = sys.modules.get("streamlit")  # only when the app has already loaded it: no heavy import here
    if streamlit is not None:
        try:
            value = streamlit.secrets.get(name)
            if value:
                return str(value)
        except Exception:  # no Secrets box on the laptop: that is normal
            pass
    return os.getenv(name)


ANTHROPIC_API_KEY = _secret("ANTHROPIC_API_KEY")
QDRANT_URL = _secret("QDRANT_URL")
QDRANT_API_KEY = _secret("QDRANT_API_KEY")
SUPABASE_URL = _secret("SUPABASE_URL")
SUPABASE_KEY = _secret("SUPABASE_KEY")
LANGSMITH_API_KEY = _secret("LANGSMITH_API_KEY")
GEMINI_API_KEY = _secret("GEMINI_API_KEY")
GROQ_API_KEY = _secret("GROQ_API_KEY")

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
COLLECTION = "filings_v1"
PIECE_TOKENS = 400  # target size of one piece
PIECE_HARD_STOP = 480  # no piece may be longer (bge small stops at 512)
TOP_K = 5
PAGE_OFFSET = 1  # the question file counts pages from 0, our pages start at 1 (checked on all 103 questions)
QDRANT_SEARCH = "query_points"  # or "search"; set after Day 1 check 2
FILTER_MODE = "company+year"  # experiment 4 on 85 questions: company+year 51 found, company 46, off 43 

# ---- What the bot does with a question before and after the search (bot/lookup.py) ----
QUERY_REWRITE = True  # experiment 5: Claude writes the search query in a filing's words. 64 of 85 right pages found, 51 without
QUERY_HINTS = True  # a short finance word list in that request (experiment 8 measures it)
QUERY_COUNT = 1  # queries per question. 2 or 3 give a question that needs several pages more than one (experiment 8 measures it)
PAGE_CONTEXT = True  # send the whole page of the best hits, not only the piece that matched (a cut table, case 07966)
PAGE_CONTEXT_PAGES = 3  # at most this many whole pages for one question
REWRITE_CACHE = None  # a file: rewrites are kept there and reused. run_eval and the experiments set it, the app does not
NEIGHBOR_PAGES = 0  # fix neighbors: also send this many pages before and after each whole page (0 = off). 1 sends about 2.6 times the text: a full run costs about 18 cents instead of about 10
QUERY_CLEAN = False  # build rewrite: take markdown, a label such as "Query:" and a last word "search" out of the rewritten query. Test it first: python -m evals.rewrite_test
REWRITE_PROMPT = "rewrite_v2"  # the rewrite request: rewrite_v2 (in use) or rewrite_v3 (a line in the words of the filing). Test it first: python -m evals.rewrite_test --new

# ---- Money (cents) ----
PRICE_IN_CENTS_PER_MTOK = 10.0  # 0.10 dollars per million tokens in
PRICE_OUT_CENTS_PER_MTOK = 50.0  # 0.50 dollars per million tokens out
BUDGET_CENTS = 100  # the whole project
FUSE_CENTS = 90  # llm.py refuses Claude calls above this


def _number(name, default):
    try:
        return float(os.getenv(name) or default)
    except ValueError:
        return default


# ---- Public demo limits (step 4.4) ----
DEMO_DAILY_CENTS = _number("DEMO_DAILY_CENTS", 2.0)  # Claude spend of the demo per day (UTC), checked against the log before each answer
DEMO_TOTAL_CENTS = _number("DEMO_TOTAL_CENTS", 15.0)  # and in all; this comes out of BUDGET_CENTS
DEMO_RESERVE_CENTS = 0.15  # the most one agent answer can cost: a question is refused when its answer could cross a limit
DEMO_SESSION_QUESTIONS = 10  # questions in one visit (a page refresh starts a new visit: the two money limits are the hard ones)
MAX_QUESTION_CHARS = 500  # the input guard refuses longer questions
DEMO_NOTICE = "Public SEC filings. Not investment advice. The role switch is a demo, not a login."
TRACE_ONE_IN = 5  # the public app sends 1 trace in 5 to LangSmith; eval runs send every one
SHOW_QUESTIONS = _secret("SHOW_QUESTIONS") == "1"  # the dashboard lists what visitors typed only when this is set (your laptop), never on the public page

# ---- Pass lines (step 2.9): a score in points out of 100 that must be reached ----
# Each one is the baseline minus 5 points. After step 2.8 run:  python scripts\check_scores.py --show-lines
PASS_LINES = {
    "correct": 65.7,
    "number_match": 64.4,
    "found_page": 71.0,
    "supported": 93.7,
    "citation_valid": 91.0,
    "citation_hit": 66.2,
    "json_shape": 95.0,
    "length": 93.7,
}


def need(name):
    """Return a key from .env, or stop with a plain message. Never prints the value."""
    value = _secret(name)
    if not value:
        raise SystemExit(f"Missing {name}. Add it to .env (see .env.example).")
    return value
