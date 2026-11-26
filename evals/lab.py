"""The lab: what the Colab notebook builds and how it chooses. The same code runs on a laptop.

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
