"""Tests for the upgrades of 26 November (apply_upgrades.py): two embedding models, a chosen piece size, context lines, the compare tool and the lab.

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
