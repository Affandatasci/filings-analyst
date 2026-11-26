"""Tests for step 3.2: roles and levels. The check must happen in the Qdrant filter, never in the prompt.

No network, no key, no model: an in memory Qdrant stands in. Run from the project folder:
    python -m pytest -q
"""
import sys
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from security import access  # noqa: E402

# Qdrant in memory says payload indexes do nothing. True, and harmless here.
pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes")


def test_the_three_roles_and_what_each_may_read():
    assert access.allowed_levels("guest") == ["public"]
    assert access.allowed_levels("analyst") == ["public", "internal"]
    assert access.allowed_levels("hr") == ["public", "internal", "hr"]


def test_the_default_role_is_the_weakest_one():
    assert access.DEFAULT_ROLE == "guest"
    assert access.allowed_levels(access.DEFAULT_ROLE) == ["public"]


def test_role_names_ignore_case_and_spaces():
    assert access.check_role("  HR ") == "hr"
    assert access.check_role("Analyst") == "analyst"


@pytest.mark.parametrize("role", ["admin", "", None, 5, "root", "gue st", "hr;analyst", ["hr"], "hr analyst", "superuser"])
def test_an_unknown_role_is_refused_and_never_becomes_a_stronger_one(role):
    with pytest.raises(access.AccessError):
        access.check_role(role)
    with pytest.raises(access.AccessError):
        access.allowed_levels(role)
    with pytest.raises(access.AccessError):
        access.qdrant_filter(role)


@pytest.mark.parametrize("role", list(access.ROLES))
@pytest.mark.parametrize("level", access.LEVELS)
def test_each_role_reads_exactly_its_own_levels(role, level):
    assert access.can_read(role, level) == (level in access.ROLES[role])


def test_require_refuses_a_level_above_the_role():
    access.require("hr", "hr")
    access.require("analyst", "internal")
    with pytest.raises(access.AccessError, match="guest may not read hr"):
        access.require("guest", "hr")
    with pytest.raises(access.AccessError):
        access.require("analyst", "hr")


def test_the_levels_come_in_order_and_each_role_is_a_prefix_of_the_next():
    assert access.LEVELS == ("public", "internal", "hr")
    guest, analyst, hr = (access.ROLES[name] for name in ("guest", "analyst", "hr"))
    assert analyst[: len(guest)] == guest and hr[: len(analyst)] == analyst


def test_allowed_levels_gives_a_copy_that_can_not_change_the_rules():
    levels = access.allowed_levels("guest")
    levels.append("hr")
    assert access.allowed_levels("guest") == ["public"]


def test_the_filter_holds_the_levels_of_the_role():
    condition = access.qdrant_filter("analyst").must[0]
    assert condition.key == "access"
    assert condition.match.any == ["public", "internal"]


@pytest.fixture()
def qdrant():
    client = QdrantClient(":memory:")
    client.create_collection("c", vectors_config=VectorParams(size=2, distance=Distance.COSINE))
    client.upsert("c", points=[
        PointStruct(id=number, vector=[1.0, 0.1 * number], payload={"doc": f"DOC_{level}", "access": level})
        for number, level in enumerate(["public", "internal", "hr", "hr", "public"], start=1)
    ])
    return client


@pytest.mark.parametrize("role, expected", [("guest", {"public"}), ("analyst", {"public", "internal"}), ("hr", {"public", "internal", "hr"})])
def test_qdrant_itself_only_returns_what_the_role_may_read(qdrant, role, expected):
    points, _ = qdrant.scroll("c", scroll_filter=access.qdrant_filter(role), limit=20)
    assert {point.payload["access"] for point in points} == expected
    found = qdrant.query_points("c", query=[1.0, 0.0], query_filter=access.qdrant_filter(role), limit=20).points
    assert {point.payload["access"] for point in found} == expected


def test_a_guest_never_gets_an_hr_piece_however_close_it_is(qdrant):
    # the hr pieces are the closest to the question, and still they never come back
    found = qdrant.query_points("c", query=[1.0, 0.25], query_filter=access.qdrant_filter("guest"), limit=1).points
    assert [point.payload["access"] for point in found] == ["public"]
