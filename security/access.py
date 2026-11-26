"""Step 3.2: who may read what.

    from security.access import allowed_levels
    allowed_levels("guest")      # ["public"]
    allowed_levels("analyst")    # ["public", "internal"]
    allowed_levels("hr")         # ["public", "internal", "hr"]

The role comes from the app (a switch on the page), never from the model and never from the question.
What matters is where the check happens: in the Qdrant filter, not in the prompt.
search(question, access=allowed_levels(role)) sends that filter with the search.
"""
ROLES = {
    "guest": ("public",),  # the default visitor: the 37 real filings and the 3 poisoned notes
    "analyst": ("public", "internal"),
    "hr": ("public", "internal", "hr"),  # sees personal data unmasked
}
LEVELS = ("public", "internal", "hr")
DEFAULT_ROLE = "guest"


class AccessError(ValueError):
    """An unknown role, or a role asking for a file above its level."""


def check_role(role):
    """The role in lower case, or AccessError. There is no silent fallback to a stronger role."""
    name = role.strip().lower() if isinstance(role, str) else ""
    if name not in ROLES:
        raise AccessError(f"Unknown role {role!r}. The roles are: {', '.join(ROLES)}.")
    return name


def allowed_levels(role):
    return list(ROLES[check_role(role)])


def can_read(role, level):
    return level in ROLES[check_role(role)]


def require(role, level):
    """Refuse when the role may not read this level."""
    if not can_read(role, level):
        raise AccessError(f"The role {check_role(role)} may not read {level} files.")


def qdrant_filter(role):
    """The Qdrant filter for a role: only pieces whose access label is one of its levels."""
    from qdrant_client.models import FieldCondition, Filter, MatchAny

    return Filter(must=[FieldCondition(key="access", match=MatchAny(any=allowed_levels(role)))])
