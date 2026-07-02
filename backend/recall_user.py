"""User id validation for Triton hotword recall pools."""

from __future__ import annotations

import re

DEFAULT_RECALL_USER_ID = "default"
_USER_ID_RE = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]*$")


class RecallUserIdError(ValueError):
    """Raised when a recall hotword-pool user id is not safe to forward."""


def normalize_recall_user_id(
    value: object | None = None,
    *,
    default: str = DEFAULT_RECALL_USER_ID,
) -> str:
    """Return a validated recall user id, using ``default`` for empty input."""
    fallback = str(default or DEFAULT_RECALL_USER_ID).strip() or DEFAULT_RECALL_USER_ID
    user_id = str(value or "").strip() or fallback
    if user_id in {".", ".."} or not _USER_ID_RE.fullmatch(user_id):
        raise RecallUserIdError(
            "USER_ID must contain only letters, digits, dot, underscore, or hyphen"
        )
    return user_id
