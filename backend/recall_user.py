"""Hotword-pool id validation for Triton recall pools."""

from __future__ import annotations

import re

DEFAULT_HOTWORD_POOL_ID = "default"
DEFAULT_RECALL_USER_ID = DEFAULT_HOTWORD_POOL_ID
_HOTWORD_POOL_ID_RE = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]*$")
_USER_ID_RE = _HOTWORD_POOL_ID_RE


class HotwordPoolIdError(ValueError):
    """Raised when a hotword-pool id is not safe to forward."""


RecallUserIdError = HotwordPoolIdError


def normalize_hotword_pool_id(
    value: object | None = None,
    *,
    default: str = DEFAULT_HOTWORD_POOL_ID,
) -> str:
    """Return a validated hotword-pool id, using ``default`` for empty input."""
    fallback = str(default or DEFAULT_HOTWORD_POOL_ID).strip() or DEFAULT_HOTWORD_POOL_ID
    hotword_pool_id = str(value or "").strip() or fallback
    if (
        hotword_pool_id in {".", ".."}
        or not _HOTWORD_POOL_ID_RE.fullmatch(hotword_pool_id)
    ):
        raise HotwordPoolIdError(
            "HOTWORD_POOL_ID/USER_ID must contain only letters, digits, dot, "
            "underscore, or hyphen"
        )
    return hotword_pool_id


def normalize_recall_user_id(
    value: object | None = None,
    *,
    default: str = DEFAULT_RECALL_USER_ID,
) -> str:
    """Backward-compatible alias for hotword-pool id normalization."""
    return normalize_hotword_pool_id(value, default=default)
