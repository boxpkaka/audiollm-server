"""Hotword-pool id validation for Triton recall pools."""

from __future__ import annotations

import re

DEFAULT_HOTWORD_POOL_ID = "default"
_HOTWORD_POOL_ID_RE = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]*$")


class HotwordPoolIdError(ValueError):
    """Raised when a hotword-pool id is not safe to forward."""


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
            "HOTWORD_POOL_ID must contain only letters, digits, dot, "
            "underscore, or hyphen"
        )
    return hotword_pool_id
