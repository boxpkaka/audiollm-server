"""Per-session debug dump: persist each final segment's audio + metadata.

Purpose
-------
When ``debug_dump_enabled`` is on, every finalized ASR segment is written to
``<debug_dump_dir>/<session_id>/<seg_id>.{wav,json}``:

- the WAV is the exact PCM that fed inference, which the previous audit
  confirmed is byte-identical to the client's replay audio (``audio_b64``);
- the JSON carries the final text, the partial history seen during that
  utterance, primary/secondary raw outputs, the model + feature-flag snapshot
  and timing.

This is a troubleshooting aid (e.g. the "replay says a word the final text
dropped" investigation), so it lives off the hot path: metadata is assembled
in the event loop, file IO is pushed to a worker thread, and any write failure
is logged and swallowed — it must never break the ASR response.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .audio.utils import pcm_to_wav_bytes
from .config import SAMPLE_RATE

logger = logging.getLogger(__name__)

# debug_dump.py lives in backend/, so parent.parent is the project root that
# also hosts config.yaml; a relative ``debug_dump_dir`` resolves against it so
# the dump location is independent of the process CWD.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def new_session_id() -> str:
    """A sortable, collision-resistant id for one WebSocket connection.

    Format ``YYMMDD-HHMMSS-xxxx`` (xxxx = 2 random bytes) keeps directory
    listings chronological while staying unique across sessions that open in
    the same second.
    """
    return datetime.now().strftime("%y%m%d-%H%M%S-") + secrets.token_hex(2)


def _safe(name: str | None, fallback: str) -> str:
    cleaned = _UNSAFE.sub("_", str(name or "").strip()).strip("_.")
    return cleaned or fallback


class SessionDumper:
    """Persist per-segment audio + metadata for one streaming session."""

    def __init__(self, dump_dir: str, session_id: str, *, engine: str = "") -> None:
        base = Path(dump_dir).expanduser()
        if not base.is_absolute():
            base = _PROJECT_ROOT / base
        self._session_id = session_id
        self._engine = engine
        self._dir = base / session_id
        # Partial transcripts accumulated per segment key, attached to the
        # segment's JSON when its final lands. Keyed identically to write_final
        # so the two sides always line up (see ``_partial_key``).
        self._partials: dict[str, list[dict[str, Any]]] = {}
        self._seg_counter = 0

    @property
    def base_dir(self) -> str:
        """Absolute directory this session writes into (sent in ``ready``)."""
        return str(self._dir)

    @staticmethod
    def _partial_key(seg_id: str | None) -> str:
        return _safe(seg_id, "unknown") if seg_id else "unknown"

    def record_partial(self, seg_id: str | None, text: str) -> None:
        """Append one partial transcript; cheap, runs on the event loop."""
        self._partials.setdefault(self._partial_key(seg_id), []).append(
            {
                "ts": datetime.now().isoformat(timespec="milliseconds"),
                "text": text,
            }
        )

    async def write_final(
        self,
        *,
        seg_id: str | None,
        pcm: np.ndarray,
        meta: dict[str, Any],
    ) -> str:
        """Write ``<seg>.wav`` + ``<seg>.json`` and return the dump id.

        The dump id (``<session_id>/<seg_id>``) doubles as the relative path
        stem, so a client that copies it from the bubble can locate the files
        directly under :attr:`base_dir`'s parent.
        """
        if seg_id:
            stem = _safe(seg_id, "seg")
            partials = self._partials.pop(self._partial_key(seg_id), [])
        else:
            self._seg_counter += 1
            stem = f"seg-{self._seg_counter}"
            partials = self._partials.pop("unknown", [])

        dump_id = f"{self._session_id}/{stem}"
        record: dict[str, Any] = {
            "dump_id": dump_id,
            "session_id": self._session_id,
            "seg_id": seg_id or stem,
            "engine": self._engine,
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
            "partials": partials,
            **meta,
        }
        pcm_arr = np.asarray(pcm, dtype=np.float32).reshape(-1)
        try:
            await asyncio.to_thread(self._write, stem, pcm_arr, record)
        except Exception as exc:  # never break the ASR response on a dump error
            logger.warning("debug dump write failed for %s: %s", dump_id, exc)
        return dump_id

    def _write(self, stem: str, pcm: np.ndarray, record: dict[str, Any]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        wav_path = self._dir / f"{stem}.wav"
        json_path = self._dir / f"{stem}.json"
        wav_path.write_bytes(pcm_to_wav_bytes(pcm, SAMPLE_RATE))
        record["wav_file"] = str(wav_path)
        json_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
