"""Unit tests for the target-speaker enrollment feature.

Covers
------
1. The v4-aligned prompt builder in ``backend/asr/client.py`` — verifies
   the exact ``messages`` structure for the four shapes the model was
   trained on (plain ASR, ASR + hotwords, TS-ASR, TS-ASR + hotwords).
   These tests are byte-for-byte assertions because v4 SFT is sensitive
   to whitespace ordering (the ``\\n`` lives at the *start* of the second
   text block in the TS-ASR shapes).
2. The in-memory ``EnrollmentStore`` — duration validation, tail-trim,
   TTL eviction, LRU overflow, and round-trip get/delete.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.asr.client import (  # noqa: E402
    build_audio_only_messages,
    build_primary_messages,
)
from backend.asr.enrollment import (  # noqa: E402
    EnrollmentError,
    _Store,
    decode_and_validate,
)
from backend.audio.utils import pcm_to_wav_base64  # noqa: E402
from backend.config import SAMPLE_RATE  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wav_b64(seconds: float, sr: int = SAMPLE_RATE) -> str:
    n = max(1, int(round(sr * seconds)))
    t = np.arange(n, dtype=np.float32) / sr
    sig = 0.3 * np.sin(2 * np.pi * 440 * t)
    return pcm_to_wav_base64(sig.astype(np.float32), sr)


# ---------------------------------------------------------------------------
# Prompt builder — must match v4 task 1/2/5/6 byte-for-byte
# ---------------------------------------------------------------------------


def test_primary_messages_task1_plain_asr():
    """Task 1: ``Transcribe the following audio.`` + <audio>."""
    msgs = build_primary_messages("TARGET_B64")
    assert len(msgs) == 1
    content = msgs[0]["content"]
    assert content == [
        {"type": "text", "text": "Transcribe the following audio."},
        {"type": "input_audio", "input_audio": {"data": "TARGET_B64", "format": "wav"}},
    ]


def test_primary_messages_task2_asr_hotwords():
    """Task 2: hotwords joined with ``,`` (no spaces), comma after Hotwords:."""
    msgs = build_primary_messages(
        "TARGET_B64", hotwords=["江门", "彭丽媛", "奥体中心"]
    )
    text = msgs[0]["content"][0]["text"]
    assert text == "Transcribe the following audio.\nHotwords: 江门,彭丽媛,奥体中心"
    assert msgs[0]["content"][1]["input_audio"]["data"] == "TARGET_B64"


def test_primary_messages_task5_tsasr():
    """Task 5: dual text + dual audio with the ``\\n`` on second text head."""
    msgs = build_primary_messages(
        "TARGET_B64", enrollment_wav_base64="ENROLL_B64"
    )
    content = msgs[0]["content"]
    assert content == [
        {"type": "text", "text": "Given the speaker's voice:"},
        {"type": "input_audio", "input_audio": {"data": "ENROLL_B64", "format": "wav"}},
        {
            "type": "text",
            "text": "\nTranscribe what this speaker says in the following audio.",
        },
        {"type": "input_audio", "input_audio": {"data": "TARGET_B64", "format": "wav"}},
    ]


def test_primary_messages_task6_tsasr_hotwords():
    """Task 6: second text adds ``\\nHotwords: w1,w2`` after the transcribe line."""
    msgs = build_primary_messages(
        "TARGET_B64",
        hotwords=["北京", "清华大学"],
        enrollment_wav_base64="ENROLL_B64",
    )
    second_text = msgs[0]["content"][2]["text"]
    assert second_text == (
        "\nTranscribe what this speaker says in the following audio.\n"
        "Hotwords: 北京,清华大学"
    )


def test_primary_messages_hotword_dedup_and_strip():
    """Hotwords are stripped + deduped while preserving order, so the
    prompt bytes never gain stray whitespace from sloppy clients."""
    msgs = build_primary_messages(
        "TARGET_B64",
        hotwords=["北京", "  北京  ", "上海", "上海", "", "广州"],
    )
    text = msgs[0]["content"][0]["text"]
    assert text == "Transcribe the following audio.\nHotwords: 北京,上海,广州"


def test_primary_messages_no_language_line_ever():
    """v4 SFT has 0% Language: coverage. Even when callers pass a
    ``src_lang`` value (via the upstream ``query_audio_model``
    wrapper) the prompt must omit the line — assert via absence."""
    msgs = build_primary_messages("TARGET_B64", hotwords=["北京"])
    assembled = "".join(
        item.get("text", "")
        for item in msgs[0]["content"]
        if item.get("type") == "text"
    )
    assert "Language:" not in assembled


def test_audio_only_messages_has_no_text_item():
    """The secondary (Qwen3) path is text-free single-audio prompting."""
    msgs = build_audio_only_messages("AUDIO_B64")
    assert msgs == [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": "AUDIO_B64", "format": "wav"},
                }
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Enrollment store — duration validation + TTL + LRU
# ---------------------------------------------------------------------------


def test_decode_and_validate_happy_path():
    b64, dur = decode_and_validate(_wav_b64(3.0), min_sec=1.0, max_sec=8.0)
    assert isinstance(b64, str) and b64
    assert dur == pytest.approx(3.0, abs=0.05)


def test_decode_and_validate_too_short_raises():
    with pytest.raises(EnrollmentError) as exc:
        decode_and_validate(_wav_b64(0.4), min_sec=1.0, max_sec=8.0)
    assert exc.value.code == "too_short"


def test_decode_and_validate_tail_trims_when_too_long():
    """Overflows tail-trim rather than reject — matches the existing
    ASR / emotion upload convention."""
    b64, dur = decode_and_validate(_wav_b64(12.0), min_sec=1.0, max_sec=8.0)
    assert dur == pytest.approx(8.0, abs=0.05)


def test_decode_and_validate_rejects_empty():
    with pytest.raises(EnrollmentError) as exc:
        decode_and_validate("", min_sec=1.0, max_sec=8.0)
    assert exc.value.code == "empty"


def test_decode_and_validate_rejects_garbage_b64():
    with pytest.raises(EnrollmentError) as exc:
        decode_and_validate("not-actually-a-wav", min_sec=1.0, max_sec=8.0)
    assert exc.value.code == "decode_failed"


def test_store_put_get_delete_roundtrip():
    store = _Store(ttl_sec=10.0, max_entries=4)
    entry = store.put(_wav_b64(2.0), 2.0)
    assert entry.enrollment_id
    fetched = store.get(entry.enrollment_id)
    assert fetched is not None
    assert fetched.wav_base64 == entry.wav_base64
    assert store.delete(entry.enrollment_id) is True
    assert store.get(entry.enrollment_id) is None


def test_store_get_returns_none_for_missing_id():
    store = _Store(ttl_sec=10.0, max_entries=4)
    assert store.get("does-not-exist") is None
    assert store.get("") is None


def test_store_ttl_eviction_is_lazy():
    """The store doesn't run a sweeper thread; expiry is checked at
    read time. A get() past the TTL drops the entry."""
    store = _Store(ttl_sec=0.05, max_entries=4)
    entry = store.put(_wav_b64(2.0), 2.0)
    time.sleep(0.08)
    assert store.get(entry.enrollment_id) is None


def test_store_overflow_evicts_lru():
    store = _Store(ttl_sec=60.0, max_entries=3)
    ids = [store.put(_wav_b64(1.0), 1.0).enrollment_id for _ in range(3)]
    # Touch ids[1] and ids[2] so ids[0] becomes the LRU candidate.
    time.sleep(0.01)
    store.get(ids[1])
    time.sleep(0.01)
    store.get(ids[2])
    time.sleep(0.01)
    new_entry = store.put(_wav_b64(1.0), 1.0)
    assert store.get(ids[0]) is None, "oldest entry should have been evicted"
    assert store.get(ids[1]) is not None
    assert store.get(ids[2]) is not None
    assert store.get(new_entry.enrollment_id) is not None
