"""Unit tests for the per-session debug dumper (backend/debug_dump.py)."""

from __future__ import annotations

import asyncio
import json
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.debug_dump import SessionDumper, new_session_id  # noqa: E402


def _read_wav_mono16k(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000
        assert wf.getsampwidth() == 2
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def test_new_session_id_unique_and_shaped() -> None:
    a, b = new_session_id(), new_session_id()
    assert a != b
    # YYMMDD-HHMMSS-xxxx
    assert len(a.split("-")) == 3


def test_base_dir_is_session_dir(tmp_path: Path) -> None:
    dumper = SessionDumper(str(tmp_path), "sess", engine="asr")
    assert dumper.base_dir == str(tmp_path / "sess")


def test_write_final_persists_wav_and_json_with_partials(tmp_path: Path) -> None:
    sid = "260629-150812-abcd"
    dumper = SessionDumper(str(tmp_path), sid, engine="asr")
    pcm = (np.sin(np.linspace(0, 20, 16000, dtype=np.float32)) * 0.5).astype(
        np.float32
    )

    dumper.record_partial("seg-1", "你好")
    dumper.record_partial("seg-1", "你好世界")
    dump_id = asyncio.run(
        dumper.write_final(
            seg_id="seg-1",
            pcm=pcm,
            meta={"text": "你好世界。", "audio": {"duration_sec": 1.0}},
        )
    )

    assert dump_id == f"{sid}/seg-1"
    wav_path = tmp_path / sid / "seg-1.wav"
    json_path = tmp_path / sid / "seg-1.json"
    assert wav_path.is_file()
    assert json_path.is_file()

    # The dumped wav round-trips to the same sample count (it is the exact
    # PCM that fed inference == the client's replay audio).
    decoded = _read_wav_mono16k(wav_path)
    assert decoded.shape[0] == pcm.shape[0]

    record = json.loads(json_path.read_text(encoding="utf-8"))
    assert record["dump_id"] == dump_id
    assert record["session_id"] == sid
    assert record["seg_id"] == "seg-1"
    assert record["engine"] == "asr"
    assert record["text"] == "你好世界。"
    assert [p["text"] for p in record["partials"]] == ["你好", "你好世界"]
    assert record["wav_file"].endswith("seg-1.wav")


def test_partials_are_consumed_per_segment(tmp_path: Path) -> None:
    dumper = SessionDumper(str(tmp_path), "sess", engine="asr")
    dumper.record_partial("seg-1", "p1")
    asyncio.run(
        dumper.write_final(seg_id="seg-1", pcm=np.zeros(1600, dtype=np.float32), meta={})
    )
    # A second final for the same id must not re-attach the first segment's
    # partials.
    asyncio.run(
        dumper.write_final(seg_id="seg-1", pcm=np.zeros(1600, dtype=np.float32), meta={})
    )
    record = json.loads(
        (tmp_path / "sess" / "seg-1.json").read_text(encoding="utf-8")
    )
    assert record["partials"] == []


def test_anonymous_segment_id_uses_counter(tmp_path: Path) -> None:
    dumper = SessionDumper(str(tmp_path), "sess", engine="asr")
    dumper.record_partial(None, "p1")
    d1 = asyncio.run(
        dumper.write_final(seg_id=None, pcm=np.zeros(1600, dtype=np.float32), meta={})
    )
    d2 = asyncio.run(
        dumper.write_final(seg_id=None, pcm=np.zeros(1600, dtype=np.float32), meta={})
    )
    assert d1 == "sess/seg-1"
    assert d2 == "sess/seg-2"
    rec1 = json.loads((tmp_path / "sess" / "seg-1.json").read_text(encoding="utf-8"))
    assert [p["text"] for p in rec1["partials"]] == ["p1"]


def test_unsafe_seg_id_is_sanitized(tmp_path: Path) -> None:
    dumper = SessionDumper(str(tmp_path), "sess", engine="asr")
    dump_id = asyncio.run(
        dumper.write_final(
            seg_id="../../etc/passwd",
            pcm=np.zeros(800, dtype=np.float32),
            meta={},
        )
    )
    # No path traversal: the stem stays inside the session dir.
    stem = dump_id.split("/", 1)[1]
    assert "/" not in stem
    assert ".." not in stem
    assert (tmp_path / "sess" / f"{stem}.json").is_file()
