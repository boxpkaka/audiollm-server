"""Offline tests for the Target-Speaker ASR (TS-ASR) pipeline.

Covers:
- Prompt builder (default / hotwords / voice_traits compatibility)
- Enrollment decoder (happy path, duration bounds, bad payloads)
- Client request shape (dual-audio content, base_url override)
- TsAsrTaskEngine lifecycle (enrollment error, segment flow, fallback config)
- StreamingSession's generic extract_hotwords plumbing

Run with:
    .venv/bin/python -m pytest tests/test_tsasr.py -v
"""

from __future__ import annotations

import asyncio
import base64
import io
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.audio.utils import (  # noqa: E402
    pcm_to_wav_base64,
    wav_base64_to_pcm_16k_mono,
)
from backend.config import load_config  # noqa: E402
from backend.streaming.events import SegmentReady  # noqa: E402
from backend.streaming.session import SessionContext  # noqa: E402
from backend.tasks.ts_asr import TsAsrTaskEngine  # noqa: E402
from backend.tsasr.enrollment import EnrollmentError, decode_enrollment  # noqa: E402
from backend.tsasr.prompt import (  # noqa: E402
    ENROLL_PREFIX,
    TRANSCRIBE_PREFIX,
    build_tsasr_content,
    format_hotwords_line,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav_b64(duration_sec: float, sample_rate: int = 16000) -> str:
    n = int(duration_sec * sample_rate)
    pcm = np.zeros(n, dtype=np.float32)
    return pcm_to_wav_base64(pcm, sample_rate=sample_rate)


def _patch_secondary_nonempty(monkeypatch, text: str = "presence"):
    """Stub the Qwen3-ASR presence gate with a non-empty transcription.

    The dual-channel gate in ``TsAsrTaskEngine._dual_infer`` only emits text
    when both TS-ASR *and* the secondary ASR have non-empty output. Tests
    that focus on the TS-ASR side should call this so the gate doesn't
    short-circuit and swallow the result they want to assert on.
    """

    async def _fake(*_args, **_kwargs):
        return {"transcription": text, "raw_text": text}

    monkeypatch.setattr(
        "backend.tasks.ts_asr.query_audio_model_secondary", _fake
    )


def _make_wav_bytes_custom(
    duration_sec: float, sample_rate: int, channels: int = 1
) -> bytes:
    n = int(duration_sec * sample_rate)
    samples = np.zeros(n * channels, dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Prompt builder (aligned with v3 SFT template; see backend/tsasr/prompt.py)
# ---------------------------------------------------------------------------


def test_prompt_builder_default_layout_no_hotwords():
    """Template A: enrollment + transcribe-with-period + mixed audio."""
    content = build_tsasr_content("ENROLL_B64", "MIXED_B64")
    assert len(content) == 4
    assert content[0] == {"type": "text", "text": ENROLL_PREFIX}
    assert content[1]["type"] == "input_audio"
    assert content[1]["input_audio"]["data"] == "ENROLL_B64"
    assert content[1]["input_audio"]["format"] == "wav"
    # Transcribe text MUST end in a period (v3 standard) and MUST NOT
    # carry a Hotwords line when none were supplied.
    assert content[2] == {"type": "text", "text": "\n" + TRANSCRIBE_PREFIX}
    assert TRANSCRIBE_PREFIX.endswith(".")
    assert content[3]["type"] == "input_audio"
    assert content[3]["input_audio"]["data"] == "MIXED_B64"


def test_prompt_builder_with_hotwords_layout():
    """Template B: Hotwords line is appended to the transcribe text block.

    Critically the Hotwords line lives in the SAME ``text`` block as the
    transcribe instruction so the mixed ``<audio>`` immediately follows
    the last text line with no extra whitespace -- matching the
    ``"\\n".join(lines) + _AUDIO_PLACEHOLDER`` byte sequence emitted at
    training time.
    """
    content = build_tsasr_content(
        "ENR", "MIX", hotwords=["Alpha", " Beta ", "", "Gamma"]
    )
    assert len(content) == 4
    transcribe_text = content[2]["text"]
    assert transcribe_text == (
        "\n" + TRANSCRIBE_PREFIX + "\nHotwords: Alpha,Beta,Gamma"
    )
    # No trailing period on the Hotwords line, no spaces in the comma join.
    assert "Hotwords: Alpha,Beta,Gamma." not in transcribe_text
    assert "Hotwords: Alpha, Beta, Gamma" not in transcribe_text
    # The mixed audio is the very next chunk -- no spurious text in between.
    assert content[3]["type"] == "input_audio"


def test_prompt_builder_voice_traits_does_not_leak_into_prompt():
    """Backward-compat: voice_traits is accepted but never written to prompt.

    v3 training data has no ``Speaker traits:`` segment, so emitting one
    would push the prompt off-distribution. The parameter still exists so
    older clients keep working without errors.
    """
    content = build_tsasr_content(
        "ENR", "MIX", voice_traits="female, warm, mid-range"
    )
    texts = [c["text"] for c in content if c.get("type") == "text"]
    assert all("Speaker traits" not in t for t in texts)
    assert all("female" not in t for t in texts)
    # Layout matches the no-hotwords case exactly.
    assert content[2] == {"type": "text", "text": "\n" + TRANSCRIBE_PREFIX}


def test_prompt_builder_empty_optionals_are_ignored():
    content = build_tsasr_content(
        "E", "M", hotwords=["", "  "], voice_traits="   "
    )
    texts = [c["text"] for c in content if c.get("type") == "text"]
    assert all("Hotwords" not in t for t in texts)
    assert all("Speaker traits" not in t for t in texts)
    assert len(content) == 4


def test_format_hotwords_line_edge_cases():
    assert format_hotwords_line(None) == ""
    assert format_hotwords_line([]) == ""
    assert format_hotwords_line(["  "]) == ""
    # Trims surrounding whitespace, joins with comma, no trailing period.
    assert (
        format_hotwords_line(["foo", " bar ", "", "baz"])
        == "\nHotwords: foo,bar,baz"
    )


# ---------------------------------------------------------------------------
# Enrollment decoder
# ---------------------------------------------------------------------------


def test_decode_enrollment_happy_path_16k():
    b64 = _make_wav_b64(2.0)
    enr = decode_enrollment(b64, min_sec=1.0, max_sec=10.0)
    assert enr.pcm.dtype == np.float32
    assert abs(enr.duration_sec - 2.0) < 1e-3
    assert enr.sample_rate == 16000
    # Re-encoded canonical wav should round-trip back to 16k PCM of the same length.
    round_tripped = wav_base64_to_pcm_16k_mono(enr.wav_base64)
    assert round_tripped.shape == enr.pcm.shape


def test_decode_enrollment_accepts_48k_and_resamples():
    raw = _make_wav_bytes_custom(2.0, 48000)
    b64 = base64.b64encode(raw).decode()
    enr = decode_enrollment(b64, min_sec=1.0, max_sec=10.0)
    assert abs(enr.duration_sec - 2.0) < 0.05
    assert enr.sample_rate == 16000


def test_decode_enrollment_stereo_downmix():
    raw = _make_wav_bytes_custom(1.5, 16000, channels=2)
    b64 = base64.b64encode(raw).decode()
    enr = decode_enrollment(b64, min_sec=1.0, max_sec=10.0)
    assert abs(enr.duration_sec - 1.5) < 1e-3


def test_decode_enrollment_rejects_missing():
    with pytest.raises(EnrollmentError) as exc_info:
        decode_enrollment("", min_sec=1.0, max_sec=10.0)
    assert exc_info.value.code == "missing"


def test_decode_enrollment_rejects_too_short():
    b64 = _make_wav_b64(0.3)
    with pytest.raises(EnrollmentError) as exc_info:
        decode_enrollment(b64, min_sec=1.0, max_sec=10.0)
    assert exc_info.value.code == "too_short"


def test_decode_enrollment_trims_long_audio_instead_of_rejecting():
    # 8s input > 5s cap must no longer raise "too_long". With zero-filled
    # PCM the VAD finds no voiced frames, so the trim path falls back to
    # the leading max_sec window and decode_enrollment accepts it.
    b64 = _make_wav_b64(8.0)
    enr = decode_enrollment(b64, min_sec=0.5, max_sec=5.0)
    assert abs(enr.duration_sec - 5.0) < 1e-2
    # Canonical re-encode must round-trip to the trimmed PCM length.
    round_tripped = wav_base64_to_pcm_16k_mono(enr.wav_base64)
    assert round_tripped.shape == enr.pcm.shape


def test_decode_enrollment_short_clip_unaffected_by_trim():
    # Sanity: clips already within [min_sec, max_sec] skip the VAD trim
    # entirely and keep their original duration.
    b64 = _make_wav_b64(2.0)
    enr = decode_enrollment(b64, min_sec=1.0, max_sec=5.0)
    assert abs(enr.duration_sec - 2.0) < 1e-3


def test_decode_enrollment_rejects_garbage():
    with pytest.raises(EnrollmentError) as exc_info:
        decode_enrollment("!!!not-base64!!!", min_sec=1.0, max_sec=5.0)
    # base64 garbage either fails decode or fails WAV parse; both map cleanly.
    assert exc_info.value.code in {"decode_failed"}


def test_decode_enrollment_rejects_non_wav_format():
    b64 = _make_wav_b64(2.0)
    with pytest.raises(EnrollmentError) as exc_info:
        decode_enrollment(b64, min_sec=1.0, max_sec=5.0, audio_format="mp3")
    assert exc_info.value.code == "unsupported_format"


# ---------------------------------------------------------------------------
# vad_trim_audio
# ---------------------------------------------------------------------------

from backend.audio.vad import vad_trim_audio  # noqa: E402


def test_vad_trim_audio_passthrough_when_shorter_than_target():
    sr = 16000
    pcm = np.zeros(int(2.0 * sr), dtype=np.float32)
    out = vad_trim_audio(pcm, target_sec=5.0)
    assert out.size == pcm.size
    assert out.dtype == np.float32


def test_vad_trim_audio_caps_at_target():
    # Long input never exceeds target_sec. Silent PCM exercises the "no
    # voiced frames" fallback branch that returns the leading window.
    sr = 16000
    pcm = np.zeros(int(8.0 * sr), dtype=np.float32)
    out = vad_trim_audio(pcm, target_sec=5.0)
    assert out.size == int(5.0 * sr)
    assert out.dtype == np.float32


def test_vad_trim_audio_empty_input():
    out = vad_trim_audio(np.empty(0, dtype=np.float32), target_sec=5.0)
    assert out.size == 0


# ---------------------------------------------------------------------------
# TsAsrTaskEngine
# ---------------------------------------------------------------------------


async def _noop_send(payload: dict) -> bool:
    return True


def _ungated_cfg(**overrides):
    """Build a config that bypasses the TS-ASR speech-presence gate.

    The gate runs Silero-VAD over the PCM segment and rejects clips with
    fewer than ``tsasr_speech_gate_min_voiced_ms`` of voiced frames.
    Synthetic test PCM (constant DC, white noise, ones) doesn't look like
    speech to the VAD, so engine tests that want to drive the inference
    path must disable the gate; otherwise ``handle_segment`` /
    ``handle_partial`` short-circuit before reaching the model.
    """
    base = {"tsasr_speech_gate_enabled": False}
    base.update(overrides)
    return load_config().override(**base)


def _make_ctx(cfg=None, send_json=None):
    cfg = cfg or _ungated_cfg()
    return SessionContext(
        cfg=cfg, language="zh", src_lang="Chinese",
        hotwords=[], send_json=send_json or _noop_send,
    )


@pytest.mark.asyncio
async def test_engine_on_start_rejects_missing_enrollment():
    sent: list[dict] = []

    async def _send(payload):
        sent.append(payload)
        return True

    ctx = _make_ctx(send_json=_send)
    engine = TsAsrTaskEngine()
    await engine.on_start({"type": "start"}, ctx)
    assert sent and sent[0]["type"] == "error"
    assert sent[0]["code"] == "enrollment_missing"


@pytest.mark.asyncio
async def test_engine_on_start_caches_enrollment_and_acks():
    sent: list[dict] = []

    async def _send(payload):
        sent.append(payload)
        return True

    ctx = _make_ctx(send_json=_send)
    engine = TsAsrTaskEngine()
    await engine.on_start(
        {
            "type": "start",
            "enrollment_audio": _make_wav_b64(2.0),
            "voice_traits": "female, warm",
        },
        ctx,
    )
    assert engine._enrollment is not None
    assert engine._voice_traits == "female, warm"
    assert sent and sent[0]["type"] == "enrollment_ok"
    assert abs(sent[0]["duration_sec"] - 2.0) < 1e-2


@pytest.mark.asyncio
async def test_engine_handle_segment_calls_tsasr_with_dual_audio(monkeypatch):
    sent: list[dict] = []

    async def _send(payload):
        sent.append(payload)
        return True

    ctx = _make_ctx(send_json=_send)
    engine = TsAsrTaskEngine()
    await engine.on_start(
        {"enrollment_audio": _make_wav_b64(2.0)}, ctx
    )

    captured: dict = {}

    async def _fake_query(
        mixed_b64, enrollment_b64, *,
        hotwords=None, voice_traits=None,
        base_url=None, model_name=None, timeout=None,
        enrollment_duration_sec=None,
    ):
        captured["mixed_b64"] = mixed_b64
        captured["enrollment_b64"] = enrollment_b64
        captured["hotwords"] = hotwords
        captured["voice_traits"] = voice_traits
        captured["base_url"] = base_url
        captured["model_name"] = model_name
        captured["enrollment_duration_sec"] = enrollment_duration_sec
        return {
            "transcription": "hello world",
            "raw_text": "hello world",
            "detected_language": "English",
            "enrollment_duration_sec": enrollment_duration_sec,
        }

    monkeypatch.setattr(
        "backend.tasks.ts_asr.query_tsasr_model", _fake_query
    )
    _patch_secondary_nonempty(monkeypatch)

    seg = SegmentReady(
        pcm=np.ones(16000, dtype=np.float32) * 0.1,
        is_stop_flush=True,
    )
    ok = await engine.handle_segment(seg, ctx)
    assert ok is True

    assert captured["enrollment_b64"] == engine._enrollment.wav_base64
    assert captured["mixed_b64"]  # non-empty
    assert captured["enrollment_b64"] != captured["mixed_b64"]
    # ``tsasr_enable_hotwords`` defaults to True in v3-aligned config; with
    # an empty ctx.hotwords list the engine still passes ``[]`` through so
    # the prompt builder consistently sees a list (and decides whether to
    # emit a Hotwords line based on its contents).
    assert captured["hotwords"] == []
    # ``_resolve`` prefers ``tsasr_*`` when set (the deployed config wires
    # the dedicated TS-ASR endpoint), and falls back to ``vllm_*`` only
    # when ``tsasr_*`` is empty.
    expected_base = ctx.cfg.tsasr_base_url or ctx.cfg.vllm_base_url
    expected_model = ctx.cfg.tsasr_model_name or ctx.cfg.vllm_model_name
    assert captured["base_url"] == expected_base
    assert captured["model_name"] == expected_model
    assert abs(captured["enrollment_duration_sec"] - 2.0) < 1e-2

    final = next(m for m in sent if m["type"] == "final")
    assert final["text"] == "hello world"
    assert final["language"] == "English"
    assert final["task"] == "tsasr"


@pytest.mark.asyncio
async def test_engine_handle_segment_without_enrollment_is_noop():
    ctx = _make_ctx()
    engine = TsAsrTaskEngine()
    seg = SegmentReady(pcm=np.zeros(8000, dtype=np.float32))
    assert await engine.handle_segment(seg, ctx) is False


@pytest.mark.asyncio
async def test_engine_handle_segment_respects_tsasr_endpoint_override(monkeypatch):
    async def _send(payload):
        return True

    cfg = _ungated_cfg(
        tsasr_base_url="http://custom:9000",
        tsasr_model_name="Amphion/TS-Demo",
    )
    ctx = SessionContext(cfg=cfg, language="zh", send_json=_send)
    engine = TsAsrTaskEngine()
    await engine.on_start(
        {"enrollment_audio": _make_wav_b64(2.0)}, ctx
    )

    captured: dict = {}

    async def _fake_query(*args, **kwargs):
        captured.update(kwargs)
        return {
            "transcription": "hi",
            "raw_text": "hi",
            "detected_language": None,
            "enrollment_duration_sec": None,
        }

    monkeypatch.setattr(
        "backend.tasks.ts_asr.query_tsasr_model", _fake_query
    )
    _patch_secondary_nonempty(monkeypatch)
    seg = SegmentReady(pcm=np.ones(16000, dtype=np.float32) * 0.05)
    await engine.handle_segment(seg, ctx)
    assert captured["base_url"] == "http://custom:9000"
    assert captured["model_name"] == "Amphion/TS-Demo"


@pytest.mark.asyncio
async def test_engine_handle_partial_skipped_when_disabled(monkeypatch):
    """``tsasr_enable_partial=False`` must short-circuit before any inference."""
    async def _send(_payload):
        return True

    cfg = _ungated_cfg(tsasr_enable_partial=False)
    ctx = SessionContext(cfg=cfg, language="zh", send_json=_send)
    engine = TsAsrTaskEngine()
    await engine.on_start(
        {"enrollment_audio": _make_wav_b64(2.0)}, ctx
    )
    called = {"n": 0}

    async def _fake_query(*_args, **_kwargs):
        called["n"] += 1
        return {
            "transcription": "x", "raw_text": "x",
            "detected_language": None, "enrollment_duration_sec": None,
        }

    monkeypatch.setattr(
        "backend.tasks.ts_asr.query_tsasr_model", _fake_query
    )
    _patch_secondary_nonempty(monkeypatch)
    from backend.streaming.events import PartialSnapshot

    snap = PartialSnapshot(pcm=np.ones(8000, dtype=np.float32) * 0.05)
    await engine.handle_partial(snap, ctx)
    assert called["n"] == 0


@pytest.mark.asyncio
async def test_engine_handle_partial_enabled_via_config(monkeypatch):
    sent: list[dict] = []

    async def _send(payload):
        sent.append(payload)
        return True

    cfg = _ungated_cfg(tsasr_enable_partial=True)
    ctx = SessionContext(cfg=cfg, language="zh", send_json=_send)
    engine = TsAsrTaskEngine()
    await engine.on_start(
        {"enrollment_audio": _make_wav_b64(2.0)}, ctx
    )

    async def _fake_query(*_args, **_kwargs):
        return {
            "transcription": "hello",
            "raw_text": "hello",
            "detected_language": None,
            "enrollment_duration_sec": None,
        }

    monkeypatch.setattr(
        "backend.tasks.ts_asr.query_tsasr_model", _fake_query
    )
    _patch_secondary_nonempty(monkeypatch)
    from backend.streaming.events import PartialSnapshot

    snap = PartialSnapshot(pcm=np.ones(8000, dtype=np.float32) * 0.05)
    await engine.handle_partial(snap, ctx)
    partials = [m for m in sent if m["type"] == "partial"]
    assert partials
    assert partials[0]["text"] == "hello"
    assert partials[0]["task"] == "tsasr"
    # Same id is shared between partial and the eventual final emission.
    assert partials[0]["id"]


@pytest.mark.asyncio
async def test_engine_on_stop_emits_empty_final_when_nothing_sent():
    sent: list[dict] = []

    async def _send(payload):
        sent.append(payload)
        return True

    ctx = _make_ctx(send_json=_send)
    engine = TsAsrTaskEngine()
    await engine.on_stop(ctx, sent_any_response=False, stopped=True)
    assert sent == [
        {"type": "final", "text": "", "language": "zh", "task": "tsasr"}
    ]


@pytest.mark.asyncio
async def test_engine_on_stop_silent_on_disconnect():
    sent: list[dict] = []

    async def _send(payload):
        sent.append(payload)
        return True

    ctx = _make_ctx(send_json=_send)
    engine = TsAsrTaskEngine()
    await engine.on_stop(ctx, sent_any_response=False, stopped=False)
    assert sent == []


@pytest.mark.asyncio
async def test_engine_with_hotwords_enabled_forwards_hotwords(monkeypatch):
    async def _send(payload):
        return True

    cfg = _ungated_cfg(tsasr_enable_hotwords=True)
    ctx = SessionContext(
        cfg=cfg, language="zh", hotwords=["Alpha", "Beta"], send_json=_send
    )
    engine = TsAsrTaskEngine()
    await engine.on_start(
        {"enrollment_audio": _make_wav_b64(2.0)}, ctx
    )

    captured: dict = {}

    async def _fake_query(*_args, **kwargs):
        captured.update(kwargs)
        return {
            "transcription": "x", "raw_text": "x",
            "detected_language": None, "enrollment_duration_sec": None,
        }

    monkeypatch.setattr(
        "backend.tasks.ts_asr.query_tsasr_model", _fake_query
    )
    _patch_secondary_nonempty(monkeypatch)
    seg = SegmentReady(pcm=np.ones(16000, dtype=np.float32) * 0.05)
    await engine.handle_segment(seg, ctx)
    assert captured["hotwords"] == ["Alpha", "Beta"]


@pytest.mark.asyncio
async def test_engine_with_hotwords_disabled_does_not_forward_hotwords(monkeypatch):
    """``tsasr_enable_hotwords=False`` should pass ``None`` (not the list)."""
    async def _send(_payload):
        return True

    cfg = _ungated_cfg(tsasr_enable_hotwords=False)
    ctx = SessionContext(
        cfg=cfg, language="zh", hotwords=["Alpha", "Beta"], send_json=_send
    )
    engine = TsAsrTaskEngine()
    await engine.on_start(
        {"enrollment_audio": _make_wav_b64(2.0)}, ctx
    )

    captured: dict = {}

    async def _fake_query(*_args, **kwargs):
        captured.update(kwargs)
        return {
            "transcription": "x", "raw_text": "x",
            "detected_language": None, "enrollment_duration_sec": None,
        }

    monkeypatch.setattr(
        "backend.tasks.ts_asr.query_tsasr_model", _fake_query
    )
    _patch_secondary_nonempty(monkeypatch)
    seg = SegmentReady(pcm=np.ones(16000, dtype=np.float32) * 0.05)
    await engine.handle_segment(seg, ctx)
    assert captured["hotwords"] is None


# ---------------------------------------------------------------------------
# StreamingSession: generic extract_hotwords plumbing
# ---------------------------------------------------------------------------
#
# The realtime ASR page already supports a ``{"type":"extract_hotwords"}``
# control message that asks the backend to call an LLM and return
# extracted hotwords. We lifted that handler into ``StreamingSession`` so
# every task engine on the new session layer (TS-ASR, emotion) gets it for
# free. The two tests below drive ``_handle_text`` directly with a
# ``StreamingSession`` constructed against a fake WebSocket so we can
# assert on both the success and failure response shapes.


class _FakeWebSocket:
    """Minimal in-memory WebSocket for unit tests.

    Captures every ``send_json`` payload in ``self.sent``. ``accept`` /
    ``receive`` are stubbed because the tests drive control message
    handling directly via ``_handle_text``.
    """

    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)


async def _drain_extract_tasks(session) -> None:
    """Wait for any pending extract_hotwords background tasks to finish."""
    while session._extract_tasks:
        await asyncio.gather(*session._extract_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_streaming_session_extract_hotwords_success(monkeypatch):
    from backend.streaming import StreamingSession, VadSegmentedStream

    async def _fake_query(_text: str) -> list[str]:
        return ["北京", "清华大学"]

    monkeypatch.setattr(
        "backend.streaming.session.query_text_hotwords", _fake_query
    )

    ws = _FakeWebSocket()
    session = StreamingSession(
        ws,  # type: ignore[arg-type]
        stream=VadSegmentedStream(),
        engine=TsAsrTaskEngine(),
    )
    # Suppress the implicit ``ready`` greeting we don't care about.
    ws.sent.clear()

    await session._handle_text(
        '{"type":"extract_hotwords","request_id":"r-1","text":"北京 清华大学"}'
    )
    await _drain_extract_tasks(session)

    assert any(
        m.get("type") == "extract_hotwords_result"
        and m.get("request_id") == "r-1"
        and m.get("hotwords") == ["北京", "清华大学"]
        for m in ws.sent
    )


@pytest.mark.asyncio
async def test_streaming_session_extract_hotwords_error(monkeypatch):
    from backend.streaming import StreamingSession, VadSegmentedStream

    async def _broken_query(_text: str) -> list[str]:
        raise RuntimeError("upstream LLM exploded")

    monkeypatch.setattr(
        "backend.streaming.session.query_text_hotwords", _broken_query
    )

    ws = _FakeWebSocket()
    session = StreamingSession(
        ws,  # type: ignore[arg-type]
        stream=VadSegmentedStream(),
        engine=TsAsrTaskEngine(),
    )
    ws.sent.clear()

    await session._handle_text(
        '{"type":"extract_hotwords","request_id":"r-2","text":"x"}'
    )
    await _drain_extract_tasks(session)

    err = next(
        m for m in ws.sent if m.get("type") == "extract_hotwords_error"
    )
    assert err["request_id"] == "r-2"
    assert "upstream LLM exploded" in err["message"]


# ---------------------------------------------------------------------------
# WAV decoder (backend.audio.utils.wav_base64_to_pcm_16k_mono)
# ---------------------------------------------------------------------------


def test_wav_decoder_empty_payload_raises():
    with pytest.raises(ValueError):
        wav_base64_to_pcm_16k_mono("")


def test_wav_decoder_returns_empty_for_zero_frames():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"")
    b64 = base64.b64encode(buf.getvalue()).decode()
    pcm = wav_base64_to_pcm_16k_mono(b64)
    assert pcm.size == 0
    assert pcm.dtype == np.float32
