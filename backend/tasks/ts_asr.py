"""Target-Speaker ASR task engine.

Paired with ``VadSegmentedStream`` (to match the primary ASR pipeline so the
demo feels identical from the user side), this engine:

* decodes and caches the enrollment (reference) audio during ``on_start``;
* submits ``build_tsasr_content(enrollment, mixed)`` to the TS-capable vLLM
  endpoint on every segment;
* runs the secondary general-purpose ASR (Qwen3-ASR-1.7B) on the mixed clip
  in parallel as a *presence gate* — if either path produces empty text the
  segment / partial is suppressed (silences out interferers and TS-ASR
  hallucinations on pure noise);
* emits a throttled ``partial`` (pseudo-streaming) on every PartialSnapshot
  so the UI can update text smoothly while the user keeps speaking.

Design intent: this file is a *thin orchestrator*. All prompt / network
details belong in :mod:`backend.tsasr` so the engine can remain stable as
the prompt template evolves.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid

from ..asr.client import query_audio_model_secondary
from ..audio.utils import pcm_to_wav_base64
from ..audio.vad import analyze_speech_presence
from ..config import SAMPLE_RATE
from ..streaming.events import PartialSnapshot, SegmentReady
from ..streaming.session import SessionContext
from ..tsasr.client import query_tsasr_model
from ..tsasr.enrollment import EnrollmentAudio, EnrollmentError, decode_enrollment
from .base import BaseTaskEngine

logger = logging.getLogger(__name__)


def _resolve(cfg, tsasr_field: str, fallback_field: str) -> str:
    """Return ``cfg.tsasr_*`` when set, else fall back to ``cfg.vllm_*``."""
    value = getattr(cfg, tsasr_field, "") or ""
    if value:
        return value
    return getattr(cfg, fallback_field, "") or ""


def _passes_speech_gate(segment, cfg, *, kind: str) -> bool:
    """Reject VAD segments dominated by transient noise (e.g. keyboard taps).

    Returns True if the clip should proceed to TS-ASR inference. The gate is
    a no-op when ``tsasr_speech_gate_enabled`` is False or the configured
    minimum voiced duration is non-positive, preserving the legacy behavior
    of forwarding every VAD-emitted segment.
    """
    if not bool(getattr(cfg, "tsasr_speech_gate_enabled", True)):
        return True

    min_voiced_ms = float(getattr(cfg, "tsasr_speech_gate_min_voiced_ms", 0))
    if min_voiced_ms <= 0:
        return True

    prob_threshold = float(
        getattr(cfg, "tsasr_speech_gate_prob_threshold", 0.6)
    )
    stats = analyze_speech_presence(segment, prob_threshold=prob_threshold)
    voiced_ms = stats.voiced_sec * 1000.0
    if voiced_ms < min_voiced_ms:
        logger.info(
            "TS-ASR %s gated: voiced=%.0fms<%0.fms ratio=%.2f mean_prob=%.2f "
            "thr=%.2f total=%.2fs",
            kind,
            voiced_ms,
            min_voiced_ms,
            stats.voiced_ratio,
            stats.mean_prob,
            prob_threshold,
            stats.total_sec,
        )
        return False
    return True


class TsAsrTaskEngine(BaseTaskEngine):
    """Runs TS-ASR inference against per-segment mixed audio."""

    name = "tsasr"

    def __init__(self) -> None:
        self._enrollment: EnrollmentAudio | None = None
        self._voice_traits: str | None = None
        # Cache the resolved hotword-enable flag so ``on_start`` is the single
        # source of truth regardless of later hotword updates.
        self._hotwords_enabled: bool = False
        # Per-engine segment-ID source. The frontend keys its replayable WAV
        # cache by this id, so the value MUST be unique across every engine
        # instance the browser has ever talked to on this page — otherwise a
        # new session's ``tsasr-1`` would collide with a previous session's
        # cached blob and the replay button would play the wrong clip. We
        # mint a fresh 8-hex-char prefix on construction (one engine = one
        # WebSocket) and append a monotonic counter, giving readable ids
        # like ``tsasr-3f9a1c2e-1`` that stay ordered per session.
        self._segment_prefix: str = uuid.uuid4().hex[:8]
        self._segment_counter: int = 0
        # Active utterance id, shared between consecutive partials and the
        # final segment that closes them. Allocated on the first partial of
        # an utterance and consumed by ``handle_segment`` so the frontend can
        # replace the partial bubble in-place when ``final`` arrives. ``None``
        # means "no partial has fired for the in-progress utterance yet";
        # in that case ``handle_segment`` falls back to minting a fresh id.
        self._active_utterance_id: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def on_start(self, ctrl: dict, ctx: SessionContext) -> None:
        cfg = ctx.cfg
        min_sec = float(getattr(cfg, "tsasr_enrollment_min_sec", 1.0))
        max_sec = float(getattr(cfg, "tsasr_enrollment_max_sec", 30.0))

        audio_b64 = (
            ctrl.get("enrollment_audio")
            or ctrl.get("enrollment_wav_base64")
            or ""
        )
        audio_fmt = ctrl.get("enrollment_format", "wav")

        try:
            self._enrollment = decode_enrollment(
                audio_b64,
                min_sec=min_sec,
                max_sec=max_sec,
                audio_format=audio_fmt,
            )
        except EnrollmentError as err:
            logger.warning("Enrollment rejected [%s]: %s", err.code, err)
            # Don't re-raise: the session wraps on_start in try/except and
            # would otherwise continue accepting PCM silently. Instead we
            # leave ``_enrollment`` as None so subsequent segments are
            # short-circuited and the client sees this error + no finals.
            await ctx.send_json(
                {
                    "type": "error",
                    "code": f"enrollment_{err.code}",
                    "message": str(err),
                }
            )
            return

        # ``voice_traits`` is accepted from the client for backward
        # compatibility (older demo builds exposed it as a free-form
        # speaker description) but is intentionally NOT injected into
        # the prompt -- v3 SFT training data has no ``Speaker traits:``
        # segment, so adding one would push the prompt off-distribution.
        # We still cache it here so it shows up in session logs, which
        # helps trace any client that's still sending the field.
        voice_traits = ctrl.get("voice_traits")
        if isinstance(voice_traits, str) and voice_traits.strip():
            self._voice_traits = voice_traits.strip()
        else:
            self._voice_traits = None

        self._hotwords_enabled = bool(
            getattr(cfg, "tsasr_enable_hotwords", False)
        )

        logger.info(
            "TS-ASR session ready: enrollment=%.2fs traits=%r hotwords=%s",
            self._enrollment.duration_sec,
            self._voice_traits,
            self._hotwords_enabled,
        )
        await ctx.send_json(
            {
                "type": "enrollment_ok",
                "duration_sec": round(self._enrollment.duration_sec, 3),
                "sample_rate_hz": SAMPLE_RATE,
            }
        )

    # ------------------------------------------------------------------
    # Speech lifecycle (placeholder UI)
    # ------------------------------------------------------------------

    async def handle_speech_start(self, ctx: SessionContext) -> None:
        """Announce a placeholder utterance the moment VAD opens.

        Mints the segment id ahead of inference so the frontend can
        paint the "识别中…" bubble immediately rather than waiting for
        the segment-end -> inference -> final round trip. The same id
        is reused in ``handle_segment`` (or retracted by
        ``handle_speech_dropped``).
        """
        if self._enrollment is None:
            # Enrollment was rejected or hasn't arrived yet; suppress
            # the placeholder so we don't paint a bubble that can never
            # be filled in.
            return
        if self._active_utterance_id is not None:
            # Defensive: a previous announce wasn't consumed (shouldn't
            # happen since the stream resets ``_announced_speech`` on
            # every segment-end, but better safe than leaky).
            return
        self._segment_counter += 1
        seg_id = f"tsasr-{self._segment_prefix}-{self._segment_counter}"
        self._active_utterance_id = seg_id
        await ctx.send_json(
            {
                "type": "processing",
                "id": seg_id,
                "language": ctx.language,
                "task": "tsasr",
            }
        )

    async def handle_speech_dropped(self, ctx: SessionContext) -> None:
        """Retract a placeholder when VAD ends without a usable segment.

        The stream emits this when the in-flight utterance is too
        short / sub-threshold to forward. We send an empty ``final``
        carrying the same id so the frontend can drop the bubble.
        """
        stale_id = self._active_utterance_id
        self._active_utterance_id = None
        if stale_id is None:
            return
        await ctx.send_json(
            {
                "type": "final",
                "id": stale_id,
                "text": "",
                "language": ctx.language,
                "task": "tsasr",
            }
        )

    # ------------------------------------------------------------------
    # Per-segment inference
    # ------------------------------------------------------------------

    async def handle_segment(
        self, seg: SegmentReady, ctx: SessionContext
    ) -> bool:
        if self._enrollment is None:
            logger.warning("TS-ASR segment dropped: no enrollment cached")
            return False

        cfg = ctx.cfg
        segment = seg.pcm
        audio_duration = len(segment) / SAMPLE_RATE

        max_seconds = float(getattr(cfg, "tsasr_max_audio_seconds", 0.0))
        if max_seconds > 0 and audio_duration > max_seconds:
            max_samples = int(SAMPLE_RATE * max_seconds)
            logger.info(
                "Trimming TS-ASR segment %.1fs -> %.1fs (cap)",
                audio_duration, max_seconds,
            )
            segment = segment[-max_samples:]
            audio_duration = len(segment) / SAMPLE_RATE

        if not _passes_speech_gate(segment, cfg, kind="segment"):
            # Drop any pending utterance id since this segment is being
            # discarded. If a partial / processing bubble is already on
            # screen, ask the frontend to discard it via an empty-text
            # final carrying the same id.
            stale_id = self._active_utterance_id
            self._active_utterance_id = None
            if stale_id is not None:
                await ctx.send_json(
                    {
                        "type": "final",
                        "id": stale_id,
                        "text": "",
                        "language": ctx.language,
                        "task": "tsasr",
                    }
                )
            return False

        # Reuse the id that ``handle_speech_start`` (or a partial)
        # already minted and announced to the frontend. The placeholder
        # bubble is already on screen — we just need to upgrade it. If
        # neither path ran (e.g. WebSocket reconnect, or speech-start
        # was suppressed because enrollment wasn't ready yet), mint a
        # fresh id and announce it now as a fallback so the bubble
        # still gets created before the final text arrives.
        if self._active_utterance_id is not None:
            seg_id = self._active_utterance_id
            self._active_utterance_id = None
        else:
            self._segment_counter += 1
            seg_id = f"tsasr-{self._segment_prefix}-{self._segment_counter}"
            await ctx.send_json(
                {
                    "type": "processing",
                    "id": seg_id,
                    "language": ctx.language,
                    "task": "tsasr",
                }
            )

        t0 = time.monotonic()
        mixed_b64 = pcm_to_wav_base64(segment)

        text, sec_text, detected_lang = await self._dual_infer(mixed_b64, ctx)

        elapsed = time.monotonic() - t0
        rtf = elapsed / audio_duration if audio_duration > 0 else 0.0
        logger.info(
            "Final TS-ASR: audio=%.2fs infer=%.3fs RTF=%.3f "
            "text=%r sec=%r",
            audio_duration, elapsed, rtf,
            (text or "")[:80], (sec_text or "")[:80],
        )

        if not text and not sec_text:
            # Both paths empty (or the optional secondary gate suppressed
            # the segment). Emit an empty final for the announced id so
            # the frontend can drop the placeholder bubble it has just
            # painted — otherwise it would spin forever.
            await ctx.send_json(
                {
                    "type": "final",
                    "id": seg_id,
                    "text": "",
                    "text_secondary": "",
                    "language": ctx.language,
                    "task": "tsasr",
                }
            )
            return False

        payload: dict = {
            "type": "final",
            "id": seg_id,
            "text": text,
            "text_secondary": sec_text,
            "language": detected_lang,
            "task": "tsasr",
            # The mixed audio that was actually fed to the model. The client
            # uses it to wire up a replay button next to the transcript so
            # users can sanity-check what the target-speaker extraction
            # heard. Size is bounded by tsasr_max_audio_seconds.
            "audio_b64": mixed_b64,
            "duration_sec": round(audio_duration, 2),
        }
        return await ctx.send_json(payload)

    # ------------------------------------------------------------------
    # Pseudo-streaming partial
    # ------------------------------------------------------------------

    async def handle_partial(
        self, snap: PartialSnapshot, ctx: SessionContext
    ) -> None:
        cfg = ctx.cfg
        if not bool(getattr(cfg, "tsasr_enable_partial", True)):
            return
        if self._enrollment is None:
            return

        snapshot = snap.pcm
        audio_duration = len(snapshot) / SAMPLE_RATE

        if not _passes_speech_gate(snapshot, cfg, kind="partial"):
            return

        t0 = time.monotonic()
        mixed_b64 = pcm_to_wav_base64(snapshot)

        text, sec_text, _ = await self._dual_infer(mixed_b64, ctx)

        elapsed = time.monotonic() - t0
        rtf = elapsed / audio_duration if audio_duration > 0 else 0.0
        logger.info(
            "Partial TS-ASR: audio=%.2fs infer=%.3fs RTF=%.3f "
            "text=%r sec=%r",
            audio_duration, elapsed, rtf,
            (text or "")[:80], (sec_text or "")[:80],
        )

        if not text and not sec_text:
            # Both paths empty (or the optional secondary gate suppressed
            # the segment). If we already painted a partial in this
            # utterance, retract the bubble — the frontend treats an
            # empty-text partial with a known id as "discard". The next
            # non-empty partial mints a fresh id (and bubble) so the
            # user never sees stale text lingering through a silent gap.
            # If we never sent a partial for this utterance yet, stay
            # completely quiet.
            stale_id = self._active_utterance_id
            self._active_utterance_id = None
            if stale_id is not None:
                await ctx.send_json(
                    {
                        "type": "partial",
                        "id": stale_id,
                        "text": "",
                        "text_secondary": "",
                        "language": ctx.language,
                        "task": "tsasr",
                    }
                )
            return

        # Allocate an utterance id on the first non-empty partial of this
        # run so it can be reused by both subsequent partials and the
        # eventual final. We deliberately allocate AFTER the dual-infer
        # gate so empty partials don't burn a counter slot — keeps the
        # logged id sequence aligned with what the user actually sees.
        if self._active_utterance_id is None:
            self._segment_counter += 1
            self._active_utterance_id = (
                f"tsasr-{self._segment_prefix}-{self._segment_counter}"
            )
        utt_id = self._active_utterance_id

        await ctx.send_json(
            {
                "type": "partial",
                "id": utt_id,
                "text": text,
                "text_secondary": sec_text,
                "language": ctx.language,
                "task": "tsasr",
            }
        )

    # ------------------------------------------------------------------
    # Dual-channel inference (TS-ASR + Qwen3-ASR presence gate)
    # ------------------------------------------------------------------

    async def _dual_infer(
        self, mixed_b64: str, ctx: SessionContext
    ) -> tuple[str, str, str]:
        """Run TS-ASR inference, optionally alongside a parallel Qwen3-ASR run.

        Returns ``(ts_text, sec_text, lang)``:

        * ``ts_text`` — AmphionTSASR transcription (target speaker only)
        * ``sec_text`` — Qwen3-ASR general-purpose transcription of the
          same mixed clip; empty string when the secondary path is
          disabled, suppressed, or failed.
        * ``lang`` — detected language (from AmphionTSASR when available,
          else falls back to ``ctx.language``).

        Behavior is driven by two independent flags:

        * ``tsasr_show_secondary_text`` (default True) — surfaces the
          Qwen3 transcription to the caller (rendered as a second labeled
          row on the frontend). Always runs the secondary in parallel.
        * ``tsasr_enable_secondary_gate`` (default False) — uses Qwen3 as
          a silence/presence gate; when on, both texts are zeroed if
          either path is empty (protects against TS-ASR hallucinations
          on pure noise).

        When both flags are off, only AmphionTSASR runs, mirroring the
        original single-path behavior.
        """
        cfg = ctx.cfg
        hotwords = list(ctx.hotwords) if self._hotwords_enabled else None

        base_url = _resolve(cfg, "tsasr_base_url", "vllm_base_url")
        model_name = _resolve(cfg, "tsasr_model_name", "vllm_model_name")
        ts_timeout = float(
            getattr(cfg, "tsasr_request_timeout", 0)
            or getattr(cfg, "asr_request_timeout", 30.0)
        )

        ts_coro = query_tsasr_model(
            mixed_b64,
            self._enrollment.wav_base64,  # type: ignore[union-attr]
            hotwords=hotwords,
            voice_traits=self._voice_traits,
            base_url=base_url,
            model_name=model_name,
            timeout=ts_timeout,
            enrollment_duration_sec=self._enrollment.duration_sec,  # type: ignore[union-attr]
        )

        show_secondary = bool(
            getattr(cfg, "tsasr_show_secondary_text", True)
        )
        use_gate = bool(
            getattr(cfg, "tsasr_enable_secondary_gate", False)
        )
        run_secondary = show_secondary or use_gate

        if not run_secondary:
            try:
                ts_res = await ts_coro
            except Exception as exc:
                logger.warning("TS-ASR failed: %s", exc)
                return "", "", ctx.language
            ts_text = str((ts_res or {}).get("transcription") or "").strip()
            if not ts_text:
                logger.debug("TS-ASR suppressed: empty transcription")
                return "", "", ctx.language
            detected_lang = (
                (ts_res or {}).get("detected_language") or ctx.language
            )
            return ts_text, "", detected_lang

        sec_timeout = float(getattr(cfg, "asr_request_timeout", 30.0))
        sec_base_url = getattr(cfg, "secondary_vllm_base_url", "") or ""
        sec_model_name = getattr(cfg, "secondary_vllm_model_name", "") or ""

        ts_task = asyncio.create_task(ts_coro)
        sec_task = asyncio.create_task(
            query_audio_model_secondary(
                mixed_b64,
                hotwords=hotwords,
                base_url=sec_base_url,
                model_name=sec_model_name,
                timeout=sec_timeout,
            )
        )
        ts_res, sec_res = await asyncio.gather(
            ts_task, sec_task, return_exceptions=True
        )

        if isinstance(sec_res, Exception):
            logger.warning("TS-ASR dual: secondary ASR failed: %s", sec_res)
            sec_res = None
        if isinstance(ts_res, Exception):
            logger.warning("TS-ASR dual: TS-ASR failed: %s", ts_res)
            ts_res = None

        sec_text = str((sec_res or {}).get("transcription") or "").strip()
        ts_text = str((ts_res or {}).get("transcription") or "").strip()
        detected_lang = (
            (ts_res or {}).get("detected_language") or ctx.language
        )

        if use_gate and (not sec_text or not ts_text):
            if not sec_text:
                logger.debug("TS-ASR gate: secondary empty (silence)")
            if not ts_text:
                logger.debug("TS-ASR gate: target speaker silent")
            return "", "", ctx.language

        return ts_text, sec_text, detected_lang

    # ------------------------------------------------------------------
    # Stop guarantee
    # ------------------------------------------------------------------

    async def on_stop(
        self,
        ctx: SessionContext,
        *,
        sent_any_response: bool,
        stopped: bool,
    ) -> None:
        if stopped and not sent_any_response:
            await ctx.send_json(
                {
                    "type": "final",
                    "text": "",
                    "language": ctx.language,
                    "task": "tsasr",
                }
            )
