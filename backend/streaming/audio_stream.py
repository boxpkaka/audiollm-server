"""AudioStream strategies that decouple PCM ingestion from inference logic.

Two built-in strategies are provided:

- :class:`VadSegmentedStream` slices the input stream by VAD-detected speech
  segments and optionally emits :class:`PartialSnapshot` events while the user
  is still speaking (used for the ASR task).
- :class:`WholeUtteranceStream` accumulates everything and only emits a single
  :class:`SegmentReady` when the session is flushed (used for the emotion task).

Both classes implement the :class:`AudioStream` protocol so the
:class:`StreamingSession` can drive them without knowing the task semantics.
"""

from __future__ import annotations

import logging
import time
from typing import Iterable, Protocol, runtime_checkable

import numpy as np

from ..audio.vad import VADProcessor
from ..config import SAMPLE_RATE, Config
from .events import PartialSnapshot, SegmentReady, SpeechDropped, SpeechStarted

logger = logging.getLogger(__name__)

StreamEvent = SegmentReady | PartialSnapshot | SpeechStarted | SpeechDropped


@runtime_checkable
class AudioStream(Protocol):
    """Strategy interface for slicing the incoming PCM byte stream."""

    def configure(self, cfg: Config) -> None:
        """Apply (possibly per-session-overridden) Config knobs."""

    def feed(self, pcm_bytes: bytes) -> Iterable[StreamEvent]:
        """Push raw int16 little-endian PCM bytes; return zero or more events."""

    def flush(self, *, force: bool) -> Iterable[StreamEvent]:
        """Drain any remaining buffered audio (called on stop / disconnect).

        When ``force`` is True the implementation should always emit the
        residual audio (subject to non-empty); when False it may discard
        sub-threshold leftovers. Streams that announce ``SpeechStarted``
        must also emit ``SpeechDropped`` here whenever the in-flight
        utterance can't be recovered, so engines can retract any
        placeholder UI they painted on speech-start.
        """


def _pcm_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


class VadSegmentedStream:
    """VAD-segmented stream with optional partial snapshots.

    Replicates the segmentation behavior of the original
    ``ASRStreamingSession``: feeds PCM frames into a :class:`VADProcessor`,
    emits a :class:`SegmentReady` when speech-to-silence is detected, and
    optionally emits a throttled :class:`PartialSnapshot` while the user is
    still speaking.
    """

    def __init__(self, *, enable_partial: bool | None = None) -> None:
        self.vad = VADProcessor()
        self._pcm_carry: np.ndarray = np.empty(0, dtype=np.float32)
        # Running count of samples the VAD has processed across all feeds.
        # Used to stamp each segment with an approximate session-timeline
        # position (SegmentReady.start_ms/end_ms). Carry-over samples are
        # counted in the feed that actually processes them, so this stays a
        # continuous, monotonic clock.
        self._consumed_samples: int = 0
        self._cfg: Config | None = None
        self._partial_interval: float = 0.5
        self._last_partial_time: float = 0.0
        # ``None`` means "follow cfg.enable_pseudo_stream"; callers that know
        # their downstream engine doesn't consume partials (e.g. emotion) can
        # pass ``False`` to skip the snapshot bookkeeping entirely regardless
        # of how the deployment toggles pseudo-stream globally.
        self._partial_override: bool | None = enable_partial
        self._enable_partial: bool = True
        # Tracks whether we've already emitted ``SpeechStarted`` for the
        # current in-flight utterance. Reset every time speech ends
        # (whether the resulting segment was usable or got dropped) so
        # the next silent->speaking transition fires exactly one event.
        self._announced_speech: bool = False

    def configure(self, cfg: Config) -> None:
        self._cfg = cfg
        # Push the per-connection VAD tunables onto the live processor. Without
        # this, start.config / parameter.asr_config overrides for vad_threshold,
        # silence_duration_ms, etc. would silently no-op (the VAD was frozen to
        # process-wide defaults at construction).
        self.vad.apply_config(cfg)
        self._partial_interval = cfg.pseudo_stream_interval_ms / 1000.0
        if self._partial_override is None:
            # Partial snapshots only make sense if at least one ASR engine is
            # on AND pseudo-stream is enabled. Higher-level engines may further
            # suppress partials, but a stream that knows nothing about engines
            # uses the conservative default.
            self._enable_partial = bool(cfg.enable_pseudo_stream)
        else:
            self._enable_partial = self._partial_override

    @property
    def cfg(self) -> Config:
        if self._cfg is None:
            raise RuntimeError("VadSegmentedStream.configure() not called")
        return self._cfg

    def feed(self, pcm_bytes: bytes) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        pcm = _pcm_bytes_to_float32(pcm_bytes)

        if self._pcm_carry.size > 0:
            pcm = np.concatenate([self._pcm_carry, pcm])

        hop = self.vad.hop_size
        used = (len(pcm) // hop) * hop
        self._pcm_carry = (
            pcm[used:].copy() if used < len(pcm) else np.empty(0, dtype=np.float32)
        )

        cfg = self.cfg
        min_samples = int(SAMPLE_RATE * cfg.min_segment_duration_ms / 1000)
        # The first partial of each utterance uses its own decoupled floor:
        # lowering it speeds up the first partial without relaxing the
        # final-segment short-noise filter, which keeps using ``min_samples``
        # (line below and in ``flush``). Snapshots grow monotonically within an
        # utterance, so this floor only ever gates the first partial; later ones
        # are already longer. ``Config.__post_init__`` guarantees
        # pseudo_stream_first_partial_ms <= min_segment_duration_ms, so this
        # floor is never stricter than final's.
        first_partial_min_samples = int(
            SAMPLE_RATE * cfg.pseudo_stream_first_partial_ms / 1000
        )

        base = self._consumed_samples
        for i in range(0, used, hop):
            was_speaking = self.vad.is_speaking
            segment = self.vad.process(pcm[i : i + hop])
            now_speaking = self.vad.is_speaking

            # Silent -> speaking transition: announce immediately so any
            # downstream engine that wants to paint a placeholder ("识别
            # 中…") can do so before the user has even finished the
            # utterance.
            if not was_speaking and now_speaking and not self._announced_speech:
                self._announced_speech = True
                self._last_partial_time = 0.0
                events.append(SpeechStarted())

            if segment is None:
                continue

            # ``segment`` is the finalized speech buffer. Whether we
            # forward it as a usable ``SegmentReady`` or retract the
            # announcement via ``SpeechDropped``, the in-flight
            # announcement window has ended.
            announced = self._announced_speech
            self._announced_speech = False

            if len(segment) < min_samples:
                logger.info(
                    "Drop short segment (%.1fs)", len(segment) / SAMPLE_RATE
                )
                if announced:
                    events.append(SpeechDropped())
                continue
            end_sample = base + i + hop
            events.append(self._segment_with_timing(segment, end_sample))

        self._consumed_samples = base + used

        if self._enable_partial and self.vad.is_speaking:
            now = time.monotonic()
            if now - self._last_partial_time >= self._partial_interval:
                snapshot = self.vad.snapshot_incomplete_speech()
                if snapshot is not None and len(snapshot) >= first_partial_min_samples:
                    self._last_partial_time = now
                    events.append(PartialSnapshot(pcm=snapshot))

        return events

    def flush(self, *, force: bool) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        remaining = self.vad.flush()
        announced = self._announced_speech
        self._announced_speech = False
        if remaining is None or len(remaining) == 0:
            # No usable tail. If we'd announced speech-start the engine
            # is still showing a placeholder for it; let it retract.
            if announced:
                events.append(SpeechDropped())
            return events
        cfg = self.cfg
        min_samples = int(SAMPLE_RATE * cfg.min_segment_duration_ms / 1000)
        if not force and len(remaining) < min_samples:
            if announced:
                events.append(SpeechDropped())
            return events
        if force and len(remaining) < min_samples:
            # Force-flush still emits the residual but we drop the
            # placeholder announcement marker since this segment will
            # generate its own final.
            pass
        events.append(
            self._segment_with_timing(
                remaining, self._consumed_samples, is_stop_flush=force
            )
        )
        return events

    def _segment_with_timing(
        self,
        segment: np.ndarray,
        end_sample: int,
        *,
        is_stop_flush: bool = False,
    ) -> SegmentReady:
        start_sample = max(0, end_sample - len(segment))
        return SegmentReady(
            pcm=segment,
            is_stop_flush=is_stop_flush,
            start_ms=start_sample * 1000.0 / SAMPLE_RATE,
            end_ms=end_sample * 1000.0 / SAMPLE_RATE,
        )


class WholeUtteranceStream:
    """Accumulates the entire audio stream and emits one segment on flush.

    Used by tasks like emotion recognition where the upstream client sends a
    complete utterance and the model takes the full clip as input.
    """

    def __init__(self) -> None:
        self._buffers: list[np.ndarray] = []
        self._cfg: Config | None = None

    def configure(self, cfg: Config) -> None:
        self._cfg = cfg

    def feed(self, pcm_bytes: bytes) -> list[StreamEvent]:
        pcm = _pcm_bytes_to_float32(pcm_bytes)
        if pcm.size > 0:
            self._buffers.append(pcm)
        return []

    def flush(self, *, force: bool) -> list[StreamEvent]:
        if not self._buffers:
            return []
        merged = np.concatenate(self._buffers)
        self._buffers.clear()
        if merged.size == 0:
            return []
        return [SegmentReady(pcm=merged, is_stop_flush=force)]
