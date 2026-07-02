"""AudioStream implementation backed by the k2 gRPC streaming ASR service."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator

import grpc
import numpy as np

from ..asr.k2 import asr_pb2 as pb
from ..asr.k2.client import get_k2_stub, validate_k2_server
from ..audio.vad import TenVad, VADProcessor
from ..config import SAMPLE_RATE, Config
from .events import PartialText, SegmentReady

logger = logging.getLogger(__name__)

_BYTES_PER_SAMPLE = 2
_EVENT_SENTINEL = object()
_REQUEST_SENTINEL = object()
_IDLE_SPEECH_RMS = 0.003
_DIGITAL_SILENCE_RMS = 0.0005


class _K2VoiceGate:
    """Incremental speech-evidence gate for k2 events.

    k2 remains the endpoint authority. This gate only answers whether the
    current k2 segment has accumulated enough local speech evidence to be
    forwarded to partial/final consumers; it never trims or re-cuts the audio.
    """

    def __init__(self, cfg: Config) -> None:
        self.enabled = bool(cfg.k2_voice_gate_enabled)
        self.threshold = float(cfg.k2_voice_gate_threshold)
        self.start_frames = max(1, int(cfg.k2_voice_gate_start_frames))
        self.smoothing_alpha = min(1.0, max(0.0, float(cfg.vad_smoothing_alpha)))
        self._processor = VADProcessor(
            threshold=self.threshold,
            smoothing_alpha=self.smoothing_alpha,
            start_frames=self.start_frames,
            sample_rate=SAMPLE_RATE,
        )
        self._carry = np.empty(0, dtype=np.int16)
        self._smoothed_prob: float | None = None
        self._speech_count = 0
        self._has_voice = not self.enabled

    @property
    def has_voice(self) -> bool:
        return self._has_voice

    def reset(self) -> None:
        self._carry = np.empty(0, dtype=np.int16)
        self._smoothed_prob = None
        self._speech_count = 0
        self._has_voice = not self.enabled

    def feed(self, pcm_bytes: bytes) -> None:
        if not self.enabled or not pcm_bytes:
            return
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
        if self._carry.size:
            pcm = np.concatenate([self._carry, pcm])
        hop = self._processor.hop_size
        used = (len(pcm) // hop) * hop
        self._carry = (
            pcm[used:].copy() if used < len(pcm) else np.empty(0, dtype=np.int16)
        )
        for i in range(0, used, hop):
            self._process_frame(pcm[i : i + hop])

    def _process_frame(self, frame_i16: np.ndarray) -> None:
        vad_backend = self._processor.vad
        if TenVad is not None and isinstance(vad_backend, TenVad):
            vad_input = frame_i16
        else:
            # The energy fallback expects normalized float PCM; callers in the
            # local VAD path normally provide that already, but k2 receives raw
            # int16 bytes so we normalize explicitly here.
            vad_input = frame_i16.astype(np.float32) / 32768.0
        raw_prob = self._processor._extract_prob(vad_backend.process(vad_input))
        if self._smoothed_prob is None:
            self._smoothed_prob = raw_prob
        else:
            a = self.smoothing_alpha
            self._smoothed_prob = (a * self._smoothed_prob) + ((1.0 - a) * raw_prob)
        if self._smoothed_prob > self.threshold:
            self._speech_count += 1
        else:
            self._speech_count = 0
        if self._speech_count >= self.start_frames:
            self._has_voice = True


class K2SegmentedStream:
    """k2-backed stream that emits k2 partials and LLM-ready segments.

    k2 owns endpointing. This stream keeps a bounded copy of the PCM sent to
    k2 and emits that same audio for LLM final inference; local VAD must not
    re-decide segment starts/ends or it can drop soft onsets that k2 heard.
    """

    def __init__(self) -> None:
        self._cfg: Config | None = None
        self._request_q: asyncio.Queue[object] = asyncio.Queue(maxsize=200)
        self._event_q: asyncio.Queue[object] = asyncio.Queue()
        self._recv_task: asyncio.Task | None = None
        self._call = None
        self._started = False
        self._closing = False
        self._closed = False

        self._buffer = bytearray()
        self._buffer_start_sample = 0
        self._total_samples = 0
        self._saw_partial = False
        self._segment_seq = 0
        self._active_segment_id: str | None = None
        self._idle_seen_speech = False
        self._voice_gate: _K2VoiceGate | None = None

    def configure(self, cfg: Config) -> None:
        self._cfg = cfg
        self._voice_gate = _K2VoiceGate(cfg)

    @property
    def cfg(self) -> Config:
        if self._cfg is None:
            raise RuntimeError("K2SegmentedStream.configure() not called")
        return self._cfg

    async def start(self) -> None:
        """Open the k2 Recognize stream and start receiving events."""

        if self._started:
            return
        cfg = self.cfg
        await validate_k2_server(cfg)
        stub = get_k2_stub(cfg.k2_target)
        self._call = stub.Recognize(self._request_iter())
        self._recv_task = asyncio.create_task(self._recv_loop())
        self._started = True

    def feed(self, pcm_bytes: bytes) -> list:
        """Queue PCM for k2 and local buffering; async events arrive separately."""

        if not pcm_bytes or self._closing or self._closed:
            return []
        if len(pcm_bytes) % _BYTES_PER_SAMPLE:
            pcm_bytes = pcm_bytes[:-1]
        if not pcm_bytes:
            return []

        self._append_buffer(pcm_bytes)
        self._update_idle_speech(pcm_bytes)
        self._update_voice_gate(pcm_bytes)
        req = pb.PcmRequest(
            audio_chunk=pb.AudioChunk(
                data=pcm_bytes,
                client_send_us=int(time.time() * 1_000_000),
            )
        )
        try:
            self._request_q.put_nowait(req)
        except asyncio.QueueFull:
            raise RuntimeError("k2 request queue is full")

        self._trim_idle_buffer_if_needed()
        self._force_cut_if_needed()
        return []

    async def events(self) -> AsyncIterator[object]:
        while True:
            item = await self._event_q.get()
            if item is _EVENT_SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    async def flush(self, *, force: bool) -> list:
        """Tell k2 no more audio is coming; final events arrive via events()."""

        if self._closing or self._closed:
            return []
        self._closing = True
        try:
            await self._request_q.put(pb.PcmRequest(end_of_stream=pb.EndOfStream()))
            await self._request_q.put(_REQUEST_SENTINEL)
        except RuntimeError:
            pass
        return []

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closing = True
        try:
            await self._request_q.put(_REQUEST_SENTINEL)
        except RuntimeError:
            pass
        if self._call is not None:
            try:
                self._call.cancel()
            except Exception:
                pass
        if self._recv_task is not None:
            await asyncio.gather(self._recv_task, return_exceptions=True)
        self._closed = True

    async def _request_iter(self):
        cfg = self.cfg
        yield pb.PcmRequest(
            session_config=pb.SessionConfig(
                trace_id=f"audiollm-{int(time.time() * 1000)}",
                client_app="audiollm-demo",
                audio_format=pb.AudioFormat(
                    sample_rate=int(cfg.k2_sample_rate or SAMPLE_RATE),
                    encoding=pb.PCM_S16LE,
                    channels=1,
                ),
                decoding=pb.Decoding(method=pb.GREEDY_SEARCH),
                enable_endpoint=True,
                # k2 is intentionally plain recognition only.
                include_token_timestamps=False,
            )
        )
        while True:
            item = await self._request_q.get()
            if item is _REQUEST_SENTINEL:
                break
            yield item

    async def _recv_loop(self) -> None:
        try:
            async for ev in self._call:
                which = ev.WhichOneof("payload")
                if which == "partial":
                    text = ev.partial.text.strip()
                    is_first_partial = not self._saw_partial
                    self._saw_partial = True
                    if text and self.cfg.enable_pseudo_stream:
                        segment_id = self._current_segment_id()
                        if not self._voice_gate_has_voice():
                            if is_first_partial:
                                logger.info(
                                    "k2 first partial suppressed by voice gate "
                                    "id=%s text=%r",
                                    segment_id,
                                    text,
                                )
                            continue
                        if is_first_partial:
                            logger.info("k2 first partial id=%s text=%r", segment_id, text)
                        await self._event_q.put(
                            PartialText(text=text, id=segment_id)
                        )
                elif which in ("endpoint", "final"):
                    self._emit_buffered_segment(is_stop_flush=self._closing)
                elif which == "error":
                    self._emit_buffered_segment(is_stop_flush=True)
                    msg = ev.error.message or f"k2 error code={ev.error.code}"
                    await self._event_q.put(RuntimeError(msg))
                    break
                elif which == "session_ended":
                    self._emit_buffered_segment(is_stop_flush=True)
                    break
        except asyncio.CancelledError:
            raise
        except grpc.RpcError as exc:
            self._emit_buffered_segment(is_stop_flush=True)
            await self._event_q.put(RuntimeError(f"k2 stream failed: {exc}"))
        finally:
            self._closed = True
            await self._event_q.put(_EVENT_SENTINEL)

    def _append_buffer(self, pcm_bytes: bytes) -> None:
        if not self._buffer:
            self._buffer_start_sample = self._total_samples
        self._buffer.extend(pcm_bytes)
        self._total_samples += len(pcm_bytes) // _BYTES_PER_SAMPLE

    def _reset_buffer(self) -> None:
        self._buffer.clear()
        self._buffer_start_sample = self._total_samples
        self._saw_partial = False
        self._idle_seen_speech = False
        if self._voice_gate is not None:
            self._voice_gate.reset()

    def _current_segment_id(self) -> str:
        if self._active_segment_id is None:
            self._segment_seq += 1
            self._active_segment_id = f"seg-{self._segment_seq}"
        return self._active_segment_id

    def _finish_segment_id(self) -> str:
        segment_id = self._current_segment_id()
        self._active_segment_id = None
        return segment_id

    def _trim_idle_buffer_if_needed(self) -> None:
        if self._saw_partial or self._idle_seen_speech:
            return
        keep_samples = int(SAMPLE_RATE * self.cfg.k2_idle_keep_ms / 1000)
        if keep_samples <= 0:
            return
        keep_bytes = keep_samples * _BYTES_PER_SAMPLE
        if len(self._buffer) <= keep_bytes:
            return
        drop_bytes = len(self._buffer) - keep_bytes
        del self._buffer[:drop_bytes]
        self._buffer_start_sample += drop_bytes // _BYTES_PER_SAMPLE

    def _update_idle_speech(self, pcm_bytes: bytes) -> None:
        if self._idle_seen_speech:
            return
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if pcm.size == 0:
            return
        rms = float(np.sqrt(np.mean(np.square(pcm), dtype=np.float32)))
        if rms >= _IDLE_SPEECH_RMS:
            self._idle_seen_speech = True

    def _update_voice_gate(self, pcm_bytes: bytes) -> None:
        if self._voice_gate is not None:
            self._voice_gate.feed(pcm_bytes)

    def _voice_gate_has_voice(self) -> bool:
        if self._voice_gate is None:
            return True
        return self._voice_gate.has_voice

    def _force_cut_if_needed(self) -> None:
        max_samples = int(float(self.cfg.k2_max_segment_sec) * SAMPLE_RATE)
        if max_samples <= 0:
            return
        buffered_samples = len(self._buffer) // _BYTES_PER_SAMPLE
        if buffered_samples < max_samples:
            return
        logger.warning(
            "Force-cutting k2 segment at %.1fs (cap %.0fs)",
            buffered_samples / SAMPLE_RATE,
            float(self.cfg.k2_max_segment_sec),
        )
        self._emit_buffered_segment(is_stop_flush=False)

    def _emit_buffered_segment(self, *, is_stop_flush: bool) -> None:
        if not self._buffer:
            self._reset_buffer()
            return
        segment_id = self._finish_segment_id()
        base_sample = self._buffer_start_sample
        saw_partial = self._saw_partial
        voice_gate_has_voice = self._voice_gate_has_voice()
        pcm_i16 = np.frombuffer(bytes(self._buffer), dtype=np.int16)
        self._reset_buffer()
        if pcm_i16.size == 0:
            return
        pcm = pcm_i16.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(np.square(pcm), dtype=np.float32)))
        if not saw_partial and rms < _DIGITAL_SILENCE_RMS:
            return
        if not voice_gate_has_voice:
            logger.info(
                "k2 segment dropped by voice gate id=%s audio=%.2fs rms=%.5f",
                segment_id,
                pcm.size / SAMPLE_RATE,
                rms,
            )
            return
        end_sample = base_sample + pcm.size
        logger.info(
            "k2 segment ready id=%s audio=%.2fs start=%.0fms end=%.0fms stop_flush=%s",
            segment_id,
            pcm.size / SAMPLE_RATE,
            base_sample * 1000.0 / SAMPLE_RATE,
            end_sample * 1000.0 / SAMPLE_RATE,
            is_stop_flush,
        )
        self._event_q.put_nowait(
            SegmentReady(
                pcm=pcm,
                is_stop_flush=is_stop_flush,
                id=segment_id,
                start_ms=base_sample * 1000.0 / SAMPLE_RATE,
                end_ms=end_sample * 1000.0 / SAMPLE_RATE,
            )
        )
