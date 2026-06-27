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
from ..audio.vad import VADProcessor
from ..config import SAMPLE_RATE, Config
from .events import PartialText, SegmentReady

logger = logging.getLogger(__name__)

_BYTES_PER_SAMPLE = 2
_EVENT_SENTINEL = object()
_REQUEST_SENTINEL = object()


class K2SegmentedStream:
    """k2-backed stream that emits k2 partials and LLM-ready segments.

    k2 owns endpointing. Local VAD is used only after k2 has decided a segment
    boundary, to trim leading/trailing silence before the LLM final path sees
    the audio.
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
        self._idle_vad = VADProcessor()
        self._idle_seen_speech = False

    def configure(self, cfg: Config) -> None:
        self._cfg = cfg
        self._idle_vad.apply_config(cfg)

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
                    self._saw_partial = True
                    if text and self.cfg.enable_pseudo_stream:
                        await self._event_q.put(PartialText(text=text))
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
        self._idle_vad = VADProcessor()
        self._idle_vad.apply_config(self.cfg)
        self._idle_seen_speech = False

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
        hop = self._idle_vad.hop_size
        used = (pcm.size // hop) * hop
        for off in range(0, used, hop):
            self._idle_vad.process(pcm[off : off + hop])
            if self._idle_vad.is_speaking:
                self._idle_seen_speech = True
                return

    def _force_cut_if_needed(self) -> None:
        max_samples = int(float(self.cfg.k2_max_segment_sec) * SAMPLE_RATE)
        if max_samples <= 0:
            return
        buffered_samples = len(self._buffer) // _BYTES_PER_SAMPLE
        if buffered_samples < max_samples:
            return
        logger.info(
            "Force-cutting k2 segment at %.1fs (cap %.0fs)",
            buffered_samples / SAMPLE_RATE,
            float(self.cfg.k2_max_segment_sec),
        )
        self._emit_buffered_segment(is_stop_flush=False)

    def _emit_buffered_segment(self, *, is_stop_flush: bool) -> None:
        if not self._buffer:
            self._reset_buffer()
            return
        trimmed = self._trim_segment(bytes(self._buffer), self._buffer_start_sample)
        self._reset_buffer()
        if trimmed is None:
            return
        pcm, start_sample, end_sample = trimmed
        self._event_q.put_nowait(
            SegmentReady(
                pcm=pcm,
                is_stop_flush=is_stop_flush,
                start_ms=start_sample * 1000.0 / SAMPLE_RATE,
                end_ms=end_sample * 1000.0 / SAMPLE_RATE,
            )
        )

    def _trim_segment(
        self, pcm_bytes: bytes, base_sample: int
    ) -> tuple[np.ndarray, int, int] | None:
        pcm_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        if pcm_i16.size == 0:
            return None
        pcm = pcm_i16.astype(np.float32) / 32768.0
        spans = self._vad_speech_spans(pcm)
        if not spans:
            return None

        preroll = int(SAMPLE_RATE * self.cfg.k2_preroll_ms / 1000)
        tail = int(SAMPLE_RATE * self.cfg.vad_keep_tail_ms / 1000)
        first_start = min(start for start, _end in spans)
        last_end = max(end for _start, end in spans)
        start = max(0, first_start - preroll)
        end = min(pcm.size, last_end + tail)
        if end <= start:
            return None

        min_samples = int(SAMPLE_RATE * self.cfg.min_segment_duration_ms / 1000)
        voiced_span = sum(max(0, end - start) for start, end in spans)
        if voiced_span < min_samples:
            return None

        out = pcm[start:end].astype(np.float32, copy=True)
        return out, base_sample + start, base_sample + end

    def _vad_speech_spans(self, pcm: np.ndarray) -> list[tuple[int, int]]:
        """Return speech spans from the existing VAD state machine.

        Unlike a raw per-frame threshold, the state machine includes
        pre-speech backfill and trims trailing silence the same way the local
        fallback stream does. Multiple spans inside one k2 endpoint are kept as
        a single bounding range by the caller so natural pauses are preserved.
        """

        vad = VADProcessor()
        vad.apply_config(self.cfg)
        hop = vad.hop_size
        n_full = (pcm.size // hop) * hop
        if n_full <= 0:
            return []

        spans: list[tuple[int, int]] = []
        for off in range(0, n_full, hop):
            frame = pcm[off : off + hop]
            segment = vad.process(frame)
            if segment is not None:
                end = off + hop
                start = max(0, end - len(segment))
                spans.append((start, end))

        tail = vad.flush()
        if tail is not None and len(tail) > 0:
            end = n_full
            start = max(0, end - len(tail))
            spans.append((start, end))
        return spans
