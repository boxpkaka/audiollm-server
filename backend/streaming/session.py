"""Generic WebSocket session that wires an AudioStream to a TaskEngine.

The session owns:

- WebSocket lifecycle (ready, receive loop, error/close)
- Parsing of common control messages
  (start/stop/update_hotwords/extract_hotwords)
- Per-session config override (Config.override)
- Dispatching ``SegmentReady`` events serially through a work queue
- Throttled, non-overlapping dispatch of ``PartialSnapshot`` events

It does NOT know what "ASR" or "emotion" means; that lives in the engine.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from fastapi import WebSocket, WebSocketDisconnect

from ..asr.enrollment import get_enrollment_store
from ..asr.hotword import query_text_hotwords, sanitize_hotwords
from ..config import Config, load_config
from .audio_stream import AudioStream
from .events import PartialSnapshot, SegmentReady, SpeechDropped, SpeechStarted
from .protocol import ControlAction, NativeProtocol, PcmAction, WireProtocol

if TYPE_CHECKING:
    from ..tasks.base import TaskEngine

logger = logging.getLogger(__name__)

LANG_CODE_MAP: dict[str, str] = {
    "zh": "Chinese",
    "cn": "Chinese",
    "en": "English",
    "id": "Indonesian",
    "th": "Thai",
}


def map_language(lang_query: str) -> str:
    """Map a language code or full name to the canonical model-side string."""
    code = (lang_query or "").strip().lower()
    if not code:
        return "N/A"
    if code in LANG_CODE_MAP:
        return LANG_CODE_MAP[code]
    for full_name in ("Chinese", "English", "Indonesian", "Thai"):
        if code == full_name.lower():
            return full_name
    return "N/A"


@dataclass
class SessionContext:
    """Snapshot of common session state passed to engine callbacks.

    The session passes a *frozen* snapshot to per-segment / per-partial calls
    so concurrent updates (e.g. ``update_hotwords``) don't race with in-flight
    inference.
    """

    cfg: Config
    language: str = ""
    src_lang: str = "N/A"
    hotwords: list[str] = field(default_factory=list)
    # Optional cached target-speaker enrollment (base64 WAV). The session
    # resolves the opaque ``enrollment_id`` once at start / on every
    # ``update_hotwords`` and stores the WAV inline so per-segment
    # inference doesn't re-hit the in-memory store on every call.
    enrollment_id: str | None = None
    enrollment_b64: str | None = None
    send_json: Callable[[dict[str, Any]], Awaitable[bool]] = None  # type: ignore[assignment]

    def snapshot(self) -> "SessionContext":
        return replace(self, hotwords=list(self.hotwords))


_SENTINEL = object()


class StreamingSession:
    """Run one client connection by composing an AudioStream and a TaskEngine."""

    def __init__(
        self,
        websocket: WebSocket,
        *,
        stream: AudioStream,
        engine: "TaskEngine",
        language: str = "",
        protocol: WireProtocol | None = None,
    ) -> None:
        self.ws = websocket
        self.stream = stream
        self.engine = engine
        # The wire protocol owns on-the-wire framing; the rest of the session
        # only ever deals in control dicts + PCM bytes. NativeProtocol is the
        # historical 1:1 framing so existing endpoints are byte-for-byte
        # unchanged when no protocol is supplied.
        self.protocol: WireProtocol = protocol or NativeProtocol()

        self.cfg: Config = load_config()
        self.stream.configure(self.cfg)

        self.ctx = SessionContext(
            cfg=self.cfg,
            language=language,
            src_lang=map_language(language),
            hotwords=[],
            send_json=self._send_json,
        )

        self._work_queue: asyncio.Queue = asyncio.Queue(maxsize=40)
        self._partial_task: asyncio.Task | None = None
        # Long-text hotword extraction runs out-of-band so the receive
        # loop never blocks on an LLM round-trip; outstanding tasks
        # are tracked here and cancelled in cleanup.
        self._extract_tasks: set[asyncio.Task] = set()

        self._started = False
        self._stopped = False
        self._sent_any_response = False
        self._ws_closed = False

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        sent_ready = await self._send_json({"type": "ready"})
        if sent_ready:
            logger.info(
                "%s ready (language=%s)",
                self.engine.name,
                self.ctx.language,
            )
        try:
            await asyncio.gather(self._receive_loop(), self._work_loop())
        except Exception:
            logger.exception("StreamingSession[%s] error", self.engine.name)
        finally:
            # Protocols that frame an explicit end-of-session marker (e.g. AST
            # v3's header.status=2) emit it here, after every queued segment
            # has drained. Native framing returns None and sends nothing.
            terminal = self.protocol.encode_terminal()
            if terminal is not None:
                await self._send_wire(terminal)

    async def cleanup(self) -> None:
        if self._partial_task and not self._partial_task.done():
            self._partial_task.cancel()
        if self._extract_tasks:
            for task in self._extract_tasks:
                task.cancel()
            await asyncio.gather(*self._extract_tasks, return_exceptions=True)
        logger.info("StreamingSession[%s] ended", self.engine.name)

    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------

    async def _send_wire(self, wire: dict[str, Any]) -> bool:
        if self._ws_closed:
            return False
        try:
            await self.ws.send_json(wire)
            return True
        except (WebSocketDisconnect, RuntimeError):
            self._ws_closed = True
            return False

    async def _send_json(self, payload: dict[str, Any]) -> bool:
        """Encode an internal message via the active protocol and send it.

        A protocol may suppress a message (return ``None``) when it has no
        wire representation (e.g. AST v3 has no ``ready``); that is reported
        as success so callers don't treat the no-op as a send failure.
        """
        wire = self.protocol.encode_outbound(payload)
        if wire is None:
            return not self._ws_closed
        return await self._send_wire(wire)

    # ------------------------------------------------------------------
    # Receive loop: control messages + binary PCM
    # ------------------------------------------------------------------

    async def _receive_loop(self) -> None:
        try:
            while True:
                msg = await self.ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break

                should_stop = False
                for action in self.protocol.decode_inbound(msg):
                    if isinstance(action, ControlAction):
                        if await self._handle_control(action.ctrl):
                            should_stop = True
                            break
                    elif isinstance(action, PcmAction):
                        # PCM is only meaningful between start and stop; frames
                        # outside that window (or before the protocol has
                        # synthesized a start) are silently dropped.
                        if not self._started or self._stopped:
                            continue
                        await self._handle_pcm(action.data)
                if should_stop:
                    break
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected (%s)", self.engine.name)
        finally:
            # Always flush remaining audio so engine sees the tail.
            for ev in self.stream.flush(force=True):
                await self._dispatch_stream_event(ev)
            await self._work_queue.put(_SENTINEL)

    async def _handle_control(self, ctrl: dict) -> bool:
        """Dispatch one already-parsed control message; return True to stop."""
        msg_type = ctrl.get("type", "")
        if msg_type == "start":
            await self._handle_start(ctrl)
            return False
        if msg_type == "stop":
            await self._handle_stop()
            return True
        if msg_type == "update_hotwords":
            self._handle_update_hotwords(ctrl)
            return False
        if msg_type == "extract_hotwords":
            self._handle_extract_hotwords(ctrl)
            return False
        # Delegate unknown control messages to engine (returns truthy if handled).
        try:
            await self.engine.on_control(ctrl, self.ctx)
        except Exception:
            logger.exception("engine.on_control failed for %s", msg_type)
        return False

    async def _handle_start(self, ctrl: dict) -> None:
        if self._started:
            logger.warning("Duplicate start message, ignoring")
            return
        self._started = True

        client_config = ctrl.get("config")
        if isinstance(client_config, dict) and client_config:
            self.cfg = self.cfg.override(**client_config)
            self.ctx.cfg = self.cfg
            self.stream.configure(self.cfg)
            logger.info("Config overridden by client: %s", list(client_config.keys()))

        lang_val = str(ctrl.get("language", "")).strip()
        if lang_val:
            self.ctx.language = lang_val
            self.ctx.src_lang = map_language(lang_val)

        hw_raw = ctrl.get("hotwords")
        if isinstance(hw_raw, list):
            self.ctx.hotwords = sanitize_hotwords(hw_raw)
            logger.info("Hotwords from start: %d items", len(self.ctx.hotwords))

        if "enrollment_id" in ctrl:
            self._apply_enrollment(ctrl.get("enrollment_id"))

        fmt = ctrl.get("format", "pcm_s16le")
        sr = ctrl.get("sample_rate_hz", 16000)
        ch = ctrl.get("channels", 1)
        logger.info(
            "Start[%s] mode=%s format=%s sr=%s ch=%s language=%s",
            self.engine.name, ctrl.get("mode"), fmt, sr, ch, self.ctx.language,
        )

        try:
            await self.engine.on_start(ctrl, self.ctx)
        except Exception:
            logger.exception("engine.on_start failed")

    def _handle_update_hotwords(self, ctrl: dict) -> None:
        self.ctx.hotwords = sanitize_hotwords(ctrl.get("hotwords", []))
        if "src_lang" in ctrl:
            lang_val = str(ctrl.get("src_lang", "")).strip()
            if lang_val:
                self.ctx.language = lang_val
                self.ctx.src_lang = map_language(lang_val)
        if "enrollment_id" in ctrl:
            self._apply_enrollment(ctrl.get("enrollment_id"))
        logger.info(
            "Hotwords updated: %s (src_lang=%s, enrollment=%s)",
            self.ctx.hotwords, self.ctx.src_lang, self.ctx.enrollment_id,
        )

    def _apply_enrollment(self, raw_id: object) -> None:
        """Resolve an enrollment id (or ``None``/empty to clear) into a
        cached WAV. Unknown / expired ids are treated as "no enrollment"
        so a stale id from a long-lived tab degrades to plain ASR
        instead of breaking the WS session."""
        if raw_id is None or not isinstance(raw_id, str) or not raw_id.strip():
            self.ctx.enrollment_id = None
            self.ctx.enrollment_b64 = None
            return
        ident = raw_id.strip()
        entry = get_enrollment_store().get(ident)
        if entry is None:
            logger.warning("Enrollment id %s not found / expired", ident)
            self.ctx.enrollment_id = None
            self.ctx.enrollment_b64 = None
            return
        self.ctx.enrollment_id = ident
        self.ctx.enrollment_b64 = entry.wav_base64

    def _handle_extract_hotwords(self, ctrl: dict) -> None:
        """Schedule a long-text hotword extraction in the background.

        The receive loop returns immediately so further audio frames /
        control messages are not blocked by the LLM round-trip; the
        eventual ``extract_hotwords_result`` (or ``..._error``) is
        sent through the same WebSocket from the spawned task.
        """
        request_id = str(ctrl.get("request_id", "")).strip()
        source_text = str(ctrl.get("text", ""))
        task = asyncio.create_task(
            self._run_extract_hotwords(request_id, source_text)
        )
        self._extract_tasks.add(task)
        task.add_done_callback(self._extract_tasks.discard)

    async def _run_extract_hotwords(
        self, request_id: str, source_text: str
    ) -> None:
        try:
            extracted = await query_text_hotwords(source_text)
            await self._send_json(
                {
                    "type": "extract_hotwords_result",
                    "request_id": request_id,
                    "hotwords": extracted,
                }
            )
        except WebSocketDisconnect:
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "extract_hotwords failed (request_id=%s)", request_id or "n/a"
            )
            await self._send_json(
                {
                    "type": "extract_hotwords_error",
                    "request_id": request_id,
                    "message": str(exc),
                }
            )

    async def _handle_stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        logger.info("Stop received (%s), flushing", self.engine.name)
        for ev in self.stream.flush(force=True):
            await self._dispatch_stream_event(ev)

    # ------------------------------------------------------------------
    # PCM dispatch
    # ------------------------------------------------------------------

    async def _handle_pcm(self, pcm_bytes: bytes) -> None:
        for ev in self.stream.feed(pcm_bytes):
            await self._dispatch_stream_event(ev)

    async def _dispatch_stream_event(self, ev) -> None:
        # Heavy work (full-segment inference) goes through the queue so
        # it stays serialized; lightweight notifications (speech start /
        # dropped, partial snapshot) fan out directly without queuing
        # so the placeholder UI shows up before the segment finishes.
        if isinstance(ev, SegmentReady):
            self._enqueue_segment(ev)
        elif isinstance(ev, PartialSnapshot):
            self._maybe_launch_partial(ev)
        elif isinstance(ev, SpeechStarted):
            await self._safe_speech_start()
        elif isinstance(ev, SpeechDropped):
            await self._safe_speech_dropped()

    def _enqueue_segment(self, ev: SegmentReady) -> None:
        snapshot = self.ctx.snapshot()
        try:
            self._work_queue.put_nowait((ev, snapshot))
        except asyncio.QueueFull:
            logger.warning("Work queue full, dropping segment")

    def _maybe_launch_partial(self, snap: PartialSnapshot) -> None:
        if self._partial_task is not None and not self._partial_task.done():
            return
        snapshot_ctx = self.ctx.snapshot()
        self._partial_task = asyncio.create_task(self._safe_partial(snap, snapshot_ctx))

    async def _safe_partial(self, snap: PartialSnapshot, ctx: SessionContext) -> None:
        try:
            await self.engine.handle_partial(snap, ctx)
        except WebSocketDisconnect:
            self._ws_closed = True
        except Exception:
            logger.debug("engine.handle_partial failed", exc_info=True)

    async def _safe_speech_start(self) -> None:
        try:
            await self.engine.handle_speech_start(self.ctx.snapshot())
        except WebSocketDisconnect:
            self._ws_closed = True
        except Exception:
            logger.debug("engine.handle_speech_start failed", exc_info=True)

    async def _safe_speech_dropped(self) -> None:
        try:
            await self.engine.handle_speech_dropped(self.ctx.snapshot())
        except WebSocketDisconnect:
            self._ws_closed = True
        except Exception:
            logger.debug("engine.handle_speech_dropped failed", exc_info=True)

    # ------------------------------------------------------------------
    # Work loop: drain final segments serially
    # ------------------------------------------------------------------

    async def _work_loop(self) -> None:
        while True:
            item = await self._work_queue.get()
            if item is _SENTINEL:
                break
            seg, ctx = item
            try:
                sent = await self.engine.handle_segment(seg, ctx)
                if sent:
                    self._sent_any_response = True
            except WebSocketDisconnect:
                self._ws_closed = True
                break
            except Exception as e:
                logger.exception("engine.handle_segment failed")
                if not await self._send_json(
                    {"type": "error", "message": str(e)}
                ):
                    break

        try:
            await self.engine.on_stop(
                self.ctx.snapshot(),
                sent_any_response=self._sent_any_response,
                stopped=self._stopped,
            )
        except Exception:
            logger.exception("engine.on_stop failed")
