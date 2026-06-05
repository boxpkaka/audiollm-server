import asyncio
import json
import logging
import time

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from .asr.client import query_audio_model, query_audio_model_secondary
from .asr.enrollment import get_enrollment_store
from .asr.fusion import choose_fused_result
from .asr.hotword import query_text_hotwords, sanitize_hotwords
from .asr.itn import normalize_final_text
from .audio.utils import Resampler48to16, pcm_to_wav_base64
from .audio.vad import VADProcessor
from .config import SAMPLE_RATE, default_config
from .emotion_spec.client import query_emotion_spec_model

logger = logging.getLogger(__name__)

MIN_SEGMENT_SAMPLES = int(SAMPLE_RATE * default_config.min_segment_duration_ms / 1000)

VALID_SRC_LANG = frozenset({"N/A", "Chinese", "English", "Indonesian", "Thai"})


def normalize_client_src_lang(value: object) -> str:
    s = str(value or "").strip()
    if not s:
        return "N/A"
    if s in VALID_SRC_LANG:
        return s
    logger.warning("Unknown src_lang %r, using N/A", s)
    return "N/A"


def _generate_segment_id() -> str:
    return f"seg-{int(time.time() * 1000)}"


class AudioSession:
    """Manages one WebSocket session: VAD ingestion + ASR pipeline."""

    def __init__(self, websocket: WebSocket) -> None:
        self.ws = websocket
        self.segment_queue: asyncio.Queue[tuple | None] = asyncio.Queue(maxsize=20)
        self.vad = VADProcessor()
        self.resampler = Resampler48to16()
        self._pcm_carry = np.empty(0, dtype=np.float32)
        self.hotwords: list[str] = []
        self.src_lang: str = "N/A"
        self.enable_emotion: bool = False
        # Optional target-speaker enrollment. ``enrollment_id`` is the
        # opaque handle returned by ``POST /api/asr/enrollment``;
        # ``enrollment_b64`` caches the resolved WAV so we don't take a
        # store hit on every segment. Both reset to ``None`` when the
        # client clears the enrollment via ``update_hotwords``.
        self.enrollment_id: str | None = None
        self.enrollment_b64: str | None = None
        self.stop_event = asyncio.Event()
        self.extract_tasks: set[asyncio.Task] = set()
        self._ws_closed = False

        # Pseudo-streaming state
        self._utterance_id: str | None = None
        self._partial_seq: int = 0
        self._last_partial_time: float = 0.0
        self._partial_task: asyncio.Task | None = None
        self._pseudo_stream_interval: float = default_config.pseudo_stream_interval_ms / 1000.0

    async def _send_json(self, data: dict) -> bool:
        """Send JSON over WebSocket. Returns False if the connection is gone."""
        if self._ws_closed:
            return False
        try:
            await self.ws.send_json(data)
            return True
        except (WebSocketDisconnect, RuntimeError):
            self._ws_closed = True
            self.stop_event.set()
            return False

    async def run(self) -> None:
        try:
            await asyncio.gather(self._vad_loop(), self._asr_loop())
        except Exception:
            logger.exception("Session error")

    async def cleanup(self) -> None:
        if self._partial_task and not self._partial_task.done():
            self._partial_task.cancel()
        if self.extract_tasks:
            for task in self.extract_tasks:
                task.cancel()
            await asyncio.gather(*self.extract_tasks, return_exceptions=True)
        logger.info("Session ended")

    # ------------------------------------------------------------------
    # VAD loop: receive audio frames + control messages
    # ------------------------------------------------------------------

    async def _vad_loop(self) -> None:
        try:
            while not self.stop_event.is_set():
                msg = await self.ws.receive()

                if msg.get("type") == "websocket.disconnect":
                    break

                if "bytes" in msg and msg["bytes"]:
                    self._ingest_audio(msg["bytes"])
                elif "text" in msg and msg["text"]:
                    self._handle_control_message(msg["text"])

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected (vad_loop)")
        finally:
            remaining = self.vad.flush()
            if remaining is not None and len(remaining) >= MIN_SEGMENT_SAMPLES:
                seg_id = self._utterance_id or _generate_segment_id()
                self._utterance_id = None
                await self.segment_queue.put(
                    (
                        seg_id,
                        remaining,
                        list(self.hotwords),
                        self.src_lang,
                    )
                )
            self.stop_event.set()
            await self.segment_queue.put(None)

    def _ingest_audio(self, raw_bytes: bytes) -> None:
        pcm_48k = (
            np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        )
        pcm = self.resampler.process(pcm_48k)
        if pcm.size == 0:
            return
        if self._pcm_carry.size > 0:
            pcm = np.concatenate([self._pcm_carry, pcm])
        hop = self.vad.hop_size
        used = (len(pcm) // hop) * hop
        self._pcm_carry = pcm[used:].copy() if used < len(pcm) else np.empty(0, dtype=np.float32)
        for i in range(0, used, hop):
            segment = self.vad.process(pcm[i : i + hop])
            if segment is not None:
                self._enqueue_segment(segment)

        # Pseudo-streaming partials live on their own switch. We just need
        # at least one decoder online — `_emit_partial` picks primary text
        # when available and falls back to secondary, so disabling either
        # model individually keeps the live caption working.
        if not default_config.enable_pseudo_stream:
            return
        if not default_config.enable_primary_asr and not default_config.enable_secondary_asr:
            return

        if self.vad.is_speaking:
            if self._utterance_id is None:
                self._utterance_id = _generate_segment_id()
                self._partial_seq = 0
                self._last_partial_time = 0.0
                logger.debug("Utterance started: %s", self._utterance_id)

            now = time.monotonic()
            if now - self._last_partial_time >= self._pseudo_stream_interval:
                snapshot = self.vad.snapshot_incomplete_speech()
                if snapshot is not None and len(snapshot) >= MIN_SEGMENT_SAMPLES:
                    if self._partial_task is None or self._partial_task.done():
                        self._last_partial_time = now
                        self._partial_seq += 1
                        self._partial_task = asyncio.create_task(
                            self._emit_partial(
                                self._utterance_id,
                                snapshot,
                                self._partial_seq,
                            )
                        )
        else:
            if self._utterance_id is not None and not self.vad.is_speaking:
                self._utterance_id = None

    def _enqueue_segment(self, segment: np.ndarray) -> None:
        if len(segment) < MIN_SEGMENT_SAMPLES:
            logger.info(
                "Drop short segment (%.1fs < %.1fs)",
                len(segment) / SAMPLE_RATE,
                default_config.min_segment_duration_ms / 1000.0,
            )
            self._utterance_id = None
            return
        seg_id = self._utterance_id or _generate_segment_id()
        self._utterance_id = None
        try:
            self.segment_queue.put_nowait(
                (seg_id, segment, list(self.hotwords), self.src_lang)
            )
        except asyncio.QueueFull:
            logger.warning("Segment queue full, dropping %s", seg_id)

    def _handle_control_message(self, text: str) -> None:
        try:
            ctrl = json.loads(text)
        except json.JSONDecodeError:
            return

        if ctrl.get("type") == "update_hotwords":
            self.hotwords = sanitize_hotwords(ctrl.get("hotwords", []))
            if "src_lang" in ctrl:
                self.src_lang = normalize_client_src_lang(ctrl.get("src_lang"))
            if "enable_emotion" in ctrl:
                self.enable_emotion = bool(ctrl.get("enable_emotion"))
            if "enrollment_id" in ctrl:
                self._apply_enrollment(ctrl.get("enrollment_id"))
            logger.info(
                "Hotwords updated: %s (src_lang=%s, emotion=%s, enrollment=%s)",
                self.hotwords,
                self.src_lang,
                self.enable_emotion,
                self.enrollment_id,
            )

        elif ctrl.get("type") == "update_emotion":
            self.enable_emotion = bool(ctrl.get("enabled"))
            logger.info("Emotion recognition toggled: %s", self.enable_emotion)

        elif ctrl.get("type") == "set_enrollment":
            self._apply_enrollment(ctrl.get("enrollment_id"))
            logger.info("Enrollment set to %s", self.enrollment_id)

        elif ctrl.get("type") == "extract_hotwords":
            request_id = str(ctrl.get("request_id", "")).strip()
            source_text = str(ctrl.get("text", ""))
            task = asyncio.create_task(
                self._extract_hotwords(request_id, source_text)
            )
            self.extract_tasks.add(task)
            task.add_done_callback(self.extract_tasks.discard)

        elif ctrl.get("type") == "flush":
            # Client manually stopped the mic. Drain VAD's pending audio
            # so the trailing in-progress utterance gets transcribed
            # promptly instead of waiting for the WebSocket to actually
            # disconnect (the long-lived WS in the realtime ASR page
            # stays open across recording sessions, so without an
            # explicit flush the tail just sits in VAD's buffer until
            # the next disconnect or the next utterance overwrites it).
            self._flush_pending_audio()

    def _apply_enrollment(self, raw_id: object) -> None:
        """Resolve an enrollment id (or ``None`` to clear) into the cached WAV.

        Unknown / expired ids fall back to "no enrollment" rather than
        erroring out the WS session — they get reported via the regular
        log path; the client can re-upload if needed.
        """
        if raw_id is None:
            self.enrollment_id = None
            self.enrollment_b64 = None
            return
        if not isinstance(raw_id, str) or not raw_id.strip():
            self.enrollment_id = None
            self.enrollment_b64 = None
            return
        ident = raw_id.strip()
        entry = get_enrollment_store().get(ident)
        if entry is None:
            logger.warning("Enrollment id %s not found / expired", ident)
            self.enrollment_id = None
            self.enrollment_b64 = None
            return
        self.enrollment_id = ident
        self.enrollment_b64 = entry.wav_base64

    def _flush_pending_audio(self) -> None:
        """Drain VAD's pending speech and enqueue it as a final segment.

        Mirrors the ``finally`` branch of ``_vad_loop`` so a manual stop
        from the client (``{"type": "flush"}``) produces the same
        ``vad_event`` + ``response`` pair as a natural VAD-detected end.
        Returning quietly when there's nothing buffered (or the residual
        is shorter than the minimum segment length) is by design — silent
        no-ops keep the protocol idempotent for spurious flush messages.
        """
        was_speaking = self.vad.is_speaking
        remaining = self.vad.flush()
        if remaining is None:
            logger.info(
                "Flush: nothing to drain (was_speaking=%s, utterance=%s)",
                was_speaking,
                self._utterance_id,
            )
            self._utterance_id = None
            return
        if len(remaining) < MIN_SEGMENT_SAMPLES:
            logger.info(
                "Flush: segment too short (%.2fs < %.2fs)",
                len(remaining) / SAMPLE_RATE,
                default_config.min_segment_duration_ms / 1000.0,
            )
            self._utterance_id = None
            return
        seg_id = self._utterance_id or _generate_segment_id()
        self._utterance_id = None
        logger.info(
            "Flush: enqueueing segment %s (%.2fs)",
            seg_id,
            len(remaining) / SAMPLE_RATE,
        )
        try:
            self.segment_queue.put_nowait(
                (seg_id, remaining, list(self.hotwords), self.src_lang)
            )
        except asyncio.QueueFull:
            logger.warning("Segment queue full, dropping flush %s", seg_id)

    async def _extract_hotwords(self, request_id: str, source_text: str) -> None:
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
        except Exception as e:
            logger.exception(
                "extract_hotwords failed (request_id=%s)", request_id or "n/a"
            )
            await self._send_json(
                {
                    "type": "extract_hotwords_error",
                    "request_id": request_id,
                    "message": str(e),
                }
            )

    # ------------------------------------------------------------------
    # Pseudo-streaming: partial secondary ASR while user is speaking
    # ------------------------------------------------------------------

    async def _emit_partial(
        self, utterance_id: str, snapshot: np.ndarray, seq: int
    ) -> None:
        """Emit a rolling partial using whichever decoder(s) are online.

        - Both on  : run in parallel; secondary acts as a noise gate
                     (primary tends to hallucinate on near-silence), and
                     the partial text comes from primary so hotwords /
                     language / target-speaker prompts apply.
        - Primary only : no noise gate; better to leak an occasional
                         hallucinated partial than to have the live
                         caption go completely silent.
        - Secondary only : use the secondary text directly. It is
                           weaker (no hotwords / TS), but still useful
                           as a live caption when primary is offline.
        """
        try:
            wav_b64 = pcm_to_wav_base64(snapshot)

            primary_task = None
            secondary_task = None

            if default_config.enable_primary_asr:
                primary_task = asyncio.create_task(
                    query_audio_model(
                        wav_b64,
                        hotwords=list(self.hotwords),
                        src_lang=self.src_lang,
                        enrollment_wav_base64=self.enrollment_b64,
                    )
                )
            if default_config.enable_secondary_asr:
                secondary_task = asyncio.create_task(
                    query_audio_model_secondary(wav_b64)
                )

            secondary_text = ""
            if secondary_task is not None:
                secondary_res = await secondary_task
                secondary_text = str(
                    (secondary_res or {}).get("transcription") or ""
                ).strip()
                # When secondary is the gate (primary present) or the
                # sole source, an empty result means "silence" — skip.
                if not secondary_text:
                    if primary_task is not None:
                        primary_task.cancel()
                    return

            if primary_task is not None:
                primary_res = await primary_task
                text = str(
                    (primary_res or {}).get("transcription") or ""
                ).strip()
            else:
                text = secondary_text

            if not text:
                return

            # Utterance lifecycle gate: _enqueue_segment zeroes out
            # _utterance_id when the final segment goes into the ASR
            # queue (and a fresh speech burst would set it to a *new*
            # id). Either way, an inflight partial whose utterance no
            # longer matches is racing the final response — letting it
            # through flips the bubble back to "streaming" on the client
            # and silently overwrites the finalized text. This is the
            # only point where we can serialize partial/final because the
            # async task creation-to-completion interval is not lockable.
            if self._utterance_id != utterance_id:
                logger.debug(
                    "Drop stale partial for %s seq=%s (utterance finalized)",
                    utterance_id,
                    seq,
                )
                return

            await self._send_json(
                {
                    "type": "partial_transcript",
                    "utterance_id": utterance_id,
                    "text": text,
                    "seq": seq,
                }
            )
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.debug("Partial ASR failed for %s seq=%s", utterance_id, seq)

    # ------------------------------------------------------------------
    # ASR loop: consume segments, query models, send results
    # ------------------------------------------------------------------

    async def _asr_loop(self) -> None:
        while True:
            item = await self.segment_queue.get()
            if item is None:
                break

            seg_id, segment, hw_snapshot, lang_snapshot = item
            logger.info(
                "Processing segment %s (%.1fs, hotwords=%s, src_lang=%s)",
                seg_id,
                len(segment) / SAMPLE_RATE,
                hw_snapshot,
                lang_snapshot,
            )

            try:
                await self._process_segment(
                    seg_id, segment, hw_snapshot, lang_snapshot
                )
            except (WebSocketDisconnect, RuntimeError):
                break
            except Exception as e:
                logger.exception("LLM query failed for %s", seg_id)
                await self._send_vad_event(seg_id, segment)
                if not await self._send_json(
                    {"type": "error", "id": seg_id, "message": str(e)}
                ):
                    break

    async def _process_segment(
        self,
        seg_id: str,
        segment: np.ndarray,
        hw_snapshot: list[str],
        lang_snapshot: str,
    ) -> None:
        wav_b64 = pcm_to_wav_base64(segment)
        primary_res: object = None
        secondary_res: object = None
        # secondary 在 dual fusion 路径下兼做噪声门：如果 secondary 自己
        # 判 silence，`_dual_asr_pipeline` 直接返回 (None, None) 并取消
        # primary。这里把 "silence 信号" 与 "ASR 真失败" 区分开，避免下面
        # 的失败分支把 silence 也当成 error。
        is_silence = False

        # enable_dual_asr_fusion (validated against enable_secondary_asr at
        # config load time) decides whether final segments go through the
        # dual-model pipeline. With fusion off we still emit a final from
        # the primary alone, saving one vLLM call per segment.
        if default_config.enable_dual_asr_fusion:
            secondary_res, primary_res = await self._dual_asr_pipeline(
                seg_id, wav_b64, hw_snapshot, lang_snapshot
            )
            if secondary_res is None and primary_res is None:
                is_silence = True
        else:
            if default_config.enable_primary_asr:
                primary_res = await asyncio.wait_for(
                    query_audio_model(
                        wav_b64,
                        hotwords=hw_snapshot,
                        src_lang=lang_snapshot,
                        enrollment_wav_base64=self.enrollment_b64,
                    ),
                    timeout=default_config.primary_asr_timeout,
                )

        primary_result = None if isinstance(primary_res, Exception) else primary_res
        secondary_result = None if isinstance(secondary_res, Exception) else secondary_res

        if isinstance(primary_res, Exception):
            logger.warning("Primary ASR failed for %s: %s", seg_id, primary_res)
        if isinstance(secondary_res, Exception):
            logger.warning("Secondary ASR failed for %s: %s", seg_id, secondary_res)

        asr_failed = (
            not is_silence
            and primary_result is None
            and secondary_result is None
        )

        if is_silence or asr_failed:
            text = ""
            fused: dict | None = None
        else:
            fused = choose_fused_result(
                primary_result, secondary_result, hotwords=hw_snapshot
            )
            text = str(fused.get("text") or "").strip()
            if text:
                # Final-only display transform (ITN + plate); partials emitted
                # by _emit_partial deliberately stay spoken-form.
                lang = (
                    primary_result.get("detected_language")
                    if primary_result
                    else None
                ) or ""
                text = normalize_final_text(text, lang, default_config)

        # Emotion is a parallel semantic channel: prosody / paralinguistic
        # cues exist independently of whether the ASR head emitted text.
        # Run the SPEC model when the client opted in, even on silence /
        # ASR-failure paths — the engine itself returns None when the
        # signal is genuinely empty, so we don't fabricate output.
        emotion_payload: dict | None = None
        if self.enable_emotion:
            emotion_payload = await self._run_emotion(segment)

        if asr_failed and not emotion_payload:
            # Real upstream failure with no other channel to fall back on:
            # surface as an error to the client (matches pre-decoupling
            # behavior) instead of silently dropping the segment.
            raise RuntimeError("Both ASR models failed for this segment.")
        if asr_failed:
            # ASR genuinely broken but emotion still produced a signal —
            # we degrade to emotion-only instead of erroring, which is
            # nicer UX but hides the ASR fault from the user. Log loud so
            # ops can spot the upstream regression even when clients are
            # happy seeing emotion bubbles.
            logger.warning(
                "ASR failed for %s, degrading to emotion-only response",
                seg_id,
            )

        if not text and not emotion_payload:
            logger.info("Skip empty response for %s (silence)", seg_id)
            return

        await self._send_vad_event(seg_id, segment, wav_b64)

        payload: dict = {
            "type": "response",
            "id": seg_id,
            "text": text,
            "model_hotwords": fused["model_hotwords"] if fused else [],
        }
        if primary_result and primary_result.get("detected_language"):
            payload["src_lang_detected"] = primary_result["detected_language"]
        if default_config.debug_show_dual_asr and fused:
            payload.update(
                {
                    "text_primary": fused["primary_text"],
                    "text_secondary": fused["secondary_text"],
                    "fusion_meta": fused["fusion"],
                }
            )
        if emotion_payload:
            payload["emotion"] = emotion_payload

        await self._send_json(payload)

    async def _run_emotion(self, segment: np.ndarray) -> dict | None:
        """Run SER + SEPC in parallel on the final segment via AmphionSPEC.

        Best-effort: any single inference failure degrades to omitting that
        field rather than failing the whole ASR response.
        """
        audio_duration = len(segment) / SAMPLE_RATE
        clip = segment
        if (
            default_config.emotion_spec_max_audio_seconds > 0
            and audio_duration > default_config.emotion_spec_max_audio_seconds
        ):
            max_samples = int(SAMPLE_RATE * default_config.emotion_spec_max_audio_seconds)
            clip = segment[-max_samples:]

        try:
            wav_b64 = pcm_to_wav_base64(clip)
        except Exception:
            logger.exception("Emotion: failed to encode wav")
            return None

        async def _call(mode: str):
            return await query_emotion_spec_model(
                wav_b64,
                mode=mode,
                base_url=default_config.emotion_spec_vllm_base_url,
                model_name=default_config.emotion_spec_vllm_model_name,
                timeout=default_config.emotion_spec_request_timeout,
            )

        ser_res, sepc_res = await asyncio.gather(
            _call("ser"), _call("sepc"), return_exceptions=True
        )

        if isinstance(ser_res, Exception):
            logger.warning("SER inference failed: %s", ser_res)
            ser_res = None
        if isinstance(sepc_res, Exception):
            logger.warning("SEPC inference failed: %s", sepc_res)
            sepc_res = None

        ser_label = str((ser_res or {}).get("label", "") or "").strip()
        sepc_text = str((sepc_res or {}).get("text", "") or "").strip()
        sepc_label = str((sepc_res or {}).get("label", "") or "").strip()

        if not ser_label and not sepc_text and not sepc_label:
            return None

        return {
            "ser_label": ser_label,
            "sepc_text": sepc_text,
            "sepc_label": sepc_label,
        }

    async def _send_vad_event(
        self,
        seg_id: str,
        segment: np.ndarray,
        wav_b64: str | None = None,
    ) -> None:
        if wav_b64 is None:
            wav_b64 = pcm_to_wav_base64(segment)
        await self._send_json(
            {
                "type": "vad_event",
                "event": "segment_detected",
                "id": seg_id,
                "duration": f"{len(segment) / SAMPLE_RATE:.1f}s",
                "audio_b64": wav_b64,
            }
        )

    async def _dual_asr_pipeline(
        self,
        seg_id: str,
        wav_b64: str,
        hw_snapshot: list[str],
        lang_snapshot: str,
    ) -> tuple:
        """Run both ASR models in parallel, wait for both, return results.

        Returns (secondary_res, primary_res).  Returns (None, None) when the
        segment is silence.
        """
        secondary_task = asyncio.create_task(
            query_audio_model_secondary(wav_b64, hotwords=hw_snapshot)
        )
        primary_task = None
        if default_config.enable_primary_asr:
            primary_task = asyncio.create_task(
                asyncio.wait_for(
                    query_audio_model(
                        wav_b64,
                        hotwords=hw_snapshot,
                        src_lang=lang_snapshot,
                        enrollment_wav_base64=self.enrollment_b64,
                    ),
                    timeout=default_config.primary_asr_timeout,
                )
            )

        secondary_res = await secondary_task
        primary_res: object = None

        if isinstance(secondary_res, Exception):
            logger.warning("Secondary ASR failed for %s: %s", seg_id, secondary_res)
            secondary_res = None
            if primary_task is not None:
                try:
                    primary_res = await primary_task
                except Exception as err:
                    primary_res = err
            if primary_res is None or isinstance(primary_res, Exception):
                raise RuntimeError("Both ASR models failed for this segment.")
            return secondary_res, primary_res

        secondary_text = str(
            (secondary_res or {}).get("transcription") or ""
        ).strip()

        if not secondary_text:
            logger.info("Skip empty response for %s (secondary silence)", seg_id)
            if primary_task is not None:
                primary_task.cancel()
            return None, None

        if primary_task is not None:
            try:
                primary_res = await primary_task
            except Exception as err:
                primary_res = err

        return secondary_res, primary_res
