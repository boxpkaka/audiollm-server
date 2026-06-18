"""ASR task engine: dual-model inference + fusion + pseudo-streaming partials."""

from __future__ import annotations

import asyncio
import logging
import time

from ..asr.client import query_audio_model, query_audio_model_secondary
from ..asr.fusion import choose_fused_result
from ..asr.itn import normalize_final_text
from ..audio.utils import pcm_to_wav_base64
from ..config import SAMPLE_RATE
from ..streaming.events import PartialSnapshot, SegmentReady
from ..streaming.session import SessionContext
from .base import BaseTaskEngine

logger = logging.getLogger(__name__)


class AsrTaskEngine(BaseTaskEngine):
    """Drives the existing dual-ASR pipeline against the streaming session."""

    name = "asr"

    def __init__(self, *, emit_timing: bool = False) -> None:
        # When True, ``final`` messages carry the segment's session-timeline
        # position as ``bg_ms`` / ``ed_ms``. The native ``/transcribe-streaming``
        # contract is plain ``{type,text,language}``, so this stays off by
        # default and is only enabled for protocols that surface segment timing
        # (AST v3's ``bg`` / ``ed``). The wire protocol consumes these internal
        # fields; they never reach a native-framed client.
        self._emit_timing = emit_timing

    # ------------------------------------------------------------------
    # Final segment -> final_asr / final
    # ------------------------------------------------------------------

    async def handle_segment(
        self, seg: SegmentReady, ctx: SessionContext
    ) -> bool:
        cfg = ctx.cfg
        segment = seg.pcm
        audio_duration = len(segment) / SAMPLE_RATE
        t0 = time.monotonic()
        wav_b64 = pcm_to_wav_base64(segment)
        hw_snapshot = ctx.hotwords

        primary_res: object = None
        secondary_res: object = None

        # Final segments only run the dual pipeline when fusion is on.
        # With fusion off (but secondary still online for partial gating)
        # we save one vLLM call per segment by running primary-only.
        if cfg.enable_dual_asr_fusion:
            secondary_res, primary_res = await self._dual_asr(
                wav_b64, hw_snapshot, ctx
            )
            if secondary_res is None and primary_res is None:
                return False
        elif cfg.enable_primary_asr:
            primary_res = await asyncio.wait_for(
                query_audio_model(
                    wav_b64,
                    hotwords=hw_snapshot,
                    src_lang=ctx.src_lang,
                    enrollment_wav_base64=ctx.enrollment_b64,
                    base_url=cfg.vllm_base_url,
                    model_name=cfg.vllm_model_name,
                    prompt_template=cfg.vllm_prompt_template,
                    timeout=cfg.asr_request_timeout,
                ),
                timeout=cfg.primary_asr_timeout,
            )

        primary_result = (
            None if isinstance(primary_res, Exception) else primary_res
        )
        secondary_result = (
            None if isinstance(secondary_res, Exception) else secondary_res
        )

        if isinstance(primary_res, Exception):
            logger.warning("Primary ASR failed: %s", primary_res)
        if isinstance(secondary_res, Exception):
            logger.warning("Secondary ASR failed: %s", secondary_res)
        if primary_result is None and secondary_result is None:
            raise RuntimeError("Both ASR models failed for this segment.")

        text, detected_lang = self._select_text(
            primary_result, secondary_result, hw_snapshot, ctx
        )
        # ITN + plate normalization is a final-only display transform; partials
        # stay spoken-form (see handle_partial).
        if text:
            text = normalize_final_text(text, detected_lang, cfg)

        elapsed = time.monotonic() - t0
        rtf = elapsed / audio_duration if audio_duration > 0 else 0.0
        logger.info(
            "Final ASR: audio=%.2fs infer=%.3fs RTF=%.3f text=%r",
            audio_duration, elapsed, rtf, text[:80],
        )

        if not text:
            return False

        payload: dict = {
            "type": "final",
            "text": text,
            "language": detected_lang,
        }
        if self._emit_timing:
            if seg.start_ms is not None:
                payload["bg_ms"] = seg.start_ms
            if seg.end_ms is not None:
                payload["ed_ms"] = seg.end_ms
        return await ctx.send_json(payload)

    # ------------------------------------------------------------------
    # Pseudo-streaming partial
    # ------------------------------------------------------------------

    async def handle_partial(
        self, snap: PartialSnapshot, ctx: SessionContext
    ) -> None:
        cfg = ctx.cfg
        if not cfg.enable_pseudo_stream:
            return
        if not (cfg.enable_primary_asr or cfg.enable_secondary_asr):
            return

        snapshot = snap.pcm
        audio_duration = len(snapshot) / SAMPLE_RATE
        t0 = time.monotonic()
        wav_b64 = pcm_to_wav_base64(snapshot)
        hw_snapshot = ctx.hotwords

        primary_res: object = None
        secondary_res: object = None

        if cfg.enable_secondary_asr and cfg.enable_primary_asr:
            secondary_res, primary_res = await self._dual_asr(
                wav_b64, hw_snapshot, ctx
            )
            if secondary_res is None and primary_res is None:
                return
        elif cfg.enable_primary_asr:
            primary_res = await asyncio.wait_for(
                query_audio_model(
                    wav_b64,
                    hotwords=hw_snapshot,
                    src_lang=ctx.src_lang,
                    enrollment_wav_base64=ctx.enrollment_b64,
                    base_url=cfg.vllm_base_url,
                    model_name=cfg.vllm_model_name,
                    prompt_template=cfg.vllm_prompt_template,
                    timeout=cfg.asr_request_timeout,
                ),
                timeout=cfg.primary_asr_timeout,
            )
        elif cfg.enable_secondary_asr:
            secondary_res = await query_audio_model_secondary(
                wav_b64,
                hotwords=hw_snapshot,
                base_url=cfg.secondary_vllm_base_url,
                model_name=cfg.secondary_vllm_model_name,
                timeout=cfg.asr_request_timeout,
            )

        primary_result = (
            None if isinstance(primary_res, Exception) else primary_res
        )
        secondary_result = (
            None if isinstance(secondary_res, Exception) else secondary_res
        )

        if primary_result is None and secondary_result is None:
            return

        # Noise gate: if secondary is enabled and produced empty text, skip.
        if cfg.enable_secondary_asr:
            sec_text = str(
                (secondary_result or {}).get("transcription") or ""
            ).strip()
            if not sec_text:
                logger.debug("Partial suppressed: secondary empty (noise gate)")
                return

        text, _ = self._select_text(
            primary_result, secondary_result, hw_snapshot, ctx
        )

        elapsed = time.monotonic() - t0
        rtf = elapsed / audio_duration if audio_duration > 0 else 0.0
        logger.info(
            "Partial ASR: audio=%.2fs infer=%.3fs RTF=%.3f text=%r",
            audio_duration, elapsed, rtf, text[:80],
        )

        if not text:
            return

        await ctx.send_json(
            {
                "type": "partial",
                "text": text,
                "language": ctx.language,
            }
        )

    # ------------------------------------------------------------------
    # Stop guarantee: always emit a final after stop (possibly empty).
    # ------------------------------------------------------------------

    async def on_stop(
        self,
        ctx: SessionContext,
        *,
        sent_any_response: bool,
        stopped: bool,
    ) -> None:
        # Match the legacy behavior: only emit empty final after explicit stop
        # (not after raw socket close) when nothing was sent in this drain.
        if stopped and not sent_any_response:
            await ctx.send_json(
                {
                    "type": "final",
                    "text": "",
                    "language": ctx.language,
                }
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _dual_asr(
        self,
        wav_b64: str,
        hw_snapshot: list[str],
        ctx: SessionContext,
    ) -> tuple:
        cfg = ctx.cfg
        secondary_task = asyncio.create_task(
            query_audio_model_secondary(
                wav_b64,
                hotwords=hw_snapshot,
                base_url=cfg.secondary_vllm_base_url,
                model_name=cfg.secondary_vllm_model_name,
                timeout=cfg.asr_request_timeout,
            )
        )
        primary_task = None
        if cfg.enable_primary_asr:
            primary_task = asyncio.create_task(
                asyncio.wait_for(
                    query_audio_model(
                        wav_b64,
                        hotwords=hw_snapshot,
                        src_lang=ctx.src_lang,
                        enrollment_wav_base64=ctx.enrollment_b64,
                        base_url=cfg.vllm_base_url,
                        model_name=cfg.vllm_model_name,
                        prompt_template=cfg.vllm_prompt_template,
                        timeout=cfg.asr_request_timeout,
                    ),
                    timeout=cfg.primary_asr_timeout,
                )
            )

        secondary_res = await secondary_task
        primary_res: object = None

        if isinstance(secondary_res, Exception):
            logger.warning("Secondary ASR failed: %s", secondary_res)
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
            if primary_task is not None:
                primary_task.cancel()
            return None, None

        if primary_task is not None:
            try:
                primary_res = await primary_task
            except Exception as err:
                primary_res = err

        return secondary_res, primary_res

    def _select_text(
        self,
        primary_result,
        secondary_result,
        hw_snapshot: list[str],
        ctx: SessionContext,
    ) -> tuple[str, str]:
        cfg = ctx.cfg
        detected_lang = ctx.language

        if primary_result and not secondary_result:
            text = str(primary_result.get("transcription") or "").strip()
            detected_lang = (
                primary_result.get("detected_language") or ctx.language
            )
        elif secondary_result and not primary_result:
            text = str(secondary_result.get("transcription") or "").strip()
        else:
            fused = choose_fused_result(
                primary_result,
                secondary_result,
                hotwords=hw_snapshot,
                similarity_threshold=cfg.fusion_similarity_threshold,
                min_primary_score=cfg.fusion_min_primary_score,
                max_repetition_ratio=cfg.fusion_max_repetition_ratio,
                disagreement_threshold=cfg.fusion_disagreement_threshold,
                hotword_boost=cfg.fusion_hotword_boost,
                primary_score_margin=cfg.fusion_primary_score_margin,
            )
            text = str(fused.get("text") or "").strip()
            if primary_result and primary_result.get("detected_language"):
                detected_lang = primary_result["detected_language"]

        return text, detected_lang
