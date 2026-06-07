import asyncio
import base64
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import websockets
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.types import Receive, Scope, Send

from .asr.client import query_audio_model, query_audio_model_secondary
from .asr.enrollment import (
    EnrollmentError,
    decode_and_validate,
    get_enrollment_store,
)
from .asr.fusion import choose_fused_result
from .asr.itn import normalize_final_text
from .audio.utils import wav_base64_to_pcm_16k_mono
from .config import SAMPLE_RATE, load_config
from .emotion.client import query_emotion_model
from .emotion.jobs import JobQueueFullError, get_emotion_job_store
from .emotion.service import EmotionDecodeError, decode_wav_capped
from .emotion_spec.jobs import get_emotion_spec_job_store
from .http_client import close_client
from .session import AudioSession
from .streaming import AstV3Protocol, StreamingSession, VadSegmentedStream
from .tasks import AsrTaskEngine, EmotionTaskEngine
from .text_cleanup import clean_asr_text
from .text_cleanup.client import TextCleanupConfigError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_client()


app = FastAPI(title="AudioLLM Server", lifespan=lifespan)


@app.websocket("/ws/audio")
async def audio_ws(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected (/ws/audio)")
    session = AudioSession(websocket)
    try:
        await session.run()
    finally:
        await session.cleanup()


@app.websocket("/transcribe-streaming")
async def transcribe_streaming_ws(websocket: WebSocket, language: str = ""):
    await websocket.accept()
    logger.info("Transcribe-streaming connected (language=%s)", language)
    session = StreamingSession(
        websocket,
        stream=VadSegmentedStream(),
        engine=AsrTaskEngine(),
        language=language,
    )
    try:
        await session.run()
    finally:
        await session.cleanup()


@app.websocket("/tuling/ast/v3")
async def tuling_ast_v3_ws(websocket: WebSocket):
    """iFlytek Tuling AST v3 streaming ASR.

    Same VAD-segmented dual-ASR pipeline as ``/transcribe-streaming``, but the
    on-the-wire framing is the AST v3 ``header/parameter/payload`` envelope:
    audio arrives base64-encoded inside JSON frames, ``header.status`` (0/1/2)
    drives start/stop, and results are repackaged into the ``payload.result``
    lattice. ``AstV3Protocol`` owns all of that translation; the session,
    stream, and engine are the shared ones. ``emit_timing`` lets the engine
    surface segment ``bg``/``ed`` to the protocol.

    See ``docs/tuling-ast-v3-protocol.md``.
    """
    await websocket.accept()
    logger.info("Tuling AST v3 connected (/tuling/ast/v3)")
    # Endpoint policy: primary-only (no secondary / no local Qwen / no fusion),
    # with the primary pinned to the AST v3-specific upstream when configured
    # (empty astv3_vllm_* falls back to the global primary). These are forced
    # overrides (see StreamingSession._config_overrides), re-applied after the
    # client's start.config so a client cannot re-enable secondary via
    # parameter.asr_config.
    cfg = load_config()
    astv3_overrides: dict[str, object] = {"enable_secondary_asr": False}
    if cfg.astv3_vllm_base_url:
        astv3_overrides["vllm_base_url"] = cfg.astv3_vllm_base_url
    if cfg.astv3_vllm_model_name:
        astv3_overrides["vllm_model_name"] = cfg.astv3_vllm_model_name
    session = StreamingSession(
        websocket,
        stream=VadSegmentedStream(),
        engine=AsrTaskEngine(emit_timing=True),
        protocol=AstV3Protocol(),
        config_overrides=astv3_overrides,
    )
    try:
        await session.run()
    finally:
        await session.cleanup()


# Hard-coded remote AST v3 backend that the "实时语音识别（测试用）" page targets.
# It speaks plaintext ws://, so an HTTPS-served frontend (playground.amphion.top)
# cannot open it directly — the browser's mixed-content policy forbids ws:// from
# an https:// page. The same-origin proxy below bridges the browser's wss://
# connection to it. Temporary test scaffolding: the address is intentionally
# pinned here, not exposed as config.
ASTV3_TEST_PROXY_UPSTREAM = "ws://159.138.9.106:18082/tuling/ast/v3"


async def _astv3_proxy_pump_to_upstream(client: WebSocket, upstream) -> None:
    """Relay browser -> upstream frames verbatim (text or binary)."""
    try:
        while True:
            message = await client.receive()
            if message["type"] == "websocket.disconnect":
                break
            text = message.get("text")
            if text is not None:
                await upstream.send(text)
                continue
            data = message.get("bytes")
            if data is not None:
                await upstream.send(data)
    except (WebSocketDisconnect, websockets.ConnectionClosed, RuntimeError):
        pass
    finally:
        # Closing the upstream unblocks the opposite pump's async-for so the
        # whole proxy tears down once either side goes away.
        await upstream.close()


async def _astv3_proxy_pump_to_client(upstream, client: WebSocket) -> None:
    """Relay upstream -> browser frames verbatim (text or binary)."""
    try:
        async for message in upstream:
            if isinstance(message, (bytes, bytearray)):
                await client.send_bytes(bytes(message))
            else:
                await client.send_text(message)
    except (websockets.ConnectionClosed, WebSocketDisconnect, RuntimeError):
        pass
    finally:
        try:
            await client.close()
        except RuntimeError:
            # client transport already closed; nothing to do
            pass


@app.websocket("/astv3-test-proxy")
async def astv3_test_proxy_ws(websocket: WebSocket):
    """Same-origin WebSocket proxy for the AST v3 test page.

    The "实时语音识别（测试用）" page is served over HTTPS but its target backend
    speaks plaintext ws:// (``ASTV3_TEST_PROXY_UPSTREAM``), which the browser's
    mixed-content policy forbids opening from an HTTPS page. This endpoint accepts
    the browser's same-origin (wss://) connection and relays every frame, in both
    directions, to/from that upstream without inspecting the AST v3 envelope. It
    is a transparent byte pump, so the on-the-wire contract is identical to
    ``/tuling/ast/v3`` (see ``docs/tuling-ast-v3-protocol.md``).

    Temporary test scaffolding: the upstream address is hard-coded.
    """
    await websocket.accept()
    logger.info("AST v3 test proxy connected -> %s", ASTV3_TEST_PROXY_UPSTREAM)
    try:
        async with websockets.connect(
            ASTV3_TEST_PROXY_UPSTREAM, max_size=None, open_timeout=10
        ) as upstream:
            await asyncio.gather(
                _astv3_proxy_pump_to_upstream(websocket, upstream),
                _astv3_proxy_pump_to_client(upstream, websocket),
            )
    except WebSocketDisconnect:
        # Browser hung up before/while the upstream was being dialed.
        pass
    except Exception as exc:  # upstream connect / handshake failure
        logger.warning("AST v3 test proxy upstream error: %s", exc)
        try:
            await websocket.close(code=1011)
        except RuntimeError:
            pass


@app.websocket("/emotion-segmented-streaming")
async def emotion_segmented_streaming_ws(websocket: WebSocket, language: str = ""):
    await websocket.accept()
    logger.info(
        "Emotion-segmented-streaming connected (language=%s)", language
    )
    session = StreamingSession(
        websocket,
        # Emotion has no partial output, so disable VAD's snapshot bookkeeping
        # regardless of the global pseudo-stream toggle.
        stream=VadSegmentedStream(enable_partial=False),
        engine=EmotionTaskEngine(streaming=True),
        language=language,
    )
    try:
        await session.run()
    finally:
        await session.cleanup()


# ---------------------------------------------------------------------------
# One-shot upload endpoints
# ---------------------------------------------------------------------------
# The /api/* routes power the "Upload audio file" buttons in the demos.
# They deliberately bypass the WebSocket/VAD pipeline: the frontend hands us
# a fully-decoded 16 kHz mono WAV (produced via the browser's Web Audio API),
# and we forward the bytes to the same vLLM endpoints the streaming engines
# call. This keeps the upload flow as "send the whole clip, get one final
# result" — no chunking, no VAD segmentation, no partials.
#
# All caps that the streaming pipeline normally enforces server-side
# (emotion 20s tail, ASR 60s tail) are still applied here so a malicious or
# buggy client can't bypass them by switching from WS to REST.

# Hard cap on multipart upload bytes. ~16-bit / 16 kHz mono WAV at 60 s is
# ~1.9 MB; this 25 MB ceiling is generous for any clip the model would
# realistically be asked to handle.
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024

# Server-side trim caps (mirror the streaming pipeline's behaviour so REST
# and WS produce identical model inputs for the same recording).
_ASR_MAX_SECONDS = 60.0


def _parse_csv(raw: str | None) -> list[str]:
    """Parse a ``"a,b ,c"`` form field into a clean string list."""
    if not raw:
        return []
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


async def _read_audio_bytes(audio: UploadFile) -> bytes:
    """Read a multipart audio upload, enforcing the global byte cap.

    UploadFile.read with no argument loads into memory; the size check is
    primarily a guard against accidental huge uploads, not a streaming
    safeguard (we need the full payload for vLLM anyway).
    """
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="audio file is empty")
    if len(raw) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"audio file exceeds {_MAX_UPLOAD_BYTES} bytes",
        )
    return raw


def _wav_to_pcm_capped(raw: bytes, max_seconds: float) -> tuple[bytes, float]:
    """Decode a WAV blob to 16 kHz mono and tail-trim to ``max_seconds``.

    Returns (re_encoded_wav_bytes, duration_sec). When no trim is needed the
    re-encoded WAV is byte-equivalent to ``pcm_to_wav_base64(pcm)``.
    """
    import io
    import wave

    import numpy as np

    from .audio.utils import pcm_to_wav_base64

    wav_b64 = base64.b64encode(raw).decode("ascii")
    try:
        pcm = wav_base64_to_pcm_16k_mono(wav_b64)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"could not decode audio: {exc}"
        ) from exc
    if pcm.size == 0:
        raise HTTPException(status_code=400, detail="audio decoded to empty PCM")
    duration = pcm.size / SAMPLE_RATE
    if max_seconds > 0 and duration > max_seconds:
        # Match streaming engines: keep the trailing window. Emotion picks
        # the tail because the most recent emotion is what users care about;
        # we use the same convention for ASR for consistency.
        keep = int(SAMPLE_RATE * max_seconds)
        pcm = pcm[-keep:]
        duration = pcm.size / SAMPLE_RATE
    new_b64 = pcm_to_wav_base64(pcm.astype(np.float32, copy=False))
    new_bytes = base64.b64decode(new_b64)
    # Sanity: re-encoded WAV should still parse.
    with wave.open(io.BytesIO(new_bytes), "rb") as wf:
        assert wf.getframerate() == SAMPLE_RATE
    return new_bytes, duration


def _model_result_payload(result: object | None) -> dict | None:
    if result is None:
        return None
    if isinstance(result, Exception):
        return {
            "error": str(result),
            "error_type": result.__class__.__name__,
        }
    if isinstance(result, dict):
        return dict(result)
    return {"raw": str(result)}


def _emotion_result_payload(
    result: object,
    *,
    mode: str,
    duration_sec: float,
    language: str,
) -> dict:
    if isinstance(result, Exception):
        return {
            "type": "error",
            "mode": mode,
            "message": str(result),
            "error_type": result.__class__.__name__,
        }
    payload = {
        "type": "final_emotion",
        "mode": mode,
        "label": result.get("label", ""),
        "text": result.get("text", ""),
        "duration_sec": round(duration_sec, 3),
    }
    if language:
        payload["language"] = language
    return payload


def _public_asr_payload(result: dict) -> dict:
    return {
        "text": result.get("text", ""),
        "language": result.get("language", ""),
    }


def _public_cleanup_payload(result: dict) -> dict:
    return {
        "text": result.get("text", ""),
    }


def _resolve_enrollment_b64(enrollment_id: str | None) -> str | None:
    """Look up an enrollment id, refreshing its TTL. Missing/expired ids
    return ``None`` (caller decides to error or silently fall back)."""
    if not enrollment_id:
        return None
    entry = get_enrollment_store().get(enrollment_id)
    return entry.wav_base64 if entry is not None else None


async def _run_dual_asr_upload(
    wav_b64: str,
    *,
    cfg,
    hotwords: list[str],
    language: str,
    enrollment_b64: str | None = None,
) -> dict:
    primary_task = None
    secondary_task = None
    if cfg.enable_primary_asr:
        primary_task = asyncio.create_task(
            asyncio.wait_for(
                query_audio_model(
                    wav_b64,
                    hotwords=hotwords,
                    src_lang=language or "N/A",
                    enrollment_wav_base64=enrollment_b64,
                    base_url=cfg.vllm_base_url,
                    model_name=cfg.vllm_model_name,
                    timeout=cfg.asr_request_timeout,
                ),
                timeout=cfg.primary_asr_timeout,
            )
        )
    # Secondary (Qwen3) keeps single-audio prompting regardless of enrollment;
    # it is trained as a plain ASR model with no target-speaker channel, so
    # forcing enrollment audio in front of the mixed clip would push the model
    # out of distribution. The fusion stage still benefits from Qwen's
    # parallel transcription as a sanity check on the primary's output.
    #
    # REST uploads have no partial channel, so the secondary only earns its
    # keep when fusion is on. Gating on `enable_dual_asr_fusion` here (rather
    # than `enable_secondary_asr`) lets operators keep the secondary online
    # for streaming partials while skipping it on one-shot uploads.
    if cfg.enable_dual_asr_fusion:
        secondary_task = asyncio.create_task(
            query_audio_model_secondary(
                wav_b64,
                hotwords=hotwords,
                base_url=cfg.secondary_vllm_base_url,
                model_name=cfg.secondary_vllm_model_name,
                timeout=cfg.asr_request_timeout,
            )
        )

    primary_res: object | None = None
    secondary_res: object | None = None
    if primary_task is not None:
        try:
            primary_res = await primary_task
        except Exception as err:  # noqa: BLE001 - preserve failure details
            primary_res = err
            logger.warning("Primary ASR failed: %s", err)
    if secondary_task is not None:
        try:
            secondary_res = await secondary_task
        except Exception as err:  # noqa: BLE001
            secondary_res = err
            logger.warning("Secondary ASR failed: %s", err)

    primary_result = None if isinstance(primary_res, Exception) else primary_res
    secondary_result = None if isinstance(secondary_res, Exception) else secondary_res
    if primary_result is None and secondary_result is None:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "all configured ASR models failed",
                "primary": _model_result_payload(primary_res),
                "secondary": _model_result_payload(secondary_res),
            },
        )

    detected_lang = language or ""
    fusion_payload: dict | None = None
    if primary_result and not secondary_result:
        text = str(primary_result.get("transcription") or "").strip()
        detected_lang = primary_result.get("detected_language") or detected_lang
    elif secondary_result and not primary_result:
        text = str(secondary_result.get("transcription") or "").strip()
    else:
        fusion_payload = choose_fused_result(
            primary_result,
            secondary_result,
            hotwords=hotwords,
            similarity_threshold=cfg.fusion_similarity_threshold,
            min_primary_score=cfg.fusion_min_primary_score,
            max_repetition_ratio=cfg.fusion_max_repetition_ratio,
            disagreement_threshold=cfg.fusion_disagreement_threshold,
            hotword_boost=cfg.fusion_hotword_boost,
            primary_score_margin=cfg.fusion_primary_score_margin,
        )
        text = str(fusion_payload.get("text") or "").strip()
        if primary_result and primary_result.get("detected_language"):
            detected_lang = primary_result["detected_language"]

    # Final-only display transform (ITN + plate normalization), matching the
    # streaming engines so REST and WS clients see the same written form.
    if text:
        text = normalize_final_text(text, detected_lang, cfg)

    raw_text = ""
    if primary_result:
        raw_text = str(primary_result.get("raw_text") or "")
    elif secondary_result:
        raw_text = str(secondary_result.get("raw_text") or "")

    return {
        "text": text,
        "language": detected_lang,
        "raw_text": raw_text,
        "primary": _model_result_payload(primary_res),
        "secondary": _model_result_payload(secondary_res),
        "fusion": fusion_payload,
    }


@app.post("/api/asr/enrollment")
async def asr_enrollment_create(audio: UploadFile = File(...)):
    """Cache a target-speaker enrollment clip and return its opaque id.

    The frontend uploads a 1–8 s clip once (file or mic recording) and
    then passes the returned ``enrollment_id`` on every ``/api/asr/upload``
    call and the WS ``start`` payload. The server validates duration,
    canonicalises to 16 kHz mono WAV, and stores the base64 result so
    primary inference can splice it into the dual-audio prompt without
    re-decoding on every segment.
    """
    raw = await _read_audio_bytes(audio)
    wav_b64 = base64.b64encode(raw).decode("ascii")
    cfg = load_config()
    try:
        canonical_b64, duration_sec = decode_and_validate(
            wav_b64,
            min_sec=cfg.asr_enrollment_min_sec,
            max_sec=cfg.asr_enrollment_max_sec,
        )
    except EnrollmentError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc

    store = get_enrollment_store()
    store.configure(
        ttl_sec=cfg.asr_enrollment_ttl_sec,
        max_entries=cfg.asr_enrollment_max_entries,
    )
    entry = store.put(canonical_b64, duration_sec)
    return {
        "enrollment_id": entry.enrollment_id,
        "duration_sec": round(duration_sec, 3),
    }


@app.delete("/api/asr/enrollment/{enrollment_id}")
async def asr_enrollment_delete(enrollment_id: str):
    """Drop a previously uploaded enrollment clip.

    Returning 204 on missing ids keeps the frontend's "clear" button
    idempotent — repeated clears never error out.
    """
    get_enrollment_store().delete(enrollment_id)
    return JSONResponse(status_code=204, content=None)


@app.post("/api/asr/upload")
async def asr_upload(
    audio: UploadFile = File(...),
    language: str = Form(""),
    hotwords: str = Form(""),
    enrollment_id: str = Form(""),
):
    """One-shot ASR over an uploaded clip.

    Mirrors :class:`AsrTaskEngine.handle_segment` but operates on the entire
    clip in a single dual-model call (no VAD segmentation, no partials).
    Returns the same fields the streaming ``final`` event carries.

    When ``enrollment_id`` resolves to a cached enrollment clip the primary
    model is prompted with the dual-audio TS-ASR template (task 5/6 of the
    v4 prompt spec). Unknown / expired ids fall back to plain ASR so a
    stale id from a long-running tab does not break uploads.
    """
    raw = await _read_audio_bytes(audio)
    wav_bytes, duration_sec = _wav_to_pcm_capped(raw, _ASR_MAX_SECONDS)
    wav_b64 = base64.b64encode(wav_bytes).decode("ascii")
    cfg = load_config()
    hw_list = _parse_csv(hotwords)
    enrollment_b64 = _resolve_enrollment_b64(enrollment_id)
    asr_result = await _run_dual_asr_upload(
        wav_b64,
        cfg=cfg,
        hotwords=hw_list,
        language=language,
        enrollment_b64=enrollment_b64,
    )

    return {
        "type": "final",
        "text": asr_result["text"],
        "language": asr_result["language"],
        "duration_sec": round(duration_sec, 3),
        "enrollment_used": enrollment_b64 is not None,
    }


@app.post("/api/emotion/jobs", status_code=202)
async def emotion_create_job(
    audio: UploadFile = File(...),
    mode: str = Form(""),
    language: str = Form(""),
):
    """Enqueue whole-utterance emotion inference; poll GET /api/emotion/jobs/{id}."""
    raw = await _read_audio_bytes(audio)
    cfg = load_config()
    cap = float(getattr(cfg, "emotion_max_audio_seconds", 0.0))
    try:
        decode_wav_capped(raw, cap)
    except EmotionDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    store = get_emotion_job_store()
    store.configure(cfg)
    try:
        job = await store.submit(
            raw,
            mode=mode,
            language=language,
            cfg=cfg,
        )
    except JobQueueFullError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
            headers={"Retry-After": "5"},
        ) from exc

    poll_url = f"/api/emotion/jobs/{job.job_id}"
    return JSONResponse(
        status_code=202,
        content={
            "job_id": job.job_id,
            "status": job.status,
            "poll_url": poll_url,
        },
    )


@app.get("/api/emotion/jobs/{job_id}")
async def emotion_get_job(job_id: str):
    """Poll async emotion job status and result."""
    store = get_emotion_job_store()
    job = await store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_poll_dict()


@app.post("/api/emotion-spec/jobs", status_code=202)
async def emotion_spec_create_job(
    audio: UploadFile = File(...),
    mode: str = Form(""),
    language: str = Form(""),
):
    """Enqueue whole-utterance AmphionSPEC inference; poll GET /api/emotion-spec/jobs/{id}.

    Independent of ``/api/emotion/jobs`` — separate queue, separate
    concurrency budget, separate vLLM endpoint (cfg.emotion_spec_vllm_*).
    ``mode`` accepts ``ser`` or ``sepc`` (alias ``spec`` is normalized to
    ``sepc``); empty falls back to ``cfg.emotion_spec_task_mode``.
    """
    raw = await _read_audio_bytes(audio)
    cfg = load_config()
    cap = float(getattr(cfg, "emotion_spec_max_audio_seconds", 0.0))
    try:
        decode_wav_capped(raw, cap)
    except EmotionDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    store = get_emotion_spec_job_store()
    store.configure(cfg)
    try:
        job = await store.submit(
            raw,
            mode=mode,
            language=language,
            cfg=cfg,
        )
    except JobQueueFullError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
            headers={"Retry-After": "5"},
        ) from exc

    poll_url = f"/api/emotion-spec/jobs/{job.job_id}"
    return JSONResponse(
        status_code=202,
        content={
            "job_id": job.job_id,
            "status": job.status,
            "poll_url": poll_url,
        },
    )


@app.get("/api/emotion-spec/jobs/{job_id}")
async def emotion_spec_get_job(job_id: str):
    """Poll async AmphionSPEC job status and result."""
    store = get_emotion_spec_job_store()
    job = await store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_poll_dict()


@app.post("/api/audio/analyze")
async def audio_analyze(
    audio: UploadFile = File(...),
    language: str = Form(""),
    hotwords: str = Form(""),
    emotion_mode: str = Form("both"),
    enrollment_id: str = Form(""),
):
    """One-shot audio analysis: ASR raw output + cleaned text + emotion."""
    raw = await _read_audio_bytes(audio)
    cfg = load_config()
    hw_list = _parse_csv(hotwords)
    enrollment_b64 = _resolve_enrollment_b64(enrollment_id)

    asr_wav_bytes, duration_sec = _wav_to_pcm_capped(raw, _ASR_MAX_SECONDS)
    asr_wav_b64 = base64.b64encode(asr_wav_bytes).decode("ascii")

    emotion_cap = float(getattr(cfg, "emotion_max_audio_seconds", 0.0))
    emotion_wav_bytes, emotion_duration_sec = _wav_to_pcm_capped(raw, emotion_cap)
    emotion_wav_b64 = base64.b64encode(emotion_wav_bytes).decode("ascii")

    asr_task = asyncio.create_task(
        _run_dual_asr_upload(
            asr_wav_b64,
            cfg=cfg,
            hotwords=hw_list,
            language=language,
            enrollment_b64=enrollment_b64,
        )
    )
    emotion_ser_task = asyncio.create_task(
        query_emotion_model(
            emotion_wav_b64,
            mode="ser",
            base_url=cfg.emotion_vllm_base_url,
            model_name=cfg.emotion_vllm_model_name,
            timeout=cfg.emotion_request_timeout,
        )
    )
    emotion_sec_task = asyncio.create_task(
        query_emotion_model(
            emotion_wav_b64,
            mode="sec",
            base_url=cfg.emotion_vllm_base_url,
            model_name=cfg.emotion_vllm_model_name,
            timeout=cfg.emotion_request_timeout,
            max_tokens=256,
        )
    )

    asr_out, emotion_ser_out, emotion_sec_out = await asyncio.gather(
        asr_task,
        emotion_ser_task,
        emotion_sec_task,
        return_exceptions=True,
    )
    if isinstance(asr_out, HTTPException):
        raise asr_out
    if isinstance(asr_out, Exception):
        logger.error("Audio analyze ASR failed: %s", asr_out)
        raise HTTPException(status_code=502, detail=str(asr_out)) from asr_out
    asr_result = asr_out

    if isinstance(emotion_ser_out, Exception):
        logger.error("Audio analyze SER inference failed: %s", emotion_ser_out)
    if isinstance(emotion_sec_out, Exception):
        logger.error("Audio analyze SEC inference failed: %s", emotion_sec_out)
    emotion_ser = _emotion_result_payload(
        emotion_ser_out,
        mode="ser",
        duration_sec=emotion_duration_sec,
        language=language,
    )
    emotion_sec = _emotion_result_payload(
        emotion_sec_out,
        mode="sec",
        duration_sec=emotion_duration_sec,
        language=language,
    )
    emotion_payload = {
        "type": "final_emotion_pair",
        "mode": "both",
        "ser": emotion_ser,
        "sec": emotion_sec,
    }

    try:
        cleaned = await clean_asr_text(
            str(asr_result.get("text") or ""),
            hotwords=[],
            language=str(asr_result.get("language") or language or ""),
            emotion=emotion_payload,
            cfg=cfg,
        )
    except TextCleanupConfigError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Audio analyze text cleanup failed")
        raise HTTPException(
            status_code=502,
            detail=f"text cleanup model failed: {exc}",
        ) from exc

    return {
        "type": "audio_analysis",
        "duration_sec": round(duration_sec, 3),
        "language": asr_result.get("language") or language or "",
        "hotwords": hw_list,
        "asr": _public_asr_payload(asr_result),
        "cleaned_asr": _public_cleanup_payload(cleaned),
        "emotion": emotion_payload,
    }


class _RevalidateStaticFiles(StaticFiles):
    """Static files with tiered caching.

    Browsers were aggressively caching ``app.js`` / ``style.css`` because
    starlette's default ``StaticFiles`` ships no explicit ``Cache-Control``
    header, leaving the heuristic up to the client. That made shipping
    frontend fixes during a session unreliable — users had to hard-reload
    to pick up changes. We inject ``no-cache`` so the browser still uses
    its disk copy, but always revalidates with the server's ``ETag`` (set
    by starlette from mtime+size); unchanged files come back as 304 so
    bandwidth stays cheap. Cache-Control is omitted on non-200 responses
    to avoid pinning errors.

    For the assets the demo loads on every page (CSS, JS) we additionally
    grant a short ``max-age`` so the browser can serve repeat requests
    from disk cache without a conditional round-trip. Ten seconds is long
    enough to cover all the navigations of a single user session yet
    short enough that any real frontend change still appears within a
    blink — and the ``ETag`` revalidation kicks back in once the window
    expires. HTML is intentionally left at plain ``no-cache`` so users
    always see the latest markup.
    """

    _CACHEABLE_EXTS = (".css", ".js")

    @staticmethod
    def _cache_header_for(path: str) -> bytes:
        if path.endswith(_RevalidateStaticFiles._CACHEABLE_EXTS):
            return b"no-cache, max-age=10"
        return b"no-cache"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        cache_header = self._cache_header_for((scope.get("path") or "").lower())

        async def send_with_cache(message: dict) -> None:
            if message["type"] == "http.response.start":
                status = message.get("status", 0)
                if 200 <= status < 300:
                    headers = list(message.get("headers", []))
                    headers = [
                        (k, v) for (k, v) in headers if k.lower() != b"cache-control"
                    ]
                    headers.append((b"cache-control", cache_header))
                    message["headers"] = headers
            await send(message)

        await super().__call__(scope, receive, send_with_cache)


# Static mount comes last so it doesn't shadow the /api routes above.
app.mount(
    "/", _RevalidateStaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend"
)
