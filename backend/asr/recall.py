"""Triton hotword recall and RAG-ASR management client.

AudioLLM stays a thin gateway: final ASR sends PCM to Triton for top-K hotword
recall and optional vLLM-ready audio_embeds.  Hotword-pool management and
enrollment embedding storage can be routed to the optional RAG-ASR HTTP
management service; when that service is not configured, legacy Triton
management and local enrollment fallback remain available.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
import numpy as np

from ..config import Config, Upstream, get_service_upstream
from ..recall_user import DEFAULT_HOTWORD_POOL_ID, normalize_hotword_pool_id

DEFAULT_RECALL_MODEL = "rag_asr_retrieve"
SAMPLE_RATE = 16000


@dataclass(frozen=True)
class RecallResult:
    words: list[str]
    audio_embeds_b64: str | None
    projector_len: int | None
    uuid: str
    enrollment_audio_embeds_b64: str | None = None
    enrollment_projector_len: int | None = None
    message: dict[str, object] | None = None


def stable_audio_uuid(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Stable id for vLLM's multimodal cache."""
    audio = np.asarray(pcm, dtype=np.float32)
    digest = hashlib.sha1()
    digest.update(str(int(sample_rate)).encode("ascii"))
    digest.update(audio.tobytes())
    return f"triton-audio-{digest.hexdigest()[:16]}"


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _triton_url(base_url: str) -> str:
    value = base_url.strip().rstrip("/")
    parsed = urlparse(value if "://" in value else f"http://{value}")
    return parsed.netloc or parsed.path


def _recall_upstream() -> Upstream:
    upstream = get_service_upstream("recall")
    if upstream is None or not upstream.base_url:
        raise RuntimeError("Triton recall service is not configured")
    return upstream


def _management_upstream() -> Upstream | None:
    upstream = get_service_upstream("recall_management")
    if upstream is None or not upstream.base_url:
        return None
    return upstream


def _management_url(upstream: Upstream, path: str) -> str:
    return f"{upstream.base_url.rstrip('/')}/{path.lstrip('/')}"


def _client_for(upstream: Upstream):
    import tritonclient.http as httpclient

    return httpclient, httpclient.InferenceServerClient(url=_triton_url(upstream.base_url))


def _string_input(httpclient, name: str, value: str):
    tensor = httpclient.InferInput(name, [1], "BYTES")
    tensor.set_data_from_numpy(np.array([value], dtype=object))
    return tensor


def _int_input(httpclient, name: str, value: int):
    tensor = httpclient.InferInput(name, [1], "INT32")
    tensor.set_data_from_numpy(np.array([int(value)], dtype=np.int32))
    return tensor


def _control_request_id(payload: dict[str, object]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return f"ctl:{encoded}"


def _model_name(upstream: Upstream) -> str:
    return upstream.model_name or DEFAULT_RECALL_MODEL


def _infer_sync(
    pcm: np.ndarray,
    *,
    sample_rate: int,
    top_k: int,
    want_audio_embeds: bool,
    hotword_pool_id: str,
    enable_hotword_recall: bool = True,
    enrollment_id: str | None = None,
    enrollment_user_id: str | None = None,
    want_enrollment_audio_embeds: bool = False,
) -> RecallResult:
    upstream = _recall_upstream()
    httpclient, client = _client_for(upstream)
    wav = np.asarray(pcm, dtype=np.float32).reshape(-1)
    wav_input = httpclient.InferInput("WAV", wav.shape, "FP32")
    control: dict[str, object] = {
        "action": "infer",
        "hotword_pool_id": hotword_pool_id,
    }
    if enrollment_id:
        control["enrollment_id"] = enrollment_id
    if enrollment_user_id:
        control["enrollment_user_id"] = enrollment_user_id
    inputs = [
        wav_input,
        _int_input(httpclient, "SAMPLE_RATE", sample_rate),
        _int_input(httpclient, "TOP_K", top_k),
    ]
    if not enable_hotword_recall:
        inputs.append(_int_input(httpclient, "ENABLE_HOTWORD_RECALL", 0))
    if want_enrollment_audio_embeds:
        inputs.append(_int_input(httpclient, "WANT_ENROLLMENT_AUDIO_EMBEDS", 1))
    wav_input.set_data_from_numpy(wav)

    outputs = [
        httpclient.InferRequestedOutput("WORD_LIST"),
        httpclient.InferRequestedOutput("PROJECTOR_LEN"),
    ]
    if want_audio_embeds:
        outputs.append(httpclient.InferRequestedOutput("AUDIO_EMBEDS_B64"))
    if want_enrollment_audio_embeds:
        outputs.extend(
            [
                httpclient.InferRequestedOutput("ENROLLMENT_AUDIO_EMBEDS_B64"),
                httpclient.InferRequestedOutput("ENROLLMENT_PROJECTOR_LEN"),
                httpclient.InferRequestedOutput("MESSAGE"),
            ]
        )

    result = client.infer(
        _model_name(upstream),
        inputs,
        outputs=outputs,
        request_id=_control_request_id(control),
    )
    words = json.loads(_decode(result.as_numpy("WORD_LIST")[0]))
    projector_len = int(result.as_numpy("PROJECTOR_LEN")[0])
    audio_embeds_b64 = None
    if want_audio_embeds:
        audio_embeds_b64 = _decode(result.as_numpy("AUDIO_EMBEDS_B64")[0])
    enrollment_audio_embeds_b64 = None
    enrollment_projector_len = None
    message: dict[str, object] | None = None
    if want_enrollment_audio_embeds:
        enrollment_audio_embeds_b64 = _decode(
            result.as_numpy("ENROLLMENT_AUDIO_EMBEDS_B64")[0]
        )
        enrollment_projector_len = int(result.as_numpy("ENROLLMENT_PROJECTOR_LEN")[0])
        message_raw = _decode(result.as_numpy("MESSAGE")[0])
        parsed = json.loads(message_raw) if message_raw else {}
        message = parsed if isinstance(parsed, dict) else {"message": parsed}
    return RecallResult(
        words=[str(word) for word in words],
        audio_embeds_b64=audio_embeds_b64,
        projector_len=projector_len,
        uuid=stable_audio_uuid(wav, sample_rate),
        enrollment_audio_embeds_b64=enrollment_audio_embeds_b64,
        enrollment_projector_len=enrollment_projector_len,
        message=message,
    )


async def recall_audio(
    pcm: np.ndarray,
    cfg: Config,
    *,
    sample_rate: int = SAMPLE_RATE,
    want_audio_embeds: bool = True,
    hotword_pool_id: str | None = None,
    user_id: str | None = None,
    enable_hotword_recall: bool = True,
    enrollment_id: str | None = None,
    enrollment_user_id: str | None = None,
    want_enrollment_audio_embeds: bool = False,
) -> RecallResult:
    """Recall hotwords for one audio segment."""
    top_k = max(int(cfg.recall_top_k), 0) if enable_hotword_recall else 0
    resolved_hotword_pool_id = normalize_hotword_pool_id(
        hotword_pool_id if hotword_pool_id is not None else user_id,
        default=cfg.hotword_pool_id,
    )
    if top_k == 0 and not (enrollment_id or want_enrollment_audio_embeds):
        return RecallResult(
            words=[],
            audio_embeds_b64=None,
            projector_len=None,
            uuid=stable_audio_uuid(pcm, sample_rate),
        )
    return await asyncio.to_thread(
        _infer_sync,
        pcm,
        sample_rate=sample_rate,
        top_k=top_k,
        want_audio_embeds=want_audio_embeds,
        hotword_pool_id=resolved_hotword_pool_id,
        enable_hotword_recall=enable_hotword_recall,
        enrollment_id=enrollment_id,
        enrollment_user_id=enrollment_user_id,
        want_enrollment_audio_embeds=want_enrollment_audio_embeds,
    )


def _enrollment_sync(
    action: str,
    *,
    enrollment_id: str,
    enrollment_user_id: str | None = None,
    user_id: str | None = None,
    pcm: np.ndarray | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> dict[str, object]:
    management = _management_upstream()
    if management is not None:
        resolved_enrollment_user_id = normalize_hotword_pool_id(
            enrollment_user_id if enrollment_user_id is not None else user_id,
            default=DEFAULT_HOTWORD_POOL_ID,
        )
        with httpx.Client(timeout=management.timeout) as client:
            if action == "upsert_enrollment":
                if pcm is None:
                    raise ValueError("pcm is required for upsert_enrollment")
                wav = np.asarray(pcm, dtype=np.float32).reshape(-1)
                response = client.post(
                    _management_url(management, f"/enrollments/{enrollment_id}"),
                    data={
                        "enrollment_user_id": resolved_enrollment_user_id,
                        "sample_rate": str(int(sample_rate)),
                    },
                    files={
                        "file": (
                            "audio.f32",
                            wav.tobytes(),
                            "application/octet-stream",
                        )
                    },
                )
            elif action == "get_enrollment":
                response = client.get(
                    _management_url(management, f"/enrollments/{enrollment_id}"),
                    params={"enrollment_user_id": resolved_enrollment_user_id},
                )
            elif action == "delete_enrollment":
                response = client.delete(
                    _management_url(management, f"/enrollments/{enrollment_id}"),
                    params={"enrollment_user_id": resolved_enrollment_user_id},
                )
            else:
                raise ValueError(f"unknown enrollment action: {action}")
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {"status": "ok", "data": data}

    upstream = _recall_upstream()
    httpclient, client = _client_for(upstream)
    resolved_enrollment_user_id = normalize_hotword_pool_id(
        enrollment_user_id if enrollment_user_id is not None else user_id,
        default=DEFAULT_HOTWORD_POOL_ID,
    )
    control: dict[str, object] = {
        "action": action,
        "enrollment_id": enrollment_id,
        "enrollment_user_id": resolved_enrollment_user_id,
    }
    inputs = []
    if pcm is not None:
        wav = np.asarray(pcm, dtype=np.float32).reshape(-1)
        wav_input = httpclient.InferInput("WAV", wav.shape, "FP32")
        wav_input.set_data_from_numpy(wav)
        inputs.extend([wav_input, _int_input(httpclient, "SAMPLE_RATE", sample_rate)])
    else:
        inputs.append(_int_input(httpclient, "OFFSET", 0))
    outputs = [
        httpclient.InferRequestedOutput("STATUS"),
        httpclient.InferRequestedOutput("MESSAGE"),
        httpclient.InferRequestedOutput("HOTWORD_COUNT"),
        httpclient.InferRequestedOutput("HOTWORD_LIST"),
    ]
    result = client.infer(
        _model_name(upstream),
        inputs,
        outputs=outputs,
        request_id=_control_request_id(control),
    )
    status = _decode(result.as_numpy("STATUS")[0])
    message_raw = _decode(result.as_numpy("MESSAGE")[0])
    message = json.loads(message_raw) if message_raw else {}
    if not isinstance(message, dict):
        message = {"message": message}
    return {"status": status, **message}


async def upsert_enrollment(
    pcm: np.ndarray,
    *,
    enrollment_id: str,
    enrollment_user_id: str | None = None,
    user_id: str | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> dict[str, object]:
    return await asyncio.to_thread(
        _enrollment_sync,
        "upsert_enrollment",
        enrollment_id=enrollment_id,
        enrollment_user_id=enrollment_user_id,
        user_id=user_id,
        pcm=pcm,
        sample_rate=sample_rate,
    )


async def get_enrollment(
    *,
    enrollment_id: str,
    enrollment_user_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, object]:
    return await asyncio.to_thread(
        _enrollment_sync,
        "get_enrollment",
        enrollment_id=enrollment_id,
        enrollment_user_id=enrollment_user_id,
        user_id=user_id,
    )


async def delete_enrollment(
    *,
    enrollment_id: str,
    enrollment_user_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, object]:
    return await asyncio.to_thread(
        _enrollment_sync,
        "delete_enrollment",
        enrollment_id=enrollment_id,
        enrollment_user_id=enrollment_user_id,
        user_id=user_id,
    )


def _management_sync(
    action: str,
    *,
    hotwords: list[str] | None = None,
    query: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    hotword_pool_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, object]:
    management = _management_upstream()
    if management is not None:
        resolved_hotword_pool_id = normalize_hotword_pool_id(
            hotword_pool_id if hotword_pool_id is not None else user_id,
            default=DEFAULT_HOTWORD_POOL_ID,
        )
        with httpx.Client(timeout=management.timeout) as client:
            if action == "list":
                response = client.get(
                    _management_url(management, "/hotword-pool"),
                    params={
                        "hotword_pool_id": resolved_hotword_pool_id,
                        "query": query,
                        "limit": limit,
                        "offset": offset,
                    },
                )
            elif action == "add":
                response = client.post(
                    _management_url(management, "/hotword-pool"),
                    json={
                        "hotword_pool_id": resolved_hotword_pool_id,
                        "hotwords": hotwords or [],
                    },
                )
            elif action == "delete":
                response = client.request(
                    "DELETE",
                    _management_url(management, "/hotword-pool"),
                    json={
                        "hotword_pool_id": resolved_hotword_pool_id,
                        "hotwords": hotwords or [],
                    },
                )
            elif action == "reload":
                response = client.post(
                    _management_url(management, "/hotword-pool/reload"),
                    json={"hotword_pool_id": resolved_hotword_pool_id},
                )
            else:
                raise ValueError(f"unknown hotword action: {action}")
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {"status": "ok", "data": data}

    upstream = _recall_upstream()
    httpclient, client = _client_for(upstream)
    resolved_hotword_pool_id = normalize_hotword_pool_id(
        hotword_pool_id if hotword_pool_id is not None else user_id,
        default=DEFAULT_HOTWORD_POOL_ID,
    )
    control: dict[str, object] = {
        "action": action,
        "hotword_pool_id": resolved_hotword_pool_id,
    }
    if hotwords is not None:
        control["hotwords"] = hotwords
    if query:
        control["query"] = query
    inputs = []
    if limit is not None:
        inputs.append(_int_input(httpclient, "LIMIT", limit))
    inputs.append(_int_input(httpclient, "OFFSET", offset))

    outputs = [
        httpclient.InferRequestedOutput("STATUS"),
        httpclient.InferRequestedOutput("MESSAGE"),
        httpclient.InferRequestedOutput("HOTWORD_COUNT"),
        httpclient.InferRequestedOutput("HOTWORD_LIST"),
    ]
    result = client.infer(
        _model_name(upstream),
        inputs,
        outputs=outputs,
        request_id=_control_request_id(control),
    )
    status = _decode(result.as_numpy("STATUS")[0])
    message_raw = _decode(result.as_numpy("MESSAGE")[0])
    hotwords_raw = _decode(result.as_numpy("HOTWORD_LIST")[0])
    message = json.loads(message_raw) if message_raw else {}
    if not isinstance(message, dict):
        message = {"message": message}
    return {
        "status": status,
        **message,
        "hotwords": json.loads(hotwords_raw) if hotwords_raw else [],
        "total_count": int(result.as_numpy("HOTWORD_COUNT")[0]),
    }


async def list_hotword_pool(
    *,
    query: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    hotword_pool_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, object]:
    return await asyncio.to_thread(
        _management_sync,
        "list",
        query=query,
        limit=limit,
        offset=offset,
        hotword_pool_id=hotword_pool_id,
        user_id=user_id,
    )


async def add_hotwords(
    words: list[str],
    *,
    hotword_pool_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, object]:
    return await asyncio.to_thread(
        _management_sync,
        "add",
        hotwords=words,
        hotword_pool_id=hotword_pool_id,
        user_id=user_id,
    )


async def delete_hotwords(
    words: list[str],
    *,
    hotword_pool_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, object]:
    return await asyncio.to_thread(
        _management_sync,
        "delete",
        hotwords=words,
        hotword_pool_id=hotword_pool_id,
        user_id=user_id,
    )


async def reload_hotword_pool(
    *,
    hotword_pool_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, object]:
    return await asyncio.to_thread(
        _management_sync,
        "reload",
        hotword_pool_id=hotword_pool_id,
        user_id=user_id,
    )
