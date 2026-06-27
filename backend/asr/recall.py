"""Triton hotword recall client.

The deployed RAG-ASR Triton model owns a process-wide hotword pool.  This module
keeps audiollm-demo as a thin async gateway: it sends PCM to Triton, receives the
recalled top-K words plus vLLM-ready audio_embeds, and proxies pool management
operations.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from urllib.parse import urlparse

import numpy as np

from ..config import Config, Upstream, get_service_upstream

DEFAULT_RECALL_MODEL = "rag_asr_retrieve"
SAMPLE_RATE = 16000


@dataclass(frozen=True)
class RecallResult:
    words: list[str]
    audio_embeds_b64: str | None
    projector_len: int | None
    uuid: str


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


def _model_name(upstream: Upstream) -> str:
    return upstream.model_name or DEFAULT_RECALL_MODEL


def _infer_sync(
    pcm: np.ndarray,
    *,
    sample_rate: int,
    top_k: int,
    want_audio_embeds: bool,
) -> RecallResult:
    upstream = _recall_upstream()
    httpclient, client = _client_for(upstream)
    wav = np.asarray(pcm, dtype=np.float32).reshape(-1)
    inputs = [
        _string_input(httpclient, "ACTION", "infer"),
        httpclient.InferInput("WAV", wav.shape, "FP32"),
        _int_input(httpclient, "SAMPLE_RATE", sample_rate),
        _int_input(httpclient, "TOP_K", top_k),
    ]
    inputs[1].set_data_from_numpy(wav)

    outputs = [
        httpclient.InferRequestedOutput("WORD_LIST"),
        httpclient.InferRequestedOutput("PROJECTOR_LEN"),
    ]
    if want_audio_embeds:
        outputs.append(httpclient.InferRequestedOutput("AUDIO_EMBEDS_B64"))

    result = client.infer(_model_name(upstream), inputs, outputs=outputs)
    words = json.loads(_decode(result.as_numpy("WORD_LIST")[0]))
    projector_len = int(result.as_numpy("PROJECTOR_LEN")[0])
    audio_embeds_b64 = None
    if want_audio_embeds:
        audio_embeds_b64 = _decode(result.as_numpy("AUDIO_EMBEDS_B64")[0])
    return RecallResult(
        words=[str(word) for word in words],
        audio_embeds_b64=audio_embeds_b64,
        projector_len=projector_len,
        uuid=stable_audio_uuid(wav, sample_rate),
    )


async def recall_audio(
    pcm: np.ndarray,
    cfg: Config,
    *,
    sample_rate: int = SAMPLE_RATE,
    want_audio_embeds: bool = True,
) -> RecallResult:
    """Recall hotwords for one audio segment."""
    top_k = max(int(cfg.recall_top_k), 0)
    if top_k == 0:
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
    )


def _management_sync(
    action: str,
    *,
    hotwords: list[str] | None = None,
    query: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> dict[str, object]:
    upstream = _recall_upstream()
    httpclient, client = _client_for(upstream)
    inputs = [_string_input(httpclient, "ACTION", action)]
    if hotwords is not None:
        inputs.append(
            _string_input(
                httpclient,
                "HOTWORDS",
                json.dumps(hotwords, ensure_ascii=False),
            )
        )
    if query:
        inputs.append(_string_input(httpclient, "QUERY", query))
    if limit is not None:
        inputs.append(_int_input(httpclient, "LIMIT", limit))
    inputs.append(_int_input(httpclient, "OFFSET", offset))

    outputs = [
        httpclient.InferRequestedOutput("STATUS"),
        httpclient.InferRequestedOutput("MESSAGE"),
        httpclient.InferRequestedOutput("HOTWORD_COUNT"),
        httpclient.InferRequestedOutput("HOTWORD_LIST"),
    ]
    result = client.infer(_model_name(upstream), inputs, outputs=outputs)
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
) -> dict[str, object]:
    return await asyncio.to_thread(
        _management_sync,
        "list",
        query=query,
        limit=limit,
        offset=offset,
    )


async def add_hotwords(words: list[str]) -> dict[str, object]:
    return await asyncio.to_thread(_management_sync, "add", hotwords=words)


async def delete_hotwords(words: list[str]) -> dict[str, object]:
    return await asyncio.to_thread(_management_sync, "delete", hotwords=words)


async def reload_hotword_pool() -> dict[str, object]:
    return await asyncio.to_thread(_management_sync, "reload")
