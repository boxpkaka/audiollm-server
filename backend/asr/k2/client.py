"""Shared grpc.aio client for the k2 streaming ASR service."""

from __future__ import annotations

import json
import logging
from typing import Any

import grpc

from ...config import SAMPLE_RATE, Config
from . import asr_pb2 as pb
from . import asr_pb2_grpc as pb_grpc

logger = logging.getLogger(__name__)

_channels: dict[str, grpc.aio.Channel] = {}

_GRPC_OPTIONS: tuple[tuple[str, int], ...] = (
    ("grpc.max_send_message_length", 8 * 1024 * 1024),
    ("grpc.max_receive_message_length", 8 * 1024 * 1024),
)


class K2UnavailableError(RuntimeError):
    """Raised when k2 is enabled but cannot be used safely."""


def get_k2_channel(target: str) -> grpc.aio.Channel:
    """Return a shared plaintext gRPC channel for ``host:port`` targets."""

    target = target.strip()
    if not target:
        raise K2UnavailableError("k2_target is empty")
    channel = _channels.get(target)
    if channel is None:
        channel = grpc.aio.insecure_channel(target, options=_GRPC_OPTIONS)
        _channels[target] = channel
    return channel


def get_k2_stub(target: str) -> pb_grpc.AsrServiceStub:
    return pb_grpc.AsrServiceStub(get_k2_channel(target))


async def close_k2_channels() -> None:
    """Close all shared k2 channels during FastAPI shutdown."""

    channels = list(_channels.values())
    _channels.clear()
    for channel in channels:
        await channel.close()


def _manifest_sample_rate(raw: str) -> int | None:
    if not raw:
        return None
    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("k2 ServerInfo returned non-JSON manifest: %.200s", raw)
        return None
    if not isinstance(data, dict):
        return None
    value = data.get("sample_rate")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


async def validate_k2_server(cfg: Config) -> None:
    """Fail fast when the k2 server cannot accept this service's PCM format."""

    if not cfg.k2_enabled:
        return
    stub = get_k2_stub(cfg.k2_target)
    try:
        info = await stub.ServerInfo(
            pb.ServerInfoRequest(),
            timeout=max(0.1, float(cfg.k2_connect_timeout_sec)),
        )
    except grpc.RpcError as exc:
        raise K2UnavailableError(f"k2 ServerInfo failed: {exc}") from exc

    expected = int(cfg.k2_sample_rate or SAMPLE_RATE)
    actual = _manifest_sample_rate(info.model_manifest_json)
    if actual is not None and actual != expected:
        raise K2UnavailableError(
            f"k2 sample_rate mismatch: server={actual}, expected={expected}"
        )
