from __future__ import annotations

import ssl
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
BYTES_PER_MS = SAMPLE_RATE * 2 // 1000


def make_ssl_context(url: str, insecure: bool) -> ssl.SSLContext | None:
    if not url.startswith("wss://"):
        return None
    if not insecure:
        return ssl.create_default_context()
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def join_url(base_url: str, path: str) -> str:
    """Join a base URL and a path without doubling or dropping slashes."""
    return base_url.rstrip("/") + "/" + path.lstrip("/")


def read_audio_as_pcm(path: str) -> bytes:
    suffix = Path(path).suffix.lower()
    if suffix in {".pcm", ".raw"}:
        return Path(path).read_bytes()
    return read_wav_as_pcm(path)


def read_wav_as_pcm(path: str) -> bytes:
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        frames = wf.getnframes()
        raw = wf.readframes(frames)

    if sample_width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width: {sample_width} bytes")

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    if framerate != SAMPLE_RATE:
        samples = resample_linear(samples, framerate, SAMPLE_RATE)

    pcm = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    return pcm.tobytes()


def resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if samples.size == 0:
        return samples.astype(np.float32)
    dst_len = int(round(samples.size * dst_rate / src_rate))
    if dst_len <= 0:
        return np.array([], dtype=np.float32)
    src_x = np.linspace(0, samples.size - 1, num=samples.size)
    dst_x = np.linspace(0, samples.size - 1, num=dst_len)
    return np.interp(dst_x, src_x, samples).astype(np.float32)


def chunk_bytes(chunk_ms: int) -> int:
    return BYTES_PER_MS * chunk_ms
