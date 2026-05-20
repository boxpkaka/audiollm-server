#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path

import websockets

from audio_common import chunk_bytes, make_ssl_context, read_audio_as_pcm


async def main_async(args: argparse.Namespace) -> None:
    mixed_pcm = read_audio_as_pcm(args.audio_file)
    enrollment_b64 = base64.b64encode(Path(args.enrollment).read_bytes()).decode("ascii")
    chunk_size = chunk_bytes(args.chunk_ms)
    ssl_ctx = make_ssl_context(args.url, args.insecure)
    hotwords = [item.strip() for item in args.hotwords.split(",") if item.strip()]

    async with websockets.connect(args.url, ssl=ssl_ctx, open_timeout=args.timeout) as ws:
        ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=args.timeout))
        if ready.get("type") != "ready":
            raise RuntimeError(f"unexpected first message: {ready}")
        print("<- ready")

        start_msg: dict[str, object] = {
            "type": "start",
            "format": "pcm_s16le",
            "sample_rate_hz": 16000,
            "channels": 1,
            "enrollment_audio": enrollment_b64,
            "enrollment_format": "wav",
        }
        if args.language:
            start_msg["language"] = args.language
        if hotwords:
            start_msg["hotwords"] = hotwords
        if args.enable_partial:
            start_msg["config"] = {"tsasr_enable_partial": True}

        await ws.send(json.dumps(start_msg, ensure_ascii=False))
        print("-> start")

        first = json.loads(await asyncio.wait_for(ws.recv(), timeout=args.timeout))
        if first.get("type") == "error":
            print(f"<- error: {json.dumps(first, ensure_ascii=False)}")
            return
        if first.get("type") != "enrollment_ok":
            raise RuntimeError(f"expected enrollment_ok, got: {first}")
        print(f"<- enrollment_ok duration={first.get('duration_sec', '')}")

        recv_task = asyncio.create_task(receive_messages(ws))
        for offset in range(0, len(mixed_pcm), chunk_size):
            await ws.send(mixed_pcm[offset : offset + chunk_size])
            await asyncio.sleep(args.chunk_ms / 1000)

        await ws.send(json.dumps({"type": "stop"}))
        print("-> stop")

        try:
            await asyncio.wait_for(recv_task, timeout=args.final_timeout)
        except asyncio.TimeoutError:
            recv_task.cancel()
            print(f"[timeout] no more messages within {args.final_timeout}s")


async def receive_messages(ws) -> None:
    async for raw in ws:
        try:
            msg = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            print(f"<- binary {len(raw)} bytes")
            continue

        msg_type = msg.get("type", "")
        if msg_type == "partial":
            print(f"<- partial: {msg.get('text', '')}")
        elif msg_type == "final":
            print(
                f"<- final: {msg.get('text', '')} "
                f"(secondary={msg.get('text_secondary', '')!r}, language={msg.get('language', '')})"
            )
        elif msg_type == "error":
            print(f"<- error: {json.dumps(msg, ensure_ascii=False)}")
            return
        else:
            print(f"<- {json.dumps(msg, ensure_ascii=False)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call /transcribe-target-streaming over WebSocket.")
    parser.add_argument("audio_file", help="Mixed-audio WAV file or raw 16 kHz mono s16le PCM file")
    parser.add_argument("--url", required=True, help="WebSocket URL, for example wss://host:8443/transcribe-target-streaming")
    parser.add_argument("--enrollment", required=True, help="Target speaker enrollment WAV file")
    parser.add_argument("--language", default="", help="Language code, for example zh/en/id/th")
    parser.add_argument("--hotwords", default="", help="Comma-separated hotwords")
    parser.add_argument("--enable-partial", action="store_true", help="Request TS-ASR partial messages")
    parser.add_argument("--chunk-ms", type=int, default=80, help="PCM chunk size in milliseconds")
    parser.add_argument("--timeout", type=float, default=10.0, help="Connection/read timeout in seconds")
    parser.add_argument("--final-timeout", type=float, default=60.0, help="Wait time after stop")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS certificate verification for self-signed certificates")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
