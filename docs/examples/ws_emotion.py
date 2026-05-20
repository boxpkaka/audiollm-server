#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from urllib.parse import urlencode, urlsplit, urlunsplit

import websockets

from audio_common import chunk_bytes, make_ssl_context, read_audio_as_pcm


def with_language(url: str, language: str) -> str:
    if not language:
        return url
    parts = urlsplit(url)
    query = parts.query
    extra = urlencode({"language": language})
    query = f"{query}&{extra}" if query else extra
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


async def main_async(args: argparse.Namespace) -> None:
    url = with_language(args.url, args.language)
    pcm = read_audio_as_pcm(args.audio_file)
    chunk_size = chunk_bytes(args.chunk_ms)
    ssl_ctx = make_ssl_context(url, args.insecure)

    async with websockets.connect(url, ssl=ssl_ctx, open_timeout=args.timeout) as ws:
        ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=args.timeout))
        if ready.get("type") != "ready":
            raise RuntimeError(f"unexpected first message: {ready}")
        print("<- ready")

        start_msg: dict[str, object] = {
            "type": "start",
            "format": "pcm_s16le",
            "sample_rate_hz": 16000,
            "channels": 1,
        }
        if args.mode:
            start_msg["mode"] = args.mode
        if args.language:
            start_msg["language"] = args.language
        await ws.send(json.dumps(start_msg, ensure_ascii=False))
        print("-> start")

        recv_task = asyncio.create_task(receive_messages(ws, segmented=args.segmented))
        for offset in range(0, len(pcm), chunk_size):
            await ws.send(pcm[offset : offset + chunk_size])
            await asyncio.sleep(args.chunk_ms / 1000)

        await ws.send(json.dumps({"type": "stop"}))
        print("-> stop")

        try:
            await asyncio.wait_for(recv_task, timeout=args.final_timeout)
        except asyncio.TimeoutError:
            recv_task.cancel()
            print(f"[timeout] no more messages within {args.final_timeout}s")


async def receive_messages(ws, *, segmented: bool) -> None:
    final_count = 0
    async for raw in ws:
        try:
            msg = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            print(f"<- binary {len(raw)} bytes")
            continue

        msg_type = msg.get("type", "")
        if msg_type == "final_emotion":
            final_count += 1
            prefix = f"final #{final_count}" if segmented else "final"
            print(
                f"<- {prefix}: mode={msg.get('mode', '')} "
                f"label={msg.get('label', '')!r} text={msg.get('text', '')!r} "
                f"duration={msg.get('duration_sec', '')}"
            )
            if not segmented:
                return
        elif msg_type == "error":
            print(f"<- error: {json.dumps(msg, ensure_ascii=False)}")
            return
        else:
            print(f"<- {json.dumps(msg, ensure_ascii=False)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call emotion WebSocket APIs.")
    parser.add_argument("audio_file", help="WAV file or raw 16 kHz mono s16le PCM file")
    parser.add_argument("--url", required=True, help="WebSocket URL, for example wss://host:8443/emotion-streaming")
    parser.add_argument("--segmented", action="store_true", help="Use segmented receive semantics")
    parser.add_argument("--mode", choices=("ser", "sec"), default="ser", help="Emotion mode")
    parser.add_argument("--language", default="", help="Optional language passthrough")
    parser.add_argument("--chunk-ms", type=int, default=80, help="PCM chunk size in milliseconds")
    parser.add_argument("--timeout", type=float, default=10.0, help="Connection/read timeout in seconds")
    parser.add_argument("--final-timeout", type=float, default=60.0, help="Wait time after stop")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS certificate verification for self-signed certificates")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
