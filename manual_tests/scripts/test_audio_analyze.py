#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import httpx


def post_audio_analyze(args: argparse.Namespace) -> dict:
    audio_path = Path(args.audio_file)
    if not audio_path.is_file():
        raise SystemExit(f"audio file not found: {audio_path}")

    url = args.base_url.rstrip("/") + "/api/audio/analyze"
    with audio_path.open("rb") as audio_file:
        response = httpx.post(
            url,
            files={"audio": (audio_path.name, audio_file, "audio/wav")},
            data={
                "language": args.language,
                "hotwords": args.hotwords,
            },
            timeout=args.timeout,
            verify=not args.insecure,
        )

    try:
        payload = response.json()
    except ValueError:
        raise SystemExit(
            f"HTTP {response.status_code} returned non-JSON response:\n"
            f"{response.text[:2000]}"
        ) from None

    if response.status_code >= 400:
        raise SystemExit(
            f"HTTP {response.status_code} {response.reason_phrase}\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )
    return payload


def save_output(payload: dict, output_dir: Path, audio_path: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = output_dir / f"{audio_path.stem}-{timestamp}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def print_summary(payload: dict, out_path: Path) -> None:
    asr = payload.get("asr") or {}
    cleaned = payload.get("cleaned_asr") or {}
    emotion = payload.get("emotion") or {}
    ser = emotion.get("ser") or {}
    sec = emotion.get("sec") or {}

    print("== Audio Analyze Result ==")
    print(f"type:          {payload.get('type')}")
    print(f"duration_sec:  {payload.get('duration_sec')}")
    print(f"language:      {payload.get('language')}")
    print(f"hotwords:      {payload.get('hotwords')}")
    print(f"asr_text:      {asr.get('text')}")
    print(f"cleaned_text:  {cleaned.get('text')}")
    print(f"cleanup_model: {cleaned.get('model')}")
    print(f"emotion_ser:   {ser.get('label') or ser.get('text')}")
    print(f"emotion_sec:   {sec.get('text')}")
    print(f"saved_to:      {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual test for POST /api/audio/analyze")
    parser.add_argument("audio_file", help="Path to a WAV audio file")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--hotwords", default="")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="manual_tests/outputs",
        help="Directory for full JSON responses",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = post_audio_analyze(args)
    out_path = save_output(payload, Path(args.output_dir), Path(args.audio_file))
    print_summary(payload, out_path)


if __name__ == "__main__":
    main()
