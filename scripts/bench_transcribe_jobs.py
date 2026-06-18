#!/usr/bin/env python3
"""Benchmark POST /api/asr/transcriptions (offline long-audio jobs).

For each scenario this script writes a temporary config.yaml with the
scenario's ``defaults.transcribe`` overrides, boots a dedicated uvicorn via
CONFIG_PATH (the transcribe_* knobs are process-level), submits N concurrent
jobs of the same audio file, polls them to completion and records latency /
throughput / segmentation stats. Results land in bench_results/ as JSON.

Usage:
    python scripts/bench_transcribe_jobs.py /tmp/meeting_16k.wav \
        --port 18124 --label h20_meeting32min
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

ROOT = Path(__file__).resolve().parent.parent
POLL_INTERVAL_S = 0.5
SERVER_READY_TIMEOUT_S = 40.0


# ---------------------------------------------------------------------------
# Scenario matrix
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    name: str
    overrides: dict[str, Any]
    jobs: int = 1
    record: bool = True


def build_scenarios() -> list[Scenario]:
    base = {"transcribe_silence_duration_ms": 800}

    def sc(name: str, jobs: int = 1, record: bool = True, **kw: Any) -> Scenario:
        ov = {**base, **kw}
        # Keep the queue out of the way: capacity protection is not under test.
        ov.setdefault("transcribe_job_queue_max", max(16, jobs + 2))
        return Scenario(name=name, overrides=ov, jobs=jobs, record=record)

    return [
        # vLLM warm-up (KV cache, CUDA graphs, connection pool); discarded.
        sc("warmup", record=False, transcribe_segment_concurrency=4),
        # A: single-job segment concurrency sweep.
        sc("seg1", transcribe_segment_concurrency=1),
        sc("seg2", transcribe_segment_concurrency=2),
        sc("seg4", transcribe_segment_concurrency=4),
        sc("seg4_rep", transcribe_segment_concurrency=4),  # variance check
        sc("seg8", transcribe_segment_concurrency=8),
        sc("seg16", transcribe_segment_concurrency=16),
        # B: cut-pause sweep at fixed seg_conc=4.
        sc("sil350", transcribe_segment_concurrency=4,
           transcribe_silence_duration_ms=350),
        sc("sil600", transcribe_segment_concurrency=4,
           transcribe_silence_duration_ms=600),
        sc("sil1200", transcribe_segment_concurrency=4,
           transcribe_silence_duration_ms=1200),
        # C: multi-job scaling (product = total vLLM pressure).
        sc("j2_c2_seg4", jobs=2, transcribe_max_concurrent_jobs=2,
           transcribe_segment_concurrency=4),
        sc("j4_c2_seg4", jobs=4, transcribe_max_concurrent_jobs=2,
           transcribe_segment_concurrency=4),
        sc("j4_c4_seg4", jobs=4, transcribe_max_concurrent_jobs=4,
           transcribe_segment_concurrency=4),
        sc("j2_c2_seg8", jobs=2, transcribe_max_concurrent_jobs=2,
           transcribe_segment_concurrency=8),
        sc("j8_c2_seg4", jobs=8, transcribe_max_concurrent_jobs=2,
           transcribe_segment_concurrency=4),
        sc("j8_c4_seg4", jobs=8, transcribe_max_concurrent_jobs=4,
           transcribe_segment_concurrency=4),
        # D: force-cut ceiling.
        sc("maxseg15", transcribe_segment_concurrency=4,
           transcribe_max_segment_sec=15.0),
    ]


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def write_scenario_config(base_config: Path, overrides: dict[str, Any],
                          dest: Path) -> None:
    doc = yaml.safe_load(base_config.read_text())
    transcribe = doc["defaults"]["transcribe"]
    for key, value in overrides.items():
        transcribe[key] = value
    dest.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False))


async def wait_ready(client: httpx.AsyncClient, base: str) -> None:
    deadline = time.monotonic() + SERVER_READY_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            r = await client.get(f"{base}/openapi.json")
            if r.status_code == 200:
                return
        except httpx.TransportError:
            pass
        await asyncio.sleep(0.5)
    raise RuntimeError("server did not become ready")


# ---------------------------------------------------------------------------
# Job driving
# ---------------------------------------------------------------------------

@dataclass
class JobStats:
    ok: bool
    upload_s: float
    first_running_s: float | None
    e2e_s: float
    segments_total: int | None = None
    failed_segments: int | None = None
    full_text_chars: int | None = None
    seg_dur_p50_s: float | None = None
    seg_dur_p90_s: float | None = None
    seg_dur_max_s: float | None = None
    error: str = ""


def _percentile(sorted_vals: list[float], q: float) -> float:
    idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * q))
    return sorted_vals[idx]


async def run_job(client: httpx.AsyncClient, base: str, wav: bytes,
                  language: str) -> JobStats:
    t0 = time.monotonic()
    resp = await client.post(
        f"{base}/api/asr/transcriptions",
        files={"audio": ("bench.wav", wav, "audio/wav")},
        data={"language": language},
    )
    upload_s = time.monotonic() - t0
    if resp.status_code != 202:
        return JobStats(ok=False, upload_s=upload_s, first_running_s=None,
                        e2e_s=time.monotonic() - t0,
                        error=f"submit {resp.status_code}: {resp.text[:200]}")
    poll_url = resp.json()["poll_url"]

    first_running: float | None = None
    while True:
        await asyncio.sleep(POLL_INTERVAL_S)
        r = await client.get(f"{base}{poll_url}")
        if r.status_code != 200:
            return JobStats(ok=False, upload_s=upload_s,
                            first_running_s=first_running,
                            e2e_s=time.monotonic() - t0,
                            error=f"poll {r.status_code}")
        body = r.json()
        status = body["status"]
        if status == "running" and first_running is None:
            first_running = time.monotonic() - t0
        if status in ("succeeded", "failed"):
            e2e = time.monotonic() - t0
            if status == "failed":
                return JobStats(ok=False, upload_s=upload_s,
                                first_running_s=first_running, e2e_s=e2e,
                                error=json.dumps(body.get("error"))[:200])
            result = body["result"]
            durs = sorted(
                (s["end_ms"] - s["start_ms"]) / 1000.0
                for s in result["segments"]
            )
            return JobStats(
                ok=True, upload_s=upload_s, first_running_s=first_running,
                e2e_s=e2e,
                segments_total=body["progress"]["segments_total"],
                failed_segments=result["failed_segments"],
                full_text_chars=len(result["full_text"]),
                seg_dur_p50_s=round(_percentile(durs, 0.5), 2) if durs else None,
                seg_dur_p90_s=round(_percentile(durs, 0.9), 2) if durs else None,
                seg_dur_max_s=round(durs[-1], 2) if durs else None,
            )


@dataclass
class ScenarioResult:
    name: str
    overrides: dict[str, Any]
    jobs: int
    audio_sec: float
    batch_wall_s: float = 0.0
    throughput_x: float = 0.0  # jobs * audio_sec / batch_wall_s
    job_stats: list[dict[str, Any]] = field(default_factory=list)


async def run_scenario(scenario: Scenario, *, port: int, wav: bytes,
                       audio_sec: float, language: str,
                       base_config: Path, server_log: Path) -> ScenarioResult:
    base = f"http://127.0.0.1:{port}"
    cfg_path = Path(f"/tmp/bench_transcribe_cfg_{scenario.name}.yaml")
    write_scenario_config(base_config, scenario.overrides, cfg_path)

    if not port_free(port):
        raise RuntimeError(f"port {port} already in use; stale server?")

    env = {**os.environ, "CONFIG_PATH": str(cfg_path)}
    with server_log.open("ab") as log_fp:
        log_fp.write(f"\n===== scenario {scenario.name} =====\n".encode())
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "backend.main:app",
             "--host", "127.0.0.1", "--port", str(port)],
            cwd=ROOT, env=env, stdout=log_fp, stderr=subprocess.STDOUT,
        )
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, read=120.0)
        ) as client:
            await wait_ready(client, base)
            t0 = time.monotonic()
            stats = await asyncio.gather(*[
                run_job(client, base, wav, language)
                for _ in range(scenario.jobs)
            ])
            wall = time.monotonic() - t0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

    ok_jobs = sum(1 for s in stats if s.ok)
    throughput = (ok_jobs * audio_sec / wall) if wall > 0 else 0.0
    return ScenarioResult(
        name=scenario.name, overrides=scenario.overrides, jobs=scenario.jobs,
        audio_sec=audio_sec, batch_wall_s=round(wall, 2),
        throughput_x=round(throughput, 1),
        job_stats=[asdict(s) for s in stats],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def audio_duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def summary_line(res: ScenarioResult) -> str:
    ok = [s for s in res.job_stats if s["ok"]]
    if not ok:
        return (f"{res.name:<14} jobs={res.jobs} FAILED "
                f"({res.job_stats[0]['error']})")
    e2e = sorted(s["e2e_s"] for s in ok)
    segs = ok[0]["segments_total"]
    failed = sum(s["failed_segments"] or 0 for s in ok)
    return (f"{res.name:<14} jobs={res.jobs} wall={res.batch_wall_s:7.1f}s "
            f"e2e[{e2e[0]:6.1f}..{e2e[-1]:6.1f}]s thpt={res.throughput_x:6.1f}x "
            f"segs={segs} failed={failed} p50={ok[0]['seg_dur_p50_s']}s")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="16k mono WAV test file")
    parser.add_argument("--port", type=int, default=18124)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--base-config", type=Path,
                        default=ROOT / "config.yaml")
    parser.add_argument("--label", default="bench")
    parser.add_argument("--only", default="",
                        help="comma-separated scenario names to run")
    args = parser.parse_args()

    wav = args.audio.read_bytes()
    audio_sec = audio_duration_sec(args.audio)
    scenarios = build_scenarios()
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        scenarios = [s for s in scenarios if s.name in wanted]

    server_log = Path(f"/tmp/bench_transcribe_server_{args.label}.log")
    results: list[ScenarioResult] = []
    for scenario in scenarios:
        print(f"--- running {scenario.name} (jobs={scenario.jobs}, "
              f"{scenario.overrides}) ...", flush=True)
        res = await run_scenario(
            scenario, port=args.port, wav=wav, audio_sec=audio_sec,
            language=args.language, base_config=args.base_config,
            server_log=server_log,
        )
        print("    " + summary_line(res), flush=True)
        if scenario.record:
            results.append(res)

    out = ROOT / "bench_results" / (
        f"transcribe_jobs_{args.label}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    out.write_text(json.dumps({
        "audio": str(args.audio),
        "audio_sec": audio_sec,
        "language": args.language,
        "poll_interval_s": POLL_INTERVAL_S,
        "scenarios": [asdict(r) for r in results],
    }, ensure_ascii=False, indent=2))
    print(f"\nresults -> {out}")
    print(f"server log -> {server_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
