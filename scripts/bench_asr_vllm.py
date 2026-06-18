#!/usr/bin/env python3
"""ASR vLLM 推理性能压测脚本（primary / Amphion 系列）。

用真实测试集音频直接压测 OpenAI 兼容的 ``/v1/chat/completions`` 端点，
测量端到端延迟、RTF（real-time factor，推理耗时 / 音频时长）、吞吐与错误率，
并按并发阶梯逐级加压。

为保证与线上完全一致：
  - 请求体复用 ``backend.asr.client.build_primary_messages``，并通过
    ``--prompt-template`` 显式选择模型对应模板。
  - 音频统一走 ``backend.audio.utils`` 的 ``decode -> 16 kHz mono -> WAV base64``
    管线（测试集是 24 kHz，线上同样会重采样到 16 kHz 再发）。
RTF 按每条请求单独计算（latency / 该条音频真实时长），再取分位数；
真实数据时长不一，按请求算才有意义。

用法示例
--------

本机 primary（config.yaml 默认 localhost:8009 / AmphionASR-1.7B）::

    python scripts/bench_asr_vllm.py \
        --base-url http://localhost:8009 --model AmphionASR-1.7B \
        --prompt-template amphion_asr_1.7b \
        --data-dir /home/ubuntu/data/testdata/base_v2_kespeech_gpu1 \
        --output bench_results/asr_local.json

公网 primary 同款服务::

    python scripts/bench_asr_vllm.py \
        --base-url http://159.138.9.106:8000 --model Amphion-4B \
        --data-dir /home/ubuntu/data/testdata/base_v2_kespeech_gpu1 \
        --output bench_results/asr_public.json

只跑串行单流时延（RTF 基线，不加压）::

    python scripts/bench_asr_vllm.py --concurrency-list 1 --requests-per-level 50
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

# Allow running as a standalone script without installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.asr.client import build_primary_messages, parse_model_output  # noqa: E402
from backend.asr.itn import normalize_final  # noqa: E402
from backend.audio.utils import (  # noqa: E402
    pcm_to_wav_base64,
    wav_base64_to_pcm_16k_mono,
)

SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Dataset loading (real WAVs decoded to 16 kHz mono, like production)
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    utt_id: str
    duration_s: float
    wav_b64: str  # 16 kHz mono WAV, base64 — ready to drop into the payload
    ref_text: str = ""


def _resolve_audio_path(data_dir: Path, audio_path: str) -> Path | None:
    """metadata 里的 audio_path 形如 ``<dataset>/wavs/zh/x.wav``，
    它相对的是 data_dir 的父目录；这里按几种常见布局做兜底解析。"""
    p = Path(audio_path)
    candidates = [
        data_dir.parent / p,  # audio_path 含 dataset 目录名（本数据集如此）
        data_dir / p,
        data_dir / p.name,
        data_dir / "wavs" / p.name,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _wav_file_to_sample(
    wav_path: Path, utt_id: str, ref_text: str
) -> Sample | None:
    raw = wav_path.read_bytes()
    if not raw:
        return None
    src_b64 = base64.b64encode(raw).decode("ascii")
    # 复用线上解码/重采样：任意采样率/位深 -> 16 kHz mono float32 PCM。
    pcm = wav_base64_to_pcm_16k_mono(src_b64)
    if pcm.size == 0:
        return None
    duration_s = pcm.size / SAMPLE_RATE
    wav_b64 = pcm_to_wav_base64(pcm, SAMPLE_RATE)
    return Sample(
        utt_id=utt_id, duration_s=duration_s, wav_b64=wav_b64, ref_text=ref_text
    )


def load_dataset(
    data_dir: Path,
    metadata: Path | None,
    *,
    limit: int,
    lang_filter: str,
    shuffle: bool,
    seed: int,
) -> list[Sample]:
    rows: list[dict[str, Any]] = []
    if metadata and metadata.is_file():
        for line in metadata.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    else:
        # 没有 metadata：直接 glob 所有 wav。
        for wav in sorted(data_dir.rglob("*.wav")):
            rows.append({"utt_id": wav.stem, "audio_path": str(wav), "text": ""})

    if lang_filter:
        rows = [r for r in rows if str(r.get("lang", "")) == lang_filter]

    if shuffle:
        random.Random(seed).shuffle(rows)

    samples: list[Sample] = []
    n_missing = 0
    n_bad = 0
    for r in rows:
        if limit > 0 and len(samples) >= limit:
            break
        audio_path = r.get("audio_path") or r.get("wav") or ""
        if not audio_path:
            n_missing += 1
            continue
        resolved = _resolve_audio_path(data_dir, str(audio_path))
        if resolved is None:
            n_missing += 1
            continue
        try:
            s = _wav_file_to_sample(
                resolved, str(r.get("utt_id") or resolved.stem), str(r.get("text") or "")
            )
        except Exception:  # noqa: BLE001
            s = None
        if s is None:
            n_bad += 1
            continue
        samples.append(s)

    if n_missing or n_bad:
        print(f"  [load] skipped: missing={n_missing} undecodable={n_bad}")
    return samples


# ---------------------------------------------------------------------------
# Request construction (payload identical to backend/asr/client.py)
# ---------------------------------------------------------------------------


def build_payload(
    sample: Sample,
    *,
    model: str,
    prompt_template: str,
    max_tokens: int,
    hotwords: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": build_primary_messages(
            sample.wav_b64,
            hotwords=hotwords,
            template=prompt_template,
        ),
        "max_tokens": int(max_tokens),
    }


# ---------------------------------------------------------------------------
# Per-request execution
# ---------------------------------------------------------------------------


@dataclass
class RequestResult:
    ok: bool
    latency_s: float
    audio_s: float
    status: int = 0
    error: str = ""
    completion_tokens: int = 0
    prompt_tokens: int = 0
    pred_text: str = ""
    ref_text: str = ""

    @property
    def rtf(self) -> float:
        return self.latency_s / self.audio_s if self.audio_s > 0 else float("nan")


def _extract_text(data: dict[str, Any]) -> str:
    choice = (data.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content")
    if isinstance(content, list):
        return "".join(
            (c.get("text") or "") for c in content if isinstance(c, dict)
        )
    return str(content or "")


async def do_request(
    client: httpx.AsyncClient,
    url: str,
    sample: Sample,
    payload: dict[str, Any],
    timeout: float,
) -> RequestResult:
    t0 = time.perf_counter()
    try:
        r = await client.post(url, json=payload, timeout=timeout)
        elapsed = time.perf_counter() - t0
        if r.status_code != 200:
            return RequestResult(
                ok=False,
                latency_s=elapsed,
                audio_s=sample.duration_s,
                status=r.status_code,
                error=r.text[:200],
            )
        data = r.json()
        usage = data.get("usage") or {}
        return RequestResult(
            ok=True,
            latency_s=elapsed,
            audio_s=sample.duration_s,
            status=200,
            completion_tokens=int(usage.get("completion_tokens") or 0),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            pred_text=_extract_text(data).strip(),
            ref_text=sample.ref_text,
        )
    except Exception as exc:  # noqa: BLE001
        return RequestResult(
            ok=False,
            latency_s=time.perf_counter() - t0,
            audio_s=sample.duration_s,
            status=0,
            error=f"{type(exc).__name__}: {exc}"[:200],
        )


# ---------------------------------------------------------------------------
# Accuracy (optional, rough CER for sanity / parity checks)
# ---------------------------------------------------------------------------


def _normalize_for_cer(s: str) -> str:
    out = []
    for ch in s:
        if ch.isspace():
            continue
        if ch in "，。！？、；：,.!?;:\"'“”‘’（）()【】[]…—-":
            continue
        out.append(ch)
    return "".join(out)


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(
                min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            )
        prev = cur
    return prev[-1]


def _cer(ref: str, hyp: str) -> tuple[int, int]:
    """返回 (edit_distance, ref_len)，便于跨样本聚合成语料级 CER。"""
    r = _normalize_for_cer(ref)
    h = _normalize_for_cer(hyp)
    if not r:
        return (0, 0)
    return (_edit_distance(r, h), len(r))


# ---------------------------------------------------------------------------
# Concurrency level runner
# ---------------------------------------------------------------------------


@dataclass
class LevelStats:
    concurrency: int
    n_total: int
    n_ok: int
    n_err: int
    wall_s: float
    audio_s_mean: float
    lat_p50: float
    lat_p90: float
    lat_p99: float
    lat_avg: float
    rtf_p50: float
    rtf_p90: float
    rtf_p99: float
    rtf_avg: float
    req_per_s: float
    audio_s_per_s: float
    out_tok_per_s: float
    cer: float = float("nan")
    sample_pred: str = ""
    sample_ref: str = ""
    raw_latencies: list[float] = field(default_factory=list)
    raw_rtfs: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * (p / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return xs_sorted[lo]
    return xs_sorted[lo] + (xs_sorted[hi] - xs_sorted[lo]) * (k - lo)


def _assign_samples(
    samples: list[Sample], n: int, *, shuffle: bool, rng: random.Random
) -> list[Sample]:
    """为本级取 n 条样本。真实音频各不相同，天然绕过 vLLM prefix cache。
    数据不够就循环复用。"""
    pool = samples[:]
    if shuffle:
        rng.shuffle(pool)
    if n <= len(pool):
        return pool[:n]
    out: list[Sample] = []
    while len(out) < n:
        out.extend(pool)
    return out[:n]


async def run_level(
    *,
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompt_template: str,
    max_tokens: int,
    samples: list[Sample],
    concurrency: int,
    n_requests: int,
    request_timeout: float,
    measure_cer: bool,
    shuffle: bool,
    rng: random.Random,
    apply_itn: bool = False,
    hotwords: list[str] | None = None,
) -> LevelStats:
    jobs = _assign_samples(samples, n_requests, shuffle=shuffle, rng=rng)
    sem = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []

    async def _one(sample: Sample) -> None:
        async with sem:
            payload = build_payload(
                sample,
                model=model,
                prompt_template=prompt_template,
                max_tokens=max_tokens,
                hotwords=hotwords,
            )
            res = await do_request(client, url, sample, payload, request_timeout)
            results.append(res)

    t0 = time.perf_counter()
    await asyncio.gather(*(_one(s) for s in jobs))
    wall = time.perf_counter() - t0

    ok = [r for r in results if r.ok]
    err = [r for r in results if not r.ok]
    lats = [r.latency_s for r in ok]
    rtfs = [r.rtf for r in ok]
    audio_total = sum(r.audio_s for r in ok)
    out_tokens = sum(r.completion_tokens for r in ok)

    # Score the same text the backend would expose: provider wrappers are
    # stripped first (for example Qwen3-ASR's language <Lang><asr_text> prefix),
    # then optional ITN mirrors final-display normalization.
    def _scored(pred: str) -> str:
        parsed = parse_model_output(pred)["transcription"]
        return normalize_final(parsed, "zh") if apply_itn else parsed

    cer = float("nan")
    if measure_cer and ok:
        tot_d = 0
        tot_r = 0
        for r in ok:
            d, rl = _cer(r.ref_text, _scored(r.pred_text))
            tot_d += d
            tot_r += rl
        cer = (tot_d / tot_r) if tot_r > 0 else float("nan")

    return LevelStats(
        concurrency=concurrency,
        n_total=len(results),
        n_ok=len(ok),
        n_err=len(err),
        wall_s=wall,
        audio_s_mean=(audio_total / len(ok)) if ok else float("nan"),
        lat_p50=_percentile(lats, 50),
        lat_p90=_percentile(lats, 90),
        lat_p99=_percentile(lats, 99),
        lat_avg=(statistics.fmean(lats) if lats else float("nan")),
        rtf_p50=_percentile(rtfs, 50),
        rtf_p90=_percentile(rtfs, 90),
        rtf_p99=_percentile(rtfs, 99),
        rtf_avg=(statistics.fmean(rtfs) if rtfs else float("nan")),
        req_per_s=(len(ok) / wall) if wall > 0 else 0.0,
        audio_s_per_s=(audio_total / wall) if wall > 0 else 0.0,
        out_tok_per_s=(out_tokens / wall) if wall > 0 else 0.0,
        cer=cer,
        sample_pred=(_scored(ok[0].pred_text)[:80] if ok else ""),
        sample_ref=(ok[0].ref_text[:80] if ok else ""),
        raw_latencies=lats,
        raw_rtfs=rtfs,
        errors=[r.error for r in err][:5],
    )


async def warmup(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompt_template: str,
    max_tokens: int,
    samples: list[Sample],
    n: int,
    request_timeout: float,
    hotwords: list[str] | None = None,
) -> tuple[int, int]:
    if n <= 0 or not samples:
        return 0, 0
    picks = [samples[i % len(samples)] for i in range(n)]
    results = await asyncio.gather(
        *(
            do_request(
                client,
                url,
                s,
                build_payload(
                    s,
                    model=model,
                    prompt_template=prompt_template,
                    max_tokens=max_tokens,
                    hotwords=hotwords,
                ),
                request_timeout,
            )
            for s in picks
        )
    )
    ok = sum(1 for r in results if r.ok)
    return ok, len(results) - ok


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _auto_n_requests(concurrency: int, override: int, n_samples: int) -> int:
    if override > 0:
        return override
    return min(200, max(20, 5 * concurrency))


def _format_table(stats: list[LevelStats], measure_cer: bool) -> str:
    cols = (
        f"{'conc':>4}  {'n':>4}  {'ok':>4}  {'err':>4}  {'aud_s':>6}  "
        f"{'lat_p50':>8}  {'lat_p90':>8}  {'lat_p99':>8}  "
        f"{'rtf_p50':>8}  {'rtf_p90':>8}  {'rtf_p99':>8}  "
        f"{'req/s':>7}  {'xRT':>7}  {'tok/s':>7}"
    )
    if measure_cer:
        cols += f"  {'cer%':>6}"
    sep = "-" * len(cols)
    lines = [sep, cols, sep]
    for s in stats:
        row = (
            f"{s.concurrency:>4d}  {s.n_total:>4d}  {s.n_ok:>4d}  {s.n_err:>4d}  "
            f"{s.audio_s_mean:>6.2f}  "
            f"{s.lat_p50:>8.3f}  {s.lat_p90:>8.3f}  {s.lat_p99:>8.3f}  "
            f"{s.rtf_p50:>8.3f}  {s.rtf_p90:>8.3f}  {s.rtf_p99:>8.3f}  "
            f"{s.req_per_s:>7.2f}  {s.audio_s_per_s:>7.2f}  {s.out_tok_per_s:>7.1f}"
        )
        if measure_cer:
            row += f"  {s.cer * 100:>6.2f}" if not math.isnan(s.cer) else f"  {'-':>6}"
        lines.append(row)
    lines.append(sep)
    return "\n".join(lines)


async def main_async(args: argparse.Namespace) -> int:
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    data_dir = Path(args.data_dir).expanduser().resolve()
    metadata = (
        Path(args.metadata).expanduser().resolve()
        if args.metadata
        else (data_dir / "metadata.jsonl")
    )

    print(f"Target:    {url}")
    print(f"Model:     {args.model}")
    print(f"Prompt:    {args.prompt_template}")
    print(f"Data dir:  {data_dir}")
    print(f"Metadata:  {metadata if metadata.is_file() else '(glob *.wav)'}")
    print("Loading + decoding audio to 16 kHz mono ...")

    samples = load_dataset(
        data_dir,
        metadata,
        limit=args.limit,
        lang_filter=args.lang_filter,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    if not samples:
        print("No usable audio samples found.", file=sys.stderr)
        return 2

    durs = [s.duration_s for s in samples]
    print(
        f"Loaded:    {len(samples)} clips  "
        f"total={sum(durs):.1f}s  "
        f"dur min/med/max={min(durs):.2f}/{statistics.median(durs):.2f}/{max(durs):.2f}s"
    )

    if args.concurrency_list:
        ladder = [int(x) for x in args.concurrency_list.split(",") if x.strip()]
    else:
        ladder = [1, 2, 4, 8, 16, 32, 64]
    ladder = [c for c in ladder if c <= args.max_concurrency]

    print(f"Ladder:    {ladder}")
    stop_msg = []
    if args.stop_rtf > 0:
        stop_msg.append(f"rtf_p50 > {args.stop_rtf}")
    if args.stop_error_rate > 0:
        stop_msg.append(f"error_rate > {args.stop_error_rate}")
    print(f"Stop when: {' OR '.join(stop_msg) if stop_msg else '(never, full ladder)'}")
    hotwords = [w.strip() for w in args.hotwords.split(",") if w.strip()]
    print(
        f"max_tokens={args.max_tokens}  cer={'on' if args.measure_cer else 'off'}  "
        f"apply_itn={'on' if args.apply_itn else 'off'}  "
        f"hotwords={hotwords or '(none)'}"
    )
    print()

    rng = random.Random(args.seed)

    limits = httpx.Limits(
        max_connections=max(ladder) * 2 + 8,
        max_keepalive_connections=max(ladder) * 2 + 8,
    )
    timeout = httpx.Timeout(args.request_timeout, connect=10.0)

    all_stats: list[LevelStats] = []

    async with httpx.AsyncClient(limits=limits, timeout=timeout, http2=False) as client:
        ok, err = await warmup(
            client,
            url,
            args.model,
            args.prompt_template,
            args.max_tokens,
            samples,
            args.warmup,
            args.request_timeout,
            hotwords=hotwords,
        )
        print(f"warmup: ok={ok} err={err}")
        if args.warmup > 0 and ok == 0:
            print(
                "  [abort] warmup got 0 successes — endpoint unreachable or "
                "model name wrong; not ramping.",
                file=sys.stderr,
            )
            # 仍打印一条便于看错误信息。
            probe = await run_level(
                client=client, url=url, model=args.model, max_tokens=args.max_tokens,
                prompt_template=args.prompt_template,
                samples=samples, concurrency=1, n_requests=1,
                request_timeout=args.request_timeout, measure_cer=False,
                shuffle=False, rng=rng,
            )
            if probe.errors:
                print(f"  first error: {probe.errors[0]}", file=sys.stderr)
            return 3
        print()

        for c in ladder:
            n = _auto_n_requests(c, args.requests_per_level, len(samples))
            print(f"  -> concurrency={c:<4d}  n={n:<4d}  running ...", end="", flush=True)
            stats = await run_level(
                client=client,
                url=url,
                model=args.model,
                prompt_template=args.prompt_template,
                max_tokens=args.max_tokens,
                samples=samples,
                concurrency=c,
                n_requests=n,
                request_timeout=args.request_timeout,
                measure_cer=args.measure_cer,
                shuffle=args.shuffle,
                rng=rng,
                apply_itn=args.apply_itn,
                hotwords=hotwords,
            )
            all_stats.append(stats)
            err_rate = stats.n_err / max(1, stats.n_total)
            print(
                f"  lat_p50={stats.lat_p50:.3f}s  rtf_p50={stats.rtf_p50:.3f}  "
                f"xRT={stats.audio_s_per_s:.2f}  req/s={stats.req_per_s:.2f}  "
                f"err={stats.n_err}"
            )
            if stats.sample_pred:
                print(f"     pred: {stats.sample_pred!r}")
                if args.measure_cer and stats.sample_ref:
                    print(f"     ref : {stats.sample_ref!r}")
            if stats.errors:
                print(f"     first error: {stats.errors[0]}")

            if args.stop_error_rate > 0 and err_rate > args.stop_error_rate:
                print(f"  [stop] error_rate={err_rate:.1%} > {args.stop_error_rate:.1%}")
                break
            if (
                args.stop_rtf > 0
                and not math.isnan(stats.rtf_p50)
                and stats.rtf_p50 > args.stop_rtf
            ):
                print(f"  [stop] rtf_p50={stats.rtf_p50:.2f} > {args.stop_rtf:.2f}")
                break

    print()
    print(_format_table(all_stats, args.measure_cer))
    print()
    print("Legend: rtf=latency/audio (per-request, <1 faster than realtime); "
          "xRT=audio_s/s (aggregate throughput); req/s=requests/s; tok/s=output tokens/s")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "base_url": args.base_url,
                "model": args.model,
                "prompt_template": args.prompt_template,
                "data_dir": str(data_dir),
                "n_samples": len(samples),
                "ladder": ladder,
                "warmup": args.warmup,
                "requests_per_level": args.requests_per_level,
                "max_tokens": args.max_tokens,
                "request_timeout": args.request_timeout,
                "shuffle": args.shuffle,
                "seed": args.seed,
                "lang_filter": args.lang_filter,
                "limit": args.limit,
                "measure_cer": args.measure_cer,
                "apply_itn": args.apply_itn,
                "hotwords": hotwords,
                "audio_total_s": sum(durs),
            },
            "results": [
                {
                    "concurrency": s.concurrency,
                    "n_total": s.n_total,
                    "n_ok": s.n_ok,
                    "n_err": s.n_err,
                    "wall_s": s.wall_s,
                    "audio_s_mean": s.audio_s_mean,
                    "lat_avg": s.lat_avg,
                    "lat_p50": s.lat_p50,
                    "lat_p90": s.lat_p90,
                    "lat_p99": s.lat_p99,
                    "rtf_avg": s.rtf_avg,
                    "rtf_p50": s.rtf_p50,
                    "rtf_p90": s.rtf_p90,
                    "rtf_p99": s.rtf_p99,
                    "req_per_s": s.req_per_s,
                    "audio_s_per_s": s.audio_s_per_s,
                    "out_tok_per_s": s.out_tok_per_s,
                    "cer": s.cer,
                    "sample_pred": s.sample_pred,
                    "sample_ref": s.sample_ref,
                    "raw_latencies": s.raw_latencies,
                    "raw_rtfs": s.raw_rtfs,
                    "errors": s.errors,
                }
                for s in all_stats
            ],
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"Saved JSON results -> {out_path}")

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ASR vLLM 推理性能压测（OpenAI 兼容 /v1/chat/completions，真实测试集音频）。"
    )
    p.add_argument("--base-url", default="http://localhost:8009")
    p.add_argument("--model", default="AmphionASR-1.7B")
    p.add_argument(
        "--prompt-template",
        default="amphion_asr_1.7b",
        choices=["amphion_asr", "amphion_asr_1.7b"],
        help="主 ASR prompt 模板；4.3B 用 amphion_asr，1.7B 用 amphion_asr_1.7b。",
    )
    p.add_argument(
        "--data-dir",
        default="/home/ubuntu/data/testdata/base_v2_kespeech_gpu1",
        help="测试集根目录（含 metadata.jsonl 与 wavs/）。",
    )
    p.add_argument(
        "--metadata",
        default="",
        help="metadata.jsonl 路径（默认 <data-dir>/metadata.jsonl；缺失则 glob *.wav）。",
    )
    p.add_argument(
        "--concurrency-list",
        default="",
        help="并发阶梯，逗号分隔。默认 1,2,4,8,16,32,64。",
    )
    p.add_argument("--max-concurrency", type=int, default=256)
    p.add_argument(
        "--requests-per-level",
        type=int,
        default=0,
        help="每级请求数（0=自动：max(20,5*c) 上限 200）。",
    )
    p.add_argument("--warmup", type=int, default=3, help="正式计时前的预热请求数。")
    p.add_argument("--limit", type=int, default=0, help="最多加载多少条音频（0=全部）。")
    p.add_argument("--lang-filter", default="", help="按 metadata 的 lang 字段过滤，如 zh。")
    p.add_argument("--shuffle", action="store_true", help="打乱样本顺序。")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=512, help="与线上一致，默认 512。")
    p.add_argument("--request-timeout", type=float, default=120.0)
    p.add_argument(
        "--stop-rtf",
        type=float,
        default=0.0,
        help="rtf_p50 超过此值即停止加压（0=不启用，跑完整阶梯）。",
    )
    p.add_argument(
        "--stop-error-rate",
        type=float,
        default=0.2,
        help="单级错误率超过此比例即停止（0=不启用）。",
    )
    p.add_argument(
        "--measure-cer",
        action="store_true",
        help="顺带计算粗略 CER（与 ground-truth 对比，用于正确性/同模型核验）。",
    )
    p.add_argument(
        "--apply-itn",
        action="store_true",
        help=(
            "计 CER 前对预测套用 final 的 ITN+车牌规范化；"
            "与不加该开关对比即可量化 ITN 对 CER 的影响。"
        ),
    )
    p.add_argument(
        "--hotwords",
        default="",
        help="逗号分隔热词，注入 primary prompt 的 Hotwords 行，如 冀R,辽B。",
    )
    p.add_argument("--output", default="", help="结果 JSON 落盘路径（含原始延迟/RTF）。")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        rc = asyncio.run(main_async(args))
    except KeyboardInterrupt:
        rc = 130
    sys.exit(rc)


if __name__ == "__main__":
    main()
