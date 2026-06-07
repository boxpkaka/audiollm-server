#!/usr/bin/env python3
"""/tuling/ast/v3 首字返回延迟（TTFT）基准。

用真实测试集音频，针对 tuling AST v3 WebSocket 接口测量"首字返回延迟"：
从客户端开始发送音频，到服务端返回第一个非空文本字符之间的墙钟时间。

与 scripts/bench_asr_vllm.py 的区别：后者直压 vLLM 的 HTTP /v1/chat/completions，
不含 WebSocket 协议封装、VAD 切段与伪流式（pseudo_stream）调度；本脚本走完整
/tuling/ast/v3 流水线，测到的才是集成方实际感受到的首字延迟。

指标（均给 p50/p90/p99/mean/max，单位毫秒）：
  ttft_onset  首字相对"用户开始说话(起音点)"——主指标，realtime 模式专用。
              = ttft_text - 该条音频前导静音(speech onset 偏移)。realtime 下，
              文件偏移 onset_ms 处的音频在 t_first_send+onset_ms 才上线，等价于
              用户在那一刻开口，故首字延迟应从该刻起算，剔除前导静音的干扰。
  ttft_text   首个非空文本（partial 与 final 取较早者）相对"第一帧发送"
  ttft_part   首个非空 partial（msgtype=Progressive，口语形式）相对第一帧发送
  ttft_final  首个 final（msgtype=sentence，已 ITN）相对第一帧发送
  resp_last   首个非空文本相对"末帧（status=2）发送完成"；realtime 下若首字
              在发送途中已到达则为 0（说明边发边出字）
  full        会话结束（status=2 终止帧）相对末帧发送——整段完成时延

发送模式（--send-mode）：
  realtime  按 --chunk-ms 真实节奏发送，模拟麦克风实时流（默认）。ttft_text 反映
            真实 UX，天然包含 VAD 起音确认与一次 pseudo_stream 间隔。
  fast      尽快连续发完所有帧（仍按协议分帧但不 sleep），剥离实时等待，
            ttft 更接近服务端排队 + 推理的处理速度。

用法示例：
    python scripts/bench_tuling_ttft.py \
        --url ws://localhost:8080/tuling/ast/v3 \
        --data-root ~/data/testdata/data_tts_output_audio_metadata \
        --limit 30

    # 快速发送 + 输出 JSON
    python scripts/bench_tuling_ttft.py --send-mode fast --limit 50 \
        --output bench_results/tuling_ttft.json

    # 关闭伪流式，只看 final 首字延迟
    python scripts/bench_tuling_ttft.py --no-pseudo-stream --limit 30

    # 压力下的首字延迟退化（4 路并发）
    python scripts/bench_tuling_ttft.py --concurrency 4 --limit 40
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

import websockets

# 复用已验证的 client 音频工具（解码 + 重采样到 16 kHz s16le PCM）。
_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES = _REPO_ROOT / "docs" / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from audio_common import chunk_bytes, make_ssl_context, read_audio_as_pcm  # noqa: E402

SAMPLE_RATE = 16000


def _speech_onset_ms(pcm: bytes, *, frame_ms: int = 10, ratio: float = 0.10) -> float:
    """用短时能量估计起音点（用户开始说话的文件偏移，ms）。

    把 PCM 切成 frame_ms 帧算每帧 RMS，取首个 RMS >= max_rms*ratio 的帧位置。
    用于把"相对第一帧发送"的首字延迟换算成"相对用户开始说话"，剔除前导静音。
    这是参考点定位，不是 VAD 复刻：服务端切段用的是 ten-vad，此处只为统计口径。
    """
    import numpy as _np

    frame = SAMPLE_RATE * frame_ms // 1000
    x = _np.frombuffer(pcm, dtype=_np.int16).astype(_np.float32) / 32768.0
    n = (len(x) // frame) * frame
    if n == 0:
        return 0.0
    fr = x[:n].reshape(-1, frame)
    rms = _np.sqrt((fr ** 2).mean(axis=1) + 1e-12)
    peak = float(rms.max())
    if peak <= 0:
        return 0.0
    idx = int(_np.argmax(rms >= peak * ratio))
    return float(idx * frame_ms)


# ---------------------------------------------------------------------------
# 数据集加载（真实 WAV -> 16 kHz mono s16le PCM，和线上同款解码）
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    utt_id: str
    pcm: bytes  # 16 kHz mono s16le，可直接分帧 base64 后塞进 payload
    duration_s: float
    ref_text: str = ""
    onset_ms: float = 0.0  # 起音点偏移（前导静音时长），用于 ttft_onset


def load_samples(
    data_root: Path,
    *,
    subset: str,
    limit: int,
    lang_filter: str,
    max_seconds: float,
    shuffle: bool,
    seed: int,
) -> list[Sample]:
    """从 data_root 下各子集的 metadata.jsonl 加载样本。

    audio_path 形如 ``<subset>/wavs/zh/x.wav``，相对 data_root；这与
    scripts/bench_asr_vllm.py 的解析一致（那里是 data_dir.parent / p）。
    """
    if subset:
        metas = [data_root / subset / "metadata.jsonl"]
    else:
        metas = sorted(data_root.glob("*/metadata.jsonl"))

    rows: list[dict[str, Any]] = []
    for meta in metas:
        if not meta.is_file():
            continue
        for line in meta.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

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
        wav = data_root / audio_path
        if not wav.is_file():
            n_missing += 1
            continue
        try:
            pcm = read_audio_as_pcm(str(wav))
        except Exception:  # noqa: BLE001
            n_bad += 1
            continue
        if not pcm:
            n_bad += 1
            continue
        dur = len(pcm) / 2 / SAMPLE_RATE
        if max_seconds > 0 and dur > max_seconds:
            # 与线上 ASR 60s 尾截一致：保留尾部窗口。
            keep = int(SAMPLE_RATE * max_seconds) * 2
            pcm = pcm[-keep:]
            dur = len(pcm) / 2 / SAMPLE_RATE
        samples.append(
            Sample(
                utt_id=str(r.get("utt_id") or wav.stem),
                pcm=pcm,
                duration_s=dur,
                ref_text=str(r.get("text") or ""),
                onset_ms=_speech_onset_ms(pcm),
            )
        )

    if n_missing or n_bad:
        print(f"  [load] skipped: missing={n_missing} undecodable={n_bad}")
    return samples


# ---------------------------------------------------------------------------
# AST v3 帧封装（与 docs/examples/ws_ast_v3.py 对齐）
# ---------------------------------------------------------------------------


def build_frame(
    status: int,
    *,
    audio: bytes,
    trace_id: str,
    biz_id: str,
    hotwords: str = "",
    asr_config: dict | None = None,
) -> str:
    header: dict[str, object] = {"traceId": trace_id, "status": status}
    if biz_id:
        header["bizId"] = biz_id
    parameter: dict[str, object] = {"engine": {}}
    if asr_config:
        parameter["asr_config"] = asr_config
    payload: dict[str, object] = {
        "audio": {"audio": base64.b64encode(audio).decode("ascii")}
    }
    if hotwords:
        payload["text"] = {"text": hotwords}
    return json.dumps(
        {"header": header, "parameter": parameter, "payload": payload},
        ensure_ascii=False,
    )


def _result_text(result: dict[str, Any]) -> str:
    return "".join(
        cw.get("w", "")
        for item in (result.get("ws") or [])
        for cw in (item.get("cw") or [])
    ).strip()


# ---------------------------------------------------------------------------
# 单样本测量
# ---------------------------------------------------------------------------


@dataclass
class Marks:
    t_first_send: float = math.nan
    t_last_send: float = math.nan
    t_first_partial: float = math.nan
    t_first_final: float = math.nan
    t_end: float = math.nan
    first_partial_text: str = ""
    first_final_text: str = ""
    err: str = ""
    code: int = 0


@dataclass
class SampleResult:
    utt_id: str
    duration_s: float
    status: str  # ok / miss / err
    ttft_onset_ms: float = math.nan  # 首字相对起音点（realtime 主指标）
    onset_ms: float = math.nan       # 该条前导静音时长
    ttft_text_ms: float = math.nan
    ttft_partial_ms: float = math.nan
    ttft_final_ms: float = math.nan
    resp_last_ms: float = math.nan
    full_ms: float = math.nan
    pred_text: str = ""
    ref_text: str = ""
    err: str = ""


async def _receive(ws, marks: Marks) -> None:
    """并发接收循环：记录首个非空 partial / final 与会话结束时刻。"""
    async for raw in ws:
        try:
            msg = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        header = msg.get("header", {})
        if header.get("code", 0) != 0:
            marks.err = str(header.get("message", "")) or "code!=0"
            marks.code = int(header.get("code", -1))
            return
        result = msg.get("payload", {}).get("result", {})
        text = _result_text(result)
        mtype = result.get("msgtype")
        now = time.perf_counter()
        if mtype == "Progressive" and text and math.isnan(marks.t_first_partial):
            marks.t_first_partial = now
            marks.first_partial_text = text
        elif mtype == "sentence" and text and math.isnan(marks.t_first_final):
            marks.t_first_final = now
            marks.first_final_text = text
        # 终止帧：status==2 且无 ws。
        if header.get("status") == 2 and not result.get("ws"):
            marks.t_end = now
            return


async def measure_one(
    *,
    url: str,
    sample: Sample,
    chunk_ms: int,
    send_mode: str,
    final_timeout: float,
    connect_timeout: float,
    asr_config: dict | None,
    hotwords: str,
    trace_id: str,
    biz_id: str,
    ssl_ctx,
) -> SampleResult:
    cs = chunk_bytes(chunk_ms)
    pcm = sample.pcm
    chunks = [pcm[i : i + cs] for i in range(0, max(len(pcm), 1), cs)] or [b""]
    marks = Marks()
    try:
        async with websockets.connect(
            url, ssl=ssl_ctx, open_timeout=connect_timeout, max_size=None
        ) as ws:
            recv_task = asyncio.create_task(_receive(ws, marks))
            marks.t_first_send = time.perf_counter()
            for i, ch in enumerate(chunks):
                status = 0 if i == 0 else (2 if i == len(chunks) - 1 else 1)
                await ws.send(
                    build_frame(
                        status,
                        audio=ch,
                        trace_id=trace_id,
                        biz_id=biz_id,
                        hotwords=hotwords if i == 0 else "",
                        asr_config=asr_config if i == 0 else None,
                    )
                )
                # 末帧（status=2）之后没有下一帧，不能再等一个帧间隔，否则
                # t_last_send 偏晚、full（末帧->结束帧）会算成负值。
                if send_mode == "realtime" and i < len(chunks) - 1:
                    await asyncio.sleep(chunk_ms / 1000)
            marks.t_last_send = time.perf_counter()
            try:
                await asyncio.wait_for(recv_task, timeout=final_timeout)
            except asyncio.TimeoutError:
                recv_task.cancel()
    except Exception as exc:  # noqa: BLE001
        marks.err = f"{type(exc).__name__}: {exc}"[:200]

    return _build_result(sample, marks, send_mode)


def _build_result(sample: Sample, m: Marks, send_mode: str = "realtime") -> SampleResult:
    res = SampleResult(
        utt_id=sample.utt_id,
        duration_s=sample.duration_s,
        status="err" if m.err else "ok",
        onset_ms=sample.onset_ms,
        ref_text=sample.ref_text,
        err=m.err,
    )
    if m.err:
        return res

    # 首个非空文本 = partial / final 中较早出现者。
    cands = [t for t in (m.t_first_partial, m.t_first_final) if not math.isnan(t)]
    if not cands:
        res.status = "miss"  # 正常结束但全程无文本（VAD 判定无语音等）
        if not math.isnan(m.t_end) and not math.isnan(m.t_last_send):
            res.full_ms = (m.t_end - m.t_last_send) * 1000
        return res

    t_first_text = min(cands)
    base = m.t_first_send
    res.ttft_text_ms = (t_first_text - base) * 1000
    # realtime 下：起音点音频在 base+onset_ms 才上线，等价用户此刻开口，
    # 故"用户开始说话->首字"= ttft_text - onset。fast 模式无实时节奏，不适用。
    if send_mode == "realtime":
        res.ttft_onset_ms = max(0.0, res.ttft_text_ms - sample.onset_ms)
    if not math.isnan(m.t_first_partial):
        res.ttft_partial_ms = (m.t_first_partial - base) * 1000
    if not math.isnan(m.t_first_final):
        res.ttft_final_ms = (m.t_first_final - base) * 1000
    if not math.isnan(m.t_last_send):
        res.resp_last_ms = max(0.0, (t_first_text - m.t_last_send) * 1000)
    if not math.isnan(m.t_end) and not math.isnan(m.t_last_send):
        res.full_ms = (m.t_end - m.t_last_send) * 1000
    res.pred_text = m.first_final_text or m.first_partial_text
    return res


# ---------------------------------------------------------------------------
# 统计与输出
# ---------------------------------------------------------------------------


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return math.nan
    s = sorted(xs)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


@dataclass
class Metric:
    name: str
    values: list[float] = field(default_factory=list)

    def row(self) -> str:
        if not self.values:
            return f"{self.name:>10}  {'(no data)':>8}"
        return (
            f"{self.name:>10}  "
            f"{_percentile(self.values, 50):>8.1f}  "
            f"{_percentile(self.values, 90):>8.1f}  "
            f"{_percentile(self.values, 99):>8.1f}  "
            f"{statistics.fmean(self.values):>8.1f}  "
            f"{min(self.values):>8.1f}  "
            f"{max(self.values):>8.1f}  "
            f"{len(self.values):>5d}"
        )


def summarize(results: list[SampleResult]) -> str:
    ok = [r for r in results if r.status == "ok"]
    miss = [r for r in results if r.status == "miss"]
    err = [r for r in results if r.status == "err"]

    metrics = [
        Metric("ttft_onset", [r.ttft_onset_ms for r in ok if not math.isnan(r.ttft_onset_ms)]),
        Metric("ttft_text", [r.ttft_text_ms for r in ok if not math.isnan(r.ttft_text_ms)]),
        Metric("ttft_part", [r.ttft_partial_ms for r in ok if not math.isnan(r.ttft_partial_ms)]),
        Metric("ttft_final", [r.ttft_final_ms for r in ok if not math.isnan(r.ttft_final_ms)]),
        Metric("resp_last", [r.resp_last_ms for r in ok if not math.isnan(r.resp_last_ms)]),
        Metric("full", [r.full_ms for r in results if not math.isnan(r.full_ms)]),
    ]

    header = (
        f"{'metric':>10}  {'p50':>8}  {'p90':>8}  {'p99':>8}  "
        f"{'mean':>8}  {'min':>8}  {'max':>8}  {'n':>5}   (ms)"
    )
    sep = "-" * len(header)
    lines = [
        "",
        f"样本: total={len(results)}  ok={len(ok)}  miss(无文本)={len(miss)}  err={len(err)}",
        sep,
        header,
        sep,
    ]
    lines += [m.row() for m in metrics]
    lines.append(sep)
    if err:
        lines.append(f"first error: {err[0].err}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 驱动
# ---------------------------------------------------------------------------


def build_asr_config(args: argparse.Namespace) -> dict:
    cfg: dict[str, object] = {}
    for item in args.config:
        key, sep, val = item.partition("=")
        if not sep:
            continue
        try:
            cfg[key.strip()] = json.loads(val)
        except json.JSONDecodeError:
            cfg[key.strip()] = val
    if args.language:
        cfg["language"] = args.language
    if args.no_pseudo_stream:
        cfg["enable_pseudo_stream"] = False
    return cfg


async def main_async(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root).expanduser().resolve()
    ssl_ctx = make_ssl_context(args.url, args.insecure)
    asr_config = build_asr_config(args)

    print(f"Target:    {args.url}")
    print(f"Data root: {data_root}")
    print(f"Send mode: {args.send_mode}  chunk_ms={args.chunk_ms}  concurrency={args.concurrency}")
    print(f"asr_config override: {asr_config or '(none)'}  hotwords={args.hotwords or '(none)'}")
    print("Loading + decoding audio to 16 kHz mono ...")

    samples = load_samples(
        data_root,
        subset=args.subset,
        limit=args.limit,
        lang_filter=args.lang_filter,
        max_seconds=args.max_seconds,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    if not samples:
        print("No usable audio samples found.", file=sys.stderr)
        return 2

    durs = [s.duration_s for s in samples]
    print(
        f"Loaded:    {len(samples)} clips  total={sum(durs):.1f}s  "
        f"dur min/med/max={min(durs):.2f}/{statistics.median(durs):.2f}/{max(durs):.2f}s"
    )
    if args.send_mode == "realtime":
        eta = sum(durs) / max(1, args.concurrency)
        print(f"  [realtime] 预计发送耗时约 {eta:.0f}s（+推理），fast 模式可大幅缩短")
    print()

    async def run_one(sample: Sample) -> SampleResult:
        return await measure_one(
            url=args.url,
            sample=sample,
            chunk_ms=args.chunk_ms,
            send_mode=args.send_mode,
            final_timeout=args.final_timeout,
            connect_timeout=args.connect_timeout,
            asr_config=asr_config or None,
            hotwords=args.hotwords,
            trace_id=args.trace_id,
            biz_id=args.biz_id,
            ssl_ctx=ssl_ctx,
        )

    # 预热：首条常含模型/连接冷启动，单独跑不计入。
    if args.warmup > 0:
        warm = samples[: args.warmup]
        wres = await asyncio.gather(*(run_one(s) for s in warm))
        wok = sum(1 for r in wres if r.status == "ok")
        print(f"warmup: {wok}/{len(warm)} ok"
              + (f"  (sample ttft={wres[0].ttft_text_ms:.0f}ms)" if wok else ""))
        if wok == 0:
            print(f"  [warn] warmup 全部失败，first error: {wres[0].err}", file=sys.stderr)

    results: list[SampleResult] = []
    sem = asyncio.Semaphore(args.concurrency)
    done = 0
    lock = asyncio.Lock()

    async def guarded(sample: Sample) -> None:
        nonlocal done
        async with sem:
            r = await run_one(sample)
        async with lock:
            results.append(r)
            done += 1
            tag = (
                f"{r.ttft_text_ms:7.0f}ms" if r.status == "ok"
                else f"{r.status:>9}"
            )
            print(f"  [{done:>3}/{len(samples)}] {tag}  {r.utt_id[:36]:<36}  "
                  f"{(r.pred_text or r.err)[:42]}")

    t0 = time.perf_counter()
    await asyncio.gather(*(guarded(s) for s in samples))
    wall = time.perf_counter() - t0

    print(summarize(results))
    print(f"\nwall={wall:.1f}s")
    print(
        "Legend: ttft_onset=用户开始说话->首字(realtime 主指标,已剔除前导静音);\n"
        "        ttft_text=首字相对第一帧发送; ttft_part/final=首个partial/final;\n"
        "        resp_last=末帧发完到首字(realtime下可能为0); full=整段完成时延。"
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "url": args.url,
                "data_root": str(data_root),
                "subset": args.subset,
                "send_mode": args.send_mode,
                "chunk_ms": args.chunk_ms,
                "concurrency": args.concurrency,
                "limit": args.limit,
                "lang_filter": args.lang_filter,
                "max_seconds": args.max_seconds,
                "asr_config": asr_config,
                "hotwords": args.hotwords,
                "n_samples": len(samples),
            },
            "results": [vars(r) for r in results],
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"Saved JSON -> {out}")

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="/tuling/ast/v3 首字返回延迟（TTFT）基准（真实测试集音频，走完整 WS 流水线）。"
    )
    p.add_argument("--url", default="ws://localhost:8080/tuling/ast/v3",
                   help="WebSocket URL，默认 ws://localhost:8080/tuling/ast/v3")
    p.add_argument("--data-root",
                   default="~/data/testdata/data_tts_output_audio_metadata",
                   help="测试集根目录（含若干 <subset>/metadata.jsonl 与 wavs/）。")
    p.add_argument("--subset", default="",
                   help="只测某个子集目录名（默认空=合并所有子集）。")
    p.add_argument("--limit", type=int, default=30, help="最多测多少条（0=全部）。")
    p.add_argument("--lang-filter", default="", help="按 metadata 的 lang 过滤，如 zh。")
    p.add_argument("--max-seconds", type=float, default=60.0,
                   help="超过该时长尾截（与线上 ASR 60s 一致；0=不截）。")
    p.add_argument("--send-mode", choices=["realtime", "fast"], default="realtime",
                   help="realtime=按节奏发送(默认)；fast=尽快灌完测推理。")
    p.add_argument("--chunk-ms", type=int, default=128, help="每帧音频时长 ms（~4096B）。")
    p.add_argument("--concurrency", type=int, default=1,
                   help="并发连接数（默认 1 测基线；调大看压力下首字延迟退化）。")
    p.add_argument("--warmup", type=int, default=1, help="预热条数，不计入统计。")
    p.add_argument("--no-pseudo-stream", action="store_true",
                   help="首帧关闭伪流式 enable_pseudo_stream=false（只看 final 首字）。")
    p.add_argument("--language", default="", help="会话语言，写入 asr_config.language。")
    p.add_argument("--config", action="append", default=[], metavar="KEY=VALUE",
                   help="通用 asr_config 覆写，可多次，如 --config vad_threshold=0.45。")
    p.add_argument("--hotwords", default="", help="逗号分隔热词（首帧）。")
    p.add_argument("--trace-id", default="ttft-bench", help="header.traceId。")
    p.add_argument("--biz-id", default="12345", help="header.bizId。")
    p.add_argument("--connect-timeout", type=float, default=10.0)
    p.add_argument("--final-timeout", type=float, default=30.0,
                   help="末帧发送后等待结果的最长秒数。")
    p.add_argument("--shuffle", action="store_true", help="打乱样本顺序。")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--insecure", action="store_true", help="wss 跳过证书校验。")
    p.add_argument("--output", default="", help="结果 JSON 落盘路径。")
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
