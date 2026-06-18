"""Tests for the ASR vLLM benchmark helper."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import bench_asr_vllm as bench  # noqa: E402


@pytest.mark.asyncio
async def test_bench_cer_scores_parsed_model_output(monkeypatch) -> None:
    sample = bench.Sample(
        utt_id="u1",
        duration_s=1.0,
        wav_b64="WAV_B64",
        ref_text="你好世界",
    )

    async def fake_do_request(client, url, sample, payload, timeout):
        return bench.RequestResult(
            ok=True,
            latency_s=0.1,
            audio_s=sample.duration_s,
            status=200,
            pred_text="language Chinese<asr_text>你好世界",
            ref_text=sample.ref_text,
        )

    monkeypatch.setattr(bench, "do_request", fake_do_request)

    stats = await bench.run_level(
        client=None,
        url="http://example.test/v1/chat/completions",
        model="AmphionASR-1.7B",
        prompt_template="amphion_asr_1.7b",
        max_tokens=16,
        samples=[sample],
        concurrency=1,
        n_requests=1,
        request_timeout=1.0,
        measure_cer=True,
        shuffle=False,
        rng=random.Random(0),
    )

    assert stats.cer == 0.0
    assert stats.sample_pred == "你好世界"
