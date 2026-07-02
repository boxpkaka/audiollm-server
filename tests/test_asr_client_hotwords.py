from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.asr.client as client_mod  # noqa: E402
from backend.config import Config  # noqa: E402


def test_merge_recalled_and_custom_hotwords_dedupes_and_limits_custom():
    out = client_mod.merge_recalled_and_custom_hotwords(
        ["召回A", "上灯板"],
        [" 上灯板 ", "临时一", "临时二", "临时三"],
        custom_limit=2,
    )

    assert out == ["召回A", "上灯板", "临时一", "临时二"]


@pytest.mark.asyncio
async def test_query_audio_model_merges_recalled_and_custom_hotwords(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_recall_audio(*_args, **_kwargs):
        return SimpleNamespace(
            words=["召回A", "上灯板"],
            audio_embeds_b64=None,
            uuid="triton-audio-test",
        )

    async def fake_post_chat(messages, **_kwargs):
        captured["messages"] = messages
        return {
            "transcription": "ok",
            "reported_hotwords": [],
            "raw_text": "ok",
            "detected_language": "zh",
        }

    monkeypatch.setattr(client_mod, "recall_audio", fake_recall_audio)
    monkeypatch.setattr(client_mod, "_post_chat", fake_post_chat)

    result = await client_mod.query_audio_model(
        "WAV_B64",
        hotwords=["上灯板", "临时一", "临时二"],
        audio_pcm=np.zeros(160, dtype=np.float32),
        runtime_config=Config(
            enable_hotword_recall=True,
            enable_encoder_bypass=False,
            recall_custom_hotword_limit=2,
            vllm_prompt_template="amphion_asr",
        ),
    )

    assert result["reported_hotwords"] == ["召回A", "上灯板", "临时一", "临时二"]
    messages = captured["messages"]
    assert isinstance(messages, list)
    content = messages[0]["content"]
    assert content[0]["text"] == "Transcribe the following audio.\nHotwords: 召回A,上灯板,临时一,临时二"
