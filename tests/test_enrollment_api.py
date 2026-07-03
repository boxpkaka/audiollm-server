from __future__ import annotations

import base64
import sys
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.main as main_mod  # noqa: E402
from backend.audio.utils import pcm_to_wav_base64  # noqa: E402
from backend.config import SAMPLE_RATE, Config  # noqa: E402


def _wav_bytes(seconds: float = 1.2) -> bytes:
    n = int(SAMPLE_RATE * seconds)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    pcm = 0.2 * np.sin(2 * np.pi * 440 * t)
    return base64.b64decode(pcm_to_wav_base64(pcm.astype(np.float32)))


def test_enrollment_api_triton_store_does_not_use_local_store(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_upsert(pcm, **kwargs):
        captured["pcm_len"] = int(len(pcm))
        captured["kwargs"] = kwargs
        return {"status": "ok"}

    def fail_local_store():
        raise AssertionError("local enrollment store should not be used")

    monkeypatch.setattr(
        main_mod,
        "load_config",
        lambda: Config(enable_triton_enrollment_store=True),
    )
    monkeypatch.setattr(main_mod, "upsert_triton_enrollment", fake_upsert)
    monkeypatch.setattr(main_mod, "get_enrollment_store", fail_local_store)

    with TestClient(main_mod.app) as client:
        resp = client.post(
            "/api/asr/enrollment",
            files={"audio": ("speaker.wav", _wav_bytes(), "audio/wav")},
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["enrollment_id"]
    assert payload["duration_sec"] == 1.2
    kwargs = captured["kwargs"]
    assert kwargs["enrollment_id"] == payload["enrollment_id"]
    assert kwargs["user_id"] == "default"
    assert kwargs["sample_rate"] == SAMPLE_RATE
    assert captured["pcm_len"] == int(SAMPLE_RATE * 1.2)
