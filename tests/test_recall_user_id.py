from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.asr.recall as recall_mod  # noqa: E402
from backend.config import Config, Upstream  # noqa: E402


class _FakeInput:
    def __init__(self, name: str, shape, dtype: str) -> None:
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.data = None

    def set_data_from_numpy(self, data) -> None:
        self.data = data


class _FakeHttpClient:
    def InferInput(self, name: str, shape, dtype: str) -> _FakeInput:  # noqa: N802
        return _FakeInput(name, shape, dtype)

    def InferRequestedOutput(self, name: str) -> str:  # noqa: N802
        return name


class _FakeResult:
    def as_numpy(self, name: str):
        if name == "WORD_LIST":
            return np.array(['["挚音科技"]'], dtype=object)
        if name == "PROJECTOR_LEN":
            return np.array([3], dtype=np.int32)
        if name == "STATUS":
            return np.array(["ok"], dtype=object)
        if name == "MESSAGE":
            return np.array(['{"user_id":"tenant-a"}'], dtype=object)
        if name == "HOTWORD_COUNT":
            return np.array([1], dtype=np.int32)
        if name == "HOTWORD_LIST":
            return np.array(['["挚音科技"]'], dtype=object)
        raise AssertionError(name)


class _FakeClient:
    def __init__(self, calls: list[dict[str, object]]) -> None:
        self.calls = calls

    def infer(self, model_name: str, inputs, outputs=None):
        self.calls.append(
            {
                "model_name": model_name,
                "inputs": {
                    item.name: item.data.tolist()
                    if hasattr(item.data, "tolist")
                    else item.data
                    for item in inputs
                },
                "outputs": outputs,
            }
        )
        return _FakeResult()


def _install_fake_triton(monkeypatch) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    upstream = Upstream(
        name="recall",
        base_url="http://localhost:10001",
        model_name="rag_asr_retrieve",
    )
    monkeypatch.setattr(recall_mod, "_recall_upstream", lambda: upstream)
    monkeypatch.setattr(
        recall_mod,
        "_client_for",
        lambda _upstream: (_FakeHttpClient(), _FakeClient(calls)),
    )
    return calls


@pytest.mark.asyncio
async def test_recall_audio_sends_hotword_pool_id(monkeypatch):
    calls = _install_fake_triton(monkeypatch)

    result = await recall_mod.recall_audio(
        np.zeros(160, dtype=np.float32),
        Config(recall_top_k=1, recall_user_id="default"),
        want_audio_embeds=False,
        hotword_pool_id="tenant-a",
    )

    assert result.words == ["挚音科技"]
    assert calls[0]["inputs"]["USER_ID"] == ["tenant-a"]


@pytest.mark.asyncio
async def test_hotword_management_sends_hotword_pool_id(monkeypatch):
    calls = _install_fake_triton(monkeypatch)

    result = await recall_mod.add_hotwords(
        ["挚音科技"],
        hotword_pool_id="tenant-a",
    )

    assert result["hotwords"] == ["挚音科技"]
    assert calls[0]["inputs"]["ACTION"] == ["add"]
    assert calls[0]["inputs"]["USER_ID"] == ["tenant-a"]
    assert calls[0]["inputs"]["HOTWORDS"] == ['["挚音科技"]']
