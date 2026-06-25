from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.main as main_mod  # noqa: E402


@pytest.mark.asyncio
async def test_hotword_pool_add_proxies_to_recall(monkeypatch):
    seen: dict[str, object] = {}

    async def fake_add(words):
        seen["words"] = words
        return {"status": "ok", "added": len(words), "total_count": 2}

    monkeypatch.setattr(main_mod, "add_recall_hotwords", fake_add)

    result = await main_mod.asr_hotword_pool_add(
        {"hotwords": [" 挚音科技 ", "", "张硕"]}
    )

    assert seen["words"] == ["挚音科技", "张硕"]
    assert result == {"status": "ok", "added": 2, "total_count": 2}


@pytest.mark.asyncio
async def test_hotword_pool_rejects_empty_payload():
    with pytest.raises(HTTPException) as exc:
        await main_mod.asr_hotword_pool_add({"hotwords": []})
    assert exc.value.status_code == 400
    assert exc.value.detail["code"] == "empty_hotwords"
