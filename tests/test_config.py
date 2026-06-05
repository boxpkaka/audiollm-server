"""Unit tests for the grouped-config loader and single-entry refactor.

Covers
------
1. Nested (feature-grouped) config.json flattens to exactly the same Config
   as the equivalent flat file -> the on-disk regrouping is behavior-neutral.
2. A legacy flat file still loads unchanged (backward compatibility).
3. The shipped backend/config.json decodes to the expected values and keeps
   each value's JSON type (Config does no coercion, e.g. ttl stays ``int``).
4. ``override`` still uses flat field keys; nested-style dotted keys are
   ignored, so the client contract is unchanged.
5. A leaf key colliding across groups warns (config error is not silent).
6. The fusion/secondary invariant still fires (and warns) when expressed
   through nested groups -> the load-time WARN reads the flattened values.
7. ``default_config`` is exactly ``load_config()`` -> one entry point.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import (  # noqa: E402
    CLIENT_OVERRIDABLE_FIELDS,
    Config,
    _flatten,
    default_config,
    load_config,
)


def _write_and_load(tmp_path: Path, data: dict) -> Config:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return load_config(path)


def test_nested_equals_equivalent_flat(tmp_path: Path) -> None:
    nested = {
        "asr": {
            "primary": {"vllm_base_url": "http://x:1", "primary_asr_timeout": 2.0},
            "secondary": {"enable_secondary_asr": True},
            "vad": {"vad_threshold": 0.7, "vad_start_frames": 5},
        },
        "emotion": {"emotion_task_mode": "sec"},
        "http": {"http_max_connections": 9},
    }
    flat = {
        "vllm_base_url": "http://x:1",
        "primary_asr_timeout": 2.0,
        "enable_secondary_asr": True,
        "vad_threshold": 0.7,
        "vad_start_frames": 5,
        "emotion_task_mode": "sec",
        "http_max_connections": 9,
    }
    cfg_nested = _write_and_load(tmp_path / "a", nested)
    cfg_flat = _write_and_load(tmp_path / "b", flat)
    assert dataclasses.asdict(cfg_nested) == dataclasses.asdict(cfg_flat)


def test_flat_file_backward_compatible(tmp_path: Path) -> None:
    cfg = _write_and_load(tmp_path, {"vllm_base_url": "http://legacy:9", "vad_start_frames": 11})
    assert cfg.vllm_base_url == "http://legacy:9"
    assert cfg.vad_start_frames == 11


def test_shipped_config_values_and_types() -> None:
    cfg = load_config()  # reads backend/config.json
    assert cfg.vllm_base_url == "http://localhost:8009"
    assert cfg.vllm_model_name == "AmphionASR-4.3B"
    assert cfg.vad_threshold == 0.6
    assert cfg.enable_dual_asr_fusion is False
    assert cfg.enable_secondary_asr is True
    assert cfg.emotion_task_mode == "ser"
    assert cfg.emotion_spec_task_mode == "sepc"
    # Config does no type coercion: an integer literal stays an int.
    assert isinstance(cfg.asr_enrollment_ttl_sec, int)
    assert cfg.asr_enrollment_ttl_sec == 3600


def test_vad_end_frames_field_removed() -> None:
    """vad_end_frames was merged into silence_duration_ms; it must stay gone
    from both the dataclass and the client override whitelist so it can't
    silently reappear and reintroduce the max() config-spoofing bug."""
    names = {f.name for f in dataclasses.fields(Config)}
    assert "vad_end_frames" not in names
    assert "vad_end_frames" not in CLIENT_OVERRIDABLE_FIELDS


def test_override_uses_flat_keys_only() -> None:
    cfg = load_config()
    overridden = cfg.override(vad_threshold=0.45)
    assert overridden.vad_threshold == 0.45
    # A nested/dotted key is not a dataclass field -> silently ignored,
    # so the legacy flat override contract is preserved.
    untouched = cfg.override(**{"asr.vad.vad_threshold": 0.1})
    assert untouched.vad_threshold == cfg.vad_threshold


def test_override_client_allows_whitelisted_fields() -> None:
    cfg = load_config()
    out = cfg.override_client(vad_threshold=0.42, enable_pseudo_stream=False)
    assert out.vad_threshold == 0.42
    assert out.enable_pseudo_stream is False


def test_override_client_drops_non_whitelisted_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Infra/secret fields (model URL, API key) are not client-overridable."""
    cfg = load_config()
    with caplog.at_level(logging.WARNING, logger="backend.config"):
        out = cfg.override_client(
            vllm_base_url="http://evil:1",
            text_cleanup_api_key="leaked",
            vad_threshold=0.42,
        )
    assert out.vad_threshold == 0.42  # whitelisted -> applied
    assert out.vllm_base_url == cfg.vllm_base_url  # restricted -> untouched
    assert out.text_cleanup_api_key == cfg.text_cleanup_api_key
    assert any("non-overridable config field" in r.message for r in caplog.records)


def test_override_client_ignores_invalid_value() -> None:
    """A whitelisted field with an uncoercible value keeps the server default."""
    cfg = load_config()
    out = cfg.override_client(vad_threshold="not-a-number")
    assert out.vad_threshold == cfg.vad_threshold


def test_override_client_enforces_fusion_invariant() -> None:
    """fusion=True + secondary=False downgrades fusion via __post_init__."""
    cfg = load_config().override(enable_secondary_asr=True, enable_dual_asr_fusion=True)
    out = cfg.override_client(enable_secondary_asr=False, enable_dual_asr_fusion=True)
    assert out.enable_secondary_asr is False
    assert out.enable_dual_asr_fusion is False


def test_client_overridable_fields_are_real_and_safe() -> None:
    """Whitelist references only real Config fields and excludes infra/secrets."""
    names = {f.name for f in dataclasses.fields(Config)}
    assert CLIENT_OVERRIDABLE_FIELDS <= names
    forbidden = {
        "vllm_base_url",
        "secondary_vllm_base_url",
        "emotion_vllm_base_url",
        "emotion_spec_vllm_base_url",
        "text_cleanup_base_url",
        "text_cleanup_api_key",
        "text_cleanup_api_key_env",
        "http_max_connections",
        "http_max_keepalive_connections",
    }
    assert CLIENT_OVERRIDABLE_FIELDS.isdisjoint(forbidden)


def test_duplicate_leaf_across_groups_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="backend.config"):
        flat = _flatten({"groupA": {"vad_threshold": 0.5}, "groupB": {"vad_threshold": 0.9}})
    assert flat["vad_threshold"] == 0.9  # last value wins
    assert any("Duplicate config key across groups" in r.message for r in caplog.records)


def test_nested_fusion_downgrade_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="backend.config"):
        cfg = _write_and_load(
            tmp_path,
            {
                "asr": {
                    "secondary": {"enable_secondary_asr": False},
                    "fusion": {"enable_dual_asr_fusion": True},
                }
            },
        )
    assert cfg.enable_secondary_asr is False
    assert cfg.enable_dual_asr_fusion is False  # auto-downgraded
    assert any("downgrading fusion" in r.message for r in caplog.records)


def test_default_config_is_load_config_snapshot() -> None:
    assert dataclasses.asdict(default_config) == dataclasses.asdict(load_config())
