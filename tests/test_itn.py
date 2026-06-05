"""Unit tests for final-text ITN + license-plate normalization.

Layering of the tests mirrors the module's design:

1. Plate pass (zero-dependency regex): runs with ITN disabled, so these never
   need wetext installed. Covers spoken digits, casing, separators, the
   context-anchored province-missing case, and negative (no-rewrite) cases.
2. General ITN via wetext: skipped when wetext is unavailable.
3. Safety: an ITN failure (raise) or unavailability (None) must fall back to
   the pre-ITN text while the plate pass still runs.
4. Config wiring: the three Config switches exist and drive normalize_final_text.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.asr import itn  # noqa: E402
from backend.asr.itn import normalize_final, normalize_final_text  # noqa: E402
from backend.config import Config, load_config  # noqa: E402


def _wetext_works() -> bool:
    try:
        from wetext import Normalizer

        Normalizer(lang="zh", operator="itn", enable_0_to_9=False).normalize("二零二四年")
        return True
    except Exception:
        return False


_HAS_WETEXT = _wetext_works()
requires_wetext = pytest.mark.skipif(not _HAS_WETEXT, reason="wetext not installed")


def _plate(text: str) -> str:
    """Plate pass only (ITN off) -> wetext-independent."""
    return normalize_final(text, "zh", enable_itn=False, enable_plate=True)


# --------------------------------------------------------------------------
# 1. Plate pass (no wetext needed)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("辽b二四五零七", "辽B24507"),                  # lowercase letter + spoken digits
        ("鲁A幺二三四五", "鲁A12345"),                  # 幺 -> 1
        ("沪A·1234学", "沪A·1234学"),                  # too short after clean -> untouched
        ("冀R65438", "冀R65438"),                       # already written, casing kept
    ],
)
def test_plate_province_anchored(raw: str, expected: str) -> None:
    assert _plate(raw) == expected


def test_plate_in_sentence() -> None:
    assert _plate("帮忙核查一下车牌号为冀R六五四三八的情况。") == (
        "帮忙核查一下车牌号为冀R65438的情况。"
    )


def test_plate_strips_separators() -> None:
    # In-plate separators that would otherwise block the FST are removed.
    assert _plate("辽B二四五、零七") == "辽B24507"


def test_plate_context_anchored_province_missing() -> None:
    # Known boundary: 冀 misheard as J is a recognition error. The plate pass
    # fixes digits + casing under "车牌" context but does NOT recover 冀.
    out = _plate("帮忙核查一下车牌号为JR六五、四、三八的情况。")
    assert "JR65438" in out
    assert "六五" not in out  # digits were normalized
    assert "冀" not in out    # we never guess the province back


@pytest.mark.parametrize(
    "text",
    [
        "我在北京工作",          # 京 is a province char but not a plate
        "今天天气很好",
        "他考了第一名",
        "请稍等一下",
    ],
)
def test_plate_does_not_touch_prose(text: str) -> None:
    assert _plate(text) == text


# --------------------------------------------------------------------------
# 2. General ITN (skipped without wetext)
# --------------------------------------------------------------------------


@requires_wetext
@pytest.mark.parametrize(
    "raw, must_contain, must_not_contain",
    [
        ("今天是二零二四年", "2024", "二零二四"),
        ("编号六五四三八", "65438", "六五四三八"),
        ("手机号一三八零零一三八零零零", "13800138000", "一三八"),
    ],
)
def test_general_itn(raw: str, must_contain: str, must_not_contain: str) -> None:
    out = normalize_final(raw, "zh")
    assert must_contain in out
    assert must_not_contain not in out


@requires_wetext
def test_itn_plus_plate_end_to_end() -> None:
    # ITN converts the digits, the plate pass uppercases the letter.
    assert normalize_final("麻烦核实一下辽b二四五零七这辆车。", "zh") == (
        "麻烦核实一下辽B24507这辆车。"
    )


@requires_wetext
def test_itn_skipped_for_non_chinese() -> None:
    # wetext zh-ITN must not run on declared non-Chinese output.
    text = "call me at one two three"
    assert normalize_final(text, "en") == text


# --------------------------------------------------------------------------
# 3. Safety: failures fall back to pre-ITN text
# --------------------------------------------------------------------------


def test_itn_exception_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Boom:
        def normalize(self, s: str) -> str:
            raise RuntimeError("boom")

    monkeypatch.setattr(itn, "_get_normalizer", lambda flag: _Boom())
    # ITN raises -> keep pre-ITN text; no plate present so result is unchanged.
    assert normalize_final("今天是二零二四年", "zh") == "今天是二零二四年"


def test_itn_unavailable_plate_still_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(itn, "_get_normalizer", lambda flag: None)
    # No general ITN, but the zero-dep plate pass still fixes the plate.
    assert normalize_final("辽b二四五零七", "zh") == "辽B24507"


def test_empty_text_passthrough() -> None:
    assert normalize_final("", "zh") == ""
    assert normalize_final("   ", "zh") == "   "


# --------------------------------------------------------------------------
# 4. Config switches and the config-driven wrapper
# --------------------------------------------------------------------------


def test_config_has_itn_switches() -> None:
    cfg = load_config()
    assert cfg.enable_asr_itn is True
    assert cfg.asr_itn_enable_0_to_9 is False
    assert cfg.enable_asr_plate_normalize is True


def test_both_disabled_is_noop() -> None:
    raw = "辽b二四五零七 二零二四年"
    assert normalize_final(raw, "zh", enable_itn=False, enable_plate=False) == raw


def test_plate_only_via_config_wrapper() -> None:
    cfg = Config(enable_asr_itn=False, enable_asr_plate_normalize=True)
    assert normalize_final_text("辽b二四五零七", "zh", cfg) == "辽B24507"


@requires_wetext
def test_itn_only_via_config_wrapper() -> None:
    cfg = Config(enable_asr_itn=True, enable_asr_plate_normalize=False)
    out = normalize_final_text("编号六五四三八", "zh", cfg)
    assert "65438" in out
