"""Inverse text normalization (ITN) + license-plate formatting for final ASR.

The Amphion ASR model emits spoken-form text (``六五四三八``, ``二零二四年``).
For display we want written form (``65438``, ``2024年``). This module applies:

1. General ITN via ``wetext`` (the pure-Python runtime of WeTextProcessing —
   same FST grammars, no pynini). Runs only for Chinese-eligible output.
2. A zero-dependency license-plate pass that fixes what general ITN can't:
   lowercase plate letters (``辽b24507`` -> ``辽B24507``), in-plate separators
   that block the FST (``JR六五、四、三八`` under "车牌" context -> ``JR65438``),
   and per-digit spoken numerals inside a plate.

Hard boundary (documented): a province abbreviation misheard as a Latin letter
(``冀`` -> ``J``) is a recognition error, not a formatting one. No text tool can
recover it; this module fixes the digits/case but leaves the wrong province
character as-is.

Everything is best-effort: any failure returns the pre-step text unchanged so
ITN never breaks the ASR path. Apply to ``final`` only, never to ``partial``.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# 31 mainland province abbreviations that legally lead a civilian plate.
_PROVINCE_CHARS = "京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新"

# Per-position spoken digit -> Arabic. This is a plate/sequence mapping (digit by
# digit), NOT positional arithmetic — general numbers go through wetext instead.
_CN_DIGIT = {
    "零": "0", "〇": "0", "一": "1", "幺": "1", "二": "2", "两": "2",
    "三": "3", "四": "4", "五": "5", "六": "6", "七": "7", "八": "8",
    "九": "9",
}
_CN_DIGIT_CHARS = "".join(_CN_DIGIT.keys())

# Separators that may appear inside a spoken plate and must be removed.
_SEP_CHARS = " \t、，,。.．_··-—~～:：·"

# Characters allowed inside a (pre-clean) plate body: latin alnum, Arabic
# digits, spoken digits, and separators. Built programmatically to avoid
# hand-escaping mistakes in the regex character class.
_BODY_CLASS = "A-Za-z0-9" + re.escape(_CN_DIGIT_CHARS + _SEP_CHARS)

# Province-anchored plate: <province><letter><body...>. Fires essentially only
# on real plates because a province char immediately followed by a Latin letter
# does not occur in ordinary prose.
_PROV_RE = re.compile(f"([{_PROVINCE_CHARS}])\\s*([A-Za-z][{_BODY_CLASS}]{{4,10}})")

# Context-anchored plate without a (correctly recognized) province char, e.g.
# "车牌号为JR六五、四、三八". Only fires right after an explicit plate keyword to
# avoid mangling arbitrary "letter + digits" runs elsewhere.
_CTX_RE = re.compile(
    f"((?:车牌号码|车牌号|车牌|牌照号|牌照)[为是:：]?\\s*)([A-Za-z][{_BODY_CLASS}]{{4,12}})"
)

# A cleaned, valid GB plate body: letter + 5 (normal) or 6 (new-energy) alnum.
_VALID_PROV_BODY = re.compile(r"[A-Z][A-Z0-9]{5,6}")
# Lenient body for the province-missing case (1-2 leading letters + 4-7 alnum).
_VALID_CTX_BODY = re.compile(r"[A-Z]{1,2}[A-Z0-9]{4,7}")

# Lazy wetext singletons, keyed by the enable_0_to_9 flag. ``False`` entry means
# "tried and unavailable" so we don't re-import on every call.
_normalizers: dict[bool, object] = {}
_wetext_unavailable = False


def _get_normalizer(enable_0_to_9: bool):
    global _wetext_unavailable
    if _wetext_unavailable:
        return None
    if enable_0_to_9 in _normalizers:
        return _normalizers[enable_0_to_9]
    try:
        from wetext import Normalizer

        model = Normalizer(lang="zh", operator="itn", enable_0_to_9=enable_0_to_9)
    except Exception as exc:  # noqa: BLE001 — any import/build failure -> degrade
        _wetext_unavailable = True
        logger.warning(
            "wetext ITN unavailable, falling back to plate-only normalization: %s",
            exc,
        )
        return None
    _normalizers[enable_0_to_9] = model
    return model


def _itn_eligible(language: str | None) -> bool:
    """General ITN is Chinese-only. Empty/auto/unknown are treated as eligible
    because the pipeline often emits no explicit language for zh segments."""
    v = str(language or "").strip().lower()
    if not v or v in {"auto", "unknown", "mixed"}:
        return True
    return v.startswith("zh") or v in {"chinese", "mandarin", "cn"}


def _clean_plate_body(body: str) -> str:
    out: list[str] = []
    for ch in body:
        if ch in _SEP_CHARS:
            continue
        out.append(_CN_DIGIT.get(ch, ch))
    return "".join(out).upper()


def _prov_sub(m: re.Match) -> str:
    prov, body = m.group(1), m.group(2)
    core = body.rstrip(_SEP_CHARS)
    trail = body[len(core):]
    cleaned = _clean_plate_body(core)
    if _VALID_PROV_BODY.fullmatch(cleaned):
        return f"{prov}{cleaned}{trail}"
    return m.group(0)


def _ctx_sub(m: re.Match) -> str:
    ctx, body = m.group(1), m.group(2)
    core = body.rstrip(_SEP_CHARS)
    trail = body[len(core):]
    cleaned = _clean_plate_body(core)
    if _VALID_CTX_BODY.fullmatch(cleaned):
        return f"{ctx}{cleaned}{trail}"
    return m.group(0)


def normalize_plates(text: str) -> str:
    """Fix plate casing/separators/spoken-digits. Province-anchored first, then
    the keyword-anchored province-missing case. Pure regex, zero deps."""
    out = _PROV_RE.sub(_prov_sub, text)
    out = _CTX_RE.sub(_ctx_sub, out)
    return out


def normalize_final(
    text: str,
    language: str | None = "",
    *,
    enable_itn: bool = True,
    enable_plate: bool = True,
    enable_0_to_9: bool = False,
) -> str:
    """Normalize a final transcription for display.

    Order: general ITN (Chinese only) first so plates become mostly Arabic, then
    the plate pass polishes case/separators and handles ITN-missed plates. Any
    exception falls back to the last good text.
    """
    s = str(text or "")
    if not s.strip():
        return text
    out = s
    if enable_itn and _itn_eligible(language):
        model = _get_normalizer(enable_0_to_9)
        if model is not None:
            try:
                out = model.normalize(out)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ITN failed, keeping pre-ITN text: %s", exc)
                out = s
    if enable_plate:
        try:
            out = normalize_plates(out)
        except Exception as exc:  # noqa: BLE001
            logger.warning("plate normalization failed: %s", exc)
    return out


def normalize_final_text(text: str, language: str | None, cfg) -> str:
    """Config-driven wrapper used at the streaming/upload final-emission sites."""
    return normalize_final(
        text,
        language,
        enable_itn=cfg.enable_asr_itn,
        enable_plate=cfg.enable_asr_plate_normalize,
        enable_0_to_9=cfg.asr_itn_enable_0_to_9,
    )
