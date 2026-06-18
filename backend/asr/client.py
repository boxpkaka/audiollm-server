import re
from typing import Any, TypedDict

from ..config import default_config
from ..http_client import get_client
from .prompt_templates import audio_item
from .prompt_templates import build_primary_messages as _build_primary_messages


class ASRResult(TypedDict):
    transcription: str
    reported_hotwords: list[str]
    raw_text: str
    detected_language: str | None


def build_primary_messages(
    target_wav_base64: str,
    *,
    hotwords: list[str] | None = None,
    enrollment_wav_base64: str | None = None,
    template: str | None = None,
) -> list[dict]:
    """Build primary ASR messages for the selected model prompt template."""
    return _build_primary_messages(
        target_wav_base64,
        hotwords=hotwords,
        enrollment_wav_base64=enrollment_wav_base64,
        template=template or default_config.vllm_prompt_template,
    )


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks).strip()
    return str(content or "")


def build_audio_only_messages(audio_wav_base64: str) -> list[dict]:
    """Single-audio prompt without any text — used by the Qwen3 secondary
    path, which is trained as a pure ASR model and ignores text guidance."""
    return [
        {
            "role": "user",
            "content": [audio_item(audio_wav_base64)],
        }
    ]


def _parse_hotwords_field(value: str) -> list[str]:
    text = value.strip()
    if not text:
        return []
    lowered = text.lower()
    if lowered in {"n/a", "na", "none", "null", "-"}:
        return []
    return [item.strip() for item in re.split(r"[,，;；]", text) if item.strip()]


def _parse_language_field(value: str) -> str | None:
    v = str(value or "").strip()
    if not v:
        return None
    if v.lower() in {"n/a", "na", "none", "null", "-"}:
        return None
    return v


def _postprocess_asr_text(text: str) -> str:
    """Normalize provider-specific wrappers to plain transcription text."""
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(
        r"^\s*language\s+[A-Za-z\u4e00-\u9fff_-]+\s*<asr_text>\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^\s*language\s+[A-Za-z\u4e00-\u9fff_-]+\s*[:：-]?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def detect_and_fix_repetitions(text: str, threshold: int = 20) -> str:
    """Collapse pathological decode loops while leaving normal text untouched."""
    source = str(text or "")
    if len(source) <= threshold:
        return source

    fixed = source
    max_unit = min(16, max(1, len(fixed) // (threshold + 1)))
    for unit_len in range(1, max_unit + 1):
        out: list[str] = []
        i = 0
        changed = False
        while i < len(fixed):
            unit = fixed[i : i + unit_len]
            if len(unit) < unit_len:
                out.append(fixed[i:])
                break
            count = 1
            j = i + unit_len
            while fixed[j : j + unit_len] == unit:
                count += 1
                j += unit_len
            if count > threshold:
                out.append(unit)
                i = j
                changed = True
            else:
                out.append(fixed[i])
                i += 1
        if changed:
            fixed = "".join(out)
    return fixed


def parse_model_output(
    raw_text: str,
    *,
    enable_repetition_fix: bool | None = None,
) -> ASRResult:
    """Parse model output wrappers and normalize to plain transcription text."""
    raw = str(raw_text or "").strip()
    if not raw:
        return ASRResult(
            transcription="",
            reported_hotwords=[],
            raw_text="",
            detected_language=None,
        )

    normalized = raw.replace("\\r\\n", "\n").replace("\\n", "\n")

    lang_m = re.search(
        r"(?:^|\n)\s*language\s*:\s*([^\n]*)",
        normalized,
        flags=re.IGNORECASE,
    )
    detected_language = (
        _parse_language_field(lang_m.group(1)) if lang_m else None
    )
    if detected_language is None:
        qwen_lang_m = re.search(
            r"(?:^|\n)\s*language\s+([A-Za-z\u4e00-\u9fff_-]+)\s*<asr_text>",
            normalized,
            flags=re.IGNORECASE,
        )
        detected_language = (
            _parse_language_field(qwen_lang_m.group(1)) if qwen_lang_m else None
        )

    hw_m = re.search(
        r"(?:^|\n)\s*hotwords\s*:\s*(.+?)(?=\n\s*(?:language|transcription)\s*:|\Z)",
        normalized,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not hw_m:
        hw_m = re.search(
            r"(?:^|\n)\s*hotwords\s*:\s*(.+?)(?=\n\s*[A-Za-z_]+\s*:|\Z)",
            normalized,
            flags=re.IGNORECASE | re.DOTALL,
        )
    reported_hotwords = (
        _parse_hotwords_field(hw_m.group(1)) if hw_m else []
    )

    hm = re.search(r"(?i)hotwords\s*:", normalized)
    tm = re.search(r"(?i)transcription\s*:", normalized)
    h_start = hm.start() if hm else -1
    t_start = tm.start() if tm else -1

    transcription = ""
    if tm:
        if h_start >= 0 and h_start < t_start:
            m_tr = re.search(
                r"(?:^|\n)\s*transcription\s*:\s*(.*)\Z",
                normalized,
                flags=re.IGNORECASE | re.DOTALL,
            )
            transcription = m_tr.group(1).strip() if m_tr else ""
        else:
            m_tr = re.search(
                r"(?:^|\n)\s*transcription\s*:\s*(.+?)(?=\n\s*hotwords\s*:|\Z)",
                normalized,
                flags=re.IGNORECASE | re.DOTALL,
            )
            transcription = (
                m_tr.group(1).strip() if m_tr else normalized.strip()
            )
    else:
        transcription = normalized.strip()

    transcription = _postprocess_asr_text(transcription)
    if enable_repetition_fix is None:
        enable_repetition_fix = default_config.enable_asr_repetition_fix
    if enable_repetition_fix:
        transcription = detect_and_fix_repetitions(transcription)

    return ASRResult(
        transcription=transcription,
        reported_hotwords=reported_hotwords,
        raw_text=raw,
        detected_language=detected_language,
    )


async def _post_chat(
    messages: list[dict],
    *,
    base_url: str,
    model_name: str,
    timeout: float,
) -> ASRResult:
    client = get_client()
    base = base_url.rstrip("/")
    resp = await client.post(
        f"{base}/v1/chat/completions",
        json={
            "model": model_name,
            "messages": messages,
            "max_tokens": 512,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    raw_text = _content_to_text(resp.json()["choices"][0]["message"]["content"])
    return parse_model_output(raw_text)


async def query_audio_model(
    audio_wav_base64: str,
    hotwords: list[str] | None = None,
    *,
    src_lang: str = "N/A",  # accepted for callsite compatibility, ignored
    enrollment_wav_base64: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
    prompt_template: str | None = None,
    timeout: float | None = None,
) -> ASRResult:
    """Primary ASR call.

    ``src_lang`` is intentionally not forwarded into the prompt. The
    primary model's prompt format is selected by ``prompt_template`` (or
    the configured default) so Amphion 4B and 1.7B can coexist without
    duplicating call sites.
    """
    _ = src_lang  # noqa: F841 — preserved for compatibility, see docstring
    messages = build_primary_messages(
        audio_wav_base64,
        hotwords=hotwords,
        enrollment_wav_base64=enrollment_wav_base64,
        template=prompt_template,
    )
    return await _post_chat(
        messages,
        base_url=base_url or default_config.vllm_base_url,
        model_name=model_name or default_config.vllm_model_name,
        timeout=timeout if timeout is not None else default_config.asr_request_timeout,
    )


async def query_audio_model_secondary(
    audio_wav_base64: str,
    hotwords: list[str] | None = None,
    *,
    base_url: str | None = None,
    model_name: str | None = None,
    timeout: float | None = None,
) -> ASRResult:
    _ = hotwords
    messages = build_audio_only_messages(audio_wav_base64)
    return await _post_chat(
        messages,
        base_url=base_url or default_config.secondary_vllm_base_url,
        model_name=model_name or default_config.secondary_vllm_model_name,
        timeout=timeout if timeout is not None else default_config.asr_request_timeout,
    )
