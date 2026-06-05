import re
from typing import Any, TypedDict

from ..config import default_config
from ..http_client import get_client


class ASRResult(TypedDict):
    transcription: str
    reported_hotwords: list[str]
    raw_text: str
    detected_language: str | None


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


def _sanitize_hotwords(hotwords: list[str] | None) -> list[str]:
    """Drop empties, dedup, preserve order. Caller is expected to enforce
    the count / length limits at the UI layer, but we still defensively
    strip whitespace here so prompt bytes match training data exactly."""
    if not hotwords:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for w in hotwords:
        s = str(w or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _audio_item(wav_b64: str) -> dict[str, Any]:
    return {
        "type": "input_audio",
        "input_audio": {"data": wav_b64, "format": "wav"},
    }


def build_primary_messages(
    target_wav_base64: str,
    *,
    hotwords: list[str] | None = None,
    enrollment_wav_base64: str | None = None,
) -> list[dict]:
    """Build OpenAI-style ``messages`` aligned byte-for-byte with v4 SFT.

    Four shapes are emitted based on (has_enrollment, has_hotwords):

    1. plain ASR ......... "Transcribe the following audio." + <audio>
    2. ASR + hotwords .... "Transcribe the following audio.\\nHotwords: ..." + <audio>
    3. TS-ASR ............ "Given the speaker's voice:" + <enroll>
                           + "\\nTranscribe what this speaker says in the following audio."
                           + <target>
    4. TS-ASR + hotwords . same as (3) but second text appends "\\nHotwords: ..."

    The ``Language:`` line is intentionally omitted because v4 training data
    has 0% coverage of that line; including it would push every request OOD.

    The leading ``\\n`` on the second text in TS-ASR shapes is required —
    the training pipeline assembles text by simple concatenation, so the
    newline lives at the start of the *second* text block rather than at
    the end of the first. Reversing this triggers a different token
    sequence at the model.
    """
    hws = _sanitize_hotwords(hotwords)
    hw_str = ",".join(hws) if hws else ""
    has_enroll = bool(enrollment_wav_base64)

    content: list[dict[str, Any]] = []
    if has_enroll:
        content.append({"type": "text", "text": "Given the speaker's voice:"})
        content.append(_audio_item(enrollment_wav_base64))  # type: ignore[arg-type]
        second = "\nTranscribe what this speaker says in the following audio."
        if hw_str:
            second += f"\nHotwords: {hw_str}"
        content.append({"type": "text", "text": second})
        content.append(_audio_item(target_wav_base64))
    else:
        first = "Transcribe the following audio."
        if hw_str:
            first += f"\nHotwords: {hw_str}"
        content.append({"type": "text", "text": first})
        content.append(_audio_item(target_wav_base64))

    return [{"role": "user", "content": content}]


def build_audio_only_messages(audio_wav_base64: str) -> list[dict]:
    """Single-audio prompt without any text — used by the Qwen3 secondary
    path, which is trained as a pure ASR model and ignores text guidance."""
    return [
        {
            "role": "user",
            "content": [_audio_item(audio_wav_base64)],
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


def parse_model_output(raw_text: str) -> ASRResult:
    """Parse model output with ``Language:`` / ``Hotwords:`` / ``Transcription:`` lines."""
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
    timeout: float | None = None,
) -> ASRResult:
    """Primary ASR call.

    ``src_lang`` is intentionally not forwarded into the prompt — v4 SFT
    has 0% coverage of ``Language:`` lines, so injecting one would make
    every request OOD. Callers still pass it for future-proofing and so
    the upload path can populate ``detected_language`` defaults; the
    actual language tag is read back from the model's structured
    output (``Language: ...`` if the model emits it).
    """
    _ = src_lang  # noqa: F841 — preserved for compatibility, see docstring
    messages = build_primary_messages(
        audio_wav_base64,
        hotwords=hotwords,
        enrollment_wav_base64=enrollment_wav_base64,
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
