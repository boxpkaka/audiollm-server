from __future__ import annotations

import json
import re
from typing import Any, TypedDict

import httpx

from ..config import Config, load_config
from ..http_client import get_client


class TextCleanupResult(TypedDict):
    text: str
    raw_text: str
    model: str


class TextCleanupConfigError(RuntimeError):
    """Raised when the text cleanup model cannot be called due to configuration."""


SYSTEM_PROMPT = """你是一个面向语音识别结果的文本清洗助手。

任务：
1. 只基于输入的 ASR 文本进行清洗，不补充、扩写或推测音频中没有出现的信息。
2. 修正常见错别字、同音错写、标点、空格、大小写和明显重复词。
3. 保留 ASR 文本中的人名、产品名、数字、专有名词和原有语义。
4. 若 ASR 文本为空，输出空字符串。

输出要求：
只返回 JSON：{"cleaned_text":"..."}，不要返回解释。"""


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        return "\n".join(chunks).strip()
    return str(content or "")


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL)
    return fenced.group(1).strip() if fenced else stripped


def parse_cleanup_output(raw_text: str) -> str:
    raw = _strip_code_fence(str(raw_text or "")).strip()
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(parsed, dict):
        value = parsed.get("cleaned_text")
        if isinstance(value, str):
            return value.strip()
    if isinstance(parsed, str):
        return parsed.strip()
    return raw


def build_cleanup_messages(
    *,
    asr_text: str,
    language: str,
    emotion: dict[str, Any] | None,
) -> list[dict[str, str]]:
    payload = {
        "language": language or "",
        "asr_text": asr_text,
        "emotion": emotion or {},
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "请清洗以下 ASR 文本。情感信息仅用于判断语气和标点，"
                "不得据此新增任何内容。\n"
                f"{json.dumps(payload, ensure_ascii=False)}"
            ),
        },
    ]


async def clean_asr_text(
    asr_text: str,
    *,
    hotwords: list[str] | None = None,
    language: str = "",
    emotion: dict[str, Any] | None = None,
    cfg: Config | None = None,
) -> TextCleanupResult:
    # Hotwords are intentionally not sent to the cleanup LLM. They belong to
    # the ASR prompt itself; cleanup must not correct terms that ASR did not hear.
    _ = hotwords
    cfg = cfg or load_config()
    model_name = cfg.text_cleanup_model_name
    api_key = cfg.resolved_text_cleanup_api_key
    if not api_key:
        raise TextCleanupConfigError(
            f"missing text cleanup API key; set {cfg.text_cleanup_api_key_env} "
            "or backend/config.json text_cleanup_api_key"
        )

    base = cfg.text_cleanup_base_url.rstrip("/")
    client = get_client()
    try:
        resp = await client.post(
            f"{base}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": build_cleanup_messages(
                    asr_text=asr_text,
                    language=language,
                    emotion=emotion,
                ),
                "temperature": 0.1,
                "max_tokens": int(cfg.text_cleanup_max_tokens),
            },
            timeout=float(cfg.text_cleanup_timeout),
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500]
        raise RuntimeError(
            f"text cleanup model returned HTTP {exc.response.status_code}: {detail}"
        ) from exc

    raw_text = _content_to_text(resp.json()["choices"][0]["message"]["content"])
    return TextCleanupResult(
        text=parse_cleanup_output(raw_text),
        raw_text=raw_text,
        model=model_name,
    )
