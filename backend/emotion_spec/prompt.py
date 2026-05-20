"""Prompt templates for the AmphionSPEC paralinguistic-emotion model.

The model is trained with two prompts (literal strings — do NOT rephrase):

- ``ser``  : ``Classify the emotion of the following audio:{speech}``
  Identical to the baseline emotion model; emits one of the 8-way
  :data:`backend.emotion.prompt.SER_TAXONOMY` labels.
- ``sepc`` : ``Describe the paralinguistic emotion cues of the following audio:{speech}``
  Free-form description of paralinguistic cues (prosody, tempo, voice
  quality, etc.). Note the literal training token is ``sepc`` (sic), not
  ``spec`` — the config/path prefix uses ``spec`` for readability but the
  wire/prompt label stays ``sepc``.

When serving via vLLM (OpenAI-compatible chat completions) the ``{speech}``
placeholder is replaced by an ``input_audio`` content item, so on the wire
we only need the prompt prefix as plain text.
"""

from __future__ import annotations

from typing import Literal

EmotionSpecMode = Literal["ser", "sepc"]

SER_PROMPT = "Classify the emotion of the following audio:"
SEPC_PROMPT = "Describe the paralinguistic emotion cues of the following audio:"

PROMPTS: dict[str, str] = {
    "ser": SER_PROMPT,
    "sepc": SEPC_PROMPT,
}

DEFAULT_MODE: EmotionSpecMode = "sepc"

_ALIASES: dict[str, EmotionSpecMode] = {
    "ser": "ser",
    "sepc": "sepc",
    "spec": "sepc",
    "sec": "sepc",
    "para": "sepc",
    "paralinguistic": "sepc",
}


def normalize_mode(value: object) -> EmotionSpecMode:
    """Coerce a user-supplied mode string to one of the supported modes.

    Accepts the literal training tokens (``ser``/``sepc``) as well as a
    handful of aliases (``spec``/``sec``/``para``/``paralinguistic``) so
    callers don't have to remember the typo-looking ``sepc`` spelling.
    """
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _ALIASES:
            return _ALIASES[lowered]
    return DEFAULT_MODE


def get_prompt(mode: EmotionSpecMode) -> str:
    return PROMPTS[mode]
