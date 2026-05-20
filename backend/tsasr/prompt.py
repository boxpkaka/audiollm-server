"""Prompt builder for Target-Speaker ASR (TS-ASR).

The prompt format mirrors the v3 SFT recipe used to fine-tune the
in-house Amphion-3B TS-ASR checkpoint (see
``AmphionASR/src/integrations/ms_swift/data/convert.py``'s
``build_unified_instruction``). v3 only ever saw two prompt shapes,
both with enrollment and with the sentence-final period::

    # Template A -- no hotwords
    Given the speaker's voice:<audio>
    Transcribe what this speaker says in the following audio.<audio>

    # Template B -- with hotwords
    Given the speaker's voice:<audio>
    Transcribe what this speaker says in the following audio.
    Hotwords: word1,word2,...<audio>

Crucially, the ``Hotwords:`` line goes *after* the transcribe
instruction (not between enrollment and instruction) and the hotword
list is comma-joined with no spaces and no trailing period.

Optional fields not covered by v3 training (``Language:`` line,
``Speaker traits:`` segment, the no-enrollment branch) are deliberately
excluded so we never feed the checkpoint an OOD prompt shape.
"""

from __future__ import annotations

from typing import Any

ENROLL_PREFIX = "Given the speaker's voice:"
TRANSCRIBE_PREFIX = "Transcribe what this speaker says in the following audio."


def _audio_chunk(wav_base64: str) -> dict[str, Any]:
    return {
        "type": "input_audio",
        "input_audio": {"data": wav_base64, "format": "wav"},
    }


def format_hotwords_line(hotwords: list[str] | None) -> str:
    """Format the ``Hotwords:`` line for the v3 TS-ASR prompt.

    Returns ``""`` when the list is empty / None so callers can
    unconditionally concatenate the result. The resulting line is
    prefixed with a single ``"\\n"`` so it can be appended directly
    to the transcribe instruction line; the hotword list is joined
    with ``","`` (no spaces, no trailing period) to match the
    training-time format produced by ``build_hotwords_for_sample``.
    """
    if not hotwords:
        return ""
    cleaned = [str(h).strip() for h in hotwords if str(h or "").strip()]
    if not cleaned:
        return ""
    return "\nHotwords: " + ",".join(cleaned)


def build_tsasr_content(
    enrollment_wav_b64: str,
    mixed_wav_b64: str,
    *,
    hotwords: list[str] | None = None,
    voice_traits: str | None = None,
) -> list[dict[str, Any]]:
    """Assemble the OpenAI-compatible chat ``content`` list for TS-ASR.

    Output layout (positions in ``content``)::

        [0]  text:  ENROLL_PREFIX                          # "Given the speaker's voice:"
        [1]  audio: enrollment
        [2]  text:  "\\n" + TRANSCRIBE_PREFIX + (optional "\\nHotwords: a,b,c")
        [3]  audio: mixed

    Joining the transcribe instruction and the optional hotword line
    into a single ``text`` block mirrors the ``"\\n".join(lines) +
    _AUDIO_PLACEHOLDER`` pattern used at training time -- the mixed
    ``<audio>`` token must follow the last text line with no extra
    newline, otherwise the model sees an OOD prompt shape and quietly
    degrades.

    ``voice_traits`` is accepted for backward compatibility with older
    callers (the streaming engine still caches it for logging) but is
    intentionally NOT injected into the prompt: v3 SFT data has no
    ``Speaker traits:`` segment, so adding one would push the prompt
    off the training distribution.
    """
    del voice_traits

    transcribe_text = "\n" + TRANSCRIBE_PREFIX + format_hotwords_line(hotwords)

    return [
        {"type": "text", "text": ENROLL_PREFIX},
        _audio_chunk(enrollment_wav_b64),
        {"type": "text", "text": transcribe_text},
        _audio_chunk(mixed_wav_b64),
    ]
