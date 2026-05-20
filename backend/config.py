from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

HOP_SIZE = 160  # 10ms at 16kHz, TEN VAD recommended
SAMPLE_RATE = 16000


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        logger.warning("Config file not found: %s, using built-in defaults", path)
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True)
class Config:
    # ---- ASR: primary / secondary vLLM endpoints --------------------------
    vllm_base_url: str = "http://localhost:8000"
    vllm_model_name: str = "Amphion/AmphionASR-4.3B"
    secondary_vllm_base_url: str = "http://localhost:8001"
    secondary_vllm_model_name: str = "Qwen/Qwen3-ASR-1.7B"
    enable_secondary_asr: bool = True
    enable_primary_asr: bool = True
    primary_asr_timeout: float = 4.0
    debug_show_dual_asr: bool = True

    # ---- ASR: dual-model fusion knobs ------------------------------------
    fusion_similarity_threshold: float = 0.85
    fusion_min_primary_score: float = 0.55
    fusion_max_repetition_ratio: float = 0.35
    fusion_disagreement_threshold: float = 0.55
    fusion_hotword_boost: float = 0.12
    fusion_primary_score_margin: float = 0.08

    # ---- Common: HTTP / pseudo-streaming ---------------------------------
    asr_request_timeout: float = 120
    enable_pseudo_stream: bool = True
    pseudo_stream_interval_ms: int = 500

    # ---- ASR: VAD segmentation -------------------------------------------
    vad_threshold: float = 0.5
    silence_duration_ms: int = 200
    vad_smoothing_alpha: float = 0.35
    vad_start_frames: int = 3
    vad_pre_speech_ms: int = 500
    vad_end_frames: int = 20
    vad_keep_tail_ms: int = 40
    min_segment_duration_ms: int = 350

    # ---- Emotion recognition: vLLM endpoint ------------------------------
    # The Amphion multi-task model is trained to handle SER/SEC alongside ASR
    # via different text prompts, so by default we point the emotion endpoint
    # at the same backend as the primary ASR. Override
    # ``emotion_vllm_base_url`` if you serve a dedicated emotion model.
    emotion_vllm_base_url: str = "http://localhost:8000"
    emotion_vllm_model_name: str = "Amphion/AmphionASR-4.3B"
    emotion_request_timeout: float = 30.0
    # Amphion SER/SEC training uses 1-20s utterances, so we cap longer audio
    # to the trailing 20 seconds (where the most recent speech lives).
    emotion_max_audio_seconds: float = 20.0
    # Default task variant when the client doesn't specify one in start.mode.
    # "ser" -> single label classification; "sec" -> free-form caption.
    emotion_task_mode: str = "ser"
    # Whole-utterance HTTP job API backpressure (in-process store).
    emotion_max_concurrent_jobs: int = 8
    emotion_job_queue_max: int = 64
    emotion_job_ttl_sec: float = 3600.0

    # ---- Paralinguistic emotion model (AmphionSPEC) ----------------------
    # Independent vLLM endpoint that serves the AmphionSPEC checkpoint. It
    # is trained with two prompts: ``ser`` (same 8-way label set as the
    # baseline emotion model) and ``sepc`` (free-form description of
    # paralinguistic emotion cues — prosody, tempo, voice quality, etc.).
    # Configuration mirrors the emotion knobs so it can scale and back-off
    # independently from the baseline emotion store.
    emotion_spec_vllm_base_url: str = "http://localhost:9001"
    emotion_spec_vllm_model_name: str = "AmphionSPEC"
    emotion_spec_request_timeout: float = 30.0
    emotion_spec_max_audio_seconds: float = 20.0
    # Default mode when the client omits ``mode``; the prompt label
    # ``sepc`` is the literal training token (do not rename to ``spec``).
    emotion_spec_task_mode: str = "sepc"
    emotion_spec_max_concurrent_jobs: int = 8
    emotion_spec_job_queue_max: int = 64
    emotion_spec_job_ttl_sec: float = 3600.0

    # Shared httpx pool ceiling for all vLLM / upstream HTTP calls.
    http_max_connections: int = 32
    http_max_keepalive_connections: int = 16

    # ---- Text cleanup LLM (DashScope OpenAI-compatible) -------------------
    text_cleanup_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    text_cleanup_model_name: str = "qwen2.5-32b-instruct"
    text_cleanup_api_key_env: str = "DASHSCOPE_API_KEY"
    text_cleanup_api_key: str = ""
    text_cleanup_timeout: float = 30.0
    text_cleanup_max_tokens: int = 1024

    @property
    def resolved_text_cleanup_api_key(self) -> str:
        """Return configured API key, preferring the named environment variable."""
        env_name = self.text_cleanup_api_key_env.strip()
        if env_name:
            value = os.getenv(env_name, "").strip()
            if value:
                return value
        return self.text_cleanup_api_key.strip()

    def override(self, **kwargs: Any) -> Config:
        """Return a new Config with the given fields replaced (unknown keys ignored)."""
        valid_names = {f.name for f in fields(self)}
        accepted: dict[str, Any] = {}
        for k, v in kwargs.items():
            if k not in valid_names:
                continue
            expected = type(getattr(self, k))
            try:
                accepted[k] = expected(v)
            except (TypeError, ValueError):
                logger.warning("Ignoring invalid config override %s=%r", k, v)
        return replace(self, **accepted) if accepted else self


def load_config(path: Path | None = None) -> Config:
    raw = _load_json(path or _CONFIG_PATH)
    valid_names = {f.name for f in fields(Config)}
    filtered = {k: v for k, v in raw.items() if k in valid_names}
    return Config(**filtered) if filtered else Config()


_default = load_config()

# ---------------------------------------------------------------------------
# Module-level constants for backward compatibility.
# Modules that don't need per-session override can keep importing these.
# ---------------------------------------------------------------------------
VLLM_BASE_URL = _default.vllm_base_url
VLLM_MODEL_NAME = _default.vllm_model_name
SECONDARY_VLLM_BASE_URL = _default.secondary_vllm_base_url
SECONDARY_VLLM_MODEL_NAME = _default.secondary_vllm_model_name
ENABLE_SECONDARY_ASR = _default.enable_secondary_asr
ENABLE_PRIMARY_ASR = _default.enable_primary_asr
PRIMARY_ASR_TIMEOUT = _default.primary_asr_timeout
DEBUG_SHOW_DUAL_ASR = _default.debug_show_dual_asr

FUSION_SIMILARITY_THRESHOLD = _default.fusion_similarity_threshold
FUSION_MIN_PRIMARY_SCORE = _default.fusion_min_primary_score
FUSION_MAX_REPETITION_RATIO = _default.fusion_max_repetition_ratio
FUSION_DISAGREEMENT_THRESHOLD = _default.fusion_disagreement_threshold
FUSION_HOTWORD_BOOST = _default.fusion_hotword_boost
FUSION_PRIMARY_SCORE_MARGIN = _default.fusion_primary_score_margin

ASR_REQUEST_TIMEOUT = _default.asr_request_timeout
ENABLE_PSEUDO_STREAM = _default.enable_pseudo_stream
PSEUDO_STREAM_INTERVAL_MS = _default.pseudo_stream_interval_ms

VAD_THRESHOLD = _default.vad_threshold
SILENCE_DURATION_MS = _default.silence_duration_ms
VAD_SMOOTHING_ALPHA = _default.vad_smoothing_alpha
VAD_START_FRAMES = _default.vad_start_frames
VAD_PRE_SPEECH_MS = _default.vad_pre_speech_ms
VAD_END_FRAMES = _default.vad_end_frames
VAD_KEEP_TAIL_MS = _default.vad_keep_tail_ms
MIN_SEGMENT_DURATION_MS = _default.min_segment_duration_ms

EMOTION_VLLM_BASE_URL = _default.emotion_vllm_base_url
EMOTION_VLLM_MODEL_NAME = _default.emotion_vllm_model_name
EMOTION_REQUEST_TIMEOUT = _default.emotion_request_timeout
EMOTION_MAX_AUDIO_SECONDS = _default.emotion_max_audio_seconds
EMOTION_TASK_MODE = _default.emotion_task_mode
EMOTION_MAX_CONCURRENT_JOBS = _default.emotion_max_concurrent_jobs
EMOTION_JOB_QUEUE_MAX = _default.emotion_job_queue_max
EMOTION_JOB_TTL_SEC = _default.emotion_job_ttl_sec

EMOTION_SPEC_VLLM_BASE_URL = _default.emotion_spec_vllm_base_url
EMOTION_SPEC_VLLM_MODEL_NAME = _default.emotion_spec_vllm_model_name
EMOTION_SPEC_REQUEST_TIMEOUT = _default.emotion_spec_request_timeout
EMOTION_SPEC_MAX_AUDIO_SECONDS = _default.emotion_spec_max_audio_seconds
EMOTION_SPEC_TASK_MODE = _default.emotion_spec_task_mode
EMOTION_SPEC_MAX_CONCURRENT_JOBS = _default.emotion_spec_max_concurrent_jobs
EMOTION_SPEC_JOB_QUEUE_MAX = _default.emotion_spec_job_queue_max
EMOTION_SPEC_JOB_TTL_SEC = _default.emotion_spec_job_ttl_sec

HTTP_MAX_CONNECTIONS = _default.http_max_connections
HTTP_MAX_KEEPALIVE_CONNECTIONS = _default.http_max_keepalive_connections

TEXT_CLEANUP_BASE_URL = _default.text_cleanup_base_url
TEXT_CLEANUP_MODEL_NAME = _default.text_cleanup_model_name
TEXT_CLEANUP_API_KEY_ENV = _default.text_cleanup_api_key_env
TEXT_CLEANUP_TIMEOUT = _default.text_cleanup_timeout
TEXT_CLEANUP_MAX_TOKENS = _default.text_cleanup_max_tokens
