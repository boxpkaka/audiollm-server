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


def _flatten(raw: dict[str, Any]) -> dict[str, Any]:
    """Collect leaf entries from a (possibly grouped) config mapping.

    Nested dicts are treated purely as visual grouping containers: only their
    non-dict leaves become config fields, so a flat (un-nested) file still
    loads unchanged. This keeps the on-disk file groupable by feature while
    `Config` stays a flat dataclass and `override` keeps its flat key contract.

    Constraint: config values must not be objects (the dataclass has no
    dict-typed field). A leaf key colliding across groups is a config error;
    we keep the last value seen and warn so it is not silent.
    """
    flat: dict[str, Any] = {}

    def walk(node: dict[str, Any]) -> None:
        for key, value in node.items():
            if isinstance(value, dict):
                walk(value)
                continue
            if key in flat:
                logger.warning("Duplicate config key across groups: %s", key)
            flat[key] = value

    walk(raw)
    return flat


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
    # `enable_secondary_asr` is the resource gate: when False the secondary
    # vLLM is never queried (no partial noise gate, no final fusion).
    # `enable_dual_asr_fusion` controls only the *final-segment* path:
    # True  -> run primary + secondary in parallel and merge via
    #          `choose_fused_result` (requires secondary to be on);
    # False -> final is primary-only even if secondary is reachable,
    #          which saves one vLLM call per segment while keeping the
    #          partial noise gate functional. Inconsistent combinations
    #          (fusion=True, secondary=False) are downgraded at load time.
    enable_dual_asr_fusion: bool = True

    # ---- ASR: per-endpoint primary override (/tuling/ast/v3) -------------
    # The AST v3 endpoint serves a *different* primary model than the global
    # ``vllm_base_url``. Empty string means "fall back to the global primary";
    # a non-empty value routes only the ``/tuling/ast/v3`` endpoint's primary
    # to this upstream (binding wired in ``backend/main.py``). These stay out
    # of ``CLIENT_OVERRIDABLE_FIELDS`` — model URLs are SSRF-sensitive. That
    # endpoint also runs primary-only: secondary is force-disabled at the
    # route, so the local Qwen is never queried there.
    astv3_vllm_base_url: str = ""
    astv3_vllm_model_name: str = ""

    # ---- ASR: dual-model fusion knobs ------------------------------------
    fusion_similarity_threshold: float = 0.85
    fusion_min_primary_score: float = 0.55
    fusion_max_repetition_ratio: float = 0.35
    fusion_disagreement_threshold: float = 0.55
    fusion_hotword_boost: float = 0.12
    fusion_primary_score_margin: float = 0.08

    # ---- ASR: inverse text normalization (ITN) + license plate -----------
    # The model emits spoken-form text (六五四三八, 二零二四年); for display we
    # normalize finals to written form. Two independent switches (final only —
    # partials stay spoken-form to avoid flicker):
    #   enable_asr_itn             -> general ITN via wetext (Chinese only)
    #   enable_asr_plate_normalize -> zero-dep plate pass: uppercase plate
    #                                 letters, strip in-plate separators, map
    #                                 spoken digits, GB-plate-shape validated.
    # A province abbreviation misheard as a Latin letter (冀->J) is a
    # recognition error and is intentionally NOT recovered here.
    enable_asr_itn: bool = True
    asr_itn_enable_0_to_9: bool = False
    enable_asr_plate_normalize: bool = True

    # ---- Common: HTTP / pseudo-streaming ---------------------------------
    asr_request_timeout: float = 120
    enable_pseudo_stream: bool = True
    pseudo_stream_interval_ms: int = 500

    # ---- ASR: target speaker enrollment ----------------------------------
    # When a target-speaker enrollment is uploaded the primary ASR prompt
    # switches to the dual-audio "Given the speaker's voice:<enroll>\n
    # Transcribe what this speaker says in the following audio.<target>"
    # template. The clip is cached server-side for the duration of a
    # session so the WS stream does not have to retransmit it on every
    # VAD segment. v4 SFT data trained 1–8 s enrollment clips; anything
    # outside that window is silently OOD even when the API accepts it.
    asr_enrollment_min_sec: float = 1.0
    asr_enrollment_max_sec: float = 8.0
    asr_enrollment_ttl_sec: float = 3600.0
    asr_enrollment_max_entries: int = 256

    # ---- ASR: VAD segmentation -------------------------------------------
    # These VAD defaults mirror backend/config.json's vad block. They are pure
    # tuning thresholds (no per-deployment meaning), so keeping the in-code
    # fallback equal to the shipped values avoids a confusing third number;
    # config.json still overrides them at load time.
    vad_threshold: float = 0.65
    silence_duration_ms: int = 350
    vad_smoothing_alpha: float = 0.3
    vad_start_frames: int = 20
    vad_pre_speech_ms: int = 500
    vad_keep_tail_ms: int = 40
    min_segment_duration_ms: int = 350
    # 每段语音"首个 partial(伪流式中间结果)"的触发门槛:VAD 累积音频达到它才发出
    # 第一个 partial。因一段语音内 snapshot 单调增长(audio_buffer 只增不减),过了
    # 门槛后续 partial 必然更长,该门槛只对每段的首个 partial 真正 binding,故它实质
    # 是首字延迟旋钮 —— 与 vad_start_frames 一起按 max 决定首字(见
    # docs/tuling-ast-v3-protocol.md "首字延迟优化")。从 min_segment_duration_ms 解耦
    # 的原因:后者一参多职(还管 final 段过滤、flush 残余过滤),直接调它会放松短噪声
    # 段过滤。这里的 dataclass 默认 350(= min_segment_duration_ms)是"文件缺字段时的
    # 中性兜底";随附的 backend/config.json 显式设 200,即默认部署选择全局低延迟首字
    # (对所有产 partial 的端点生效,不止 tuling)。注意只把它降到 200 而 vad_start_frames
    # 仍 20 时,首字 max 仍由起音确认(约 320ms)主导,需同时调小 vad_start_frames 才到
    # 最优。不变量 first_partial<=min_segment 见 __post_init__。
    pseudo_stream_first_partial_ms: int = 350

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

    def __post_init__(self) -> None:
        # Silently enforce the invariant that fusion requires a secondary
        # decoder. Dataclass-level so any construction path (load, override,
        # direct ctor in tests) yields a consistent Config — downstream
        # code can rely on `enable_dual_asr_fusion` being a clean truth.
        # Loud WARNING happens in `load_config` against the raw input so
        # operators notice misconfigured files; in-process overrides stay
        # silent to avoid log spam.
        if self.enable_dual_asr_fusion and not self.enable_secondary_asr:
            object.__setattr__(self, "enable_dual_asr_fusion", False)
        # 首个 partial 门槛若严于 final 段最小时长,partial 就会永远比 final 晚、失去
        # "中间结果"意义;夹到 <= min_segment_duration_ms。和 fusion 不变量一样下沉
        # 到 dataclass,确保 load/override/直接构造(测试)各路径都一致。
        if self.pseudo_stream_first_partial_ms > self.min_segment_duration_ms:
            object.__setattr__(
                self, "pseudo_stream_first_partial_ms", self.min_segment_duration_ms
            )

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
        # `__post_init__` silently fixes any inconsistent combinations,
        # which is the right behavior for in-process overrides.
        return replace(self, **accepted) if accepted else self

    def override_client(self, **kwargs: Any) -> Config:
        """Like :meth:`override` but only honors client-overridable fields.

        For untrusted per-connection overrides (WebSocket ``start.config`` and
        AST v3 ``parameter.asr_config``). Infrastructure / secret / process-wide
        fields (model URLs -> SSRF, API keys, HTTP pools, job queues) are not in
        ``CLIENT_OVERRIDABLE_FIELDS`` and are dropped with a WARN: an operator
        can see a client reaching for a restricted knob, while the long-lived
        connection is never broken. Type coercion and dataclass invariants are
        delegated to :meth:`override`.
        """
        allowed: dict[str, Any] = {}
        for k, v in kwargs.items():
            if k in CLIENT_OVERRIDABLE_FIELDS:
                allowed[k] = v
            else:
                logger.warning(
                    "Ignoring non-overridable config field from client: %s", k
                )
        return self.override(**allowed)


# Per-connection client overrides are restricted to this whitelist. Guiding
# line: a field is overridable only if it tunes how *this session* processes its
# own audio. Process-wide knobs (HTTP pools, job queues, cache capacity), backend
# routing (``*_vllm_base_url`` -> SSRF, ``*_model_name``) and secrets
# (``text_cleanup_api_key*``) are intentionally excluded so an untrusted client
# cannot reach them via ``start.config`` / ``parameter.asr_config``.
CLIENT_OVERRIDABLE_FIELDS: frozenset[str] = frozenset({
    # VAD / segmentation
    "vad_threshold",
    "silence_duration_ms",
    "vad_smoothing_alpha",
    "vad_start_frames",
    "vad_pre_speech_ms",
    "vad_keep_tail_ms",
    "min_segment_duration_ms",
    # Pseudo-streaming partials
    "enable_pseudo_stream",
    "pseudo_stream_interval_ms",
    "pseudo_stream_first_partial_ms",
    # ASR model combination / timeouts
    "enable_primary_asr",
    "enable_secondary_asr",
    "enable_dual_asr_fusion",
    "primary_asr_timeout",
    "asr_request_timeout",
    "debug_show_dual_asr",
    # Dual-model fusion thresholds
    "fusion_similarity_threshold",
    "fusion_min_primary_score",
    "fusion_max_repetition_ratio",
    "fusion_disagreement_threshold",
    "fusion_hotword_boost",
    "fusion_primary_score_margin",
    # TS-ASR enrollment bounds
    "asr_enrollment_min_sec",
    "asr_enrollment_max_sec",
    "asr_enrollment_ttl_sec",
    # Emotion (baseline + paralinguistic spec) per-request tuning
    "emotion_task_mode",
    "emotion_request_timeout",
    "emotion_max_audio_seconds",
    "emotion_spec_task_mode",
    "emotion_spec_request_timeout",
    "emotion_spec_max_audio_seconds",
})

# Fail-fast: a whitelisted name that is not a real Config field is a typo that
# would silently make a knob un-overridable forever, so reject it at import.
_unknown_overridable = CLIENT_OVERRIDABLE_FIELDS - {f.name for f in fields(Config)}
if _unknown_overridable:
    raise ValueError(
        "CLIENT_OVERRIDABLE_FIELDS has unknown Config fields: "
        f"{sorted(_unknown_overridable)}"
    )


def load_config(path: Path | None = None) -> Config:
    raw = _flatten(_load_json(path or _CONFIG_PATH))
    valid_names = {f.name for f in fields(Config)}
    filtered = {k: v for k, v in raw.items() if k in valid_names}
    # Surface a loud WARNING when the on-disk config file specifies an
    # impossible combination, so operators notice — silent downgrade by
    # `__post_init__` still happens, this is just the log line.
    if raw.get("enable_dual_asr_fusion", True) and not raw.get(
        "enable_secondary_asr", True
    ):
        logger.warning(
            "enable_dual_asr_fusion=true requires enable_secondary_asr=true; "
            "downgrading fusion to false"
        )
    return Config(**filtered) if filtered else Config()


# Process-wide default Config singleton. Modules that don't carry a
# per-session Config (module-level helpers, ``value or <default>`` fallbacks)
# read fields off this instead of re-reading the file. ``load_config()``
# stays the single entry point; this is just its cached default instance, so
# every reader reaches config exactly one way: a Config object.
default_config = load_config()
