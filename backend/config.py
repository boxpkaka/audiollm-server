from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# 配置真源: 项目根 config.yaml; CONFIG_PATH 环境变量可覆盖(测试 / 多部署)。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"

HOP_SIZE = 160  # 10ms at 16kHz, TEN VAD recommended
SAMPLE_RATE = 16000

# ${VAR} 环境变量插值; 未设置 -> 空串(import 期不因缺密钥而崩, 调用时才暴露)。
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _interpolate_env(node: Any) -> Any:
    """Recursively replace ``${VAR}`` references inside parsed YAML strings."""
    if isinstance(node, str):
        return _ENV_REF.sub(lambda m: os.getenv(m.group(1), ""), node)
    if isinstance(node, dict):
        return {k: _interpolate_env(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_interpolate_env(v) for v in node]
    return node


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        logger.warning("Config file not found: %s, using built-in defaults", path)
        return {}
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"Config root must be a mapping, got {type(raw).__name__}: {path}"
        )
    return _interpolate_env(raw)


def _flatten_leaves(node: dict[str, Any]) -> dict[str, Any]:
    """Collect non-dict leaves from the feature-grouped ``defaults`` block.

    Nested groups (vad / asr / fusion / ...) are visual grouping only; their
    leaves become flat ``Config`` field candidates. A leaf key colliding across
    groups is a config error: warn and keep the last value (never silent).
    """
    flat: dict[str, Any] = {}

    def walk(n: dict[str, Any]) -> None:
        for key, value in n.items():
            if isinstance(value, dict):
                walk(value)
                continue
            if key in flat:
                logger.warning("Duplicate config key across groups: %s", key)
            flat[key] = value

    walk(node)
    return flat


@dataclass(frozen=True)
class Upstream:
    """A named downstream service instance (vLLM or external OpenAI-compatible API).

    ``base_url`` is the service root WITHOUT ``/v1``; the unified call layer
    appends ``/v1/chat/completions``. ``api_key`` is already env-interpolated
    plaintext (empty -> no auth header). ``timeout`` is the per-request HTTP
    timeout in seconds. ``max_tokens`` is an optional per-backend response cap.
    """

    name: str
    base_url: str
    model_name: str
    api_key: str = ""
    timeout: float = 120.0
    max_tokens: int | None = None


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

    # ---- ASR: offline long-audio transcription jobs ------------------------
    # POST /api/asr/transcriptions decodes the upload, replays it through the
    # same VAD segmentation as the streaming endpoints, and transcribes the
    # segments via the shared one-shot dual-ASR path. Knobs:
    #   transcribe_max_concurrent_jobs   -> jobs running at once; total vLLM
    #   transcribe_segment_concurrency      pressure is the product of the two
    #                                       (defaults 2 x 4 = 8 in-flight
    #                                       segment requests, matching the
    #                                       emotion stores' ceiling).
    #   transcribe_max_segment_sec       -> force-cut ceiling for uninterrupted
    #                                       speech (VAD only cuts on silence);
    #                                       sized well under the 60 s one-shot
    #                                       REST cap to stay in the range the
    #                                       segment models see in streaming.
    #   transcribe_max_upload_bytes      -> multipart cap; 2 h of 16 kHz mono
    #                                       s16 WAV is ~220 MB, so 512 MB
    #                                       leaves headroom for higher-rate
    #                                       client WAVs.
    #   transcribe_max_audio_sec         -> decoded-duration cap. Unlike the
    #                                       60 s upload tail-trim, exceeding it
    #                                       REJECTS the request (400): silently
    #                                       dropping the head of a meeting
    #                                       recording is never acceptable.
    #   transcribe_silence_duration_ms   -> offline-only override of the VAD
    #                                       cut pause. 0 = follow the global
    #                                       silence_duration_ms. The global one
    #                                       is tuned for live latency (350 ms);
    #                                       minutes-style transcripts read
    #                                       better with longer pauses (~800 ms)
    #                                       and offline has no latency cost,
    #                                       but raising the GLOBAL knob would
    #                                       also delay every live endpoint's
    #                                       finals — hence the scoped override
    #                                       (same fallback pattern as
    #                                       astv3_vllm_base_url). All other VAD
    #                                       knobs stay shared: there is no
    #                                       offline reason for them to differ.
    transcribe_max_concurrent_jobs: int = 2
    transcribe_segment_concurrency: int = 4
    transcribe_job_queue_max: int = 8
    transcribe_job_ttl_sec: float = 3600.0
    transcribe_max_segment_sec: float = 30.0
    transcribe_max_upload_bytes: int = 512 * 1024 * 1024
    transcribe_max_audio_sec: float = 10800.0
    transcribe_silence_duration_ms: int = 0

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


# 合法的 (protocol, task) 组合白名单。其余组合(如 astv3 × browser_demo)在解析期
# fail-fast, 避免注册出无意义的端点。browser_demo 复用 native wire, 不是独立协议。
VALID_ENDPOINT_COMBINATIONS: frozenset[tuple[str, str]] = frozenset({
    ("native", "asr"),
    ("astv3", "asr"),
    ("native", "emotion"),
    ("native", "browser_demo"),
})

# upstream 角色: 决定一个 upstream 投影到哪些扁平 Config 字段。
_UPSTREAM_ROLES: frozenset[str] = frozenset(
    {"primary", "secondary", "emotion", "emotion_spec"}
)


@dataclass(frozen=True)
class EndpointSpec:
    """A declarative WebSocket endpoint parsed from config.yaml ``endpoints``."""

    path: str
    protocol: str
    task: str
    # role -> upstream name (primary / secondary / emotion / emotion_spec)
    upstream_roles: dict[str, str] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)  # 软默认(客户端白名单可覆盖)
    lock: dict[str, Any] = field(default_factory=dict)    # 硬锁定(客户端覆盖后重施)
    client_overridable: bool = False
    input_sample_rate: int = SAMPLE_RATE


@dataclass(frozen=True)
class ParsedConfig:
    """Fully parsed config.yaml: registries + the global (REST-bound) Config."""

    upstreams: dict[str, Upstream]
    defaults: dict[str, Any]          # flattened tuning leaves
    services: dict[str, str]          # service role -> upstream name
    rest_roles: dict[str, str]        # REST role -> upstream name
    endpoints: tuple[EndpointSpec, ...]
    http_pool: dict[str, int]
    default_config: Config
    # POST /api/asr/transcriptions runtime view: default_config plus the
    # optional `rest.transcribe` block (own model bindings + fusion switch);
    # equal to default_config when the block is absent.
    transcribe_config: Config


def _parse_upstream(name: str, raw: dict[str, Any]) -> Upstream:
    if not isinstance(raw, dict):
        raise ValueError(f"upstreams.{name} must be a mapping")
    max_tokens = raw.get("max_tokens")
    return Upstream(
        name=name,
        base_url=str(raw.get("base_url", "")).rstrip("/"),
        model_name=str(raw.get("model_name", "")),
        api_key=str(raw.get("api_key", "")),
        timeout=float(raw.get("timeout", 120.0)),
        max_tokens=int(max_tokens) if max_tokens is not None else None,
    )


def _role_fields(role: str, up: Upstream) -> dict[str, Any]:
    """Project an upstream bound to a role onto the flat Config fields it feeds."""
    if role == "primary":
        return {
            "vllm_base_url": up.base_url,
            "vllm_model_name": up.model_name,
            "asr_request_timeout": up.timeout,
        }
    if role == "secondary":
        # Secondary reuses the single asr_request_timeout (primary's) by design.
        return {
            "secondary_vllm_base_url": up.base_url,
            "secondary_vllm_model_name": up.model_name,
        }
    if role == "emotion":
        return {
            "emotion_vllm_base_url": up.base_url,
            "emotion_vllm_model_name": up.model_name,
            "emotion_request_timeout": up.timeout,
        }
    if role == "emotion_spec":
        return {
            "emotion_spec_vllm_base_url": up.base_url,
            "emotion_spec_vllm_model_name": up.model_name,
            "emotion_spec_request_timeout": up.timeout,
        }
    logger.warning("Unknown upstream role ignored: %s", role)
    return {}


def _text_cleanup_fields(up: Upstream) -> dict[str, Any]:
    """Project the text-cleanup service upstream onto the legacy text_cleanup_* fields.

    BRIDGE: ``text_cleanup/client.py`` currently builds ``{base}/chat/completions``
    expecting ``base`` to already contain ``/v1``, while ``Upstream.base_url`` is
    normalized to the service root (no ``/v1``). We re-append ``/v1`` here so the
    un-migrated client keeps working. Phase 4 routes cleanup through the unified
    ``query_chat_completion`` and removes both this bridge and the text_cleanup_*
    fields.
    """
    return {
        "text_cleanup_base_url": up.base_url + "/v1",
        "text_cleanup_model_name": up.model_name,
        "text_cleanup_api_key": up.api_key,
        "text_cleanup_api_key_env": "",  # already interpolated to plaintext on load
        "text_cleanup_timeout": up.timeout,
        "text_cleanup_max_tokens": up.max_tokens if up.max_tokens is not None else 1024,
    }


def _project_base(
    *,
    upstreams: dict[str, Upstream],
    rest_roles: dict[str, str],
    defaults: dict[str, Any],
    services: dict[str, str],
    http_pool: dict[str, int],
) -> Config:
    """Project the global (REST-bound) Config view: defaults + http_pool +
    text_cleanup service + REST role bindings. This is what ``load_config()`` and
    ``default_config`` expose; per-endpoint configs override on top of it.
    """
    values: dict[str, Any] = dict(defaults)
    values["http_max_connections"] = int(http_pool.get("max_connections", 32))
    values["http_max_keepalive_connections"] = int(
        http_pool.get("max_keepalive_connections", 16)
    )
    cleanup_name = services.get("text_cleanup")
    if cleanup_name and cleanup_name in upstreams:
        values.update(_text_cleanup_fields(upstreams[cleanup_name]))
    for role, up_name in rest_roles.items():
        if up_name in upstreams:
            values.update(_role_fields(role, upstreams[up_name]))
    valid = {f.name for f in fields(Config)}
    unknown = set(defaults) - valid
    if unknown:
        logger.warning("Unknown defaults keys ignored: %s", sorted(unknown))
    filtered = {k: v for k, v in values.items() if k in valid}
    return Config(**filtered)


def _parse_endpoint(raw: dict[str, Any], upstreams: dict[str, Upstream]) -> EndpointSpec:
    path = str(raw.get("path", "")).strip()
    protocol = str(raw.get("protocol", "")).strip()
    task = str(raw.get("task", "")).strip()
    if not path:
        raise ValueError("endpoint entry missing 'path'")
    if (protocol, task) not in VALID_ENDPOINT_COMBINATIONS:
        raise ValueError(
            f"endpoint {path}: invalid (protocol, task)=({protocol!r}, {task!r}); "
            f"allowed: {sorted(VALID_ENDPOINT_COMBINATIONS)}"
        )
    roles = dict(raw.get("upstreams", {}) or {})
    for role, up_name in roles.items():
        if role not in _UPSTREAM_ROLES:
            raise ValueError(f"endpoint {path}: unknown upstream role {role!r}")
        if up_name not in upstreams:
            raise ValueError(
                f"endpoint {path}: unknown upstream {up_name!r} for role {role!r}"
            )
    return EndpointSpec(
        path=path,
        protocol=protocol,
        task=task,
        upstream_roles=roles,
        policy=dict(raw.get("policy", {}) or {}),
        lock=dict(raw.get("lock", {}) or {}),
        client_overridable=bool(raw.get("client_overridable", False)),
        input_sample_rate=int(raw.get("input_sample_rate", SAMPLE_RATE)),
    )


def _project_transcribe(
    base: Config,
    raw_transcribe: dict[str, Any],
    upstreams: dict[str, Upstream],
) -> Config:
    """Project the `rest.transcribe` block onto the transcription Config.

    The block exists so operators can see and change, in one place, which
    model(s) the long-audio endpoint runs on — independently of the shared
    `rest.upstreams` bindings used by /api/asr/upload etc. Recognized keys:

    - ``upstreams``: role -> upstream name, roles limited to primary /
      secondary (an emotion model makes no sense for transcription);
    - ``enable_dual_asr_fusion``: per-endpoint override of the global switch.

    Unknown keys raise: a typo silently falling back to the shared bindings
    is exactly the "can't see what this endpoint uses" problem again.
    """
    allowed = {"upstreams", "enable_dual_asr_fusion"}
    unknown = set(raw_transcribe) - allowed
    if unknown:
        raise ValueError(
            f"rest.transcribe: unknown keys {sorted(unknown)}; allowed: {sorted(allowed)}"
        )

    overrides: dict[str, Any] = {}
    roles = dict(raw_transcribe.get("upstreams", {}) or {})
    for role, up_name in roles.items():
        if role not in ("primary", "secondary"):
            raise ValueError(
                f"rest.transcribe.upstreams: unknown role {role!r}; "
                "allowed: ['primary', 'secondary']"
            )
        if up_name not in upstreams:
            raise ValueError(
                f"rest.transcribe.upstreams.{role}: unknown upstream {up_name!r}"
            )
        overrides.update(_role_fields(role, upstreams[up_name]))

    if "enable_dual_asr_fusion" in raw_transcribe:
        fusion = bool(raw_transcribe["enable_dual_asr_fusion"])
        overrides["enable_dual_asr_fusion"] = fusion
        if fusion and not base.enable_secondary_asr:
            logger.warning(
                "rest.transcribe.enable_dual_asr_fusion=true requires "
                "enable_secondary_asr=true (defaults.asr); downgrading to false"
            )

    return base.override(**overrides) if overrides else base


def _parse(raw: dict[str, Any]) -> ParsedConfig:
    upstreams = {
        name: _parse_upstream(name, spec)
        for name, spec in (raw.get("upstreams", {}) or {}).items()
    }
    defaults = _flatten_leaves(raw.get("defaults", {}) or {})

    services = dict(raw.get("services", {}) or {})
    for svc, up_name in services.items():
        if up_name not in upstreams:
            raise ValueError(f"services.{svc}: unknown upstream {up_name!r}")

    rest_roles = dict((raw.get("rest", {}) or {}).get("upstreams", {}) or {})
    for role, up_name in rest_roles.items():
        if role not in _UPSTREAM_ROLES:
            raise ValueError(f"rest.upstreams: unknown role {role!r}")
        if up_name not in upstreams:
            raise ValueError(f"rest.upstreams.{role}: unknown upstream {up_name!r}")

    http_pool = dict(raw.get("http_pool", {}) or {})
    endpoints = tuple(
        _parse_endpoint(ep, upstreams) for ep in (raw.get("endpoints", []) or [])
    )

    # Loud WARN on an impossible global combo (silent downgrade still happens in
    # __post_init__; this is just the operator-facing log line).
    if defaults.get("enable_dual_asr_fusion") and not defaults.get(
        "enable_secondary_asr", True
    ):
        logger.warning(
            "enable_dual_asr_fusion=true requires enable_secondary_asr=true; "
            "downgrading fusion to false"
        )

    base = _project_base(
        upstreams=upstreams,
        rest_roles=rest_roles,
        defaults=defaults,
        services=services,
        http_pool=http_pool,
    )
    transcribe_cfg = _project_transcribe(
        base, dict((raw.get("rest", {}) or {}).get("transcribe", {}) or {}), upstreams
    )
    return ParsedConfig(
        upstreams=upstreams,
        defaults=defaults,
        services=services,
        rest_roles=rest_roles,
        endpoints=endpoints,
        http_pool=http_pool,
        default_config=base,
        transcribe_config=transcribe_cfg,
    )


def load_parsed(path: Path | None = None) -> ParsedConfig:
    """Parse the full config.yaml (registries + global Config).

    ``CONFIG_PATH`` env overrides the default path when no explicit path is given.
    """
    if path is None:
        env_path = os.getenv("CONFIG_PATH", "").strip()
        path = Path(env_path) if env_path else _DEFAULT_CONFIG_PATH
    return _parse(_load_yaml(path))


def load_config(path: Path | None = None) -> Config:
    """Backward-compatible entry point: the global (REST-bound) default Config.

    Per-endpoint runtime configs come from :func:`resolve_endpoint`.
    """
    return load_parsed(path).default_config


def load_transcribe_config(path: Path | None = None) -> Config:
    """Runtime Config for POST /api/asr/transcriptions.

    Equals :func:`load_config` unless config.yaml declares a `rest.transcribe`
    block (own primary/secondary bindings and/or fusion switch).
    """
    return load_parsed(path).transcribe_config


def resolve_endpoint(
    spec: EndpointSpec, parsed: ParsedConfig | None = None
) -> Config:
    """Project an endpoint's runtime Config.

    global default (REST-bound) + this endpoint's explicit upstream bindings +
    ``policy`` (soft) + ``lock`` (hard). The route layer re-applies ``spec.lock``
    after any client override so locks can't be undone by ``start.config``.
    """
    parsed = parsed or _PARSED
    overrides: dict[str, Any] = {}
    for role, up_name in spec.upstream_roles.items():
        if up_name in parsed.upstreams:
            overrides.update(_role_fields(role, parsed.upstreams[up_name]))
    overrides.update(spec.policy)
    overrides.update(spec.lock)
    # 端点未绑 secondary 上游 => 物理上没有副模型, 强制关闭(放在最后, 不可被
    # policy/lock 误开成一个并不存在的副模型)。
    if spec.task == "asr" and "secondary" not in spec.upstream_roles:
        overrides["enable_secondary_asr"] = False
    return parsed.default_config.override(**overrides)


def get_service_upstream(
    service: str, parsed: ParsedConfig | None = None
) -> Upstream | None:
    """Resolve a global auxiliary service (hotword / text_cleanup) to its Upstream."""
    parsed = parsed or _PARSED
    name = parsed.services.get(service)
    return parsed.upstreams.get(name) if name else None


# 进程级解析结果(基于默认路径)。下面的注册表都从此派生, 全进程一份。
_PARSED: ParsedConfig = load_parsed()
UPSTREAMS: dict[str, Upstream] = _PARSED.upstreams
ENDPOINTS: tuple[EndpointSpec, ...] = _PARSED.endpoints
SERVICES: dict[str, str] = _PARSED.services
REST_ROLES: dict[str, str] = _PARSED.rest_roles

# Process-wide default Config singleton (REST-bound projection). Modules without a
# per-session Config (module-level helpers, ``value or <default>`` fallbacks) read
# fields off this. Single entry point preserved: every reader reaches config via a
# Config object.
default_config: Config = _PARSED.default_config
