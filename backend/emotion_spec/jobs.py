"""In-process async job queue for the AmphionSPEC HTTP API.

Independent twin of :mod:`backend.emotion.jobs` — uses its own
:class:`asyncio.Semaphore` sized by ``emotion_spec_max_concurrent_jobs``
so SPEC traffic cannot starve the baseline emotion store (and vice
versa). The shared :class:`JobQueueFullError` is re-exported from the
baseline module to keep route-level ``except`` clauses single-line.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import Config, load_config
from ..emotion.jobs import JobQueueFullError  # noqa: F401 — public re-export
from .service import EmotionDecodeError, empty_final_emotion_spec, infer_emotion_spec_from_wav

logger = logging.getLogger(__name__)

JOB_STATUS_QUEUED = "queued"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_SUCCEEDED = "succeeded"
JOB_STATUS_FAILED = "failed"

_ACTIVE_STATUSES = frozenset({JOB_STATUS_QUEUED, JOB_STATUS_RUNNING})


@dataclass
class EmotionSpecJob:
    job_id: str
    status: str
    created_at: float
    updated_at: float
    mode: str
    language: str
    wav_bytes: bytes = field(repr=False)
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

    def to_poll_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.status == JOB_STATUS_SUCCEEDED and self.result is not None:
            payload["result"] = self.result
        if self.status == JOB_STATUS_FAILED and self.error is not None:
            payload["error"] = self.error
        return payload


class EmotionSpecJobStore:
    """Process-local SPEC job store with semaphore-limited vLLM concurrency."""

    def __init__(self) -> None:
        self._jobs: dict[str, EmotionSpecJob] = {}
        self._lock = asyncio.Lock()
        self._semaphore: asyncio.Semaphore | None = None
        self._max_concurrent = 8
        self._queue_max = 64
        self._ttl_sec = 3600.0
        self._cfg: Config = load_config()

    def configure(self, cfg: Config | None = None) -> None:
        if cfg is not None:
            self._cfg = cfg
        self._max_concurrent = max(
            1, int(getattr(self._cfg, "emotion_spec_max_concurrent_jobs", 8))
        )
        self._queue_max = max(
            1, int(getattr(self._cfg, "emotion_spec_job_queue_max", 64))
        )
        self._ttl_sec = max(
            60.0, float(getattr(self._cfg, "emotion_spec_job_ttl_sec", 3600))
        )
        self._semaphore = asyncio.Semaphore(self._max_concurrent)

    async def submit(
        self,
        wav_bytes: bytes,
        *,
        mode: str = "",
        language: str = "",
        cfg: Config | None = None,
    ) -> EmotionSpecJob:
        if cfg is not None:
            self.configure(cfg)
        elif self._semaphore is None:
            self.configure()

        job_id = f"es_{secrets.token_hex(12)}"
        now = time.time()
        job = EmotionSpecJob(
            job_id=job_id,
            status=JOB_STATUS_QUEUED,
            created_at=now,
            updated_at=now,
            mode=mode,
            language=language,
            wav_bytes=wav_bytes,
        )

        async with self._lock:
            await self._purge_expired_locked()
            active = sum(
                1 for j in self._jobs.values() if j.status in _ACTIVE_STATUSES
            )
            if active >= self._queue_max:
                raise JobQueueFullError(
                    f"emotion-spec job queue full ({self._queue_max} active jobs)"
                )
            self._jobs[job_id] = job

        asyncio.create_task(self._run_job(job_id))
        logger.info("Emotion-SPEC job %s queued (active=%s)", job_id, active + 1)
        return job

    async def get(self, job_id: str) -> EmotionSpecJob | None:
        async with self._lock:
            await self._purge_expired_locked()
            return self._jobs.get(job_id)

    async def _purge_expired_locked(self) -> None:
        if self._ttl_sec <= 0:
            return
        cutoff = time.time() - self._ttl_sec
        expired = [
            jid
            for jid, job in self._jobs.items()
            if job.updated_at < cutoff
            and job.status not in _ACTIVE_STATUSES
        ]
        for jid in expired:
            del self._jobs[jid]

    async def _run_job(self, job_id: str) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return

        assert self._semaphore is not None
        try:
            async with self._semaphore:
                await self._set_status(job_id, JOB_STATUS_RUNNING)
                await self._execute(job_id)
        except Exception:
            logger.exception("Unexpected error running emotion-spec job %s", job_id)
            await self._set_failed(
                job_id,
                message="internal job runner error",
                code="internal_error",
            )

    async def _execute(self, job_id: str) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return

        chosen_mode = job.mode or getattr(self._cfg, "emotion_spec_task_mode", "sepc")
        try:
            result = await infer_emotion_spec_from_wav(
                job.wav_bytes,
                mode=chosen_mode,
                language=job.language,
                cfg=self._cfg,
            )
        except EmotionDecodeError as exc:
            await self._set_failed(
                job_id,
                message=str(exc),
                code="decode_error",
            )
            return
        except asyncio.TimeoutError:
            await self._set_failed(
                job_id,
                message="emotion-spec model request timed out",
                code="inference_timeout",
            )
            return
        except Exception as exc:
            logger.exception("Emotion-SPEC job %s inference failed", job_id)
            await self._set_failed(
                job_id,
                message=str(exc),
                code="inference_failed",
            )
            return

        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JOB_STATUS_SUCCEEDED
            job.result = result
            job.updated_at = time.time()
            job.wav_bytes = b""

    async def _set_status(self, job_id: str, status: str) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = status
            job.updated_at = time.time()

    async def _set_failed(
        self, job_id: str, *, message: str, code: str
    ) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JOB_STATUS_FAILED
            job.error = {"message": message, "code": code}
            job.updated_at = time.time()
            job.wav_bytes = b""


_spec_store: EmotionSpecJobStore | None = None


def get_emotion_spec_job_store() -> EmotionSpecJobStore:
    global _spec_store
    if _spec_store is None:
        _spec_store = EmotionSpecJobStore()
        _spec_store.configure()
    return _spec_store


__all__ = [
    "EmotionSpecJob",
    "EmotionSpecJobStore",
    "JobQueueFullError",
    "JOB_STATUS_FAILED",
    "JOB_STATUS_QUEUED",
    "JOB_STATUS_RUNNING",
    "JOB_STATUS_SUCCEEDED",
    "get_emotion_spec_job_store",
]
