"""In-process async job queue for whole-utterance emotion HTTP API.

Thin shell over the generic :class:`backend.jobstore.JobStore`: this module
only owns the emotion-specific bits (job fields, config knobs, inference
call + error translation). Queue/TTL/concurrency mechanics live in the
shared skeleton.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import Config, load_config
from ..jobstore import (
    JOB_STATUS_FAILED,
    JOB_STATUS_QUEUED,
    JOB_STATUS_RUNNING,
    JOB_STATUS_SUCCEEDED,
    BaseJob,
    JobExecutionError,
    JobQueueFullError,
    JobStore,
)
from .service import EmotionDecodeError, infer_emotion_from_wav

logger = logging.getLogger(__name__)

__all__ = [
    "EmotionJob",
    "EmotionJobStore",
    "JobQueueFullError",
    "JOB_STATUS_FAILED",
    "JOB_STATUS_QUEUED",
    "JOB_STATUS_RUNNING",
    "JOB_STATUS_SUCCEEDED",
    "get_emotion_job_store",
]


@dataclass(kw_only=True)
class EmotionJob(BaseJob):
    mode: str = ""
    language: str = ""
    wav_bytes: bytes = field(default=b"", repr=False)

    def release_payload(self) -> None:
        self.wav_bytes = b""


class EmotionJobStore(JobStore[EmotionJob]):
    """Process-local emotion job store with semaphore-limited vLLM concurrency."""

    id_prefix = "em"
    label = "Emotion"

    def __init__(self) -> None:
        super().__init__()
        self._cfg: Config = load_config()

    def configure(self, cfg: Config | None = None) -> None:
        if cfg is not None:
            self._cfg = cfg
        self.configure_limits(
            max_concurrent=getattr(self._cfg, "emotion_max_concurrent_jobs", 8),
            queue_max=getattr(self._cfg, "emotion_job_queue_max", 64),
            ttl_sec=getattr(self._cfg, "emotion_job_ttl_sec", 3600),
        )

    async def submit(
        self,
        wav_bytes: bytes,
        *,
        mode: str = "",
        language: str = "",
        cfg: Config | None = None,
    ) -> EmotionJob:
        if cfg is not None:
            self.configure(cfg)
        elif not self.is_configured:
            self.configure()

        now = time.time()
        job = EmotionJob(
            job_id=self.new_job_id(),
            status=JOB_STATUS_QUEUED,
            created_at=now,
            updated_at=now,
            mode=mode,
            language=language,
            wav_bytes=wav_bytes,
        )
        return await self.enqueue(job)

    async def run_job_payload(self, job: EmotionJob) -> dict[str, Any]:
        chosen_mode = job.mode or getattr(self._cfg, "emotion_task_mode", "ser")
        try:
            return await infer_emotion_from_wav(
                job.wav_bytes,
                mode=chosen_mode,
                language=job.language,
                cfg=self._cfg,
            )
        except EmotionDecodeError as exc:
            raise JobExecutionError(str(exc), code="decode_error") from exc
        except asyncio.TimeoutError as exc:
            raise JobExecutionError(
                "emotion model request timed out", code="inference_timeout"
            ) from exc


_store: EmotionJobStore | None = None


def get_emotion_job_store() -> EmotionJobStore:
    global _store
    if _store is None:
        _store = EmotionJobStore()
        _store.configure()
    return _store
