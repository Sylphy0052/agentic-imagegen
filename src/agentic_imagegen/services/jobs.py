"""非同期生成ジョブの管理。

MCPのtool呼び出しは短時間で返す必要がある一方、生成は数十秒から数分かかる。
投入と状態問い合わせを分けるため、実行中のジョブをここで保持する。

状態はプロセス内のメモリにのみ持つ。サーバーを再起動すると消えるが、
生成物と metadata.json はディスクに残るため、結果自体は失われない。
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

logger: Final = logging.getLogger(__name__)


class JobStatus(StrEnum):
    """ジョブの状態。"""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job[T]:
    """1回分のジョブ。単発生成なら結果は GenerationResult、一括生成なら結果の一覧になる。"""

    job_id: str
    status: JobStatus = JobStatus.RUNNING
    result: T | None = None
    error: BaseException | None = None
    task: asyncio.Task[None] | None = field(default=None, repr=False)


class JobRegistry[T]:
    """実行中および完了済みのジョブを保持する。

    完了したジョブも保持し続ける。呼び出し側は完了を検知したあとに
    結果を取りに来るため、終わった時点で捨てると結果を渡せない。

    結果の型を固定しないのは、単発生成と一括生成で結果の形が違うため。
    投入と状態問い合わせを分ける仕組み自体は両者で同じものを使う。
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job[T]] = {}

    def submit(self, factory: Callable[[], Awaitable[T]]) -> str:
        """処理を投入し、job_idを返す。

        factoryは呼び出し時点でコルーチンを生成する。実行中の例外は
        ジョブの状態として記録し、投入側へは伝播させない。
        """
        job_id = uuid.uuid4().hex
        job: Job[T] = Job(job_id=job_id)
        self._jobs[job_id] = job
        job.task = asyncio.create_task(self._run(job, factory))
        return job_id

    def get(self, job_id: str) -> Job[T] | None:
        """ジョブを取り出す。存在しなければ None。"""
        return self._jobs.get(job_id)

    async def wait(self, job_id: str) -> Job[T]:
        """ジョブの完了を待つ。主にテストと後始末のために使う。"""
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.task is not None:
            await job.task
        return job

    async def _run(self, job: Job[T], factory: Callable[[], Awaitable[T]]) -> None:
        try:
            job.result = await factory()
        except Exception as exc:
            job.status = JobStatus.FAILED
            job.error = exc
            logger.warning("generation job failed: job_id=%s (%s)", job.job_id, type(exc).__name__)
        else:
            job.status = JobStatus.COMPLETED
            logger.info("generation job completed: job_id=%s", job.job_id)


__all__ = ["Job", "JobRegistry", "JobStatus"]
