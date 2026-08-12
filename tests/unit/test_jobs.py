"""非同期生成ジョブの管理。

MCPのtool呼び出しは短時間で返す必要があるため、生成は投入だけして
状態を問い合わせる形にする。その土台をここでテストする。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agentic_imagegen.domain.results import GenerationResult
from agentic_imagegen.errors import GenerationFailed, GenerationTimeout
from agentic_imagegen.services.jobs import JobRegistry, JobStatus


def _result(tmp_path: Path) -> GenerationResult:
    return GenerationResult(
        prompt_id="p-1",
        seed=42,
        directory=tmp_path,
        files=(tmp_path / "image_0001.png",),
        metadata_path=tmp_path / "metadata.json",
    )


async def test_submit_returns_job_id(tmp_path: Path) -> None:
    registry = JobRegistry()

    async def run() -> GenerationResult:
        return _result(tmp_path)

    job_id = registry.submit(run)

    assert job_id
    assert registry.get(job_id) is not None


async def test_completes_and_keeps_result(tmp_path: Path) -> None:
    registry = JobRegistry()
    started = asyncio.Event()

    async def run() -> GenerationResult:
        started.set()
        return _result(tmp_path)

    job_id = registry.submit(run)
    await started.wait()
    await registry.wait(job_id)

    job = registry.get(job_id)
    assert job is not None
    assert job.status is JobStatus.COMPLETED
    assert job.result is not None
    assert job.result.seed == 42
    assert job.error is None


async def test_running_before_completion(tmp_path: Path) -> None:
    registry = JobRegistry()
    release = asyncio.Event()

    async def run() -> GenerationResult:
        await release.wait()
        return _result(tmp_path)

    job_id = registry.submit(run)
    await asyncio.sleep(0)

    job = registry.get(job_id)
    assert job is not None
    assert job.status is JobStatus.RUNNING

    release.set()
    await registry.wait(job_id)
    assert registry.get(job_id).status is JobStatus.COMPLETED  # type: ignore[union-attr]


async def test_records_failure_without_raising() -> None:
    """失敗はジョブの状態として残す。投入側へ例外を伝播させない。"""
    registry = JobRegistry()

    async def run() -> GenerationResult:
        raise GenerationFailed("ComfyUI側で失敗しました")

    job_id = registry.submit(run)
    await registry.wait(job_id)

    job = registry.get(job_id)
    assert job is not None
    assert job.status is JobStatus.FAILED
    assert isinstance(job.error, GenerationFailed)
    assert job.result is None


async def test_keeps_exit_code_of_error() -> None:
    """CLIのexit code体系をそのままMCPの応答へ持ち込めるようにする。"""
    registry = JobRegistry()

    async def run() -> GenerationResult:
        raise GenerationTimeout("時間切れ")

    job_id = registry.submit(run)
    await registry.wait(job_id)

    job = registry.get(job_id)
    assert job is not None
    assert job.error is not None
    assert job.error.exit_code == 6


async def test_unexpected_error_is_recorded() -> None:
    """ImageGenError以外の例外もジョブを壊さずに記録する。"""
    registry = JobRegistry()

    async def run() -> GenerationResult:
        raise RuntimeError("想定外")

    job_id = registry.submit(run)
    await registry.wait(job_id)

    job = registry.get(job_id)
    assert job is not None
    assert job.status is JobStatus.FAILED
    assert isinstance(job.error, RuntimeError)


async def test_unknown_job_id_returns_none() -> None:
    registry = JobRegistry()

    assert registry.get("does-not-exist") is None


async def test_wait_on_unknown_job_raises() -> None:
    registry = JobRegistry()

    with pytest.raises(KeyError):
        await registry.wait("does-not-exist")


async def test_job_ids_are_unique(tmp_path: Path) -> None:
    registry = JobRegistry()

    async def run() -> GenerationResult:
        return _result(tmp_path)

    ids = {registry.submit(run) for _ in range(5)}

    assert len(ids) == 5
