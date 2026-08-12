"""MCP toolとして公開する操作の中身。

MCP層 (mcp_server) は薄いアダプタに留め、ロジックはここへ置く。
CLIと同じ Service / Domain を使うため、経路が違っても検証は同一になる。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

from agentic_imagegen.adapters.comfyui.client import ComfyUIClient
from agentic_imagegen.config import Settings
from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.domain.policy import validate_against_limits
from agentic_imagegen.domain.results import GenerationResult
from agentic_imagegen.errors import ImageGenError, InvalidGenerationSpec
from agentic_imagegen.services.batch import BatchItem, BatchOutcome, BatchRunner, expand_seeds
from agentic_imagegen.services.batch import run_batch as run_batch_items
from agentic_imagegen.services.generation import generate
from agentic_imagegen.services.jobs import JobRegistry, JobStatus
from agentic_imagegen.services.spec_loader import parse_spec
from agentic_imagegen.workflows.injector import ALLOWED_WORKFLOWS, resolve_workflow_name

#: 生成の実体。テストではComfyUIへ接続しないものへ差し替える。
type Runner = Callable[[GenerationSpec], Awaitable[GenerationResult]]


def list_workflows() -> list[str]:
    """実行を許可しているWorkflowテンプレート名を返す。"""
    return sorted(ALLOWED_WORKFLOWS)


async def list_models(settings: Settings) -> list[str]:
    """ComfyUIが持っているcheckpoint名を返す。"""
    async with ComfyUIClient(settings) as client:
        return list(await client.available_checkpoints())


async def list_loras(settings: Settings) -> list[str]:
    """ComfyUIが持っているLoRA名を返す。"""
    async with ComfyUIClient(settings) as client:
        return list(await client.available_loras())


async def list_controlnets(settings: Settings) -> list[str]:
    """ComfyUIが持っているControlNetモデル名を返す。"""
    async with ComfyUIClient(settings) as client:
        return list(await client.available_controlnets())


async def _default_runner(
    spec: GenerationSpec, *, settings: Settings, project_root: Path
) -> GenerationResult:
    """実際にComfyUIへ接続して生成する。テストではここを差し替える。"""
    async with ComfyUIClient(settings) as client:
        return await generate(spec, settings, backend=client, project_root=project_root)


def submit_generation(
    spec: Any,
    *,
    settings: Settings,
    project_root: Path,
    registry: JobRegistry[GenerationResult],
    runner: Runner | None = None,
) -> dict[str, Any]:
    """Specを検証したうえで生成を投入し、job_idを返す。

    生成には数十秒から数分かかるため、完了は待たない。状態は
    get_generation_status で問い合わせる。

    検証はここで済ませる。不正なSpecはジョブにせず例外にすることで、
    「投入はできたが即座に失敗する」状態を作らない。
    """
    if not isinstance(spec, dict):
        raise InvalidGenerationSpec("Specはマッピングである必要があります")

    parsed = parse_spec(spec, presets_root=settings.presets_root)
    validate_against_limits(parsed, settings)

    execute = runner if runner is not None else _default_runner_for(settings, project_root)

    async def factory() -> GenerationResult:
        return await execute(parsed)

    job_id = registry.submit(factory)
    return {
        "job_id": job_id,
        "status": JobStatus.RUNNING.value,
        "workflow": resolve_workflow_name(parsed),
    }


def _default_runner_for(settings: Settings, project_root: Path) -> Runner:
    async def run(spec: GenerationSpec) -> GenerationResult:
        return await _default_runner(spec, settings=settings, project_root=project_root)

    return run


def submit_batch(
    specs: Any,
    *,
    seeds: Sequence[int] | None = None,
    settings: Settings,
    project_root: Path,
    registry: JobRegistry[list[BatchOutcome]],
    runner: BatchRunner | None = None,
) -> dict[str, Any]:
    """複数のSpecを検証したうえで一括生成を投入し、job_idを返す。

    CLIの `imagegen batch` と同じ services.batch を通す。検証は投入前に全件行い、
    1件でも不正なら1件も投入しない。途中まで生成された状態を作らないため。

    seedsを指定するとSpecごとに各seedを当てたものへ展開する。
    """
    items = _batch_items(specs, seeds=seeds, settings=settings)

    async def factory() -> list[BatchOutcome]:
        if runner is not None:
            return await run_batch_items(items, runner=runner)
        # バッチ全体で1つの接続を使い回す (CLIの batch と同じ扱い)
        async with ComfyUIClient(settings) as client:

            async def run(item: BatchItem) -> GenerationResult:
                return await generate(
                    item.spec, settings, backend=client, project_root=project_root
                )

            return await run_batch_items(items, runner=run)

    job_id = registry.submit(factory)
    return {
        "job_id": job_id,
        "status": JobStatus.RUNNING.value,
        "total": len(items),
        "items": [
            {"label": item.label, "workflow": resolve_workflow_name(item.spec)} for item in items
        ],
    }


def _batch_items(specs: Any, *, seeds: Sequence[int] | None, settings: Settings) -> list[BatchItem]:
    """受け取ったSpecの並びを検証し、seed掃引まで展開する。"""
    if isinstance(specs, dict | str) or not isinstance(specs, Sequence):
        raise InvalidGenerationSpec("specs はSpecの配列である必要があります")
    if not specs:
        raise InvalidGenerationSpec("specs には1件以上のSpecを指定してください")

    pairs: list[tuple[str, GenerationSpec]] = []
    for index, raw in enumerate(specs):
        if not isinstance(raw, dict):
            raise InvalidGenerationSpec(f"spec[{index}] はマッピングである必要があります")
        parsed = parse_spec(raw, presets_root=settings.presets_root)
        validate_against_limits(parsed, settings)
        # ファイルとして存在しないため、並びの位置を出どころとして残す
        pairs.append((f"spec[{index}]", parsed))

    return expand_seeds(pairs, seeds=seeds)


def get_batch_status(
    job_id: str, *, registry: JobRegistry[list[BatchOutcome]], project_root: Path
) -> dict[str, Any]:
    """投入済みの一括生成の状態を返す。

    1件失敗しても残りは続くため、完了時の status は completed のままにし、
    内訳を succeeded / failed と items で示す。
    """
    job = registry.get(job_id)
    if job is None:
        raise ValueError(f"不明な job_id です: {job_id}")

    if job.status is JobStatus.RUNNING:
        return _batch_payload(JobStatus.RUNNING)

    if job.status is JobStatus.FAILED:
        error = job.error
        return _batch_payload(
            JobStatus.FAILED,
            error=str(error) if error is not None else "原因不明の失敗",
            exit_code=getattr(error, "exit_code", 1),
        )

    outcomes = job.result
    if outcomes is None:  # pragma: no cover - COMPLETEDなら必ず結果がある
        return _batch_payload(JobStatus.FAILED, error="結果が記録されていません", exit_code=1)

    succeeded = sum(1 for outcome in outcomes if outcome.succeeded)
    return _batch_payload(
        JobStatus.COMPLETED,
        total=len(outcomes),
        succeeded=succeeded,
        failed=len(outcomes) - succeeded,
        items=[_outcome_payload(outcome, project_root) for outcome in outcomes],
    )


def _outcome_payload(outcome: BatchOutcome, project_root: Path) -> dict[str, Any]:
    """1件分の結果。単発生成と同じ鍵にラベルを足しただけの形にする。"""
    result = outcome.result
    if result is None:
        error = outcome.error
        body = _status_payload(
            JobStatus.FAILED,
            error=str(error) if error is not None else "原因不明の失敗",
            exit_code=getattr(error, "exit_code", 1),
        )
    else:
        body = _status_payload(
            JobStatus.COMPLETED,
            seed=result.seed,
            prompt_id=result.prompt_id,
            directory=_relative(result.directory, project_root),
            files=[_relative(path, project_root) for path in result.files],
            metadata_path=_relative(result.metadata_path, project_root),
        )
    return {"label": outcome.item.label, **body}


def _batch_payload(
    status: JobStatus,
    *,
    total: int | None = None,
    succeeded: int | None = None,
    failed: int | None = None,
    items: list[dict[str, Any]] | None = None,
    error: str | None = None,
    exit_code: int | None = None,
) -> dict[str, Any]:
    return {
        "status": status.value,
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "items": items or [],
        "error": error,
        "exit_code": exit_code,
    }


def get_generation_status(
    job_id: str, *, registry: JobRegistry[GenerationResult], project_root: Path
) -> dict[str, Any]:
    """投入済みジョブの状態を返す。

    パスは作業ルートからの相対で返す。絶対パスは実行環境の構成を露出するため。
    """
    job = registry.get(job_id)
    if job is None:
        raise ValueError(f"不明な job_id です: {job_id}")

    if job.status is JobStatus.RUNNING:
        return _status_payload(JobStatus.RUNNING)

    if job.status is JobStatus.FAILED:
        error = job.error
        return _status_payload(
            JobStatus.FAILED,
            error=str(error) if error is not None else "原因不明の失敗",
            exit_code=getattr(error, "exit_code", 1),
        )

    result = job.result
    if result is None:  # pragma: no cover - COMPLETEDなら必ず結果がある
        return _status_payload(JobStatus.FAILED, error="結果が記録されていません", exit_code=1)

    return _status_payload(
        JobStatus.COMPLETED,
        seed=result.seed,
        prompt_id=result.prompt_id,
        directory=_relative(result.directory, project_root),
        files=[_relative(path, project_root) for path in result.files],
        metadata_path=_relative(result.metadata_path, project_root),
    )


def _status_payload(
    status: JobStatus,
    *,
    seed: int | None = None,
    prompt_id: str | None = None,
    directory: str | None = None,
    files: list[str] | None = None,
    metadata_path: str | None = None,
    error: str | None = None,
    exit_code: int | None = None,
) -> dict[str, Any]:
    return {
        "status": status.value,
        "seed": seed,
        "prompt_id": prompt_id,
        "directory": directory,
        "files": files or [],
        "metadata_path": metadata_path,
        "error": error,
        "exit_code": exit_code,
    }


def _relative(path: Path, project_root: Path) -> str:
    """作業ルートからの相対パスへ直す。外にある場合はファイル名だけ返す。"""
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.name


def validate_generation(spec: Any, *, settings: Settings, project_root: Path) -> dict[str, Any]:
    """Specを検証し、結果を構造化して返す。

    このtoolの目的は検証結果を得ることなので、不正なSpecでも例外にせず
    `valid: false` と理由を返す。生成は行わない。
    """
    if not isinstance(spec, dict):
        return _failure(["Specはマッピングである必要があります"])

    try:
        parsed = parse_spec(spec, presets_root=settings.presets_root)
        validate_against_limits(parsed, settings)
    except ImageGenError as exc:
        return _failure(str(exc).splitlines())

    return _success(parsed)


def _failure(errors: list[str]) -> dict[str, Any]:
    return {
        "valid": False,
        "errors": [line.strip() for line in errors if line.strip()],
        "workflow": None,
        "resolution": None,
        "checkpoint": None,
        "loras": [],
        "presets": {},
        "prompt": None,
        "source": None,
        "control": None,
        "upscale": None,
    }


def _success(spec: GenerationSpec) -> dict[str, Any]:
    params = spec.generation
    # img2imgは入力画像のサイズを使うため、解像度を返しても意味がない
    resolution = (
        None
        if spec.source is not None
        else {
            "width": params.width,
            "height": params.height,
            "batch_size": params.batch_size,
        }
    )

    return {
        "valid": True,
        "errors": [],
        "workflow": resolve_workflow_name(spec),
        "resolution": resolution,
        "checkpoint": spec.model.checkpoint,
        "loras": [lora.model_dump(mode="json") for lora in spec.model.loras],
        "presets": {
            key: value
            for key, value in spec.presets.model_dump(mode="json").items()
            if value is not None
        },
        "prompt": spec.prompt.model_dump(mode="json"),
        "source": spec.source.model_dump(mode="json") if spec.source is not None else None,
        # workflow名だけでは、どのパラメータで効いているのかまでは分からない
        "control": spec.control.model_dump(mode="json") if spec.control is not None else None,
        "upscale": (params.upscale.model_dump(mode="json") if params.upscale is not None else None),
    }


__all__ = [
    "Runner",
    "get_batch_status",
    "get_generation_status",
    "list_controlnets",
    "list_loras",
    "list_models",
    "list_workflows",
    "submit_batch",
    "submit_generation",
    "validate_generation",
]
