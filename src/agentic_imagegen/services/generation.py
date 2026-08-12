"""画像生成のユースケース。

Spec -> Workflow -> バックエンド実行 -> 保存 の流れをここで組み立てる。
バックエンドは Protocol 越しに扱い、ComfyUI固有の型には依存しない。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Protocol

from agentic_imagegen.config import Settings
from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.domain.policy import resolve_output_directory
from agentic_imagegen.domain.results import GenerationResult, HealthStatus, ImageRef
from agentic_imagegen.workflows.injector import prepare_workflow

logger: Final = logging.getLogger(__name__)

METADATA_FILENAME: Final = "metadata.json"

#: 同じ日・同じprefixで再実行したときに、既存の結果を上書きしないための試行上限。
_MAX_DIRECTORY_SUFFIX: Final = 1000


class GenerationBackend(Protocol):
    """画像生成バックエンドに求める操作。

    ComfyUIClient はこのProtocolを構造的に満たす。
    将来 diffusers や remote API を足す場合も、この形に合わせれば
    Service層を変更せずに差し替えられる。
    """

    async def submit(self, workflow: dict[str, Any]) -> str: ...

    async def wait_for_completion(
        self, prompt_id: str, *, timeout: float | None = None
    ) -> None: ...

    async def fetch_outputs(self, prompt_id: str) -> tuple[ImageRef, ...]: ...

    async def download(self, ref: ImageRef) -> bytes: ...

    async def health(self) -> HealthStatus: ...


async def generate(
    spec: GenerationSpec,
    settings: Settings,
    *,
    backend: GenerationBackend,
    project_root: Path,
    timeout: float | None = None,
    workflows_dir: Path | None = None,
) -> GenerationResult:
    """Specに従って画像を生成し、結果をプロジェクト配下へ保存する。"""
    directory = _prepare_directory(spec, settings, project_root)

    prepared = prepare_workflow(spec, workflows_dir=workflows_dir)
    seed = prepared.seed
    logger.info(
        "generation start: workflow=%s prefix=%s seed=%s", spec.task, spec.output.prefix, seed
    )

    prompt_id = await backend.submit(prepared.workflow)
    await backend.wait_for_completion(
        prompt_id, timeout=timeout if timeout is not None else float(settings.timeout_seconds)
    )
    refs = await backend.fetch_outputs(prompt_id)

    directory.mkdir(parents=True, exist_ok=True)
    files = tuple(
        [await _save_image(backend, ref, directory, index) for index, ref in enumerate(refs, 1)]
    )

    metadata_path = _write_metadata(
        directory,
        spec=spec,
        prompt_id=prompt_id,
        seed=seed,
        files=files,
        workflow_name=prepared.workflow_name,
        workflow_hash=prepared.template_hash,
        backend_info=await _collect_backend_info(backend),
    )
    logger.info("generation done: prompt_id=%s files=%d dir=%s", prompt_id, len(files), directory)

    return GenerationResult(
        prompt_id=prompt_id,
        seed=seed,
        directory=directory,
        files=files,
        metadata_path=metadata_path,
    )


def _prepare_directory(spec: GenerationSpec, settings: Settings, project_root: Path) -> Path:
    """`<出力ルート>/<日付>/<prefix>` を作業ルート内に解決する。

    同じ日に同じprefixで再実行した場合は連番を付け、既存の結果を上書きしない。
    """
    directory = spec.output.directory or str(settings.output_root)
    base = resolve_output_directory(directory, project_root)
    day = datetime.now().astimezone().strftime("%Y-%m-%d")
    candidate = base / day / spec.output.prefix

    if not candidate.exists():
        return candidate
    for suffix in range(2, _MAX_DIRECTORY_SUFFIX):
        numbered = candidate.with_name(f"{spec.output.prefix}-{suffix}")
        if not numbered.exists():
            return numbered
    return candidate


async def _save_image(
    backend: GenerationBackend, ref: ImageRef, directory: Path, index: int
) -> Path:
    data = await backend.download(ref)
    suffix = Path(ref.filename).suffix or ".png"
    path = directory / f"image_{index:04d}{suffix}"
    path.write_bytes(data)
    return path


async def _collect_backend_info(backend: GenerationBackend) -> dict[str, Any] | None:
    """metadataへ残す実行基盤の情報を集める。

    ここでの失敗は生成そのものを巻き戻す理由にならない (画像は既に取得済み)。
    記録を諦めるだけにして、理由はログへ残す。
    """
    try:
        status = await backend.health()
    except Exception:
        logger.warning(
            "実行基盤の情報を取得できませんでした。metadataへは記録しません", exc_info=True
        )
        return None
    return {"comfyui_version": status.comfyui_version, "devices": list(status.devices)}


def _write_metadata(
    directory: Path,
    *,
    spec: GenerationSpec,
    prompt_id: str,
    seed: int,
    files: tuple[Path, ...],
    workflow_name: str,
    workflow_hash: str,
    backend_info: dict[str, Any] | None,
) -> Path:
    metadata = {
        "prompt_id": prompt_id,
        # 論理タスク名 (spec.task) ではなく、実際に使ったテンプレート名を残す
        "workflow": workflow_name,
        "workflow_hash": workflow_hash,
        "created_at": datetime.now().astimezone().isoformat(),
        "resolved_seed": seed,
        "backend": backend_info,
        "spec": spec.model_dump(mode="json"),
        "outputs": [path.name for path in files],
    }
    path = directory / METADATA_FILENAME
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


__all__ = ["GenerationBackend", "generate"]
