"""generate_image / get_generation_status の中身。

生成の実体は差し替えられる形にして、ここではtoolの応答契約だけを見る。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agentic_imagegen.config import Settings
from agentic_imagegen.domain.results import GenerationResult
from agentic_imagegen.errors import GenerationTimeout, InvalidGenerationSpec
from agentic_imagegen.services.jobs import JobRegistry
from agentic_imagegen.services.mcp_tools import get_generation_status, submit_generation

VALID_SPEC: dict[str, Any] = {
    "version": "1",
    "task": "txt2img",
    "prompt": {"positive": "1girl"},
    "model": {"checkpoint": "v1-5-pruned-emaonly.safetensors"},
}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        comfyui_base_url="http://127.0.0.1:8188",
        max_width=2048,
        max_height=2048,
        max_pixels=4194304,
        max_batch=4,
        timeout_seconds=30,
        output_root=Path("outputs"),
        presets_root=tmp_path / "presets",
    )


def _result(root: Path) -> GenerationResult:
    directory = root / "outputs" / "2026-08-12" / "sample"
    directory.mkdir(parents=True, exist_ok=True)
    image = directory / "image_0001.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    metadata = directory / "metadata.json"
    metadata.write_text("{}", encoding="utf-8")
    return GenerationResult(
        prompt_id="p-1",
        seed=4242,
        directory=directory,
        files=(image,),
        metadata_path=metadata,
    )


class TestSubmitGeneration:
    async def test_returns_job_id_and_workflow(self, settings: Settings, tmp_path: Path) -> None:
        registry = JobRegistry[GenerationResult]()

        async def runner(spec: Any) -> GenerationResult:
            return _result(tmp_path)

        response = submit_generation(
            VALID_SPEC,
            settings=settings,
            project_root=tmp_path,
            registry=registry,
            runner=runner,
        )

        assert response["status"] == "running"
        assert response["job_id"]
        assert response["workflow"] == "txt2img"
        await registry.wait(response["job_id"])

    async def test_rejects_invalid_spec(self, settings: Settings, tmp_path: Path) -> None:
        """生成を行うtoolなので、不正な入力は例外にして投入させない。"""
        registry = JobRegistry[GenerationResult]()

        with pytest.raises(InvalidGenerationSpec):
            submit_generation(
                {"task": "txt2img"},
                settings=settings,
                project_root=tmp_path,
                registry=registry,
                runner=None,
            )

    async def test_rejects_policy_violation(self, settings: Settings, tmp_path: Path) -> None:
        registry = JobRegistry[GenerationResult]()
        oversized = {**VALID_SPEC, "generation": {"width": 4096, "height": 4096}}

        with pytest.raises(InvalidGenerationSpec):
            submit_generation(
                oversized,
                settings=settings,
                project_root=tmp_path,
                registry=registry,
                runner=None,
            )


class TestGetGenerationStatus:
    async def test_reports_completed_with_relative_paths(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        """絶対パスを返すと環境の情報が漏れるため、作業ルートからの相対で返す。"""
        registry = JobRegistry[GenerationResult]()

        async def runner(spec: Any) -> GenerationResult:
            return _result(tmp_path)

        job_id = submit_generation(
            VALID_SPEC,
            settings=settings,
            project_root=tmp_path,
            registry=registry,
            runner=runner,
        )["job_id"]
        await registry.wait(job_id)

        status = get_generation_status(job_id, registry=registry, project_root=tmp_path)

        assert status["status"] == "completed"
        assert status["seed"] == 4242
        assert status["files"] == ["outputs/2026-08-12/sample/image_0001.png"]
        assert status["metadata_path"] == "outputs/2026-08-12/sample/metadata.json"
        assert status["error"] is None

    async def test_reports_running(self, settings: Settings, tmp_path: Path) -> None:
        import asyncio

        registry = JobRegistry[GenerationResult]()
        release = asyncio.Event()

        async def runner(spec: Any) -> GenerationResult:
            await release.wait()
            return _result(tmp_path)

        job_id = submit_generation(
            VALID_SPEC,
            settings=settings,
            project_root=tmp_path,
            registry=registry,
            runner=runner,
        )["job_id"]
        await asyncio.sleep(0)

        status = get_generation_status(job_id, registry=registry, project_root=tmp_path)
        assert status["status"] == "running"
        assert status["files"] == []

        release.set()
        await registry.wait(job_id)

    async def test_reports_failure_with_exit_code(self, settings: Settings, tmp_path: Path) -> None:
        """CLIのexit code体系をそのまま応答へ持ち込む。"""
        registry = JobRegistry[GenerationResult]()

        async def runner(spec: Any) -> GenerationResult:
            raise GenerationTimeout("生成がタイムアウトしました")

        job_id = submit_generation(
            VALID_SPEC,
            settings=settings,
            project_root=tmp_path,
            registry=registry,
            runner=runner,
        )["job_id"]
        await registry.wait(job_id)

        status = get_generation_status(job_id, registry=registry, project_root=tmp_path)

        assert status["status"] == "failed"
        assert status["exit_code"] == 6
        assert "タイムアウト" in status["error"]

    async def test_unexpected_error_maps_to_exit_code_one(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        registry = JobRegistry[GenerationResult]()

        async def runner(spec: Any) -> GenerationResult:
            raise RuntimeError("想定外")

        job_id = submit_generation(
            VALID_SPEC,
            settings=settings,
            project_root=tmp_path,
            registry=registry,
            runner=runner,
        )["job_id"]
        await registry.wait(job_id)

        status = get_generation_status(job_id, registry=registry, project_root=tmp_path)

        assert status["status"] == "failed"
        assert status["exit_code"] == 1

    async def test_unknown_job_id_raises(self, tmp_path: Path) -> None:
        registry = JobRegistry[GenerationResult]()

        with pytest.raises(ValueError, match="job_id"):
            get_generation_status("nope", registry=registry, project_root=tmp_path)
