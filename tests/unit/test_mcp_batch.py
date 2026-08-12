"""generate_batch / get_batch_status の中身。

一括生成の実体は差し替えられる形にして、ここではtoolの応答契約だけを見る。
CLIの batch と同じ services.batch を通ることも合わせて確認する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agentic_imagegen.config import Settings
from agentic_imagegen.domain.results import GenerationResult
from agentic_imagegen.errors import GenerationTimeout, InvalidGenerationSpec
from agentic_imagegen.services.batch import BatchItem, BatchOutcome
from agentic_imagegen.services.jobs import JobRegistry
from agentic_imagegen.services.mcp_tools import get_batch_status, submit_batch

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


@pytest.fixture
def registry() -> JobRegistry[list[BatchOutcome]]:
    return JobRegistry[list[BatchOutcome]]()


def _result(root: Path, name: str, seed: int) -> GenerationResult:
    directory = root / "outputs" / "2026-08-12" / name
    directory.mkdir(parents=True, exist_ok=True)
    image = directory / "image_0001.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    metadata = directory / "metadata.json"
    metadata.write_text("{}", encoding="utf-8")
    return GenerationResult(
        prompt_id=f"p-{seed}",
        seed=seed,
        directory=directory,
        files=(image,),
        metadata_path=metadata,
    )


class TestSubmitBatch:
    async def test_expands_seeds_into_items(
        self, settings: Settings, tmp_path: Path, registry: JobRegistry[list[BatchOutcome]]
    ) -> None:
        seen: list[int] = []

        async def runner(item: BatchItem) -> GenerationResult:
            seen.append(item.spec.generation.seed)
            return _result(tmp_path, f"s{item.spec.generation.seed}", item.spec.generation.seed)

        payload = submit_batch(
            [VALID_SPEC],
            seeds=[11, 22, 33],
            settings=settings,
            project_root=tmp_path,
            registry=registry,
            runner=runner,
        )

        assert payload["status"] == "running"
        assert payload["total"] == 3
        assert [item["label"] for item in payload["items"]] == [
            "spec[0] (seed=11)",
            "spec[0] (seed=22)",
            "spec[0] (seed=33)",
        ]
        assert {item["workflow"] for item in payload["items"]} == {"txt2img"}

        await registry.wait(payload["job_id"])
        assert seen == [11, 22, 33]

    async def test_runs_multiple_specs_without_seeds(
        self, settings: Settings, tmp_path: Path, registry: JobRegistry[list[BatchOutcome]]
    ) -> None:
        async def runner(item: BatchItem) -> GenerationResult:
            return _result(tmp_path, "x", 1)

        payload = submit_batch(
            [VALID_SPEC, {**VALID_SPEC, "prompt": {"positive": "1boy"}}],
            settings=settings,
            project_root=tmp_path,
            registry=registry,
            runner=runner,
        )

        assert payload["total"] == 2
        assert [item["label"] for item in payload["items"]] == ["spec[0]", "spec[1]"]

        await registry.wait(payload["job_id"])

    def test_rejects_invalid_spec_before_submitting(
        self, settings: Settings, tmp_path: Path, registry: JobRegistry[list[BatchOutcome]]
    ) -> None:
        """1件でも不正なら1件も投入しない。途中まで生成された状態を作らないため。"""
        with pytest.raises(InvalidGenerationSpec):
            submit_batch(
                [VALID_SPEC, {**VALID_SPEC, "generation": {"width": 511}}],
                settings=settings,
                project_root=tmp_path,
                registry=registry,
            )

    def test_rejects_non_sequence(
        self, settings: Settings, tmp_path: Path, registry: JobRegistry[list[BatchOutcome]]
    ) -> None:
        with pytest.raises(InvalidGenerationSpec):
            submit_batch(
                VALID_SPEC,
                settings=settings,
                project_root=tmp_path,
                registry=registry,
            )

    def test_rejects_empty(
        self, settings: Settings, tmp_path: Path, registry: JobRegistry[list[BatchOutcome]]
    ) -> None:
        with pytest.raises(InvalidGenerationSpec):
            submit_batch([], settings=settings, project_root=tmp_path, registry=registry)

    def test_rejects_policy_violation(
        self, settings: Settings, tmp_path: Path, registry: JobRegistry[list[BatchOutcome]]
    ) -> None:
        oversized = {**VALID_SPEC, "generation": {"width": 4096, "height": 4096}}

        with pytest.raises(InvalidGenerationSpec):
            submit_batch([oversized], settings=settings, project_root=tmp_path, registry=registry)


class TestGetBatchStatus:
    async def test_reports_running(
        self, settings: Settings, tmp_path: Path, registry: JobRegistry[list[BatchOutcome]]
    ) -> None:
        import asyncio

        release = asyncio.Event()

        async def runner(item: BatchItem) -> GenerationResult:
            await release.wait()
            return _result(tmp_path, "x", 1)

        payload = submit_batch(
            [VALID_SPEC],
            settings=settings,
            project_root=tmp_path,
            registry=registry,
            runner=runner,
        )

        status = get_batch_status(payload["job_id"], registry=registry, project_root=tmp_path)
        assert status["status"] == "running"
        assert status["items"] == []

        release.set()
        await registry.wait(payload["job_id"])

    async def test_reports_completed_items(
        self, settings: Settings, tmp_path: Path, registry: JobRegistry[list[BatchOutcome]]
    ) -> None:
        async def runner(item: BatchItem) -> GenerationResult:
            seed = item.spec.generation.seed
            return _result(tmp_path, f"s{seed}", seed)

        payload = submit_batch(
            [VALID_SPEC],
            seeds=[7, 8],
            settings=settings,
            project_root=tmp_path,
            registry=registry,
            runner=runner,
        )
        await registry.wait(payload["job_id"])

        status = get_batch_status(payload["job_id"], registry=registry, project_root=tmp_path)

        assert status["status"] == "completed"
        assert status["total"] == 2
        assert status["succeeded"] == 2
        assert status["failed"] == 0
        first = status["items"][0]
        assert first["status"] == "completed"
        assert first["seed"] == 7
        assert first["files"] == ["outputs/2026-08-12/s7/image_0001.png"]
        assert first["metadata_path"] == "outputs/2026-08-12/s7/metadata.json"
        assert first["error"] is None

    async def test_partial_failure_keeps_going(
        self, settings: Settings, tmp_path: Path, registry: JobRegistry[list[BatchOutcome]]
    ) -> None:
        """1件失敗しても残りは続き、結果には両方が並ぶ。"""

        async def runner(item: BatchItem) -> GenerationResult:
            seed = item.spec.generation.seed
            if seed == 8:
                raise GenerationTimeout("時間切れ")
            return _result(tmp_path, f"s{seed}", seed)

        payload = submit_batch(
            [VALID_SPEC],
            seeds=[7, 8, 9],
            settings=settings,
            project_root=tmp_path,
            registry=registry,
            runner=runner,
        )
        await registry.wait(payload["job_id"])

        status = get_batch_status(payload["job_id"], registry=registry, project_root=tmp_path)

        assert status["status"] == "completed"
        assert status["succeeded"] == 2
        assert status["failed"] == 1
        failed = status["items"][1]
        assert failed["status"] == "failed"
        assert failed["exit_code"] == GenerationTimeout.exit_code
        assert "時間切れ" in failed["error"]
        assert failed["files"] == []

    def test_rejects_unknown_job(
        self, tmp_path: Path, registry: JobRegistry[list[BatchOutcome]]
    ) -> None:
        with pytest.raises(ValueError, match="job_id"):
            get_batch_status("nope", registry=registry, project_root=tmp_path)
