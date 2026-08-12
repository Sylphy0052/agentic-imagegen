"""MCP tool がテキスト合成に追随していることの検証。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic_imagegen.config import Settings
from agentic_imagegen.domain.results import GenerationResult
from agentic_imagegen.services.jobs import JobRegistry, JobStatus
from agentic_imagegen.services.mcp_tools import get_generation_status, validate_generation


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        comfyui_base_url="http://127.0.0.1:8188",
        max_width=2048,
        max_height=2048,
        max_pixels=4194304,
        max_batch=4,
        timeout_seconds=30,
        output_root=tmp_path / "outputs",
        presets_root=tmp_path / "presets",
    )


def _payload(text: dict[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "prompt": {"positive": "a street"},
        "model": {"checkpoint": "v1-5-pruned-emaonly.safetensors"},
    }
    if text is not None:
        payload["text"] = text
    return payload


def _text() -> dict[str, Any]:
    return {"layers": [{"content": "秋葉原駅", "font": "NotoSansJP.ttf", "size": 48}]}


class TestValidateGeneration:
    def test_validates_text(self, tmp_path: Path) -> None:
        result = validate_generation(
            _payload(_text()), settings=_settings(tmp_path), project_root=tmp_path
        )

        assert result["valid"] is True
        assert result["text"] == {"layers": 1, "fonts": ["NotoSansJP.ttf"]}

    def test_returns_none_without_text(self, tmp_path: Path) -> None:
        result = validate_generation(
            _payload(None), settings=_settings(tmp_path), project_root=tmp_path
        )

        assert result["text"] is None

    def test_reports_reason_for_invalid_text(self, tmp_path: Path) -> None:
        result = validate_generation(
            _payload({"layers": [{"content": "x", "font": "a.woff2", "size": 10}]}),
            settings=_settings(tmp_path),
            project_root=tmp_path,
        )

        assert result["valid"] is False
        assert result["text"] is None
        assert any("font" in error for error in result["errors"])


async def _completed_status(tmp_path: Path, result: GenerationResult) -> dict[str, Any]:
    registry = JobRegistry()

    async def run() -> GenerationResult:
        return result

    job_id = registry.submit(run)
    await registry.wait(job_id)
    return get_generation_status(job_id, registry=registry, project_root=tmp_path)


def _result(directory: Path, *, text_files: tuple[Path, ...] = ()) -> GenerationResult:
    return GenerationResult(
        prompt_id="p1",
        seed=7,
        directory=directory,
        files=(directory / "image_0001.png",),
        metadata_path=directory / "metadata.json",
        text_files=text_files,
    )


class TestGetGenerationStatus:
    async def test_returns_composed_files(self, tmp_path: Path) -> None:
        directory = tmp_path / "outputs" / "2026-08-12" / "imagegen"

        status = await _completed_status(
            tmp_path,
            _result(directory, text_files=(directory / "image_0001_text.png",)),
        )

        assert status["status"] == JobStatus.COMPLETED.value
        assert status["text_files"] == ["outputs/2026-08-12/imagegen/image_0001_text.png"]

    async def test_returns_empty_without_compose(self, tmp_path: Path) -> None:
        directory = tmp_path / "outputs" / "2026-08-12" / "imagegen"

        status = await _completed_status(tmp_path, _result(directory))

        assert status["text_files"] == []
