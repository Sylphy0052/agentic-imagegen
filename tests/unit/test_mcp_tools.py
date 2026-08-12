"""MCP toolの中身 (services層) のテスト。

MCP層は薄いアダプタに留め、ロジックはここでテストできる形にしておく。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agentic_imagegen.config import Settings
from agentic_imagegen.services.mcp_tools import (
    list_workflows,
    validate_generation,
)

VALID_SPEC: dict[str, Any] = {
    "version": "1",
    "task": "txt2img",
    "prompt": {"positive": "1girl, blue hair"},
    "generation": {"width": 512, "height": 768, "seed": 42},
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


class TestValidateGeneration:
    def test_valid_spec(self, settings: Settings, tmp_path: Path) -> None:
        result = validate_generation(VALID_SPEC, settings=settings, project_root=tmp_path)

        assert result["valid"] is True
        assert result["workflow"] == "txt2img"
        assert result["resolution"] == {"width": 512, "height": 768, "batch_size": 1}
        assert result["checkpoint"] == "v1-5-pruned-emaonly.safetensors"
        assert result["errors"] == []

    def test_reports_errors_instead_of_raising(self, settings: Settings, tmp_path: Path) -> None:
        """検証結果を得るのがtoolの目的なので、不正でも例外にせず結果として返す。"""
        broken = {**VALID_SPEC, "generation": {"width": 511}}

        result = validate_generation(broken, settings=settings, project_root=tmp_path)

        assert result["valid"] is False
        assert result["errors"]
        assert any("width" in message for message in result["errors"])

    def test_reports_policy_violation(self, settings: Settings, tmp_path: Path) -> None:
        """設定由来の上限超過も検証結果として返す。"""
        oversized = {**VALID_SPEC, "generation": {"width": 4096, "height": 4096}}

        result = validate_generation(oversized, settings=settings, project_root=tmp_path)

        assert result["valid"] is False
        assert result["errors"]

    def test_reports_lora_workflow(self, settings: Settings, tmp_path: Path) -> None:
        spec = {
            **VALID_SPEC,
            "model": {
                "checkpoint": "v1-5-pruned-emaonly.safetensors",
                "loras": [{"name": "add_detail.safetensors", "strength_model": 0.8}],
            },
        }

        result = validate_generation(spec, settings=settings, project_root=tmp_path)

        assert result["valid"] is True
        assert result["workflow"] == "txt2img_lora"
        assert result["loras"] == [
            {"name": "add_detail.safetensors", "strength_model": 0.8, "strength_clip": 1.0}
        ]

    def test_expands_presets(self, settings: Settings, tmp_path: Path) -> None:
        characters = settings.presets_root / "characters"
        characters.mkdir(parents=True)
        (characters / "kaede.yaml").write_text(
            "prompt:\n  positive: 1girl, solo, blue hair\n", encoding="utf-8"
        )
        spec = {**VALID_SPEC, "presets": {"character": "kaede"}}

        result = validate_generation(spec, settings=settings, project_root=tmp_path)

        assert result["valid"] is True
        assert result["presets"] == {"character": "kaede"}
        assert result["prompt"]["positive"].startswith("1girl, solo, blue hair")

    def test_reports_img2img_source(self, settings: Settings, tmp_path: Path) -> None:
        spec = {
            **VALID_SPEC,
            "task": "img2img",
            "generation": {"seed": 42},
            "source": {"image": "inputs/ref.png", "denoise": 0.4},
        }

        result = validate_generation(spec, settings=settings, project_root=tmp_path)

        assert result["valid"] is True
        assert result["workflow"] == "img2img"
        assert result["source"] == {"image": "inputs/ref.png", "denoise": 0.4}
        # img2imgは入力画像のサイズを使うため解像度は返さない
        assert result["resolution"] is None

    def test_rejects_non_mapping(self, settings: Settings, tmp_path: Path) -> None:
        result = validate_generation(
            ["not", "a", "mapping"], settings=settings, project_root=tmp_path
        )

        assert result["valid"] is False
        assert result["errors"]


class TestListWorkflows:
    def test_returns_allowed_workflows(self) -> None:
        names = list_workflows()

        assert set(names) == {"txt2img", "txt2img_lora", "img2img", "img2img_lora"}

    def test_is_sorted(self) -> None:
        names = list_workflows()

        assert names == sorted(names)
