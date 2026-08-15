"""エラーメッセージへ作業ルート配下の絶対パスが出ないことのテスト。

MCP tool の応答やジョブの error はそのままエージェントへ渡るため、
利用者のディレクトリ構成が読み取れる形で絶対パスを載せない。
作業ルートの外を指す場合だけは、どこを指しているか分からなくなるのを
避けるため絶対パスのまま示す。
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from agentic_imagegen.adapters.comfyui.client import ComfyUIClient
from agentic_imagegen.config import Settings
from agentic_imagegen.domain.presets import PresetKind
from agentic_imagegen.errors import InvalidGenerationSpec, WorkflowValidationError
from agentic_imagegen.services.mcp_tools import validate_generation
from agentic_imagegen.services.preset_loader import load_preset
from agentic_imagegen.services.spec_loader import load_spec
from agentic_imagegen.workflows.injector import load_workflow_template

SPEC_WITH_PRESETS = """
version: "1"
task: txt2img

presets:
  character: missing

prompt:
  positive: looking at viewer

model:
  checkpoint: meinamix_v12Final.safetensors
"""


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        comfyui_base_url="http://127.0.0.1:8188",
        max_width=2048,
        max_height=2048,
        max_pixels=4194304,
        max_batch=4,
        timeout_seconds=5,
        output_root=Path("outputs"),
        presets_root=tmp_path / "presets",
        max_source_bytes=1024,
        fonts_root=tmp_path / "fonts",
    )


class TestPresetLoader:
    def test_missing_preset_message_is_relative_to_project_root(self, tmp_path: Path) -> None:
        root = tmp_path / "presets"
        (root / "characters").mkdir(parents=True)

        with pytest.raises(InvalidGenerationSpec) as excinfo:
            load_preset(PresetKind.CHARACTER, "missing", root=root, project_root=tmp_path)

        message = str(excinfo.value)
        assert "presets/characters/missing.yaml" in message
        assert str(tmp_path) not in message

    def test_broken_preset_message_is_relative_to_project_root(self, tmp_path: Path) -> None:
        root = tmp_path / "presets"
        (root / "characters").mkdir(parents=True)
        (root / "characters" / "kaede.yaml").write_text("prompt: [\n", encoding="utf-8")

        with pytest.raises(InvalidGenerationSpec) as excinfo:
            load_preset(PresetKind.CHARACTER, "kaede", root=root, project_root=tmp_path)

        message = str(excinfo.value)
        assert "presets/characters/kaede.yaml" in message
        assert str(tmp_path) not in message

    def test_invalid_preset_document_message_is_relative_to_project_root(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "presets"
        (root / "characters").mkdir(parents=True)
        (root / "characters" / "kaede.yaml").write_text("unknown: 1\n", encoding="utf-8")

        with pytest.raises(InvalidGenerationSpec) as excinfo:
            load_preset(PresetKind.CHARACTER, "kaede", root=root, project_root=tmp_path)

        assert str(tmp_path) not in str(excinfo.value)

    def test_keeps_absolute_path_outside_project_root(self, tmp_path: Path) -> None:
        root = tmp_path / "outside" / "presets"
        (root / "characters").mkdir(parents=True)
        project_root = tmp_path / "project"
        project_root.mkdir()

        with pytest.raises(InvalidGenerationSpec) as excinfo:
            load_preset(PresetKind.CHARACTER, "missing", root=root, project_root=project_root)

        assert str(root.resolve()) in str(excinfo.value)

    def test_without_project_root_keeps_absolute_path(self, tmp_path: Path) -> None:
        root = tmp_path / "presets"
        (root / "characters").mkdir(parents=True)

        with pytest.raises(InvalidGenerationSpec) as excinfo:
            load_preset(PresetKind.CHARACTER, "missing", root=root)

        assert str(root) in str(excinfo.value)


class TestSpecLoader:
    def test_load_spec_passes_project_root_down_to_preset_loader(self, tmp_path: Path) -> None:
        (tmp_path / "presets" / "characters").mkdir(parents=True)
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(SPEC_WITH_PRESETS, encoding="utf-8")

        with pytest.raises(InvalidGenerationSpec) as excinfo:
            load_spec(
                spec_path,
                presets_root=tmp_path / "presets",
                project_root=tmp_path,
            )

        message = str(excinfo.value)
        assert "presets/characters/missing.yaml" in message
        assert str(tmp_path) not in message


class TestWorkflowTemplate:
    def test_missing_template_message_is_relative_to_project_root(self, tmp_path: Path) -> None:
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()

        with pytest.raises(WorkflowValidationError) as excinfo:
            load_workflow_template("txt2img", workflows_dir=workflows_dir, project_root=tmp_path)

        message = str(excinfo.value)
        assert "workflows/txt2img.json" in message
        assert str(tmp_path) not in message

    def test_broken_template_message_is_relative_to_project_root(self, tmp_path: Path) -> None:
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        (workflows_dir / "txt2img.json").write_text("{", encoding="utf-8")

        with pytest.raises(WorkflowValidationError) as excinfo:
            load_workflow_template("txt2img", workflows_dir=workflows_dir, project_root=tmp_path)

        assert str(tmp_path) not in str(excinfo.value)

    def test_non_mapping_template_message_is_relative_to_project_root(self, tmp_path: Path) -> None:
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        (workflows_dir / "txt2img.json").write_text("[]", encoding="utf-8")

        with pytest.raises(WorkflowValidationError) as excinfo:
            load_workflow_template("txt2img", workflows_dir=workflows_dir, project_root=tmp_path)

        assert str(tmp_path) not in str(excinfo.value)


class TestUploadImage:
    @pytest.mark.asyncio
    async def test_unreadable_image_message_has_no_absolute_path(self, tmp_path: Path) -> None:
        missing = tmp_path / "inputs" / "reference.png"

        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={}))
        async with ComfyUIClient(_settings(tmp_path), transport=transport) as client:
            with pytest.raises(InvalidGenerationSpec) as excinfo:
                await client.upload_image(missing)

        message = str(excinfo.value)
        assert "reference.png" in message
        assert str(tmp_path) not in message


class TestMcpTools:
    def test_validate_generation_reports_relative_preset_path(self, tmp_path: Path) -> None:
        (tmp_path / "presets" / "characters").mkdir(parents=True)
        spec = {
            "version": "1",
            "task": "txt2img",
            "presets": {"character": "missing"},
            "prompt": {"positive": "1girl"},
            "model": {"checkpoint": "meinamix_v12Final.safetensors"},
        }

        result = validate_generation(spec, settings=_settings(tmp_path), project_root=tmp_path)

        assert result["valid"] is False
        errors = "\n".join(result["errors"])
        assert "presets/characters/missing.yaml" in errors
        assert str(tmp_path) not in errors


#: アップロード失敗時に label 付きメッセージへ包み直す処理は
#: ComfyUIアダプタの execute() 内 (_upload_spec_image) へ移った (Issue #31)。
#: 対応するテストは
#: tests/unit/test_comfyui_execute.py の
#: test_execute_wraps_upload_failure_with_label_and_no_absolute_path を参照。
