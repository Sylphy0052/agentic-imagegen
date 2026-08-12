"""img2img用テンプレートへの注入と、入力画像のアップロード連携。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agentic_imagegen.adapters.comfyui.workflow import IMG2IMG_BINDING, build_workflow
from agentic_imagegen.config import Settings
from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.domain.policy import resolve_source_image
from agentic_imagegen.domain.results import HealthStatus, ImageRef
from agentic_imagegen.errors import InvalidGenerationSpec, WorkflowValidationError
from agentic_imagegen.services.generation import generate
from agentic_imagegen.workflows.injector import load_workflow_template, prepare_workflow

PNG = b"\x89PNG\r\n\x1a\n"


def _spec(**overrides: Any) -> GenerationSpec:
    payload: dict[str, Any] = {
        "version": "1",
        "task": "img2img",
        "prompt": {"positive": "1girl, blue hair"},
        "source": {"image": "inputs/ref.png", "denoise": 0.45},
        "generation": {"seed": 12345},
        "model": {"checkpoint": "v1-5-pruned-emaonly.safetensors"},
        "output": {"prefix": "img2img_test"},
    }
    payload.update(overrides)
    return GenerationSpec.model_validate(payload)


@pytest.fixture
def template() -> dict[str, Any]:
    return load_workflow_template("img2img")


def _inputs(workflow: dict[str, Any], role: str) -> dict[str, Any]:
    node_id = IMG2IMG_BINDING.nodes[role].node_id
    inputs: dict[str, Any] = workflow[node_id]["inputs"]
    return inputs


class TestInjection:
    def test_injects_uploaded_name_and_denoise(self, template: dict[str, Any]) -> None:
        workflow = build_workflow(
            template,
            _spec(),
            seed=1,
            binding=IMG2IMG_BINDING,
            source_image_name="imagegen_abc123_ref.png",
        )

        assert _inputs(workflow, "source_image")["image"] == "imagegen_abc123_ref.png"
        assert _inputs(workflow, "ksampler")["denoise"] == 0.45

    def test_requires_uploaded_name(self, template: dict[str, Any]) -> None:
        """アップロード前の名前でLoadImageを埋めるとComfyUI側で失敗するため、先に落とす。"""
        with pytest.raises(WorkflowValidationError, match="アップロード"):
            build_workflow(template, _spec(), seed=1, binding=IMG2IMG_BINDING)

    def test_other_parameters_are_injected(self, template: dict[str, Any]) -> None:
        workflow = build_workflow(
            template, _spec(), seed=777, binding=IMG2IMG_BINDING, source_image_name="a.png"
        )

        assert _inputs(workflow, "ksampler")["seed"] == 777
        assert _inputs(workflow, "positive_prompt")["text"] == "1girl, blue hair"
        assert _inputs(workflow, "save_image")["filename_prefix"] == "img2img_test"

    def test_template_has_no_latent_node(self, template: dict[str, Any]) -> None:
        """img2imgは入力画像のサイズを使うため EmptyLatentImage を持たない。"""
        assert all(node["class_type"] != "EmptyLatentImage" for node in template.values())

    def test_detects_ksampler_not_using_vae_encode(self, template: dict[str, Any]) -> None:
        broken = json.loads(json.dumps(template))
        ksampler = IMG2IMG_BINDING.nodes["ksampler"].node_id
        broken[ksampler]["inputs"]["latent_image"] = ["4", 0]

        with pytest.raises(WorkflowValidationError):
            build_workflow(
                broken, _spec(), seed=1, binding=IMG2IMG_BINDING, source_image_name="a.png"
            )

    def test_selects_img2img_template(self) -> None:
        prepared = prepare_workflow(_spec(), source_image_name="a.png")

        assert prepared.workflow_name == "img2img"


class TestSourceImagePolicy:
    def test_resolves_existing_file(self, tmp_path: Path) -> None:
        (tmp_path / "inputs").mkdir()
        image = tmp_path / "inputs" / "ref.png"
        image.write_bytes(PNG)

        resolved = resolve_source_image("inputs/ref.png", tmp_path, max_bytes=1024)

        assert resolved == image.resolve()

    def test_rejects_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidGenerationSpec, match="見つかりません"):
            resolve_source_image("inputs/absent.png", tmp_path, max_bytes=1024)

    def test_rejects_oversized_file(self, tmp_path: Path) -> None:
        image = tmp_path / "big.png"
        image.write_bytes(PNG * 100)

        with pytest.raises(InvalidGenerationSpec, match="大きすぎます"):
            resolve_source_image("big.png", tmp_path, max_bytes=16)

    def test_rejects_escape_via_symlink_target(self, tmp_path: Path) -> None:
        """解決後のパスが作業ルートの外なら拒否する。"""
        outside = tmp_path.parent / "outside.png"
        outside.write_bytes(PNG)
        root = tmp_path / "root"
        root.mkdir()
        (root / "link.png").symlink_to(outside)

        with pytest.raises(InvalidGenerationSpec, match="外を指しています"):
            resolve_source_image("link.png", root, max_bytes=1024)


class _FakeBackend:
    def __init__(self) -> None:
        self.uploaded: Path | None = None
        self.submitted: dict[str, Any] | None = None

    async def submit(self, workflow: dict[str, Any]) -> str:
        self.submitted = workflow
        return "prompt-1"

    async def wait_for_completion(self, prompt_id: str, *, timeout: float | None = None) -> None:
        return None

    async def fetch_outputs(self, prompt_id: str) -> tuple[ImageRef, ...]:
        return (ImageRef(filename="out.png", subfolder="", type="output"),)

    async def download(self, ref: ImageRef) -> bytes:
        return PNG

    async def health(self) -> HealthStatus:
        return HealthStatus(base_url="http://127.0.0.1:8188", comfyui_version="0.32.0", devices=())

    async def upload_image(self, path: Path) -> str:
        self.uploaded = path
        return f"imagegen_test_{path.name}"


async def test_generate_uploads_source_image(tmp_path: Path) -> None:
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "ref.png").write_bytes(PNG)
    backend = _FakeBackend()
    settings = Settings(
        comfyui_base_url="http://127.0.0.1:8188",
        max_width=2048,
        max_height=2048,
        max_pixels=4194304,
        max_batch=4,
        timeout_seconds=30,
        output_root=Path("outputs"),
    )

    result = await generate(_spec(), settings, backend=backend, project_root=tmp_path)

    assert backend.uploaded == (tmp_path / "inputs" / "ref.png").resolve()
    assert backend.submitted is not None
    load_image = IMG2IMG_BINDING.nodes["source_image"].node_id
    assert backend.submitted[load_image]["inputs"]["image"] == "imagegen_test_ref.png"

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["workflow"] == "img2img"
    assert metadata["spec"]["source"]["denoise"] == 0.45
