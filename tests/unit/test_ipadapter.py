"""IPAdapter (reference) の注入とテンプレート選択。"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agentic_imagegen.adapters.comfyui.workflow import (
    REFERENCE_APPLY_ROLE,
    REFERENCE_CLIP_VISION_ROLE,
    REFERENCE_IMAGE_ROLE,
    REFERENCE_LOADER_ROLE,
    TXT2IMG_IPADAPTER_BINDING,
    build_workflow,
)
from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.errors import WorkflowValidationError
from agentic_imagegen.workflows.injector import (
    ALLOWED_WORKFLOWS,
    load_workflow_template,
    prepare_workflow,
    resolve_workflow_name,
)

REFERENCE = {
    "image": "inputs/character.png",
    "model": "ip-adapter-plus_sd15.safetensors",
    "clip_vision": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
    "weight": 0.8,
    "weight_type": "style transfer",
    "start_percent": 0.1,
    "end_percent": 0.9,
}
CONTROL = {
    "image": "inputs/pose.png",
    "model": "control_v11p_sd15_canny_fp16.safetensors",
}
LORA = {"name": "add_detail.safetensors"}


def _spec(
    *,
    task: str = "txt2img",
    reference: dict[str, Any] | None = REFERENCE,
    loras: list[dict[str, Any]] | None = None,
    **extra: Any,
) -> GenerationSpec:
    model: dict[str, Any] = {"checkpoint": "v1-5-pruned-emaonly.safetensors"}
    if loras is not None:
        model["loras"] = loras
    payload: dict[str, Any] = {
        "version": "1",
        "task": task,
        "prompt": {"positive": "1girl"},
        "generation": {"seed": 5},
        "model": model,
    }
    if reference is not None:
        payload["reference"] = reference
    payload.update(extra)
    return GenerationSpec.model_validate(payload)


@pytest.fixture
def template() -> dict[str, Any]:
    return load_workflow_template("txt2img_ipadapter")


def _inputs(workflow: dict[str, Any], role: str) -> dict[str, Any]:
    node_id = TXT2IMG_IPADAPTER_BINDING.nodes[role].node_id
    inputs: dict[str, Any] = workflow[node_id]["inputs"]
    return inputs


class TestTemplateSelection:
    def test_reference_switches_template(self) -> None:
        assert resolve_workflow_name(_spec()) == "txt2img_ipadapter"

    def test_combines_with_lora(self) -> None:
        assert resolve_workflow_name(_spec(loras=[LORA])) == "txt2img_lora_ipadapter"

    def test_combines_with_control(self) -> None:
        """構図 (ControlNet) と人物特徴 (IPAdapter) は併用できる。"""
        assert resolve_workflow_name(_spec(control=CONTROL)) == "txt2img_controlnet_ipadapter"

    def test_combines_with_lora_and_control(self) -> None:
        assert (
            resolve_workflow_name(_spec(loras=[LORA], control=CONTROL))
            == "txt2img_lora_controlnet_ipadapter"
        )

    def test_img2img_variant(self) -> None:
        spec = _spec(task="img2img", source={"image": "inputs/base.png"})

        assert resolve_workflow_name(spec) == "img2img_ipadapter"

    def test_all_variants_are_allowed(self) -> None:
        for name in (
            "txt2img_ipadapter",
            "txt2img_lora_ipadapter",
            "txt2img_controlnet_ipadapter",
            "txt2img_lora_controlnet_ipadapter",
            "img2img_ipadapter",
            "img2img_lora_ipadapter",
            "img2img_controlnet_ipadapter",
            "img2img_lora_controlnet_ipadapter",
        ):
            assert name in ALLOWED_WORKFLOWS

    def test_templates_exist(self) -> None:
        for name in ALLOWED_WORKFLOWS:
            if "ipadapter" in name:
                assert load_workflow_template(name)


class TestInjection:
    def test_injects_reference_values(self, template: dict[str, Any]) -> None:
        workflow = build_workflow(
            template,
            _spec(),
            seed=5,
            binding=TXT2IMG_IPADAPTER_BINDING,
            reference_image_name="uploaded_character.png",
        )

        assert _inputs(workflow, REFERENCE_IMAGE_ROLE)["image"] == "uploaded_character.png"
        assert (
            _inputs(workflow, REFERENCE_LOADER_ROLE)["ipadapter_file"]
            == "ip-adapter-plus_sd15.safetensors"
        )
        assert (
            _inputs(workflow, REFERENCE_CLIP_VISION_ROLE)["clip_name"]
            == "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
        )

        apply_node = _inputs(workflow, REFERENCE_APPLY_ROLE)
        assert apply_node["weight"] == 0.8
        assert apply_node["weight_type"] == "style transfer"
        assert apply_node["start_at"] == 0.1
        assert apply_node["end_at"] == 0.9

    def test_ksampler_receives_model_from_ipadapter(self, template: dict[str, Any]) -> None:
        """KSamplerのMODELがIPAdapter経由になっていないと、参照画像が効かない。"""
        workflow = build_workflow(
            template,
            _spec(),
            seed=5,
            binding=TXT2IMG_IPADAPTER_BINDING,
            reference_image_name="uploaded.png",
        )

        ksampler = _inputs(workflow, "ksampler")
        apply_id = TXT2IMG_IPADAPTER_BINDING.nodes[REFERENCE_APPLY_ROLE].node_id
        assert ksampler["model"] == [apply_id, 0]

    def test_rejects_missing_reference(self, template: dict[str, Any]) -> None:
        with pytest.raises(WorkflowValidationError, match="reference"):
            build_workflow(
                template,
                _spec(reference=None),
                seed=5,
                binding=TXT2IMG_IPADAPTER_BINDING,
                reference_image_name="uploaded.png",
            )

    def test_rejects_missing_uploaded_image(self, template: dict[str, Any]) -> None:
        with pytest.raises(WorkflowValidationError, match="アップロード"):
            build_workflow(
                template,
                _spec(),
                seed=5,
                binding=TXT2IMG_IPADAPTER_BINDING,
                reference_image_name=None,
            )

    def test_template_is_not_mutated(self, template: dict[str, Any]) -> None:
        before = json.dumps(template, sort_keys=True)

        build_workflow(
            template,
            _spec(),
            seed=5,
            binding=TXT2IMG_IPADAPTER_BINDING,
            reference_image_name="uploaded.png",
        )

        assert json.dumps(template, sort_keys=True) == before


class TestStructureValidation:
    def test_rejects_wrong_class_type(self, template: dict[str, Any]) -> None:
        broken = json.loads(json.dumps(template))
        node_id = TXT2IMG_IPADAPTER_BINDING.nodes[REFERENCE_APPLY_ROLE].node_id
        broken[node_id]["class_type"] = "IPAdapter"

        with pytest.raises(WorkflowValidationError):
            build_workflow(
                broken,
                _spec(),
                seed=5,
                binding=TXT2IMG_IPADAPTER_BINDING,
                reference_image_name="uploaded.png",
            )

    def test_rejects_ksampler_bypassing_ipadapter(self, template: dict[str, Any]) -> None:
        """KSamplerがcheckpointから直接MODELを受けていると、参照画像が効かないまま成功する。

        形は正しいので、構造検証で落とさないと気づけない。
        """
        broken = json.loads(json.dumps(template))
        ksampler_id = TXT2IMG_IPADAPTER_BINDING.nodes["ksampler"].node_id
        checkpoint_id = TXT2IMG_IPADAPTER_BINDING.nodes["checkpoint"].node_id
        broken[ksampler_id]["inputs"]["model"] = [checkpoint_id, 0]

        with pytest.raises(WorkflowValidationError):
            build_workflow(
                broken,
                _spec(),
                seed=5,
                binding=TXT2IMG_IPADAPTER_BINDING,
                reference_image_name="uploaded.png",
            )


class TestCombinedWithControlNet:
    def test_both_are_injected(self) -> None:
        """併用時、KSamplerはmodelをIPAdapterから、条件をControlNetから受ける。"""
        name = "txt2img_controlnet_ipadapter"
        binding = ALLOWED_WORKFLOWS[name]
        spec = _spec(control=CONTROL)

        prepared = prepare_workflow(
            spec,
            control_image_name="uploaded_pose.png",
            reference_image_name="uploaded_character.png",
        )

        assert prepared.workflow_name == name
        workflow = prepared.workflow
        ksampler_id = binding.nodes["ksampler"].node_id
        apply_id = binding.nodes[REFERENCE_APPLY_ROLE].node_id
        control_apply_id = binding.nodes["control_apply"].node_id

        assert workflow[ksampler_id]["inputs"]["model"] == [apply_id, 0]
        assert workflow[ksampler_id]["inputs"]["positive"] == [control_apply_id, 0]
        assert workflow[ksampler_id]["inputs"]["negative"] == [control_apply_id, 1]


def test_rejects_upscale_and_reference_together() -> None:
    """両方かけると生成時間が現実的でないため、テンプレートを用意していない。"""
    with pytest.raises(ValueError, match="upscale"):
        _spec(generation={"seed": 5, "upscale": {"scale": 1.5}})
