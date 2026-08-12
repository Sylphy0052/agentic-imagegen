"""ControlNet の注入とテンプレート選択。"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agentic_imagegen.adapters.comfyui.workflow import (
    CONTROL_APPLY_ROLE,
    CONTROL_IMAGE_ROLE,
    CONTROL_LOADER_ROLE,
    CONTROL_PREPROCESSOR_ROLE,
    TXT2IMG_CONTROLNET_BINDING,
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

CONTROL = {
    "image": "inputs/pose.png",
    "model": "control_v11p_sd15_canny_fp16.safetensors",
    "strength": 0.8,
    "start_percent": 0.1,
    "end_percent": 0.9,
    "low_threshold": 0.3,
    "high_threshold": 0.7,
}
LORA = {"name": "add_detail.safetensors"}


def _spec(
    *,
    task: str = "txt2img",
    control: dict[str, Any] | None = CONTROL,
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
    if control is not None:
        payload["control"] = control
    if task == "img2img":
        payload["source"] = {"image": "inputs/ref.png"}
    payload.update(extra)
    return GenerationSpec.model_validate(payload)


@pytest.fixture
def template() -> dict[str, Any]:
    return load_workflow_template("txt2img_controlnet")


def _inputs(workflow: dict[str, Any], role: str) -> dict[str, Any]:
    node_id = TXT2IMG_CONTROLNET_BINDING.nodes[role].node_id
    inputs: dict[str, Any] = workflow[node_id]["inputs"]
    return inputs


class TestTemplateSelection:
    @pytest.mark.parametrize(
        ("task", "loras", "expected"),
        [
            ("txt2img", None, "txt2img_controlnet"),
            ("txt2img", [LORA], "txt2img_lora_controlnet"),
            ("img2img", None, "img2img_controlnet"),
            ("img2img", [LORA], "img2img_lora_controlnet"),
        ],
    )
    def test_matrix(self, task: str, loras: list[dict[str, Any]] | None, expected: str) -> None:
        spec = _spec(task=task, loras=loras)

        assert resolve_workflow_name(spec) == expected
        assert expected in ALLOWED_WORKFLOWS

    def test_without_control_uses_plain_template(self) -> None:
        assert resolve_workflow_name(_spec(control=None)) == "txt2img"


class TestInjection:
    def test_injects_all_parameters(self, template: dict[str, Any]) -> None:
        workflow = build_workflow(
            template,
            _spec(),
            seed=1,
            binding=TXT2IMG_CONTROLNET_BINDING,
            control_image_name="imagegen_abc_pose.png",
        )

        assert _inputs(workflow, CONTROL_IMAGE_ROLE)["image"] == "imagegen_abc_pose.png"
        assert (
            _inputs(workflow, CONTROL_LOADER_ROLE)["control_net_name"]
            == "control_v11p_sd15_canny_fp16.safetensors"
        )
        preprocessor = _inputs(workflow, CONTROL_PREPROCESSOR_ROLE)
        assert preprocessor["low_threshold"] == 0.3
        assert preprocessor["high_threshold"] == 0.7
        apply_node = _inputs(workflow, CONTROL_APPLY_ROLE)
        assert apply_node["strength"] == 0.8
        assert apply_node["start_percent"] == 0.1
        assert apply_node["end_percent"] == 0.9

    def test_requires_uploaded_image(self, template: dict[str, Any]) -> None:
        with pytest.raises(WorkflowValidationError, match="アップロード"):
            build_workflow(template, _spec(), seed=1, binding=TXT2IMG_CONTROLNET_BINDING)

    def test_requires_control_in_spec(self, template: dict[str, Any]) -> None:
        with pytest.raises(WorkflowValidationError, match="control"):
            build_workflow(
                template,
                _spec(control=None),
                seed=1,
                binding=TXT2IMG_CONTROLNET_BINDING,
                control_image_name="a.png",
            )


class TestStructureValidation:
    @pytest.mark.parametrize("key", ["positive", "negative"])
    def test_detects_ksampler_bypassing_controlnet(
        self, template: dict[str, Any], key: str
    ) -> None:
        """片方だけ元のCLIPTextEncodeへ繋がっていると条件が食い違う。"""
        broken = json.loads(json.dumps(template))
        ksampler = TXT2IMG_CONTROLNET_BINDING.nodes["ksampler"].node_id
        source = TXT2IMG_CONTROLNET_BINDING.nodes[f"{key}_prompt"].node_id
        broken[ksampler]["inputs"][key] = [source, 0]

        with pytest.raises(WorkflowValidationError):
            build_workflow(
                broken,
                _spec(),
                seed=1,
                binding=TXT2IMG_CONTROLNET_BINDING,
                control_image_name="a.png",
            )

    def test_detects_apply_not_fed_by_preprocessor(self, template: dict[str, Any]) -> None:
        broken = json.loads(json.dumps(template))
        apply_node = TXT2IMG_CONTROLNET_BINDING.nodes[CONTROL_APPLY_ROLE].node_id
        load = TXT2IMG_CONTROLNET_BINDING.nodes[CONTROL_IMAGE_ROLE].node_id
        broken[apply_node]["inputs"]["image"] = [load, 0]

        with pytest.raises(WorkflowValidationError):
            build_workflow(
                broken,
                _spec(),
                seed=1,
                binding=TXT2IMG_CONTROLNET_BINDING,
                control_image_name="a.png",
            )

    def test_prepare_workflow_selects_and_injects(self) -> None:
        prepared = prepare_workflow(_spec(), control_image_name="a.png")

        assert prepared.workflow_name == "txt2img_controlnet"
        assert _inputs(prepared.workflow, CONTROL_IMAGE_ROLE)["image"] == "a.png"


def test_rejects_upscale_and_control_together() -> None:
    """両方かけると生成時間が現実的でないためテンプレートを用意していない。"""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="同時指定"):
        _spec(generation={"seed": 5, "upscale": {"scale": 1.5}})
