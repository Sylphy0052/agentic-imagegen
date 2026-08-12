"""hires fix の注入とテンプレート選択。

テンプレートが task × LoRA有無 × upscale有無 の8通りになったため、
選択規則をここで固定する。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agentic_imagegen.adapters.comfyui.workflow import (
    HIRES_KSAMPLER_ROLE,
    TXT2IMG_HIRES_BINDING,
    UPSCALE_ROLE,
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

LORA = {"name": "add_detail.safetensors"}
UPSCALE = {"scale": 2.0, "denoise": 0.4, "steps": 6, "method": "bicubic"}


def _spec(
    *,
    task: str = "txt2img",
    loras: list[dict[str, Any]] | None = None,
    upscale: dict[str, Any] | None = None,
) -> GenerationSpec:
    generation: dict[str, Any] = {"steps": 20, "cfg": 7.0, "seed": 999}
    if upscale is not None:
        generation["upscale"] = upscale
    payload: dict[str, Any] = {
        "version": "1",
        "task": task,
        "prompt": {"positive": "1girl"},
        "generation": generation,
        "model": {"checkpoint": "v1-5-pruned-emaonly.safetensors"},
    }
    if task == "img2img":
        payload["source"] = {"image": "inputs/ref.png"}
        generation.pop("steps", None)
    if loras is not None:
        payload["model"]["loras"] = loras
    return GenerationSpec.model_validate(payload)


@pytest.fixture
def template() -> dict[str, Any]:
    return load_workflow_template("txt2img_hires")


def _inputs(workflow: dict[str, Any], role: str) -> dict[str, Any]:
    node_id = TXT2IMG_HIRES_BINDING.nodes[role].node_id
    inputs: dict[str, Any] = workflow[node_id]["inputs"]
    return inputs


class TestTemplateSelection:
    @pytest.mark.parametrize(
        ("task", "loras", "upscale", "expected"),
        [
            ("txt2img", None, None, "txt2img"),
            ("txt2img", [LORA], None, "txt2img_lora"),
            ("txt2img", None, UPSCALE, "txt2img_hires"),
            ("txt2img", [LORA], UPSCALE, "txt2img_lora_hires"),
            ("img2img", None, None, "img2img"),
            ("img2img", [LORA], None, "img2img_lora"),
            ("img2img", None, UPSCALE, "img2img_hires"),
            ("img2img", [LORA], UPSCALE, "img2img_lora_hires"),
        ],
    )
    def test_matrix(
        self,
        task: str,
        loras: list[dict[str, Any]] | None,
        upscale: dict[str, Any] | None,
        expected: str,
    ) -> None:
        spec = _spec(task=task, loras=loras, upscale=upscale)

        assert resolve_workflow_name(spec) == expected

    def test_all_selectable_names_are_allowed(self) -> None:
        for task in ("txt2img", "img2img"):
            for loras in (None, [LORA]):
                for upscale in (None, UPSCALE):
                    spec = _spec(task=task, loras=loras, upscale=upscale)
                    assert resolve_workflow_name(spec) in ALLOWED_WORKFLOWS

    def test_all_templates_exist(self) -> None:
        for name in ALLOWED_WORKFLOWS:
            assert load_workflow_template(name)


class TestInjection:
    def test_injects_scale_and_method(self, template: dict[str, Any]) -> None:
        workflow = build_workflow(
            template, _spec(upscale=UPSCALE), seed=1, binding=TXT2IMG_HIRES_BINDING
        )

        upscale = _inputs(workflow, UPSCALE_ROLE)
        assert upscale["scale_by"] == 2.0
        assert upscale["upscale_method"] == "bicubic"

    def test_second_pass_uses_own_steps_and_denoise(self, template: dict[str, Any]) -> None:
        workflow = build_workflow(
            template, _spec(upscale=UPSCALE), seed=1, binding=TXT2IMG_HIRES_BINDING
        )

        second = _inputs(workflow, HIRES_KSAMPLER_ROLE)
        assert second["steps"] == 6
        assert second["denoise"] == 0.4

    def test_second_pass_falls_back_to_base_steps(self, template: dict[str, Any]) -> None:
        workflow = build_workflow(
            template, _spec(upscale={"scale": 1.5}), seed=1, binding=TXT2IMG_HIRES_BINDING
        )

        assert _inputs(workflow, HIRES_KSAMPLER_ROLE)["steps"] == 20

    def test_second_pass_shares_seed(self, template: dict[str, Any]) -> None:
        """2段目でseedを変えると1段目の絵から離れてしまう。"""
        workflow = build_workflow(
            template, _spec(upscale=UPSCALE), seed=4242, binding=TXT2IMG_HIRES_BINDING
        )

        assert _inputs(workflow, "ksampler")["seed"] == 4242
        assert _inputs(workflow, HIRES_KSAMPLER_ROLE)["seed"] == 4242

    def test_rejects_spec_without_upscale(self, template: dict[str, Any]) -> None:
        with pytest.raises(WorkflowValidationError, match="upscale"):
            build_workflow(template, _spec(), seed=1, binding=TXT2IMG_HIRES_BINDING)


class TestStructureValidation:
    def test_detects_vae_decode_bypassing_second_pass(self, template: dict[str, Any]) -> None:
        """VAEDecodeが1段目から直接受けていると拡大が効かない。"""
        broken = json.loads(json.dumps(template))
        vae = TXT2IMG_HIRES_BINDING.nodes["vae_decode"].node_id
        ksampler = TXT2IMG_HIRES_BINDING.nodes["ksampler"].node_id
        broken[vae]["inputs"]["samples"] = [ksampler, 0]

        with pytest.raises(WorkflowValidationError):
            build_workflow(broken, _spec(upscale=UPSCALE), seed=1, binding=TXT2IMG_HIRES_BINDING)

    def test_detects_upscale_not_fed_by_first_pass(self, template: dict[str, Any]) -> None:
        broken = json.loads(json.dumps(template))
        upscale = TXT2IMG_HIRES_BINDING.nodes[UPSCALE_ROLE].node_id
        broken[upscale]["inputs"]["samples"] = ["5", 0]

        with pytest.raises(WorkflowValidationError):
            build_workflow(broken, _spec(upscale=UPSCALE), seed=1, binding=TXT2IMG_HIRES_BINDING)

    def test_prepare_workflow_selects_and_injects(self) -> None:
        prepared = prepare_workflow(_spec(loras=[LORA], upscale=UPSCALE))

        assert prepared.workflow_name == "txt2img_lora_hires"
