"""LoRA用Workflowテンプレートへの注入とテンプレート選択。

テンプレートは LoraLoader を3段直列に持つ。指定が3件未満のときは
余ったスロットを無効化する必要があり、その扱いをここで固定する。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agentic_imagegen.adapters.comfyui.workflow import (
    LORA_SLOT_ROLES,
    TXT2IMG_LORA_BINDING,
    build_workflow,
)
from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.errors import WorkflowValidationError
from agentic_imagegen.workflows.injector import load_workflow_template, prepare_workflow


def _spec(loras: list[dict[str, Any]] | None = None, **overrides: Any) -> GenerationSpec:
    model: dict[str, Any] = {"checkpoint": "v1-5-pruned-emaonly.safetensors"}
    if loras is not None:
        model["loras"] = loras
    payload: dict[str, Any] = {
        "version": "1",
        "task": "txt2img",
        "prompt": {"positive": "1girl, blue hair"},
        "generation": {"seed": 12345},
        "model": model,
    }
    payload.update(overrides)
    return GenerationSpec.model_validate(payload)


@pytest.fixture
def lora_template() -> dict[str, Any]:
    return load_workflow_template("txt2img_lora")


def _slot_inputs(workflow: dict[str, Any], index: int) -> dict[str, Any]:
    role = LORA_SLOT_ROLES[index]
    node_id = TXT2IMG_LORA_BINDING.nodes[role].node_id
    inputs: dict[str, Any] = workflow[node_id]["inputs"]
    return inputs


class TestInjection:
    def test_single_lora_fills_first_slot(self, lora_template: dict[str, Any]) -> None:
        spec = _spec([{"name": "add_detail.safetensors", "strength_model": 0.8}])

        workflow = build_workflow(lora_template, spec, seed=1, binding=TXT2IMG_LORA_BINDING)

        first = _slot_inputs(workflow, 0)
        assert first["lora_name"] == "add_detail.safetensors"
        assert first["strength_model"] == 0.8
        assert first["strength_clip"] == 1.0

    def test_unused_slots_are_disabled(self, lora_template: dict[str, Any]) -> None:
        """未使用スロットは強度0で無効化する。lora_nameは空にできないため使い回す。"""
        spec = _spec([{"name": "add_detail.safetensors"}])

        workflow = build_workflow(lora_template, spec, seed=1, binding=TXT2IMG_LORA_BINDING)

        for index in (1, 2):
            slot = _slot_inputs(workflow, index)
            assert slot["strength_model"] == 0.0
            assert slot["strength_clip"] == 0.0
            assert slot["lora_name"] == "add_detail.safetensors"

    def test_fills_all_slots_in_order(self, lora_template: dict[str, Any]) -> None:
        spec = _spec(
            [
                {"name": "a.safetensors", "strength_model": 0.1, "strength_clip": 0.2},
                {"name": "b.safetensors", "strength_model": 0.3, "strength_clip": 0.4},
                {"name": "c.safetensors", "strength_model": 0.5, "strength_clip": 0.6},
            ]
        )

        workflow = build_workflow(lora_template, spec, seed=1, binding=TXT2IMG_LORA_BINDING)

        assert [_slot_inputs(workflow, i)["lora_name"] for i in range(3)] == [
            "a.safetensors",
            "b.safetensors",
            "c.safetensors",
        ]
        assert _slot_inputs(workflow, 2)["strength_clip"] == 0.6

    def test_rejects_spec_without_loras(self, lora_template: dict[str, Any]) -> None:
        """LoRAテンプレートにLoRA未指定のSpecを流すのは選択ミス。黙って通さない。"""
        with pytest.raises(WorkflowValidationError, match="LoRA"):
            build_workflow(lora_template, _spec(), seed=1, binding=TXT2IMG_LORA_BINDING)

    def test_does_not_mutate_template(self, lora_template: dict[str, Any]) -> None:
        original = json.loads(json.dumps(lora_template))
        spec = _spec([{"name": "add_detail.safetensors"}])

        build_workflow(lora_template, spec, seed=1, binding=TXT2IMG_LORA_BINDING)

        assert lora_template == original

    def test_other_parameters_are_still_injected(self, lora_template: dict[str, Any]) -> None:
        spec = _spec([{"name": "add_detail.safetensors"}])

        workflow = build_workflow(lora_template, spec, seed=777, binding=TXT2IMG_LORA_BINDING)

        ksampler_id = TXT2IMG_LORA_BINDING.nodes["ksampler"].node_id
        checkpoint_id = TXT2IMG_LORA_BINDING.nodes["checkpoint"].node_id
        assert workflow[ksampler_id]["inputs"]["seed"] == 777
        assert workflow[checkpoint_id]["inputs"]["ckpt_name"] == "v1-5-pruned-emaonly.safetensors"


class TestStructureValidation:
    def test_detects_broken_lora_chain(self, lora_template: dict[str, Any]) -> None:
        """2段目が1段目ではなくcheckpointから来ている場合を検出する。"""
        broken = json.loads(json.dumps(lora_template))
        second = TXT2IMG_LORA_BINDING.nodes[LORA_SLOT_ROLES[1]].node_id
        checkpoint = TXT2IMG_LORA_BINDING.nodes["checkpoint"].node_id
        broken[second]["inputs"]["model"] = [checkpoint, 0]

        spec = _spec([{"name": "add_detail.safetensors"}])
        with pytest.raises(WorkflowValidationError):
            build_workflow(broken, spec, seed=1, binding=TXT2IMG_LORA_BINDING)

    def test_detects_ksampler_bypassing_loras(self, lora_template: dict[str, Any]) -> None:
        """KSamplerがLoRAを経由せずcheckpointへ直結している場合を検出する。"""
        broken = json.loads(json.dumps(lora_template))
        ksampler = TXT2IMG_LORA_BINDING.nodes["ksampler"].node_id
        checkpoint = TXT2IMG_LORA_BINDING.nodes["checkpoint"].node_id
        broken[ksampler]["inputs"]["model"] = [checkpoint, 0]

        spec = _spec([{"name": "add_detail.safetensors"}])
        with pytest.raises(WorkflowValidationError):
            build_workflow(broken, spec, seed=1, binding=TXT2IMG_LORA_BINDING)

    def test_detects_missing_lora_node(self, lora_template: dict[str, Any]) -> None:
        broken = json.loads(json.dumps(lora_template))
        del broken[TXT2IMG_LORA_BINDING.nodes[LORA_SLOT_ROLES[2]].node_id]

        spec = _spec([{"name": "add_detail.safetensors"}])
        with pytest.raises(WorkflowValidationError):
            build_workflow(broken, spec, seed=1, binding=TXT2IMG_LORA_BINDING)


class TestTemplateSelection:
    def test_uses_plain_template_without_loras(self) -> None:
        prepared = prepare_workflow(_spec())

        assert prepared.workflow_name == "txt2img"

    def test_uses_lora_template_with_loras(self) -> None:
        prepared = prepare_workflow(_spec([{"name": "add_detail.safetensors"}]))

        assert prepared.workflow_name == "txt2img_lora"

    def test_lora_template_hash_differs(self) -> None:
        plain = prepare_workflow(_spec())
        with_lora = prepare_workflow(_spec([{"name": "add_detail.safetensors"}]))

        assert plain.template_hash != with_lora.template_hash

    def test_lora_values_reach_the_workflow(self) -> None:
        prepared = prepare_workflow(
            _spec([{"name": "more_details.safetensors", "strength_clip": 0.25}])
        )

        first = _slot_inputs(prepared.workflow, 0)
        assert first["lora_name"] == "more_details.safetensors"
        assert first["strength_clip"] == 0.25


def test_template_file_exists() -> None:
    path = Path("workflows/txt2img_lora.json")
    assert path.is_file(), "LoRA用テンプレートが配置されていません"
