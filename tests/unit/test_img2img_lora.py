"""img2img と LoRA の併用。

テンプレート選択が task と LoRA指定の2軸で決まるようになったため、
4通りの組み合わせをここで固定する。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agentic_imagegen.adapters.comfyui.workflow import (
    IMG2IMG_LORA_BINDING,
    LORA_SLOT_ROLES,
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

LORA = {"name": "add_detail.safetensors", "strength_model": 0.7, "strength_clip": 0.6}


def _spec(*, task: str = "img2img", loras: list[dict[str, Any]] | None = None) -> GenerationSpec:
    payload: dict[str, Any] = {
        "version": "1",
        "task": task,
        "prompt": {"positive": "1girl, blue hair"},
        "generation": {"seed": 12345},
        "model": {"checkpoint": "v1-5-pruned-emaonly.safetensors"},
    }
    if task == "img2img":
        payload["source"] = {"image": "inputs/ref.png", "denoise": 0.5}
    if loras is not None:
        payload["model"]["loras"] = loras
    return GenerationSpec.model_validate(payload)


@pytest.fixture
def template() -> dict[str, Any]:
    return load_workflow_template("img2img_lora")


def _inputs(workflow: dict[str, Any], role: str) -> dict[str, Any]:
    node_id = IMG2IMG_LORA_BINDING.nodes[role].node_id
    inputs: dict[str, Any] = workflow[node_id]["inputs"]
    return inputs


class TestTemplateSelection:
    @pytest.mark.parametrize(
        ("task", "loras", "expected"),
        [
            ("txt2img", None, "txt2img"),
            ("txt2img", [LORA], "txt2img_lora"),
            ("img2img", None, "img2img"),
            ("img2img", [LORA], "img2img_lora"),
        ],
    )
    def test_matrix(self, task: str, loras: list[dict[str, Any]] | None, expected: str) -> None:
        assert resolve_workflow_name(_spec(task=task, loras=loras)) == expected

    def test_all_selectable_names_are_allowed(self) -> None:
        """選択されうる名前がすべて allowlist にあること。"""
        for task in ("txt2img", "img2img"):
            for loras in (None, [LORA]):
                assert resolve_workflow_name(_spec(task=task, loras=loras)) in ALLOWED_WORKFLOWS


class TestTemplateStructure:
    def test_keeps_load_image_and_vae_encode(self, template: dict[str, Any]) -> None:
        """LoRAノードを足す際に、img2img固有のノードを潰していないこと。"""
        class_types = {node["class_type"] for node in template.values()}

        assert "LoadImage" in class_types
        assert "VAEEncode" in class_types

    def test_lora_nodes_do_not_collide(self, template: dict[str, Any]) -> None:
        lora_ids = {IMG2IMG_LORA_BINDING.nodes[role].node_id for role in LORA_SLOT_ROLES}
        source_id = IMG2IMG_LORA_BINDING.nodes["source_image"].node_id
        vae_id = IMG2IMG_LORA_BINDING.nodes["vae_encode"].node_id

        assert source_id not in lora_ids
        assert vae_id not in lora_ids

    def test_vae_encode_uses_checkpoint_vae(self, template: dict[str, Any]) -> None:
        """LoraLoaderはVAEを出さないため、VAEはcheckpoint直結のままであること。"""
        vae_id = IMG2IMG_LORA_BINDING.nodes["vae_encode"].node_id
        checkpoint_id = IMG2IMG_LORA_BINDING.nodes["checkpoint"].node_id

        assert template[vae_id]["inputs"]["vae"][0] == checkpoint_id


class TestInjection:
    def test_injects_lora_and_source(self, template: dict[str, Any]) -> None:
        workflow = build_workflow(
            template,
            _spec(loras=[LORA]),
            seed=1,
            binding=IMG2IMG_LORA_BINDING,
            source_image_name="imagegen_abc_ref.png",
        )

        assert _inputs(workflow, "source_image")["image"] == "imagegen_abc_ref.png"
        assert _inputs(workflow, "ksampler")["denoise"] == 0.5
        first = _inputs(workflow, LORA_SLOT_ROLES[0])
        assert first["lora_name"] == "add_detail.safetensors"
        assert first["strength_model"] == 0.7
        assert first["strength_clip"] == 0.6

    def test_unused_slots_are_disabled(self, template: dict[str, Any]) -> None:
        workflow = build_workflow(
            template,
            _spec(loras=[LORA]),
            seed=1,
            binding=IMG2IMG_LORA_BINDING,
            source_image_name="a.png",
        )

        for role in LORA_SLOT_ROLES[1:]:
            assert _inputs(workflow, role)["strength_model"] == 0.0

    def test_detects_ksampler_bypassing_loras(self, template: dict[str, Any]) -> None:
        broken = json.loads(json.dumps(template))
        ksampler = IMG2IMG_LORA_BINDING.nodes["ksampler"].node_id
        checkpoint = IMG2IMG_LORA_BINDING.nodes["checkpoint"].node_id
        broken[ksampler]["inputs"]["model"] = [checkpoint, 0]

        with pytest.raises(WorkflowValidationError):
            build_workflow(
                broken,
                _spec(loras=[LORA]),
                seed=1,
                binding=IMG2IMG_LORA_BINDING,
                source_image_name="a.png",
            )

    def test_detects_latent_not_from_vae_encode(self, template: dict[str, Any]) -> None:
        """LoRAを足したあとも、入力画像経由のlatentであることを検証し続ける。"""
        broken = json.loads(json.dumps(template))
        ksampler = IMG2IMG_LORA_BINDING.nodes["ksampler"].node_id
        broken[ksampler]["inputs"]["latent_image"] = ["4", 0]

        with pytest.raises(WorkflowValidationError):
            build_workflow(
                broken,
                _spec(loras=[LORA]),
                seed=1,
                binding=IMG2IMG_LORA_BINDING,
                source_image_name="a.png",
            )

    def test_prepare_workflow_selects_and_injects(self) -> None:
        prepared = prepare_workflow(_spec(loras=[LORA]), source_image_name="a.png")

        assert prepared.workflow_name == "img2img_lora"
        assert _inputs(prepared.workflow, LORA_SLOT_ROLES[0])["lora_name"] == LORA["name"]
