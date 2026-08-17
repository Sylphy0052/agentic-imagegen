"""DiT系モデル (unet / clip / vae) とLoRAの併用の検査。

DiT系のLoRAは `LoraLoader` を挟むだけで当てられるが、checkpoint系と違って
MODEL と CLIP の供給元が別ノード (UNETLoader / CLIPLoader) に分かれている。
1段でも迂回していればLoRAが効かないため、結線まで確かめる。
"""

from __future__ import annotations

import itertools
from typing import Any

import pytest
from pydantic import ValidationError

from agentic_imagegen.adapters.comfyui.workflow import (
    ALL_BINDINGS,
    CLIP_LOADER_ROLE,
    LORA_SLOT_ROLES,
    UNET_LOADER_ROLE,
)
from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.workflows.injector import prepare_workflow, resolve_workflow_name

SEPARATE_MODEL = {
    "unet": "hassakuAnima_v13_int8.safetensors",
    "clip": "qwen_3_06b_base.safetensors",
    "vae": "qwen_image_vae.safetensors",
}
LORA = {"name": "anima_context_detailer_base10.safetensors", "strength_model": 0.8}


def _spec(*, loras: list[dict[str, Any]] | None = None, **generation: Any) -> GenerationSpec:
    model: dict[str, Any] = dict(SEPARATE_MODEL)
    if loras is not None:
        model["loras"] = loras
    payload: dict[str, Any] = {
        "version": "1",
        "task": "txt2img",
        "prompt": {"positive": "1girl", "negative": "worst quality"},
        "generation": {"seed": 42, "steps": 32, "cfg": 4.5, "sampler": "er_sde", **generation},
        "model": model,
    }
    return GenerationSpec.model_validate(payload)


class TestSpecAcceptance:
    def test_lora_is_accepted(self) -> None:
        """DiT系向けのLoRAが出回ったため、拒否をやめる (Issue #39)。"""
        spec = _spec(loras=[LORA])

        assert spec.model.loras[0].name == LORA["name"]

    def test_clip_skip_is_still_rejected(self) -> None:
        """Qwen3 text encoderには「最終層を打ち切る」がそのままの意味を持たない。"""
        with pytest.raises(ValidationError, match="clip_skip"):
            GenerationSpec.model_validate(
                {
                    "version": "1",
                    "task": "txt2img",
                    "prompt": {"positive": "1girl"},
                    "model": {**SEPARATE_MODEL, "clip_skip": 2},
                }
            )


class TestTemplateSelection:
    def test_lora_switches_to_the_lora_template(self) -> None:
        assert resolve_workflow_name(_spec(loras=[LORA])) == "txt2img_unet_lora"

    def test_without_lora_keeps_the_plain_template(self) -> None:
        assert resolve_workflow_name(_spec()) == "txt2img_unet"

    def test_lora_combines_with_hires_fix(self) -> None:
        spec = _spec(loras=[LORA], upscale={"scale": 1.5})

        assert resolve_workflow_name(spec) == "txt2img_unet_lora_hires"

    def test_lora_combines_with_beta57(self) -> None:
        spec = _spec(loras=[LORA], scheduler="beta57")

        assert resolve_workflow_name(spec) == "txt2img_unet_beta57_lora"


class TestBindingStructure:
    """MODEL と CLIP が別ノードから来る点を、結線で確かめる。"""

    @pytest.fixture
    def binding(self) -> Any:
        return ALL_BINDINGS["txt2img_unet_lora"]

    def test_first_lora_reads_the_two_loaders(self, binding: Any) -> None:
        first = LORA_SLOT_ROLES[0]
        sources = {
            link.input_key: link.expected_role
            for link in binding.links
            if link.source_node == first
        }

        assert sources["model"] == UNET_LOADER_ROLE
        assert sources["clip"] == CLIP_LOADER_ROLE

    def test_loras_are_chained_in_order(self, binding: Any) -> None:
        for upstream, downstream in itertools.pairwise(LORA_SLOT_ROLES):
            sources = {
                link.input_key: link.expected_role
                for link in binding.links
                if link.source_node == downstream
            }

            assert sources["model"] == upstream
            assert sources["clip"] == upstream

    def test_ksampler_and_text_encodes_read_the_last_lora(self, binding: Any) -> None:
        last = LORA_SLOT_ROLES[-1]
        consumers = {
            (link.source_node, link.input_key)
            for link in binding.links
            if link.expected_role == last
        }

        assert ("ksampler", "model") in consumers
        assert ("positive_prompt", "clip") in consumers
        assert ("negative_prompt", "clip") in consumers

    def test_no_clip_skip_node_is_introduced(self, binding: Any) -> None:
        """DiT系はQwen3のため CLIPSetLastLayer を通さない (#126で確認済み)。"""
        assert all(node.class_type != "CLIPSetLastLayer" for node in binding.nodes.values())


class TestInjection:
    def test_lora_name_and_strengths_reach_the_chain(self) -> None:
        spec = _spec(loras=[LORA])

        workflow = prepare_workflow(spec).workflow

        slot = ALL_BINDINGS["txt2img_unet_lora"].nodes[LORA_SLOT_ROLES[0]]
        inputs = workflow[slot.node_id]["inputs"]
        assert inputs["lora_name"] == LORA["name"]
        assert inputs["strength_model"] == pytest.approx(0.8)

    def test_unused_slots_are_disabled(self) -> None:
        """3枠のうち使わない段は強度0で無効化する (checkpoint系と同じ扱い)。"""
        spec = _spec(loras=[LORA])

        workflow = prepare_workflow(spec).workflow

        binding = ALL_BINDINGS["txt2img_unet_lora"]
        for role in LORA_SLOT_ROLES[1:]:
            inputs = workflow[binding.nodes[role].node_id]["inputs"]
            assert inputs["strength_model"] == pytest.approx(0.0)
            assert inputs["strength_clip"] == pytest.approx(0.0)
