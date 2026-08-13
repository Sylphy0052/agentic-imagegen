"""外部VAE (checkpoint + VAELoader) を使うWorkflowの注入と構造検証。

DiT系 (`*_unet`) と違い、checkpointはそのまま残り、VAEの参照元だけを
VAELoaderへ差し替える (Issue #57)。
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from agentic_imagegen.adapters.comfyui.workflow import (
    IMG2IMG_VAE_BINDING,
    TXT2IMG_VAE_BINDING,
    TXT2IMG_VAE_HIRES_MODEL_BINDING,
    TXT2IMG_VAE_LORA_BINDING,
    UPSCALE_MODEL_DECODE_ROLE,
    UPSCALE_MODEL_ENCODE_ROLE,
    VAE_LOADER_ROLE,
    build_workflow,
    validate_structure,
)
from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.errors import WorkflowValidationError
from agentic_imagegen.workflows.injector import (
    ALLOWED_WORKFLOWS,
    load_workflow_template,
    prepare_workflow,
)

CHECKPOINT = "meinamix_v12Final.safetensors"
EXTERNAL_VAE = "vae-ft-mse-840000-ema-pruned.safetensors"


def _spec(**extra: Any) -> GenerationSpec:
    payload: dict[str, Any] = {
        "version": "1",
        "task": "txt2img",
        "prompt": {"positive": "1girl", "negative": "low quality"},
        "model": {"checkpoint": CHECKPOINT, "vae": EXTERNAL_VAE},
    }
    payload.update(extra)
    return GenerationSpec.model_validate(payload)


@pytest.fixture
def template() -> dict[str, Any]:
    return load_workflow_template("txt2img_vae")


def _inputs(workflow: dict[str, Any], binding: Any, role: str) -> dict[str, Any]:
    node_id = binding.nodes[role].node_id
    inputs: dict[str, Any] = workflow[node_id]["inputs"]
    return inputs


class TestInjection:
    def test_injects_checkpoint_and_vae(self, template: dict[str, Any]) -> None:
        workflow = build_workflow(template, _spec(), seed=1, binding=TXT2IMG_VAE_BINDING)

        assert _inputs(workflow, TXT2IMG_VAE_BINDING, "checkpoint")["ckpt_name"] == CHECKPOINT
        assert _inputs(workflow, TXT2IMG_VAE_BINDING, VAE_LOADER_ROLE)["vae_name"] == EXTERNAL_VAE

    def test_vae_decode_points_to_vae_loader(self, template: dict[str, Any]) -> None:
        workflow = build_workflow(template, _spec(), seed=1, binding=TXT2IMG_VAE_BINDING)

        vae_loader_id = TXT2IMG_VAE_BINDING.nodes[VAE_LOADER_ROLE].node_id
        assert workflow["8"]["inputs"]["vae"] == [vae_loader_id, 0]

    def test_checkpoint_loader_remains(self, template: dict[str, Any]) -> None:
        """外部VAE版はDiT系と違い、CheckpointLoaderSimple自体は残る。"""
        assert template["4"]["class_type"] == "CheckpointLoaderSimple"

    def test_rejects_spec_without_vae(self, template: dict[str, Any]) -> None:
        """checkpointのみのSpecを外部VAE用テンプレートへ流し込ませない。"""
        spec = GenerationSpec.model_validate(
            {"prompt": {"positive": "1girl"}, "model": {"checkpoint": CHECKPOINT}}
        )

        with pytest.raises(WorkflowValidationError, match="vae"):
            build_workflow(template, spec, seed=1, binding=TXT2IMG_VAE_BINDING)

    def test_prepare_workflow_end_to_end(self) -> None:
        prepared = prepare_workflow(_spec(generation={"seed": 99}))

        assert prepared.workflow_name == "txt2img_vae"
        assert prepared.seed == 99
        vae_loader_id = TXT2IMG_VAE_BINDING.nodes[VAE_LOADER_ROLE].node_id
        assert prepared.workflow[vae_loader_id]["inputs"]["vae_name"] == EXTERNAL_VAE


class TestImg2ImgVaeEncode:
    def test_vae_encode_uses_external_vae(self) -> None:
        """入力画像をVAEEncodeする側も外部VAEを見る。"""
        template = load_workflow_template("img2img_vae")
        spec = _spec(task="img2img", source={"image": "inputs/base.png"})

        workflow = build_workflow(
            template, spec, seed=1, binding=IMG2IMG_VAE_BINDING, source_image_name="base.png"
        )

        vae_loader_id = IMG2IMG_VAE_BINDING.nodes[VAE_LOADER_ROLE].node_id
        vae_encode_id = IMG2IMG_VAE_BINDING.nodes["vae_encode"].node_id
        assert workflow[vae_encode_id]["inputs"]["vae"] == [vae_loader_id, 0]


class TestHiresModelVae:
    """アップスケールモデル版hires fixで増えるVAEDecode/VAEEncodeも差し替え対象になる。"""

    def test_extra_vae_nodes_use_external_vae(self) -> None:
        template = load_workflow_template("txt2img_vae_hires_model")
        spec = _spec(
            generation={
                "upscale": {
                    "model": "RealESRGAN_x4plus_anime_6B.pth",
                    "model_scale": 4.0,
                    "scale": 2.0,
                }
            }
        )

        workflow = build_workflow(template, spec, seed=1, binding=TXT2IMG_VAE_HIRES_MODEL_BINDING)

        vae_loader_id = TXT2IMG_VAE_HIRES_MODEL_BINDING.nodes[VAE_LOADER_ROLE].node_id
        decode_id = TXT2IMG_VAE_HIRES_MODEL_BINDING.nodes[UPSCALE_MODEL_DECODE_ROLE].node_id
        encode_id = TXT2IMG_VAE_HIRES_MODEL_BINDING.nodes[UPSCALE_MODEL_ENCODE_ROLE].node_id

        assert workflow[decode_id]["inputs"]["vae"] == [vae_loader_id, 0]
        assert workflow[encode_id]["inputs"]["vae"] == [vae_loader_id, 0]

    def test_detects_decode_bypassing_vae_loader(self) -> None:
        broken = copy.deepcopy(load_workflow_template("txt2img_vae_hires_model"))
        decode_id = TXT2IMG_VAE_HIRES_MODEL_BINDING.nodes[UPSCALE_MODEL_DECODE_ROLE].node_id
        checkpoint_id = TXT2IMG_VAE_HIRES_MODEL_BINDING.nodes["checkpoint"].node_id
        broken[decode_id]["inputs"]["vae"] = [checkpoint_id, 2]

        with pytest.raises(WorkflowValidationError):
            validate_structure(broken, TXT2IMG_VAE_HIRES_MODEL_BINDING)


class TestStructureValidation:
    def test_accepts_template(self, template: dict[str, Any]) -> None:
        validate_structure(template, TXT2IMG_VAE_BINDING)

    def test_rejects_decode_bypassing_vae_loader(self, template: dict[str, Any]) -> None:
        broken = copy.deepcopy(template)
        broken["8"]["inputs"]["vae"] = ["4", 2]

        with pytest.raises(WorkflowValidationError):
            validate_structure(broken, TXT2IMG_VAE_BINDING)

    def test_lora_binding_vae_still_uses_checkpoint_model(self) -> None:
        """外部VAEはMODEL/CLIPには影響しない。LoRA適用後のCLIPはそのまま。"""
        template = load_workflow_template("txt2img_vae_lora")

        validate_structure(template, TXT2IMG_VAE_LORA_BINDING)


class TestDerivedTemplates:
    """checkpoint系32件それぞれの `_vae` 版が構造検証を通ることを確認する。"""

    @pytest.mark.parametrize("name", sorted(k for k in ALLOWED_WORKFLOWS if "_vae" in k))
    def test_templates_pass_structure_validation(self, name: str) -> None:
        template = load_workflow_template(name)

        validate_structure(template, ALLOWED_WORKFLOWS[name])

    @pytest.mark.parametrize("name", sorted(k for k in ALLOWED_WORKFLOWS if "_vae" in k))
    def test_checkpoint_loader_is_kept(self, name: str) -> None:
        """DiT系と違い、外部VAE版はCheckpointLoaderSimpleを残したままVAEだけ差し替える。"""
        template = load_workflow_template(name)

        class_types = {node["class_type"] for node in template.values()}
        assert "CheckpointLoaderSimple" in class_types

    @pytest.mark.parametrize("name", sorted(k for k in ALLOWED_WORKFLOWS if "_vae" in k))
    def test_no_reference_to_checkpoint_vae_output_remains(self, name: str) -> None:
        """checkpointのVAE出力 ([node, 2]) を参照するノードが残っていないこと。"""
        template = load_workflow_template(name)
        checkpoint_id = next(
            node_id
            for node_id, node in template.items()
            if node["class_type"] == "CheckpointLoaderSimple"
        )

        for node in template.values():
            for value in node["inputs"].values():
                if isinstance(value, list) and len(value) == 2:
                    assert value != [checkpoint_id, 2]

    def test_vae_count_matches_checkpoint_template_count(self) -> None:
        """checkpoint系32件それぞれに `_vae` 版が1つずつ対応する。"""
        checkpoint_names = {
            name for name in ALLOWED_WORKFLOWS if "unet" not in name and "vae" not in name
        }
        vae_names = {name for name in ALLOWED_WORKFLOWS if "_vae" in name}

        assert len(checkpoint_names) == 32
        assert len(vae_names) == 32
