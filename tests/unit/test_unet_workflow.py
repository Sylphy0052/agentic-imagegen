"""UNet / CLIP / VAE を分けて読むWorkflow (DiT系) の注入と構造検証。"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from agentic_imagegen.adapters.comfyui.workflow import (
    CLIP_LOADER_ROLE,
    HIRES_KSAMPLER_ROLE,
    TXT2IMG_UNET_BINDING,
    UNET_LOADER_ROLE,
    UPSCALE_ROLE,
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
    resolve_workflow_name,
)

SEPARATE_MODEL = {
    "unet": "hassakuAnima_v13_int8.safetensors",
    "clip": "qwen_3_06b_base.safetensors",
    "vae": "qwen_image_vae.safetensors",
}


def _spec(**extra: Any) -> GenerationSpec:
    payload: dict[str, Any] = {
        "version": "1",
        "task": "txt2img",
        "prompt": {"positive": "1girl", "negative": "low quality"},
        "generation": {
            "width": 832,
            "height": 1216,
            "steps": 28,
            "cfg": 4.0,
            "seed": 7,
            "sampler": "er_sde",
            "scheduler": "simple",
        },
        "model": dict(SEPARATE_MODEL),
    }
    payload.update(extra)
    return GenerationSpec.model_validate(payload)


@pytest.fixture
def template() -> dict[str, Any]:
    return load_workflow_template("txt2img_unet")


def _inputs(workflow: dict[str, Any], role: str) -> dict[str, Any]:
    node_id = TXT2IMG_UNET_BINDING.nodes[role].node_id
    inputs: dict[str, Any] = workflow[node_id]["inputs"]
    return inputs


class TestTemplateSelection:
    def test_separate_loaders_switch_template(self) -> None:
        assert resolve_workflow_name(_spec()) == "txt2img_unet"

    def test_checkpoint_keeps_default_template(self) -> None:
        spec = GenerationSpec.model_validate(
            {
                "version": "1",
                "prompt": {"positive": "1girl"},
                "model": {"checkpoint": "meinamix_v12Final.safetensors"},
            }
        )

        assert resolve_workflow_name(spec) == "txt2img"

    def test_is_allowed(self) -> None:
        assert "txt2img_unet" in ALLOWED_WORKFLOWS

    def test_template_exists(self) -> None:
        assert load_workflow_template("txt2img_unet")


class TestInjection:
    def test_injects_loader_filenames(self, template: dict[str, Any]) -> None:
        workflow = build_workflow(template, _spec(), seed=7, binding=TXT2IMG_UNET_BINDING)

        assert _inputs(workflow, UNET_LOADER_ROLE)["unet_name"] == SEPARATE_MODEL["unet"]
        assert _inputs(workflow, CLIP_LOADER_ROLE)["clip_name"] == SEPARATE_MODEL["clip"]
        assert _inputs(workflow, VAE_LOADER_ROLE)["vae_name"] == SEPARATE_MODEL["vae"]

    def test_keeps_clip_loader_type(self, template: dict[str, Any]) -> None:
        """text encoderの種別はテンプレート側の固定値をそのまま使う。"""
        workflow = build_workflow(template, _spec(), seed=7, binding=TXT2IMG_UNET_BINDING)

        assert _inputs(workflow, CLIP_LOADER_ROLE)["type"] == "stable_diffusion"

    def test_injects_generation_params(self, template: dict[str, Any]) -> None:
        workflow = build_workflow(template, _spec(), seed=7, binding=TXT2IMG_UNET_BINDING)

        latent = _inputs(workflow, "latent")
        assert (latent["width"], latent["height"]) == (832, 1216)

        ksampler = _inputs(workflow, "ksampler")
        assert ksampler["seed"] == 7
        assert ksampler["steps"] == 28
        assert ksampler["cfg"] == 4.0
        assert ksampler["sampler_name"] == "er_sde"
        assert ksampler["scheduler"] == "simple"

    def test_rejects_checkpoint_spec(self, template: dict[str, Any]) -> None:
        """checkpoint指定のSpecをUNet用テンプレートへ流し込ませない。"""
        spec = GenerationSpec.model_validate(
            {
                "version": "1",
                "prompt": {"positive": "1girl"},
                "model": {"checkpoint": "meinamix_v12Final.safetensors"},
            }
        )

        with pytest.raises(WorkflowValidationError, match="unet"):
            build_workflow(template, spec, seed=7, binding=TXT2IMG_UNET_BINDING)

    def test_prepare_workflow_end_to_end(self) -> None:
        prepared = prepare_workflow(_spec())

        assert prepared.workflow_name == "txt2img_unet"
        assert prepared.seed == 7
        assert prepared.template_hash.startswith("sha256:")


class TestStructureValidation:
    def test_accepts_template(self, template: dict[str, Any]) -> None:
        validate_structure(template, TXT2IMG_UNET_BINDING)

    def test_rejects_ksampler_bypassing_unet_loader(self, template: dict[str, Any]) -> None:
        broken = copy.deepcopy(template)
        broken["3"]["inputs"]["model"] = ["61", 0]

        with pytest.raises(WorkflowValidationError):
            validate_structure(broken, TXT2IMG_UNET_BINDING)

    def test_rejects_prompt_bypassing_clip_loader(self, template: dict[str, Any]) -> None:
        broken = copy.deepcopy(template)
        broken["6"]["inputs"]["clip"] = ["60", 0]

        with pytest.raises(WorkflowValidationError):
            validate_structure(broken, TXT2IMG_UNET_BINDING)

    def test_rejects_decode_bypassing_vae_loader(self, template: dict[str, Any]) -> None:
        broken = copy.deepcopy(template)
        broken["8"]["inputs"]["vae"] = ["60", 0]

        with pytest.raises(WorkflowValidationError):
            validate_structure(broken, TXT2IMG_UNET_BINDING)

    def test_rejects_wrong_class_type(self, template: dict[str, Any]) -> None:
        broken = copy.deepcopy(template)
        broken["60"]["class_type"] = "CheckpointLoaderSimple"

        with pytest.raises(WorkflowValidationError):
            validate_structure(broken, TXT2IMG_UNET_BINDING)


class TestDerivedTemplates:
    """img2img / hires fix との組み合わせ。CheckpointLoaderが残っていないことまで見る。"""

    @pytest.mark.parametrize(
        "name",
        ["txt2img_unet", "txt2img_unet_hires", "img2img_unet", "img2img_unet_hires"],
    )
    def test_templates_pass_structure_validation(self, name: str) -> None:
        template = load_workflow_template(name)

        validate_structure(template, ALLOWED_WORKFLOWS[name])

    @pytest.mark.parametrize(
        "name",
        ["txt2img_unet", "txt2img_unet_hires", "img2img_unet", "img2img_unet_hires"],
    )
    def test_no_checkpoint_loader_remains(self, name: str) -> None:
        """1ファイルから MODEL / CLIP / VAE を取り出すノードが残っていてはいけない。"""
        template = load_workflow_template(name)

        class_types = {node["class_type"] for node in template.values()}
        assert "CheckpointLoaderSimple" not in class_types

    def test_img2img_vae_encode_uses_separate_vae(self) -> None:
        """入力画像をVAEEncodeする側も3ローダーのVAEを見る。"""
        binding = ALLOWED_WORKFLOWS["img2img_unet"]
        template = load_workflow_template("img2img_unet")
        vae_encode = binding.nodes["vae_encode"].node_id
        vae_loader = binding.nodes[VAE_LOADER_ROLE].node_id

        assert template[vae_encode]["inputs"]["vae"] == [vae_loader, 0]

    def test_second_pass_uses_unet_loader(self) -> None:
        """2段目のKSamplerもUNETLoaderから受ける。"""
        binding = ALLOWED_WORKFLOWS["txt2img_unet_hires"]
        template = load_workflow_template("txt2img_unet_hires")
        second = binding.nodes[HIRES_KSAMPLER_ROLE].node_id
        unet = binding.nodes[UNET_LOADER_ROLE].node_id

        assert template[second]["inputs"]["model"] == [unet, 0]

    def test_detects_second_pass_bypassing_unet_loader(self) -> None:
        binding = ALLOWED_WORKFLOWS["txt2img_unet_hires"]
        broken = copy.deepcopy(load_workflow_template("txt2img_unet_hires"))
        second = binding.nodes[HIRES_KSAMPLER_ROLE].node_id
        broken[second]["inputs"]["model"] = [binding.nodes[CLIP_LOADER_ROLE].node_id, 0]

        with pytest.raises(WorkflowValidationError):
            validate_structure(broken, binding)

    def test_prepare_workflow_for_img2img_and_hires(self) -> None:
        prepared = prepare_workflow(
            _spec(
                task="img2img",
                source={"image": "inputs/base.png"},
                generation={"seed": 7, "upscale": {"scale": 1.5, "denoise": 0.4, "steps": 6}},
            ),
            source_image_name="base.png",
        )
        binding = ALLOWED_WORKFLOWS["img2img_unet_hires"]
        upscale = prepared.workflow[binding.nodes[UPSCALE_ROLE].node_id]["inputs"]

        assert prepared.workflow_name == "img2img_unet_hires"
        assert upscale["scale_by"] == 1.5
