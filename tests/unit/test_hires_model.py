"""アップスケールモデルを使うhires fix の注入とテンプレート選択。

latent拡大 (tests/unit/test_hires.py) とは別テンプレートになるため、
選択規則と結線をここで固定する。拡大の経路が1箇所でも元のまま残っていると、
絵は出るのに拡大が効かない状態になり、出力を見ても気づきにくい。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agentic_imagegen.adapters.comfyui.workflow import (
    HIRES_KSAMPLER_ROLE,
    TXT2IMG_HIRES_MODEL_BINDING,
    UPSCALE_MODEL_APPLY_ROLE,
    UPSCALE_MODEL_DECODE_ROLE,
    UPSCALE_MODEL_ENCODE_ROLE,
    UPSCALE_MODEL_LOADER_ROLE,
    UPSCALE_MODEL_RESIZE_ROLE,
    build_workflow,
)
from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.errors import WorkflowValidationError
from agentic_imagegen.workflows.injector import (
    ALLOWED_WORKFLOWS,
    load_workflow_template,
    resolve_workflow_name,
)

LORA = {"name": "add_detail.safetensors"}
UPSCALE_MODEL = "RealESRGAN_x4plus_anime_6B.pth"
UPSCALE = {"model": UPSCALE_MODEL, "scale": 2.0, "denoise": 0.35, "steps": 6}
LATENT_UPSCALE = {"scale": 2.0}


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
    if loras is not None:
        payload["model"]["loras"] = loras
    return GenerationSpec.model_validate(payload)


@pytest.fixture
def template() -> dict[str, Any]:
    return load_workflow_template("txt2img_hires_model")


def _inputs(workflow: dict[str, Any], role: str) -> dict[str, Any]:
    node_id = TXT2IMG_HIRES_MODEL_BINDING.nodes[role].node_id
    inputs: dict[str, Any] = workflow[node_id]["inputs"]
    return inputs


class TestTemplateSelection:
    @pytest.mark.parametrize(
        ("task", "loras", "expected"),
        [
            ("txt2img", None, "txt2img_hires_model"),
            ("txt2img", [LORA], "txt2img_lora_hires_model"),
            ("img2img", None, "img2img_hires_model"),
            ("img2img", [LORA], "img2img_lora_hires_model"),
        ],
    )
    def test_model_upscale_selects_its_own_template(
        self, task: str, loras: list[dict[str, Any]] | None, expected: str
    ) -> None:
        spec = _spec(task=task, loras=loras, upscale=UPSCALE)

        assert resolve_workflow_name(spec) == expected
        assert expected in ALLOWED_WORKFLOWS

    def test_latent_upscale_keeps_the_existing_template(self) -> None:
        """model未指定のSpecの意味を変えない。"""
        spec = _spec(upscale=LATENT_UPSCALE)

        assert resolve_workflow_name(spec) == "txt2img_hires"

    def test_combines_with_controlnet(self) -> None:
        payload = _spec(upscale=UPSCALE).model_dump(mode="json", exclude_none=True)
        payload["control"] = {
            "image": "inputs/pose.png",
            "model": "control_v11p_sd15_canny_fp16.safetensors",
        }
        spec = GenerationSpec.model_validate(payload)

        assert resolve_workflow_name(spec) == "txt2img_hires_model_controlnet"


class TestInjection:
    def test_injects_model_name(self, template: dict[str, Any]) -> None:
        workflow = build_workflow(
            template, _spec(upscale=UPSCALE), seed=1, binding=TXT2IMG_HIRES_MODEL_BINDING
        )

        assert _inputs(workflow, UPSCALE_MODEL_LOADER_ROLE)["model_name"] == UPSCALE_MODEL

    def test_resize_brings_model_output_to_requested_scale(self, template: dict[str, Any]) -> None:
        """4xのモデルで2倍が欲しいので、拡大後に0.5倍へ戻す。"""
        workflow = build_workflow(
            template, _spec(upscale=UPSCALE), seed=1, binding=TXT2IMG_HIRES_MODEL_BINDING
        )

        assert _inputs(workflow, UPSCALE_MODEL_RESIZE_ROLE)["scale_by"] == pytest.approx(0.5)

    def test_resize_uses_declared_model_scale(self, template: dict[str, Any]) -> None:
        workflow = build_workflow(
            template,
            _spec(upscale={"model": UPSCALE_MODEL, "scale": 2.0, "model_scale": 2.0}),
            seed=1,
            binding=TXT2IMG_HIRES_MODEL_BINDING,
        )

        assert _inputs(workflow, UPSCALE_MODEL_RESIZE_ROLE)["scale_by"] == pytest.approx(1.0)

    def test_injects_method_into_the_pixel_resize(self, template: dict[str, Any]) -> None:
        workflow = build_workflow(
            template,
            _spec(upscale={**UPSCALE, "method": "lanczos"}),
            seed=1,
            binding=TXT2IMG_HIRES_MODEL_BINDING,
        )

        assert _inputs(workflow, UPSCALE_MODEL_RESIZE_ROLE)["upscale_method"] == "lanczos"

    def test_second_pass_uses_own_steps_and_denoise(self, template: dict[str, Any]) -> None:
        workflow = build_workflow(
            template, _spec(upscale=UPSCALE), seed=1, binding=TXT2IMG_HIRES_MODEL_BINDING
        )

        second = _inputs(workflow, HIRES_KSAMPLER_ROLE)
        assert second["steps"] == 6
        assert second["denoise"] == 0.35

    def test_second_pass_shares_seed(self, template: dict[str, Any]) -> None:
        workflow = build_workflow(
            template, _spec(upscale=UPSCALE), seed=4242, binding=TXT2IMG_HIRES_MODEL_BINDING
        )

        assert _inputs(workflow, "ksampler")["seed"] == 4242
        assert _inputs(workflow, HIRES_KSAMPLER_ROLE)["seed"] == 4242

    def test_rejects_spec_without_upscale(self, template: dict[str, Any]) -> None:
        with pytest.raises(WorkflowValidationError, match="upscale"):
            build_workflow(template, _spec(), seed=1, binding=TXT2IMG_HIRES_MODEL_BINDING)

    def test_rejects_latent_spec_on_a_model_template(self, template: dict[str, Any]) -> None:
        """テンプレートとSpecが食い違ったまま投入しない。"""
        with pytest.raises(WorkflowValidationError, match="model"):
            build_workflow(
                template,
                _spec(upscale=LATENT_UPSCALE),
                seed=1,
                binding=TXT2IMG_HIRES_MODEL_BINDING,
            )

    def test_rejects_model_spec_on_a_latent_template(self) -> None:
        binding = ALLOWED_WORKFLOWS["txt2img_hires"]

        with pytest.raises(WorkflowValidationError, match="model"):
            build_workflow(
                load_workflow_template("txt2img_hires"),
                _spec(upscale=UPSCALE),
                seed=1,
                binding=binding,
            )


class TestStructureValidation:
    def _broken(self, role: str, key: str, value: list[Any]) -> dict[str, Any]:
        broken: dict[str, Any] = json.loads(
            json.dumps(load_workflow_template("txt2img_hires_model"))
        )
        node_id = TXT2IMG_HIRES_MODEL_BINDING.nodes[role].node_id
        broken[node_id]["inputs"][key] = value
        return broken

    def test_detects_decode_not_fed_by_first_pass(self) -> None:
        ksampler = TXT2IMG_HIRES_MODEL_BINDING.nodes[HIRES_KSAMPLER_ROLE].node_id
        broken = self._broken(UPSCALE_MODEL_DECODE_ROLE, "samples", [ksampler, 0])

        with pytest.raises(WorkflowValidationError):
            build_workflow(
                broken, _spec(upscale=UPSCALE), seed=1, binding=TXT2IMG_HIRES_MODEL_BINDING
            )

    def test_detects_second_pass_bypassing_the_upscale_chain(self) -> None:
        """2段目が拡大前のlatentを受けていると、拡大が効かないまま絵が出る。"""
        broken = self._broken(HIRES_KSAMPLER_ROLE, "latent_image", ["5", 0])

        with pytest.raises(WorkflowValidationError):
            build_workflow(
                broken, _spec(upscale=UPSCALE), seed=1, binding=TXT2IMG_HIRES_MODEL_BINDING
            )

    def test_detects_resize_bypassing_the_upscale_model(self) -> None:
        decode = TXT2IMG_HIRES_MODEL_BINDING.nodes[UPSCALE_MODEL_DECODE_ROLE].node_id
        broken = self._broken(UPSCALE_MODEL_RESIZE_ROLE, "image", [decode, 0])

        with pytest.raises(WorkflowValidationError):
            build_workflow(
                broken, _spec(upscale=UPSCALE), seed=1, binding=TXT2IMG_HIRES_MODEL_BINDING
            )

    def test_detects_encode_bypassing_the_resize(self) -> None:
        apply_node = TXT2IMG_HIRES_MODEL_BINDING.nodes[UPSCALE_MODEL_APPLY_ROLE].node_id
        broken = self._broken(UPSCALE_MODEL_ENCODE_ROLE, "pixels", [apply_node, 0])

        with pytest.raises(WorkflowValidationError):
            build_workflow(
                broken, _spec(upscale=UPSCALE), seed=1, binding=TXT2IMG_HIRES_MODEL_BINDING
            )


class TestSeparateLoaderTemplates:
    """DiT系ではVAEを別ローダーから受ける。増やしたVAEノードも追随していること。"""

    @pytest.mark.parametrize("name", ["txt2img_unet_hires_model", "img2img_unet_hires_model"])
    def test_added_vae_nodes_read_the_vae_loader(self, name: str) -> None:
        binding = ALLOWED_WORKFLOWS[name]
        template = load_workflow_template(name)
        loader = binding.nodes["vae_loader"].node_id

        for role in (UPSCALE_MODEL_DECODE_ROLE, UPSCALE_MODEL_ENCODE_ROLE):
            node_id = binding.nodes[role].node_id
            assert template[node_id]["inputs"]["vae"] == [loader, 0]

    def test_detects_added_vae_node_left_on_the_checkpoint(self) -> None:
        """CheckpointLoaderが無い構成でVAEの供給元が変わっていれば落ちる。"""
        name = "txt2img_unet_hires_model"
        binding = ALLOWED_WORKFLOWS[name]
        broken: dict[str, Any] = json.loads(json.dumps(load_workflow_template(name)))
        node_id = binding.nodes[UPSCALE_MODEL_ENCODE_ROLE].node_id
        broken[node_id]["inputs"]["vae"] = [binding.nodes["clip_loader"].node_id, 0]

        payload: dict[str, Any] = {
            "version": "1",
            "task": "txt2img",
            "prompt": {"positive": "1girl"},
            "generation": {"steps": 20, "cfg": 4.0, "seed": 1, "upscale": UPSCALE},
            "model": {
                "unet": "hassakuAnima_v13_int8.safetensors",
                "clip": "qwen_3_06b_base.safetensors",
                "vae": "qwen_image_vae.safetensors",
            },
        }
        spec = GenerationSpec.model_validate(payload)

        with pytest.raises(WorkflowValidationError):
            build_workflow(broken, spec, seed=1, binding=binding)


class TestCheckpointTemplates:
    """checkpoint系ではVAEがcheckpointへ同梱される。増やしたVAEノードもそれを見ていること。"""

    @pytest.mark.parametrize(
        "name",
        ["txt2img_hires_model", "txt2img_lora_hires_model", "img2img_hires_model"],
    )
    def test_added_vae_nodes_read_the_checkpoint(self, name: str) -> None:
        binding = ALLOWED_WORKFLOWS[name]
        template = load_workflow_template(name)
        checkpoint = binding.nodes["checkpoint"].node_id

        for role in (UPSCALE_MODEL_DECODE_ROLE, UPSCALE_MODEL_ENCODE_ROLE):
            node_id = binding.nodes[role].node_id
            assert template[node_id]["inputs"]["vae"][0] == checkpoint

    def test_detects_added_vae_node_pointing_elsewhere(self) -> None:
        """checkpoint系でもVAEの供給元が変わっていれば注入前に落ちる。"""
        broken: dict[str, Any] = json.loads(
            json.dumps(load_workflow_template("txt2img_hires_model"))
        )
        node_id = TXT2IMG_HIRES_MODEL_BINDING.nodes[UPSCALE_MODEL_ENCODE_ROLE].node_id
        broken[node_id]["inputs"]["vae"] = [
            TXT2IMG_HIRES_MODEL_BINDING.nodes[UPSCALE_MODEL_LOADER_ROLE].node_id,
            0,
        ]

        with pytest.raises(WorkflowValidationError):
            build_workflow(
                broken, _spec(upscale=UPSCALE), seed=1, binding=TXT2IMG_HIRES_MODEL_BINDING
            )


class TestSpecRoundTrip:
    """metadata.json へ書いたSpecを読み直せること。

    model_scale へ既定値を埋めるとlatent拡大のSpecをdumpしたときにも値が乗り、
    読み直しで「model と一緒に指定してください」に弾かれる。
    """

    @pytest.mark.parametrize("upscale", [LATENT_UPSCALE, UPSCALE])
    def test_dumped_spec_is_loadable(self, upscale: dict[str, Any]) -> None:
        spec = _spec(upscale=upscale)

        dumped = spec.model_dump(mode="json")

        assert GenerationSpec.model_validate(dumped) == spec

    def test_latent_upscale_does_not_carry_model_scale(self) -> None:
        dumped = _spec(upscale=LATENT_UPSCALE).model_dump(mode="json")

        assert dumped["generation"]["upscale"]["model_scale"] is None
