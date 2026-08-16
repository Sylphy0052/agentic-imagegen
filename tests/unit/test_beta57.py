"""beta57 (BetaSamplingScheduler) 用テンプレートの選択・構造・注入の検査。

Anima系の配布元が推奨する `beta57` は beta分布の alpha=0.5 / beta=0.7 を指す
通称で、KSamplerの `scheduler` 欄からは選べない。専用テンプレートへ切り替わる
経路と、KSamplerと同じ意味になるようにパラメータが配られることを確かめる。
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from agentic_imagegen.adapters.comfyui.workflow import (
    ALL_BINDINGS,
    BETA57_GUIDER_ROLE,
    BETA57_HIRES_GUIDER_ROLE,
    BETA57_HIRES_NOISE_ROLE,
    BETA57_HIRES_SAMPLER_ROLE,
    BETA57_HIRES_SAMPLER_SELECT_ROLE,
    BETA57_HIRES_SIGMAS_ROLE,
    BETA57_HIRES_SPLIT_ROLE,
    BETA57_NOISE_ROLE,
    BETA57_SAMPLER_ROLE,
    BETA57_SAMPLER_SELECT_ROLE,
    BETA57_SIGMAS_ROLE,
)
from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.workflows.injector import prepare_workflow, resolve_workflow_name

SEPARATE_MODEL = {
    "unet": "hassakuAnima_v13_int8.safetensors",
    "clip": "qwen_3_06b_base.safetensors",
    "vae": "qwen_image_vae.safetensors",
}


def _spec(**generation: Any) -> GenerationSpec:
    payload: dict[str, Any] = {
        "version": "1",
        "task": "txt2img",
        "prompt": {"positive": "1girl", "negative": "worst quality"},
        "generation": {"seed": 42, "steps": 32, "cfg": 4.5, "sampler": "er_sde", **generation},
        "model": dict(SEPARATE_MODEL),
    }
    return GenerationSpec.model_validate(payload)


class TestTemplateSelection:
    def test_beta57_switches_to_the_dedicated_template(self) -> None:
        assert resolve_workflow_name(_spec(scheduler="beta57")) == "txt2img_unet_beta57"

    def test_other_schedulers_keep_the_ksampler_template(self) -> None:
        assert resolve_workflow_name(_spec(scheduler="simple")) == "txt2img_unet"

    def test_beta57_combines_with_hires_fix(self) -> None:
        spec = _spec(scheduler="beta57", upscale={"scale": 1.5})

        assert resolve_workflow_name(spec) == "txt2img_unet_beta57_hires"

    def test_beta57_is_rejected_for_checkpoint_models(self) -> None:
        with pytest.raises(ValidationError, match="beta57"):
            GenerationSpec.model_validate(
                {
                    "version": "1",
                    "task": "txt2img",
                    "prompt": {"positive": "1girl"},
                    "generation": {"scheduler": "beta57"},
                    "model": {"checkpoint": "v1-5-pruned-emaonly.safetensors"},
                }
            )


class TestBinding:
    def test_ksampler_role_is_replaced(self) -> None:
        binding = ALL_BINDINGS["txt2img_unet_beta57"]

        assert "ksampler" not in binding.nodes
        assert BETA57_SAMPLER_ROLE in binding.nodes

    def test_guider_receives_prompts_and_model(self) -> None:
        binding = ALL_BINDINGS["txt2img_unet_beta57"]
        links = {(link.source_node, link.input_key): link.expected_role for link in binding.links}

        assert links[(BETA57_GUIDER_ROLE, "positive")] == "positive_prompt"
        assert links[(BETA57_GUIDER_ROLE, "negative")] == "negative_prompt"
        assert links[(BETA57_GUIDER_ROLE, "model")] == "unet_loader"
        # スケジュールを引くためにmodel_samplingを見るので、こちらも同じ供給元を見る
        assert links[(BETA57_SIGMAS_ROLE, "model")] == "unet_loader"

    def test_hires_stage_is_also_replaced(self) -> None:
        binding = ALL_BINDINGS["txt2img_unet_beta57_hires"]
        links = {(link.source_node, link.input_key): link.expected_role for link in binding.links}

        assert "hires_ksampler" not in binding.nodes
        # 1段目 -> 拡大 -> 2段目 -> VAEDecode の順に繋がったまま置き換わる
        assert links[("upscale", "samples")] == BETA57_SAMPLER_ROLE
        assert links[(BETA57_HIRES_SAMPLER_ROLE, "latent_image")] == "upscale"
        assert links[("vae_decode", "samples")] == BETA57_HIRES_SAMPLER_ROLE
        assert links[(BETA57_HIRES_SPLIT_ROLE, "sigmas")] == BETA57_HIRES_SIGMAS_ROLE
        assert links[(BETA57_HIRES_SAMPLER_ROLE, "sigmas")] == BETA57_HIRES_SPLIT_ROLE


class TestInjection:
    def _inputs(self, spec: GenerationSpec, role: str) -> dict[str, Any]:
        prepared = prepare_workflow(spec)
        binding = ALL_BINDINGS[prepared.workflow_name]
        node: dict[str, Any] = prepared.workflow[binding.nodes[role].node_id]
        inputs: dict[str, Any] = node["inputs"]
        return inputs

    def test_seed_goes_to_random_noise(self) -> None:
        assert self._inputs(_spec(scheduler="beta57"), BETA57_NOISE_ROLE)["noise_seed"] == 42

    def test_sampler_goes_to_ksampler_select(self) -> None:
        inputs = self._inputs(_spec(scheduler="beta57"), BETA57_SAMPLER_SELECT_ROLE)

        assert inputs["sampler_name"] == "er_sde"

    def test_steps_and_beta_parameters(self) -> None:
        inputs = self._inputs(_spec(scheduler="beta57"), BETA57_SIGMAS_ROLE)

        assert inputs["steps"] == 32
        # beta57 の実体。テンプレート固定でSpecからは動かさない
        assert (inputs["alpha"], inputs["beta"]) == (0.5, 0.7)

    def test_cfg_goes_to_guider(self) -> None:
        assert self._inputs(_spec(scheduler="beta57"), BETA57_GUIDER_ROLE)["cfg"] == 4.5

    def test_hires_stage_shares_the_seed(self) -> None:
        spec = _spec(scheduler="beta57", upscale={"scale": 1.5, "denoise": 0.3})

        assert self._inputs(spec, BETA57_HIRES_NOISE_ROLE)["noise_seed"] == 42
        assert self._inputs(spec, BETA57_HIRES_SAMPLER_SELECT_ROLE)["sampler_name"] == "er_sde"
        assert self._inputs(spec, BETA57_HIRES_GUIDER_ROLE)["cfg"] == 4.5

    def test_hires_steps_are_divided_by_denoise(self) -> None:
        """KSamplerの `int(steps / denoise)` と同じ式でスケジュールを引く。"""
        spec = _spec(scheduler="beta57", upscale={"scale": 1.5, "denoise": 0.3, "steps": 16})

        assert self._inputs(spec, BETA57_HIRES_SIGMAS_ROLE)["steps"] == 53
        assert self._inputs(spec, BETA57_HIRES_SPLIT_ROLE)["denoise"] == 0.3

    def test_hires_denoise_of_one_keeps_the_step_count(self) -> None:
        spec = _spec(scheduler="beta57", upscale={"scale": 1.5, "denoise": 1.0, "steps": 16})

        assert self._inputs(spec, BETA57_HIRES_SIGMAS_ROLE)["steps"] == 16

    def test_hires_denoise_of_zero_does_not_divide(self) -> None:
        """0除算を避ける。denoise=0 はそもそも描き足しにならない指定。"""
        spec = _spec(scheduler="beta57", upscale={"scale": 1.5, "denoise": 0.0, "steps": 16})

        assert self._inputs(spec, BETA57_HIRES_SIGMAS_ROLE)["steps"] == 16
