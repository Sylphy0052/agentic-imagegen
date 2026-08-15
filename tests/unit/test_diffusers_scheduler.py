"""Specのsampler / schedulerをdiffusersのSchedulerへ対応づける部分のテスト。

ComfyUIのsampler名は47件あるが、diffusersに同じものが揃っているわけではない。
黙って近いものへ倒すと絵が変わってしまうため、対応が無いものは拒否する。
ここではtorchを読み込まずに済むよう、クラスは名前 (文字列) で扱う。
"""

from __future__ import annotations

import pytest

from agentic_imagegen.adapters.diffusers.schedulers import (
    SUPPORTED_SAMPLERS,
    SUPPORTED_SCHEDULERS,
    resolve_scheduler,
)
from agentic_imagegen.errors import InvalidGenerationSpec


class TestResolveScheduler:
    def test_euler_normal(self) -> None:
        choice = resolve_scheduler("euler", "normal")

        assert choice.class_name == "EulerDiscreteScheduler"
        assert choice.options == {}

    def test_euler_ancestral(self) -> None:
        assert resolve_scheduler("euler_ancestral", "normal").class_name == (
            "EulerAncestralDiscreteScheduler"
        )

    def test_karras_becomes_sigma_option(self) -> None:
        """ComfyUIのschedulerはdiffusers側ではsigmaの取り方の指定になる。"""
        choice = resolve_scheduler("dpmpp_2m", "karras")

        assert choice.class_name == "DPMSolverMultistepScheduler"
        assert choice.options == {"use_karras_sigmas": True}

    @pytest.mark.parametrize(
        ("scheduler", "option"),
        [
            ("karras", "use_karras_sigmas"),
            ("exponential", "use_exponential_sigmas"),
            ("beta", "use_beta_sigmas"),
        ],
    )
    def test_sigma_options(self, scheduler: str, option: str) -> None:
        choice = resolve_scheduler("euler", scheduler)  # type: ignore[arg-type]

        assert choice.options == {option: True}

    def test_sde_variant_uses_algorithm_type(self) -> None:
        """dpmpp_2m_sde は同じクラスのalgorithm_type違いで表す。"""
        choice = resolve_scheduler("dpmpp_2m_sde", "normal")

        assert choice.class_name == "DPMSolverMultistepScheduler"
        assert choice.options == {"algorithm_type": "sde-dpmsolver++"}

    def test_third_order_sde(self) -> None:
        choice = resolve_scheduler("dpmpp_3m_sde", "karras")

        assert choice.class_name == "DPMSolverMultistepScheduler"
        assert choice.options == {
            "algorithm_type": "sde-dpmsolver++",
            "solver_order": 3,
            "use_karras_sigmas": True,
        }

    def test_unsupported_sampler_is_rejected(self) -> None:
        """ComfyUI固有のsampler (cfg_pp系) はdiffusersに対応が無い。"""
        with pytest.raises(InvalidGenerationSpec) as exc:
            resolve_scheduler("euler_cfg_pp", "normal")

        assert "euler_cfg_pp" in str(exc.value)
        # 使える名前を示して、書き直せるようにする
        assert "euler" in str(exc.value)

    def test_gpu_variant_is_rejected(self) -> None:
        """*_gpu はComfyUIのサンプリング実装の置き場所を指すもので、絵の定義ではない。"""
        with pytest.raises(InvalidGenerationSpec) as exc:
            resolve_scheduler("dpmpp_2m_sde_gpu", "normal")

        assert "dpmpp_2m_sde_gpu" in str(exc.value)

    def test_unsupported_scheduler_is_rejected(self) -> None:
        with pytest.raises(InvalidGenerationSpec) as exc:
            resolve_scheduler("euler", "sgm_uniform")

        assert "sgm_uniform" in str(exc.value)

    def test_sigma_option_on_incapable_sampler_is_rejected(self) -> None:
        """DDIMのようにsigmaの取り方を選べないSchedulerへkarrasを渡すと黙って無視される。"""
        with pytest.raises(InvalidGenerationSpec) as exc:
            resolve_scheduler("ddim", "karras")

        assert "ddim" in str(exc.value)
        assert "karras" in str(exc.value)

    def test_ddim_with_normal_is_allowed(self) -> None:
        assert resolve_scheduler("ddim", "normal").class_name == "DDIMScheduler"


class TestSupportedSets:
    def test_supported_samplers_are_valid_spec_values(self) -> None:
        """対応表のキーがSpecのsampler名からずれていないこと。"""
        from typing import get_args

        from agentic_imagegen.domain.models import SamplerName

        assert set(SUPPORTED_SAMPLERS) <= set(get_args(SamplerName))

    def test_supported_schedulers_are_valid_spec_values(self) -> None:
        from typing import get_args

        from agentic_imagegen.domain.models import SchedulerName

        assert set(SUPPORTED_SCHEDULERS) <= set(get_args(SchedulerName))

    def test_every_choice_matches_real_diffusers(self) -> None:
        """対応表のクラス名とオプションが、実際のdiffusersと食い違っていないこと。

        クラス名の書き間違いも、そのSchedulerが受け付けない引数も、
        表を読んだだけでは分からない。diffusersが入っている環境でだけ実物と突き合わせる。
        """
        diffusers = pytest.importorskip("diffusers")
        import inspect

        for sampler in SUPPORTED_SAMPLERS:
            for scheduler in SUPPORTED_SCHEDULERS:
                try:
                    choice = resolve_scheduler(sampler, scheduler)  # type: ignore[arg-type]
                except InvalidGenerationSpec:
                    continue
                cls = getattr(diffusers, choice.class_name, None)
                assert cls is not None, f"{choice.class_name} がdiffusersに無い"
                accepted = inspect.signature(cls.__init__).parameters
                for option in choice.options:
                    assert option in accepted, (
                        f"{choice.class_name} は {option} を受け付けない "
                        f"(sampler={sampler} scheduler={scheduler})"
                    )

    def test_every_supported_pair_resolves(self) -> None:
        """対応表に載っている組み合わせは、拒否されずに解決できる。"""
        for sampler in SUPPORTED_SAMPLERS:
            for scheduler in SUPPORTED_SCHEDULERS:
                try:
                    resolve_scheduler(sampler, scheduler)
                except InvalidGenerationSpec:
                    # sigmaの取り方を選べないSchedulerだけは normal 以外を拒否する
                    assert scheduler != "normal"
