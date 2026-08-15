"""Specのsampler / schedulerをdiffusersのSchedulerへ対応づける。

ComfyUIのsampler名 (`SamplerName` の47件) は、ComfyUI独自のサンプリング実装まで
含んだ集合であり、diffusersに同じものが揃っているわけではない。対応が無いものを
黙って近いSchedulerへ倒すと、同じSpecでもバックエンドによって絵が変わる。
対応表に載っていない指定はここで拒否する。

torchを読み込まずに済むよう、Schedulerは名前 (文字列) で扱い、実際の解決は
呼び出し側 (backend) がdiffusersをimportしたうえで行う。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from agentic_imagegen.domain.models import SamplerName, SchedulerName
from agentic_imagegen.errors import InvalidGenerationSpec


@dataclass(frozen=True, slots=True)
class SchedulerChoice:
    """どのSchedulerをどう構成するか。

    class_name は diffusers のトップレベルにあるクラス名。options は
    `Scheduler.from_config(config, **options)` へ渡す追加設定。
    """

    class_name: str
    options: dict[str, Any]


#: Specのsampler -> (Schedulerクラス名, そのsampler固有の設定)。
#: ここに無いsamplerは拒否する。*_gpu / *_cfg_pp はComfyUIの実装都合を指す名前で、
#: 絵の定義ではないため対応させない。
_SAMPLERS: Final[dict[str, tuple[str, dict[str, Any]]]] = {
    "euler": ("EulerDiscreteScheduler", {}),
    "euler_ancestral": ("EulerAncestralDiscreteScheduler", {}),
    "heun": ("HeunDiscreteScheduler", {}),
    "dpm_2": ("KDPM2DiscreteScheduler", {}),
    "dpm_2_ancestral": ("KDPM2AncestralDiscreteScheduler", {}),
    "lms": ("LMSDiscreteScheduler", {}),
    "dpmpp_2s_ancestral": ("DPMSolverSinglestepScheduler", {}),
    "dpmpp_sde": ("DPMSolverSDEScheduler", {}),
    "dpmpp_2m": ("DPMSolverMultistepScheduler", {}),
    "dpmpp_2m_sde": ("DPMSolverMultistepScheduler", {"algorithm_type": "sde-dpmsolver++"}),
    "dpmpp_3m_sde": (
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "sde-dpmsolver++", "solver_order": 3},
    ),
    "deis": ("DEISMultistepScheduler", {}),
    "uni_pc": ("UniPCMultistepScheduler", {}),
    "uni_pc_bh2": ("UniPCMultistepScheduler", {"solver_type": "bh2"}),
    "ddim": ("DDIMScheduler", {}),
    "ddpm": ("DDPMScheduler", {}),
    "lcm": ("LCMScheduler", {}),
}

#: Specのscheduler -> sigmaの取り方を指すオプション。
#: ComfyUIのschedulerはノイズ量のスケジュールを指すもので、diffusers側では
#: Schedulerのコンストラクタ引数として表す。normalは既定なので何も足さない。
#: sgm_uniform / simple / ddim_uniform / linear_quadratic / kl_optimal は
#: 対応する引数が無いため受け付けない。
_SCHEDULERS: Final[dict[str, dict[str, Any]]] = {
    "normal": {},
    "karras": {"use_karras_sigmas": True},
    "exponential": {"use_exponential_sigmas": True},
    "beta": {"use_beta_sigmas": True},
}

#: sigmaの取り方を選べるSchedulerクラス。ここに無いクラスへ use_*_sigmas を
#: 渡しても黙って無視されるため、normal以外との組み合わせを拒否する。
#: 集合はdiffusers 0.39の実シグネチャから起こしたもので、
#: test_diffusers_scheduler.py が実物と突き合わせて守る。
#: ancestral系 (Euler / DPM++ SDE) とLMSは、A1111では karras と組めるが
#: diffusers側には対応する引数が無い。
_SIGMA_CAPABLE: Final[frozenset[str]] = frozenset(
    {
        "EulerDiscreteScheduler",
        "HeunDiscreteScheduler",
        "KDPM2DiscreteScheduler",
        "KDPM2AncestralDiscreteScheduler",
        "DPMSolverSinglestepScheduler",
        "DPMSolverMultistepScheduler",
        "DEISMultistepScheduler",
        "UniPCMultistepScheduler",
    }
)

#: 対応しているsampler名 (Specの値)。
SUPPORTED_SAMPLERS: Final[tuple[str, ...]] = tuple(_SAMPLERS)

#: 対応しているscheduler名 (Specの値)。
SUPPORTED_SCHEDULERS: Final[tuple[str, ...]] = tuple(_SCHEDULERS)


def resolve_scheduler(sampler: SamplerName, scheduler: SchedulerName) -> SchedulerChoice:
    """Specのsampler / schedulerからSchedulerの構成を決める。

    対応が無い組み合わせは InvalidGenerationSpec で拒否する。
    """
    entry = _SAMPLERS.get(sampler)
    if entry is None:
        raise InvalidGenerationSpec(
            f"diffusersバックエンドは sampler={sampler} に対応していません "
            f"(使えるもの: {', '.join(SUPPORTED_SAMPLERS)})"
        )
    class_name, options = entry

    sigma_options = _SCHEDULERS.get(scheduler)
    if sigma_options is None:
        raise InvalidGenerationSpec(
            f"diffusersバックエンドは scheduler={scheduler} に対応していません "
            f"(使えるもの: {', '.join(SUPPORTED_SCHEDULERS)})"
        )
    if sigma_options and class_name not in _SIGMA_CAPABLE:
        raise InvalidGenerationSpec(
            f"sampler={sampler} は scheduler={scheduler} と組み合わせられません "
            "(このSchedulerはsigmaの取り方を選べないため、normal を指定してください)"
        )

    return SchedulerChoice(class_name=class_name, options={**options, **sigma_options})


__all__ = [
    "SUPPORTED_SAMPLERS",
    "SUPPORTED_SCHEDULERS",
    "SchedulerChoice",
    "resolve_scheduler",
]
