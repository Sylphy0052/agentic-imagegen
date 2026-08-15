"""生成にかかる時間を、実行前にSpecだけから見積もる。

CPU推論では生成パラメータの負荷がそのまま待ち時間になる。12分待ってから
「解像度を落とせばよかった」と分かるより、validateの時点で桁を知らせたい。

係数は docs/xpu-setup.md の「所要時間とタイムアウトの目安」の実測表から起こした。
1 Mpixel・1 stepあたりの秒数で表し、pixel数とstepsに比例するとみなす。
モデルのロード時間、ControlNet / IPAdapterの上乗せ (1-2割)、img2imgのdenoiseによる
step削減は織り込まない。桁を知らせることだけが目的で、精度は求めない。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agentic_imagegen.domain.models import GenerationParams, GenerationSpec, ModelSpec

ModelFamily = Literal["sd15", "sdxl", "dit"]

#: 1 Mpixel・1 stepあたりのXPUでの秒数。
XPU_SECONDS_PER_UNIT: dict[ModelFamily, float] = {"sd15": 9.0, "sdxl": 15.0, "dit": 22.0}

#: CPUはSD1.5 / 512x768で36秒/step、Mpixel・stepあたり約92秒。
#: 他の系統はCPUで計測していないため、XPUでの比をそのまま伸ばす。
CPU_RATIO = 92.0 / XPU_SECONDS_PER_UNIT["sd15"]

PIXELS_PER_MEGAPIXEL = 1_000_000


@dataclass(frozen=True)
class Estimate:
    """実行基盤ごとの見積り。"""

    family: ModelFamily
    #: Mpixel・stepで表した仕事量。batchとhires fixの2段目を含む。
    units: float
    xpu_seconds: float
    cpu_seconds: float


def estimate_duration(spec: GenerationSpec) -> Estimate | None:
    """Specから所要時間を見積もる。見積もれないときは None。

    img2imgは入力画像のサイズで生成するため、Specの解像度からは見積もれない。
    """
    if spec.source is not None:
        return None

    family = _family(spec.model)
    units = _units(spec.generation)
    xpu = units * XPU_SECONDS_PER_UNIT[family]
    return Estimate(family=family, units=units, xpu_seconds=xpu, cpu_seconds=xpu * CPU_RATIO)


def format_duration(seconds: float) -> str:
    """「約2分」の形にする。1分未満は秒のまま。"""
    if seconds < 60:
        return f"約{round(seconds)}秒"
    return f"約{round(seconds / 60)}分"


def _family(model: ModelSpec) -> ModelFamily:
    """checkpointのファイル名から系統を推測する。

    SD1.5系とSDXL系はどちらもcheckpoint1ファイルで、Specだけでは見分けが付かない。
    配置済みのSDXL系はファイル名に XL を含むため、それを手掛かりにする。
    外した場合に起きるのは見積りが1.7倍ずれることだけで、生成には影響しない。
    """
    if model.unet is not None:
        return "dit"
    return "sdxl" if "XL" in (model.checkpoint or "").upper() else "sd15"


def _units(params: GenerationParams) -> float:
    units = _megapixels(params.width, params.height) * params.steps
    if params.upscale is not None:
        scale = params.upscale.scale
        second_steps = params.upscale.steps or params.steps
        units += _megapixels(params.width, params.height) * scale * scale * second_steps
    return units * params.batch_size


def _megapixels(width: int, height: int) -> float:
    return width * height / PIXELS_PER_MEGAPIXEL


__all__ = ["Estimate", "ModelFamily", "estimate_duration", "format_duration"]
