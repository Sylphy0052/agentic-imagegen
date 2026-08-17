"""生成にかかる時間を、実行前にSpecだけから見積もる。

CPU / XPU推論では生成パラメータの負荷がそのまま待ち時間になる。12分待ってから
「解像度を落とせばよかった」と分かるより、validateの時点で桁を知らせたい。

係数は docs/xpu-setup.md の「所要時間とタイムアウトの目安」の実測表から起こした。
1 Mpixel・1 stepあたりの秒数で表し、pixel数とstepsに比例するとみなす。
モデルのロード時間、ControlNet / IPAdapterの上乗せ (1-2割)、img2imgのdenoiseによる
step削減は織り込まない。桁を知らせることだけが目的で、精度は求めない。

実行基盤はvalidateの時点では分からない (ComfyUIへ接続しないため)。
`IMAGEGEN_DEVICE` で宣言された場合はその基盤だけを、無ければ全基盤を返す。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agentic_imagegen.config import DEVICE_NAMES, DeviceName
from agentic_imagegen.domain.models import GenerationParams, GenerationSpec, ModelSpec

ModelFamily = Literal["sd15", "sdxl", "dit"]

#: 1 Mpixel・1 stepあたりのXPUでの秒数。
XPU_SECONDS_PER_UNIT: dict[ModelFamily, float] = {"sd15": 9.0, "sdxl": 15.0, "dit": 22.0}

#: 同じくCUDAでの秒数。RTX 4070 Ti SUPER 16GB での実測から起こした
#: (計測条件と生の値は docs/xpu-setup.md の実測表を参照)。
#:
#: hires fixありの条件は1 unitあたりが1割ほど重く出るため、hires fixの有無の
#: 両方を±10%以内に収める値を採った (SD1.5は0.162と0.187、DiT系は0.297と0.353の間)。
#: SDXLはhires fixを計測していないため、実測値をそのまま使う。
CUDA_SECONDS_PER_UNIT: dict[ModelFamily, float] = {"sd15": 0.17, "sdxl": 0.23, "dit": 0.32}

#: CPUはSD1.5 / 512x768で36秒/step、Mpixel・stepあたり約92秒。
#: 他の系統はCPUで計測していないため、XPUでの比をそのまま伸ばす。
CPU_RATIO = 92.0 / XPU_SECONDS_PER_UNIT["sd15"]

PIXELS_PER_MEGAPIXEL = 1_000_000

#: 表示に使う基盤名。`IMAGEGEN_DEVICE` の値と揃える。
DEVICE_LABELS: dict[DeviceName, str] = {"cuda": "CUDA", "xpu": "XPU", "cpu": "CPU"}


@dataclass(frozen=True)
class Estimate:
    """実行基盤ごとの見積り。"""

    family: ModelFamily
    #: Mpixel・stepで表した仕事量。batchとhires fixの2段目を含む。
    units: float
    #: 基盤ごとの秒数。
    seconds: dict[DeviceName, float]

    def for_device(self, device: DeviceName) -> float:
        return self.seconds[device]

    @property
    def cuda_seconds(self) -> float:
        return self.seconds["cuda"]

    @property
    def xpu_seconds(self) -> float:
        return self.seconds["xpu"]

    @property
    def cpu_seconds(self) -> float:
        return self.seconds["cpu"]


def estimate_duration(spec: GenerationSpec) -> Estimate | None:
    """Specから所要時間を見積もる。見積もれないときは None。

    img2imgは入力画像のサイズで生成するため、Specの解像度からは見積もれない。
    """
    if spec.source is not None:
        return None

    family = _family(spec.model)
    units = _units(spec.generation)
    xpu = units * XPU_SECONDS_PER_UNIT[family]
    return Estimate(
        family=family,
        units=units,
        seconds={
            "cuda": units * CUDA_SECONDS_PER_UNIT[family],
            "xpu": xpu,
            "cpu": xpu * CPU_RATIO,
        },
    )


def format_estimate(estimate: Estimate, device: DeviceName | None) -> str:
    """`Estimate:` 行の本体を組み立てる。

    基盤が宣言されていればその1つだけを出す。分からない場合に1つへ絞ると
    外したときに桁を誤らせるため、そのときは全基盤を併記する。
    """
    targets = (device,) if device is not None else DEVICE_NAMES
    return " / ".join(
        f"{DEVICE_LABELS[name]} {format_duration(estimate.for_device(name))}" for name in targets
    )


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


__all__ = [
    "Estimate",
    "ModelFamily",
    "estimate_duration",
    "format_duration",
    "format_estimate",
]
