"""所要時間の見積り estimate_duration のテスト。

CPU推論では生成パラメータがそのまま待ち時間になる。実行してから12分待つより、
validateの時点で桁を知らせたい。値の出どころは docs/xpu-setup.md の実測表。
"""

from __future__ import annotations

import pytest

from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.services.estimate import (
    Estimate,
    estimate_duration,
    format_duration,
)

HASSAKU = "hassakuSD15_v13.safetensors"
SDXL = "waiIllustriousSDXL_v170.safetensors"
ANIMA = "hassakuAnima_v13_int8.safetensors"

#: 実測に対して許容する幅。見積りは桁を知らせるためのもので、精度は求めない。
TOLERANCE = 0.25


def _spec(**overrides: object) -> GenerationSpec:
    generation: dict[str, object] = {"width": 512, "height": 768, "steps": 20}
    generation.update(overrides.pop("generation", {}))  # type: ignore[arg-type]
    payload: dict[str, object] = {
        "version": "1",
        "task": "txt2img",
        "prompt": {"positive": "1girl"},
        "model": {"checkpoint": HASSAKU},
        "generation": generation,
    }
    payload.update(overrides)
    return GenerationSpec.model_validate(payload)


def _estimate(**overrides: object) -> Estimate:
    estimate = estimate_duration(_spec(**overrides))
    assert estimate is not None
    return estimate


class TestFamily:
    def test_checkpoint_is_sd15_by_default(self) -> None:
        assert _estimate().family == "sd15"

    def test_xl_in_the_filename_selects_sdxl(self) -> None:
        assert _estimate(model={"checkpoint": SDXL}).family == "sdxl"

    def test_separate_loaders_select_dit(self) -> None:
        estimate = _estimate(
            model={"unet": ANIMA, "clip": "qwen3.safetensors", "vae": "anima_vae.safetensors"}
        )

        assert estimate.family == "dit"


class TestMeasuredCases:
    """docs/xpu-setup.md の実測表を再現できること。"""

    def test_sd15_with_controlnet_baseline(self) -> None:
        """SD1.5 / 512x768 / 20 steps はXPUで61.3秒 (ControlNet込み・ロード済み)。"""
        estimate = _estimate()

        assert estimate.xpu_seconds == pytest.approx(71.0, rel=TOLERANCE)

    def test_sd15_hires_fix(self) -> None:
        """512x768 -> 768x1152 (2段目8 steps) はXPUで135.7秒。"""
        estimate = _estimate(generation={"upscale": {"scale": 1.5, "steps": 8}})

        assert estimate.xpu_seconds == pytest.approx(135.7, rel=TOLERANCE)

    def test_sdxl(self) -> None:
        """SDXL / 832x1216 / 24 steps はXPUで362.6秒。"""
        estimate = _estimate(
            model={"checkpoint": SDXL},
            generation={"width": 832, "height": 1216, "steps": 24},
        )

        assert estimate.xpu_seconds == pytest.approx(362.6, rel=TOLERANCE)

    def test_dit(self) -> None:
        """Anima / 640x896 / 16 steps はXPUで203.3秒。"""
        estimate = _estimate(
            model={"unet": ANIMA, "clip": "qwen3.safetensors", "vae": "anima_vae.safetensors"},
            generation={"width": 640, "height": 896, "steps": 16},
        )

        assert estimate.xpu_seconds == pytest.approx(203.3, rel=TOLERANCE)

    def test_cpu_is_an_order_slower(self) -> None:
        """SD1.5 / 512x768 / 20 steps はCPUで約12分 (36秒/step)。"""
        estimate = _estimate()

        assert estimate.cpu_seconds == pytest.approx(720.0, rel=TOLERANCE)


class TestScaling:
    def test_batch_multiplies(self) -> None:
        single = _estimate()
        quad = _estimate(generation={"batch_size": 4})

        assert quad.xpu_seconds == pytest.approx(single.xpu_seconds * 4)

    def test_upscale_steps_default_to_the_first_pass(self) -> None:
        """2段目のstepsを省くと1段目と同じ値を使う。"""
        explicit = _estimate(generation={"upscale": {"scale": 2.0, "steps": 20}})
        implicit = _estimate(generation={"upscale": {"scale": 2.0}})

        assert implicit.xpu_seconds == pytest.approx(explicit.xpu_seconds)

    def test_img2img_has_no_estimate(self) -> None:
        """入力画像のサイズで生成するため、Specの解像度からは見積もれない。

        img2imgのSpecはそもそもwidth / heightを持てない。
        """
        spec = GenerationSpec.model_validate(
            {
                "version": "1",
                "task": "img2img",
                "prompt": {"positive": "1girl"},
                "model": {"checkpoint": HASSAKU},
                "generation": {"steps": 20},
                "source": {"image": "inputs/a.png"},
            }
        )

        assert estimate_duration(spec) is None


class TestFormatDuration:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [(9.4, "約9秒"), (61.0, "約1分"), (135.7, "約2分"), (720.0, "約12分")],
    )
    def test_reads_as_a_rough_number(self, seconds: float, expected: str) -> None:
        assert format_duration(seconds) == expected
