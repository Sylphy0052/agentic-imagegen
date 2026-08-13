"""設定由来のポリシー制約 (上限値・出力先) のテスト。"""

from pathlib import Path
from typing import Any

import pytest

from agentic_imagegen.config import Settings
from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.domain.policy import resolve_output_directory, validate_against_limits
from agentic_imagegen.errors import InvalidGenerationSpec


def _spec(**generation: Any) -> GenerationSpec:
    payload: dict[str, Any] = {
        "prompt": {"positive": "a cat"},
        "model": {"checkpoint": "v1-5-pruned-emaonly.safetensors"},
        "generation": {"width": 512, "height": 512, **generation},
    }
    return GenerationSpec.model_validate(payload)


def _settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "comfyui_base_url": "http://127.0.0.1:8188",
        "max_width": 2048,
        "max_height": 2048,
        "max_pixels": 4194304,
        "max_batch": 4,
        "timeout_seconds": 300,
        "output_root": Path("outputs"),
    }
    return Settings(**{**defaults, **overrides})


def test_within_limits_passes() -> None:
    validate_against_limits(_spec(), _settings())


def test_exceeds_max_width() -> None:
    with pytest.raises(InvalidGenerationSpec, match="width"):
        validate_against_limits(_spec(width=1024), _settings(max_width=512))


def test_exceeds_max_height() -> None:
    with pytest.raises(InvalidGenerationSpec, match="height"):
        validate_against_limits(_spec(height=1024), _settings(max_height=512))


def test_exceeds_max_pixels() -> None:
    with pytest.raises(InvalidGenerationSpec, match="pixel"):
        validate_against_limits(_spec(width=2048, height=2048), _settings(max_pixels=1048576))


def test_exceeds_max_batch() -> None:
    with pytest.raises(InvalidGenerationSpec, match="batch_size"):
        validate_against_limits(_spec(batch_size=4), _settings(max_batch=1))


def test_pixels_counted_per_batch() -> None:
    """batch_sizeを掛けた総ピクセル数で判定する。"""
    with pytest.raises(InvalidGenerationSpec, match="pixel"):
        validate_against_limits(
            _spec(width=1024, height=1024, batch_size=4),
            _settings(max_pixels=2097152),
        )


class TestUpscaledPixels:
    """hires fixの拡大後の解像度も上限で縛る。

    ベース解像度の上限 (max_pixels) は生成負荷を抑えるためのもので、
    拡大後のピークメモリとは目的が違うため別の上限を持つ。
    """

    def _upscaled(self, **upscale: Any) -> GenerationSpec:
        return _spec(upscale=upscale)

    def test_latent_upscale_within_limit_passes(self) -> None:
        validate_against_limits(
            self._upscaled(scale=2.0),
            _settings(max_upscaled_pixels=1048576),
        )

    def test_latent_upscale_exceeding_limit_is_rejected(self) -> None:
        """512x512をscale 2.0で拡大すると1024x1024。上限をその手前へ置く。"""
        with pytest.raises(InvalidGenerationSpec, match="latent拡大"):
            validate_against_limits(
                self._upscaled(scale=2.0),
                _settings(max_upscaled_pixels=1048575),
            )

    def test_model_upscale_is_judged_by_peak_not_final(self) -> None:
        """モデル拡大は一度モデルの固有倍率まで広げるため、そこがピークになる。

        512x512 / scale 2.0 / model_scale 4.0 の場合、最終は1024x1024でも
        途中で2048x2048まで広がる。
        """
        upscale = {
            "model": "RealESRGAN_x4plus_anime_6B.pth",
            "model_scale": 4.0,
            "scale": 2.0,
        }

        # 最終 (1024x1024) は収まるがピーク (2048x2048) は超える上限
        with pytest.raises(InvalidGenerationSpec, match="アップスケールモデルでの拡大"):
            validate_against_limits(_spec(upscale=upscale), _settings(max_upscaled_pixels=4194303))

        validate_against_limits(_spec(upscale=upscale), _settings(max_upscaled_pixels=4194304))

    def test_model_scale_defaults_to_four_when_omitted(self) -> None:
        """model_scale未指定は4.0とみなすため、ピークも4倍で見る。"""
        upscale = {"model": "RealESRGAN_x4plus_anime_6B.pth", "scale": 2.0}

        with pytest.raises(InvalidGenerationSpec, match="アップスケールモデルでの拡大"):
            validate_against_limits(_spec(upscale=upscale), _settings(max_upscaled_pixels=4194303))

    def test_underdeclared_model_scale_is_judged_at_the_safe_floor(self) -> None:
        """model_scaleの過小申告でピーク見積もりを小さくできないこと。

        model_scale は自己申告値で、ComfyUIに置かれた実モデルの倍率とは独立している。
        小さく申告してもピークは実モデルの倍率で決まるため、検証側はESRGAN系の
        主流である4.0を下限として見積もる。
        """
        upscale = {
            "model": "RealESRGAN_x4plus_anime_6B.pth",
            "model_scale": 1.01,
            "scale": 1.01,
        }

        # 申告どおりなら 517x517 (267289) だが、4.0で見た 2048x2048 (4194304) で判定する
        with pytest.raises(InvalidGenerationSpec, match="アップスケールモデルでの拡大"):
            validate_against_limits(_spec(upscale=upscale), _settings(max_upscaled_pixels=4194303))

        validate_against_limits(_spec(upscale=upscale), _settings(max_upscaled_pixels=4194304))

    def test_larger_model_scale_is_respected(self) -> None:
        """下限より大きい申告はその値で見積もる。8xモデルなら8倍でピークを見る。"""
        upscale = {
            "model": "RealESRGAN_x8plus.pth",
            "model_scale": 8.0,
            "scale": 2.0,
        }

        # 512x512 を8倍すると 4096x4096 (16777216)
        with pytest.raises(InvalidGenerationSpec, match="アップスケールモデルでの拡大"):
            validate_against_limits(_spec(upscale=upscale), _settings(max_upscaled_pixels=16777215))

        validate_against_limits(_spec(upscale=upscale), _settings(max_upscaled_pixels=16777216))

    def test_counted_per_batch(self) -> None:
        """batch_sizeを掛けた総ピクセル数で判定する。1024x1024x4 = 4194304。"""
        with pytest.raises(InvalidGenerationSpec, match="latent拡大"):
            validate_against_limits(
                _spec(upscale={"scale": 2.0}, batch_size=4),
                _settings(max_upscaled_pixels=4194303),
            )

        validate_against_limits(
            _spec(upscale={"scale": 2.0}, batch_size=4),
            _settings(max_upscaled_pixels=4194304),
        )

    def test_without_upscale_is_not_checked(self) -> None:
        """hires fixを使わないSpecはベース側の上限だけで判定する。"""
        validate_against_limits(_spec(), _settings(max_upscaled_pixels=1))


def test_resolve_output_directory_under_root(tmp_path: Path) -> None:
    resolved = resolve_output_directory("outputs/test", tmp_path)
    assert resolved == (tmp_path / "outputs" / "test").resolve()


@pytest.mark.parametrize(
    "directory",
    ["../outside", "../../etc", "outputs/../../escape", "/etc", "~/elsewhere"],
)
def test_output_directory_escape_rejected(directory: str, tmp_path: Path) -> None:
    with pytest.raises(InvalidGenerationSpec):
        resolve_output_directory(directory, tmp_path)
