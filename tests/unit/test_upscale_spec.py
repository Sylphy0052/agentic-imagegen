"""hires fix (generation.upscale) のバリデーション。"""

from typing import Any

import pytest
from pydantic import ValidationError

from agentic_imagegen.domain.models import GenerationSpec


def _spec(**generation: Any) -> dict[str, Any]:
    return {
        "version": "1",
        "task": "txt2img",
        "prompt": {"positive": "1girl"},
        "generation": generation,
        "model": {"checkpoint": "v1-5-pruned-emaonly.safetensors"},
    }


def test_absent_by_default() -> None:
    spec = GenerationSpec.model_validate(_spec())

    assert spec.generation.upscale is None


def test_defaults() -> None:
    spec = GenerationSpec.model_validate(_spec(upscale={}))

    upscale = spec.generation.upscale
    assert upscale is not None
    assert upscale.scale == 1.5
    assert upscale.denoise == 0.5
    assert upscale.steps is None
    assert upscale.method == "nearest-exact"


def test_accepts_explicit_values() -> None:
    spec = GenerationSpec.model_validate(
        _spec(upscale={"scale": 2.0, "denoise": 0.35, "steps": 8, "method": "bicubic"})
    )

    upscale = spec.generation.upscale
    assert upscale is not None
    assert upscale.scale == 2.0
    assert upscale.denoise == 0.35
    assert upscale.steps == 8
    assert upscale.method == "bicubic"


@pytest.mark.parametrize("scale", [0.9, 1.0, 4.1, 10.0])
def test_rejects_out_of_range_scale(scale: float) -> None:
    """1.0以下は拡大にならない。上限は生成時間が現実的な範囲に絞る。"""
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec(upscale={"scale": scale}))


@pytest.mark.parametrize("denoise", [-0.1, 1.1])
def test_rejects_out_of_range_denoise(denoise: float) -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec(upscale={"denoise": denoise}))


@pytest.mark.parametrize("steps", [0, 101])
def test_rejects_out_of_range_steps(steps: int) -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec(upscale={"steps": steps}))


def test_rejects_unknown_method() -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec(upscale={"method": "catmull-rom"}))


def test_rejects_unknown_key() -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec(upscale={"scale_by": 2.0}))


def test_effective_steps_falls_back_to_generation_steps() -> None:
    """upscale.steps 未指定なら1段目と同じstepsを使う。"""
    spec = GenerationSpec.model_validate(_spec(steps=24, upscale={}))

    assert spec.generation.upscale is not None
    assert spec.generation.upscale.effective_steps(spec.generation.steps) == 24


def test_effective_steps_uses_own_value() -> None:
    spec = GenerationSpec.model_validate(_spec(steps=24, upscale={"steps": 8}))

    assert spec.generation.upscale is not None
    assert spec.generation.upscale.effective_steps(spec.generation.steps) == 8


def test_model_absent_by_default() -> None:
    """未指定ならlatent拡大のまま。既存のSpecの意味を変えない。"""
    spec = GenerationSpec.model_validate(_spec(upscale={}))

    assert spec.generation.upscale is not None
    assert spec.generation.upscale.model is None
    assert spec.generation.upscale.uses_model is False


def test_accepts_upscale_model() -> None:
    spec = GenerationSpec.model_validate(
        _spec(upscale={"model": "RealESRGAN_x4plus_anime_6B.pth", "scale": 2.0})
    )

    upscale = spec.generation.upscale
    assert upscale is not None
    assert upscale.model == "RealESRGAN_x4plus_anime_6B.pth"
    assert upscale.model_scale == 4.0
    assert upscale.uses_model is True


def test_resize_factor_brings_model_output_to_requested_scale() -> None:
    """4xのモデルで2倍が欲しければ、拡大後に0.5倍へ戻す。"""
    spec = GenerationSpec.model_validate(
        _spec(upscale={"model": "RealESRGAN_x4plus_anime_6B.pth", "scale": 2.0})
    )

    assert spec.generation.upscale is not None
    assert spec.generation.upscale.resize_factor() == pytest.approx(0.5)


def test_resize_factor_is_one_when_scale_matches_model() -> None:
    spec = GenerationSpec.model_validate(
        _spec(
            upscale={
                "model": "RealESRGAN_x4plus_anime_6B.pth",
                "scale": 4.0,
                "model_scale": 4.0,
            }
        )
    )

    assert spec.generation.upscale is not None
    assert spec.generation.upscale.resize_factor() == pytest.approx(1.0)


def test_resize_factor_requires_model() -> None:
    """latent拡大では縮小の出番が無い。呼び出し側の取り違えを落とす。"""
    spec = GenerationSpec.model_validate(_spec(upscale={}))

    assert spec.generation.upscale is not None
    with pytest.raises(ValueError, match="model"):
        spec.generation.upscale.resize_factor()


@pytest.mark.parametrize(
    "name",
    [
        "RealESRGAN_x4plus_anime_6B.pth",
        "4x-UltraSharp.safetensors",
    ],
)
def test_accepts_allowed_upscale_model_suffixes(name: str) -> None:
    spec = GenerationSpec.model_validate(_spec(upscale={"model": name, "scale": 2.0}))

    assert spec.generation.upscale is not None
    assert spec.generation.upscale.model == name


@pytest.mark.parametrize(
    "name",
    [
        "model.bin",
        "model",
        "../escape.pth",
        "a/b/deep.pth",
        "",
    ],
)
def test_rejects_unsafe_or_unknown_upscale_model_name(name: str) -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec(upscale={"model": name, "scale": 2.0}))


def test_rejects_model_scale_without_model() -> None:
    """model_scale はアップスケールモデルの固有倍率。latent拡大では意味を持たない。"""
    with pytest.raises(ValidationError, match="model_scale"):
        GenerationSpec.model_validate(_spec(upscale={"model_scale": 4.0}))


def test_rejects_scale_above_model_scale() -> None:
    """モデルの出力より大きくは引き伸ばさない。ぼけた絵を黙って返さないため。"""
    with pytest.raises(ValidationError, match="model_scale"):
        GenerationSpec.model_validate(
            _spec(
                upscale={
                    "model": "RealESRGAN_x4plus_anime_6B.pth",
                    "scale": 3.0,
                    "model_scale": 2.0,
                }
            )
        )


@pytest.mark.parametrize("model_scale", [0.9, 8.1])
def test_rejects_out_of_range_model_scale(model_scale: float) -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(
            _spec(upscale={"model": "RealESRGAN_x4plus_anime_6B.pth", "model_scale": model_scale})
        )


def test_accepts_lanczos_with_model() -> None:
    """lanczos は ImageScaleBy にだけある。モデル拡大の縮小方法として使える。"""
    spec = GenerationSpec.model_validate(
        _spec(
            upscale={
                "model": "RealESRGAN_x4plus_anime_6B.pth",
                "scale": 2.0,
                "method": "lanczos",
            }
        )
    )

    assert spec.generation.upscale is not None
    assert spec.generation.upscale.method == "lanczos"


def test_rejects_lanczos_without_model() -> None:
    """LatentUpscaleBy に lanczos は無い。投入前に落とす。"""
    with pytest.raises(ValidationError, match="lanczos"):
        GenerationSpec.model_validate(_spec(upscale={"method": "lanczos"}))


def test_rejects_bislerp_with_model() -> None:
    """ImageScaleBy に bislerp は無い。"""
    with pytest.raises(ValidationError, match="bislerp"):
        GenerationSpec.model_validate(
            _spec(upscale={"model": "RealESRGAN_x4plus_anime_6B.pth", "method": "bislerp"})
        )
