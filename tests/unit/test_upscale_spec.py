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
        GenerationSpec.model_validate(_spec(upscale={"method": "lanczos"}))


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
