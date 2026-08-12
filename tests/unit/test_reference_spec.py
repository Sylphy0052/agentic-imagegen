"""IPAdapter指定 (reference) のバリデーション。"""

from typing import Any

import pytest
from pydantic import ValidationError

from agentic_imagegen.domain.models import GenerationSpec

REFERENCE: dict[str, Any] = {
    "image": "inputs/character.png",
    "model": "ip-adapter-plus_sd15.safetensors",
    "clip_vision": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
}


def _spec(**reference: Any) -> dict[str, Any]:
    # 呼び出し側でキーを消すテストがあるため、毎回コピーを返す
    merged = {**REFERENCE, **reference}
    return {
        "version": "1",
        "task": "txt2img",
        "prompt": {"positive": "1girl"},
        "reference": merged,
        "model": {"checkpoint": "v1-5-pruned-emaonly.safetensors"},
    }


def test_absent_by_default() -> None:
    payload = _spec()
    del payload["reference"]

    assert GenerationSpec.model_validate(payload).reference is None


def test_defaults() -> None:
    reference = GenerationSpec.model_validate(_spec()).reference

    assert reference is not None
    assert reference.weight == 1.0
    assert reference.weight_type == "linear"
    assert reference.start_percent == 0.0
    assert reference.end_percent == 1.0


def test_accepts_explicit_values() -> None:
    reference = GenerationSpec.model_validate(
        _spec(weight=0.8, weight_type="style transfer", start_percent=0.1, end_percent=0.9)
    ).reference

    assert reference is not None
    assert reference.weight == 0.8
    assert reference.weight_type == "style transfer"
    assert reference.start_percent == 0.1
    assert reference.end_percent == 0.9


@pytest.mark.parametrize("missing", ["image", "model", "clip_vision"])
def test_requires_core_fields(missing: str) -> None:
    """画像・IPAdapterモデル・CLIP Visionはどれも省略できない。

    UnifiedLoaderのようにpreset名から暗黙で選ぶ経路は持たないため、
    3つとも明示する必要がある。
    """
    payload = _spec()
    del payload["reference"][missing]

    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(payload)


@pytest.mark.parametrize(
    "image",
    [
        "/etc/passwd.png",
        "../outside.png",
        "~/secret.png",
        "inputs\\ref.png",
        "inputs/ref.txt",
    ],
)
def test_rejects_unsafe_image(image: str) -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec(image=image))


@pytest.mark.parametrize(
    "model",
    [
        "../ip-adapter.safetensors",
        "/abs/ip-adapter.safetensors",
        "a/b/c/ip-adapter.safetensors",
        "ip-adapter.exe",
    ],
)
def test_rejects_unsafe_model(model: str) -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec(model=model))


def test_accepts_bin_model() -> None:
    """IPAdapterには .bin 配布があるため受け付ける (ip-adapter_sd15_light_v11.bin)。"""
    reference = GenerationSpec.model_validate(
        _spec(model="ip-adapter_sd15_light_v11.bin")
    ).reference

    assert reference is not None
    assert reference.model == "ip-adapter_sd15_light_v11.bin"


def test_accepts_uppercase_suffix() -> None:
    """実在ファイル名は大文字混じりのことがある。拡張子の大小では弾かない。"""
    reference = GenerationSpec.model_validate(
        _spec(model="IP-Adapter-Plus_SD15.SafeTensors", clip_vision="CLIP-ViT-H-14.SafeTensors")
    ).reference

    assert reference is not None
    assert reference.model == "IP-Adapter-Plus_SD15.SafeTensors"
    assert reference.clip_vision == "CLIP-ViT-H-14.SafeTensors"


@pytest.mark.parametrize(
    "clip_vision", ["../clip.safetensors", "clip.exe", "/abs/clip.safetensors"]
)
def test_rejects_unsafe_clip_vision(clip_vision: str) -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec(clip_vision=clip_vision))


@pytest.mark.parametrize("weight", [-0.1, 3.1])
def test_rejects_weight_out_of_range(weight: float) -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec(weight=weight))


def test_rejects_unknown_weight_type() -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec(weight_type="magic"))


def test_rejects_inverted_percent_range() -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec(start_percent=0.8, end_percent=0.2))


def test_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec(combine_embeds="concat"))


def test_rejects_upscale_combination() -> None:
    """IPAdapter と hires fix の同時指定は未対応 (ControlNetと同じ理由)。"""
    payload = _spec()
    payload["generation"] = {"upscale": {"scale": 1.5}}

    with pytest.raises(ValidationError, match="upscale"):
        GenerationSpec.model_validate(payload)


def test_allows_control_combination() -> None:
    """構図 (ControlNet) と人物特徴 (IPAdapter) は併用できる。

    同一キャラクタを異なる構図で出すために必要な組み合わせ。
    """
    payload = _spec()
    payload["control"] = {
        "image": "inputs/pose.png",
        "model": "control_v11p_sd15_canny_fp16.safetensors",
    }

    spec = GenerationSpec.model_validate(payload)

    assert spec.reference is not None
    assert spec.control is not None


def test_allows_img2img_combination() -> None:
    payload = _spec()
    payload["task"] = "img2img"
    payload["source"] = {"image": "inputs/base.png"}

    spec = GenerationSpec.model_validate(payload)

    assert spec.reference is not None
    assert spec.source is not None
