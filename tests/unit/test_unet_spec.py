"""UNet / CLIP / VAE を分けて指定する形式 (DiT系モデル) のバリデーション。"""

from typing import Any

import pytest
from pydantic import ValidationError

from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.workflows.injector import ALLOWED_WORKFLOWS, resolve_workflow_name

SEPARATE_MODEL: dict[str, Any] = {
    "unet": "hassakuAnima_v13_int8.safetensors",
    "clip": "qwen_3_06b_base.safetensors",
    "vae": "qwen_image_vae.safetensors",
}


def _spec(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": "1",
        "task": "txt2img",
        "prompt": {"positive": "1girl"},
        "model": dict(SEPARATE_MODEL),
    }
    payload.update(overrides)
    return payload


def test_accepts_separate_loaders() -> None:
    model = GenerationSpec.model_validate(_spec()).model

    assert model.checkpoint is None
    assert model.unet == "hassakuAnima_v13_int8.safetensors"
    assert model.clip == "qwen_3_06b_base.safetensors"
    assert model.vae == "qwen_image_vae.safetensors"


def test_uses_separate_loaders_flag() -> None:
    separate = GenerationSpec.model_validate(_spec()).model
    single = GenerationSpec.model_validate(
        _spec(model={"checkpoint": "meinamix_v12Final.safetensors"})
    ).model

    assert separate.uses_separate_loaders is True
    assert single.uses_separate_loaders is False


def test_rejects_checkpoint_with_separate_loaders() -> None:
    payload = _spec(model={**SEPARATE_MODEL, "checkpoint": "meinamix_v12Final.safetensors"})

    with pytest.raises(ValidationError, match="checkpoint"):
        GenerationSpec.model_validate(payload)


def test_rejects_empty_model() -> None:
    with pytest.raises(ValidationError, match="checkpoint"):
        GenerationSpec.model_validate(_spec(model={}))


@pytest.mark.parametrize("missing", ["unet", "clip", "vae"])
def test_rejects_partial_separate_loaders(missing: str) -> None:
    payload = dict(SEPARATE_MODEL)
    del payload[missing]

    with pytest.raises(ValidationError, match=missing):
        GenerationSpec.model_validate(_spec(model=payload))


def test_accepts_loras_with_separate_loaders() -> None:
    """DiT系向けのLoRAが出回ったため通す (Issue #39)。結線は test_dit_lora.py が見る。"""
    payload = {
        **SEPARATE_MODEL,
        "loras": [{"name": "anima_context_detailer_base10.safetensors"}],
    }

    spec = GenerationSpec.model_validate(_spec(model=payload))

    assert resolve_workflow_name(spec) == "txt2img_unet_lora"
    assert "txt2img_unet_lora" in ALLOWED_WORKFLOWS


def test_accepts_img2img_with_separate_loaders() -> None:
    """VAEEncodeを挟むだけで通る。latentのch数は入力画像から決まる。"""
    spec = GenerationSpec.model_validate(_spec(task="img2img", source={"image": "inputs/base.png"}))

    assert resolve_workflow_name(spec) == "img2img_unet"
    assert "img2img_unet" in ALLOWED_WORKFLOWS


def test_accepts_upscale_with_separate_loaders() -> None:
    """latentのch数は拡大しても変わらないため、SD1.5系と同じ構成で組める。"""
    spec = GenerationSpec.model_validate(_spec(generation={"upscale": {"scale": 1.5}}))

    assert resolve_workflow_name(spec) == "txt2img_unet_hires"
    assert "txt2img_unet_hires" in ALLOWED_WORKFLOWS


def test_accepts_img2img_and_upscale_together() -> None:
    spec = GenerationSpec.model_validate(
        _spec(
            task="img2img",
            source={"image": "inputs/base.png"},
            generation={"upscale": {"scale": 1.5}},
        )
    )

    assert resolve_workflow_name(spec) == "img2img_unet_hires"
    assert "img2img_unet_hires" in ALLOWED_WORKFLOWS


def test_rejects_control_with_separate_loaders() -> None:
    payload = _spec(
        control={
            "image": "inputs/pose.png",
            "model": "control_v11p_sd15_canny_fp16.safetensors",
        }
    )

    with pytest.raises(ValidationError, match="control"):
        GenerationSpec.model_validate(payload)


def test_rejects_reference_with_separate_loaders() -> None:
    payload = _spec(
        reference={
            "image": "inputs/character.png",
            "model": "ip-adapter-plus_sd15.safetensors",
            "clip_vision": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
        }
    )

    with pytest.raises(ValidationError, match="reference"):
        GenerationSpec.model_validate(payload)


@pytest.mark.parametrize("field", ["unet", "clip", "vae"])
def test_rejects_path_traversal(field: str) -> None:
    payload = {**SEPARATE_MODEL, field: "../secret.safetensors"}

    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec(model=payload))


def test_accepts_uppercase_suffix() -> None:
    payload = {**SEPARATE_MODEL, "unet": "Hassaku_Anima.SafeTensors"}

    assert GenerationSpec.model_validate(_spec(model=payload)).model.unet is not None


def test_accepts_er_sde_sampler() -> None:
    payload = _spec(generation={"sampler": "er_sde"})

    assert GenerationSpec.model_validate(payload).generation.sampler == "er_sde"
