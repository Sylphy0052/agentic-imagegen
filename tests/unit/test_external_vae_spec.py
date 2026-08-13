"""checkpoint + 外部VAE (VAELoader) の組み合わせを許可する `model.vae` のバリデーション。

DiT系 (unet + clip + vae) は既存のバリデーションを維持したまま、
checkpoint + vae の組み合わせだけを新たに許可する (Issue #57)。
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.workflows.injector import ALLOWED_WORKFLOWS, resolve_workflow_name

CHECKPOINT = "meinamix_v12Final.safetensors"
EXTERNAL_VAE = "vae-ft-mse-840000-ema-pruned.safetensors"

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
        "model": {"checkpoint": CHECKPOINT, "vae": EXTERNAL_VAE},
    }
    payload.update(overrides)
    return payload


# --- バリデーション ---------------------------------------------------------


def test_accepts_checkpoint_with_external_vae() -> None:
    model = GenerationSpec.model_validate(_spec()).model

    assert model.checkpoint == CHECKPOINT
    assert model.vae == EXTERNAL_VAE
    assert model.uses_external_vae is True
    assert model.uses_separate_loaders is False


def test_checkpoint_without_vae_is_not_external_vae() -> None:
    """vae未指定のcheckpointは従来どおり同梱VAEを使う (uses_external_vae は False)。"""
    model = GenerationSpec.model_validate(_spec(model={"checkpoint": CHECKPOINT})).model

    assert model.vae is None
    assert model.uses_external_vae is False


def test_dit_model_is_not_external_vae() -> None:
    """DiT系 (unet/clip/vae) は checkpoint を持たないため uses_external_vae は False。"""
    model = GenerationSpec.model_validate(_spec(model=dict(SEPARATE_MODEL))).model

    assert model.vae == SEPARATE_MODEL["vae"]
    assert model.uses_external_vae is False
    assert model.uses_separate_loaders is True


def test_rejects_vae_alone() -> None:
    """vae単体 (checkpointもunet/clipも無い) は組み合わせの元が無いため拒否する。"""
    with pytest.raises(ValidationError, match="vae"):
        GenerationSpec.model_validate(_spec(model={"vae": EXTERNAL_VAE}))


def test_rejects_checkpoint_with_unet() -> None:
    """checkpoint と unet の同時指定は従来どおり拒否される (vaeの併用可否とは独立)。"""
    payload = _spec(model={"checkpoint": CHECKPOINT, "unet": SEPARATE_MODEL["unet"]})

    with pytest.raises(ValidationError, match="checkpoint"):
        GenerationSpec.model_validate(payload)


def test_rejects_checkpoint_with_clip() -> None:
    payload = _spec(model={"checkpoint": CHECKPOINT, "clip": SEPARATE_MODEL["clip"]})

    with pytest.raises(ValidationError, match="checkpoint"):
        GenerationSpec.model_validate(payload)


def test_accepts_checkpoint_vae_with_loras() -> None:
    """DiT系と違い、checkpoint + 外部VAEはLoRAと併用できる。"""
    payload = _spec(
        model={
            "checkpoint": CHECKPOINT,
            "vae": EXTERNAL_VAE,
            "loras": [{"name": "add_detail.safetensors"}],
        }
    )

    spec = GenerationSpec.model_validate(payload)

    assert spec.model.loras[0].name == "add_detail.safetensors"


def test_accepts_checkpoint_vae_with_clip_skip() -> None:
    payload = _spec(model={"checkpoint": CHECKPOINT, "vae": EXTERNAL_VAE, "clip_skip": 2})

    spec = GenerationSpec.model_validate(payload)

    assert spec.model.clip_skip == 2


@pytest.mark.parametrize(
    "vae",
    ["../secret.safetensors", "/etc/passwd", "a/b/c.safetensors", "model.exe"],
)
def test_rejects_unsafe_vae_path(vae: str) -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec(model={"checkpoint": CHECKPOINT, "vae": vae}))


# --- Workflowテンプレートの決定 ---------------------------------------------


def test_resolves_vae_template() -> None:
    spec = GenerationSpec.model_validate(_spec())

    assert resolve_workflow_name(spec) == "txt2img_vae"
    assert "txt2img_vae" in ALLOWED_WORKFLOWS


def test_vae_suffix_precedes_lora() -> None:
    """`_vae` は `_unet` と同じ位置 (LoRAより手前) に入る。"""
    payload = _spec(model={**_spec()["model"], "loras": [{"name": "add_detail.safetensors"}]})
    spec = GenerationSpec.model_validate(payload)

    assert resolve_workflow_name(spec) == "txt2img_vae_lora"


def test_vae_suffix_with_hires() -> None:
    payload = _spec(generation={"upscale": {"scale": 1.5}})
    spec = GenerationSpec.model_validate(payload)

    assert resolve_workflow_name(spec) == "txt2img_vae_hires"


def test_vae_suffix_with_hires_model() -> None:
    payload = _spec(
        generation={"upscale": {"model": "RealESRGAN_x4plus_anime_6B.pth", "scale": 2.0}}
    )
    spec = GenerationSpec.model_validate(payload)

    assert resolve_workflow_name(spec) == "txt2img_vae_hires_model"


def test_vae_suffix_with_controlnet() -> None:
    payload = _spec(
        control={
            "image": "inputs/pose.png",
            "model": "control_v11p_sd15_canny_fp16.safetensors",
        }
    )
    spec = GenerationSpec.model_validate(payload)

    assert resolve_workflow_name(spec) == "txt2img_vae_controlnet"


def test_vae_suffix_with_ipadapter() -> None:
    payload = _spec(
        reference={
            "image": "inputs/character.png",
            "model": "ip-adapter-plus_sd15.safetensors",
            "clip_vision": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
        }
    )
    spec = GenerationSpec.model_validate(payload)

    assert resolve_workflow_name(spec) == "txt2img_vae_ipadapter"


def test_vae_suffix_with_lora_hires_model_controlnet() -> None:
    """複数軸を組み合わせても `_vae` は先頭 (task直後) に入る。"""
    payload = _spec(
        model={
            "checkpoint": CHECKPOINT,
            "vae": EXTERNAL_VAE,
            "loras": [{"name": "add_detail.safetensors"}],
        },
        generation={
            "upscale": {"model": "RealESRGAN_x4plus_anime_6B.pth", "scale": 2.0},
        },
        control={"image": "inputs/pose.png", "model": "control_v11p_sd15_canny_fp16.safetensors"},
    )
    spec = GenerationSpec.model_validate(payload)

    assert resolve_workflow_name(spec) == "txt2img_vae_lora_hires_model_controlnet"


def test_dit_model_does_not_get_vae_suffix() -> None:
    """DiT系 (uses_separate_loaders) は `_unet` になり `_vae` は付かない。"""
    spec = GenerationSpec.model_validate(_spec(model=dict(SEPARATE_MODEL)))

    assert resolve_workflow_name(spec) == "txt2img_unet"


def test_img2img_vae_template() -> None:
    payload = _spec(task="img2img", source={"image": "inputs/base.png"})
    spec = GenerationSpec.model_validate(payload)

    assert resolve_workflow_name(spec) == "img2img_vae"
    assert "img2img_vae" in ALLOWED_WORKFLOWS
