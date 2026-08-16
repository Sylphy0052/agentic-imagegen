"""clip skip (CLIPSetLastLayer) のバリデーションと注入・参照整合性。

CLIPSetLastLayerは全29テンプレートへ無条件に挿入されている前提のため、
このテストは特定のbindingだけでなく `ALLOWED_WORKFLOWS` 全件を横断して検証する。
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from agentic_imagegen.adapters.comfyui.workflow import (
    CLIP_SKIP_ROLE,
    IMG2IMG_BINDING,
    IMG2IMG_LORA_BINDING,
    TXT2IMG_BINDING,
    TXT2IMG_LORA_BINDING,
    TXT2IMG_UNET_BINDING,
    build_workflow,
    validate_structure,
)
from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.workflows.injector import ALLOWED_WORKFLOWS, load_workflow_template

SEPARATE_MODEL: dict[str, Any] = {
    "unet": "hassakuAnima_v13_int8.safetensors",
    "clip": "qwen_3_06b_base.safetensors",
    "vae": "qwen_image_vae.safetensors",
}


def _spec_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "version": "1",
        "task": "txt2img",
        "prompt": {"positive": "1girl, blue hair", "negative": "low quality"},
        "generation": {
            "width": 512,
            "height": 768,
            "steps": 20,
            "cfg": 5.5,
            "seed": -1,
            "batch_size": 1,
            "sampler": "euler",
            "scheduler": "normal",
        },
        "model": {"checkpoint": "v1-5-pruned-emaonly.safetensors"},
        "output": {"directory": "outputs", "prefix": "blue_hair"},
    }
    for section, values in overrides.items():
        if isinstance(values, dict) and isinstance(base.get(section), dict):
            base[section] = {**base[section], **values}
        else:
            base[section] = values
    return base


# --- domain: model.clip_skip のバリデーション -------------------------------


def test_clip_skip_defaults_to_none() -> None:
    spec = GenerationSpec.model_validate(_spec_dict())

    assert spec.model.clip_skip is None


@pytest.mark.parametrize("value", [1, 2, 12])
def test_clip_skip_accepts_valid_range(value: int) -> None:
    spec = GenerationSpec.model_validate(_spec_dict(model={"clip_skip": value}))

    assert spec.model.clip_skip == value


@pytest.mark.parametrize("value", [0, -1, 13, 100])
def test_clip_skip_rejects_out_of_range(value: int) -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec_dict(model={"clip_skip": value}))


def test_clip_skip_rejected_with_separate_loaders() -> None:
    """DiT系 (unet/clip/vae) とclip_skipの併用はtext encoderの構造が異なるため拒否する。"""
    payload = _spec_dict(model={**SEPARATE_MODEL, "checkpoint": None, "clip_skip": 2})

    with pytest.raises(ValidationError, match="clip_skip"):
        GenerationSpec.model_validate(payload)


def test_clip_skip_and_loras_both_reported_with_separate_loaders() -> None:
    """LoRAとclip_skipを同時指定した場合、両方の非対応理由が1回のエラーで分かる。"""
    payload = _spec_dict(
        model={
            **SEPARATE_MODEL,
            "checkpoint": None,
            "clip_skip": 2,
            "loras": [{"name": "add_detail.safetensors"}],
        }
    )

    with pytest.raises(ValidationError, match="LoRA") as excinfo:
        GenerationSpec.model_validate(payload)

    assert "clip_skip" in str(excinfo.value)


# --- adapter: build_workflowでの注入 -----------------------------------------


def test_unspecified_clip_skip_keeps_template_default() -> None:
    """未指定時は出力が現状 (clip skip 1相当) と完全に一致する。"""
    template = load_workflow_template("txt2img")

    workflow = build_workflow(template, GenerationSpec.model_validate(_spec_dict()), seed=1)

    assert workflow["70"]["inputs"]["stop_at_clip_layer"] == -1


def test_clip_skip_two_is_injected() -> None:
    template = load_workflow_template("txt2img")
    spec = GenerationSpec.model_validate(_spec_dict(model={"clip_skip": 2}))

    workflow = build_workflow(template, spec, seed=1)

    assert workflow["70"]["inputs"]["stop_at_clip_layer"] == -2


def test_clip_skip_injected_for_lora_template() -> None:
    template = load_workflow_template("txt2img_lora")
    spec = GenerationSpec.model_validate(
        _spec_dict(
            model={
                "checkpoint": "v1-5-pruned-emaonly.safetensors",
                "clip_skip": 2,
                "loras": [{"name": "add_detail.safetensors"}],
            }
        )
    )

    workflow = build_workflow(template, spec, seed=1, binding=TXT2IMG_LORA_BINDING)

    assert workflow["70"]["inputs"]["stop_at_clip_layer"] == -2
    # clip_skipの供給元はLoRAチェーンの最終段 (12番, CLIP出力スロット1)
    assert workflow["70"]["inputs"]["clip"] == ["12", 1]


def test_clip_skip_source_is_checkpoint_when_no_lora() -> None:
    template = load_workflow_template("txt2img")

    assert template["70"]["inputs"]["clip"] == ["4", 1]


def test_unet_template_has_no_clip_set_last_layer() -> None:
    """DiT系テンプレートはCLIPSetLastLayerを持たず、CLIPLoaderへ直結する。

    text encoderがCLIPではなくQwen3のため、stop_at_clip_layer=-1でも素通しにならず
    条件付けが壊れる (出力が単色や人型の崩れた塊になるのを2026-08-16に実機で確認)。
    """
    template = load_workflow_template("txt2img_unet")

    assert "70" not in template
    assert template["6"]["inputs"]["clip"] == ["61", 0]
    assert template["7"]["inputs"]["clip"] == ["61", 0]
    assert CLIP_SKIP_ROLE not in TXT2IMG_UNET_BINDING.nodes


def test_clip_text_encode_always_reads_from_clip_skip_node() -> None:
    """checkpoint系のCLIPTextEncode (positive/negative) はCLIPの供給元へ直結しない。"""
    for name in ("txt2img", "txt2img_lora", "img2img", "img2img_lora"):
        template = load_workflow_template(name)

        assert template["6"]["inputs"]["clip"] == ["70", 0], name
        assert template["7"]["inputs"]["clip"] == ["70", 0], name


# --- 参照整合性: 全29テンプレートを横断して検証 -------------------------------


@pytest.mark.parametrize("name", sorted(ALLOWED_WORKFLOWS))
def test_all_templates_pass_structure_validation(name: str) -> None:
    """全テンプレートがbinding定義 (clip_skipノードの結線含む) と一致することを保証する。

    ノード挿入時に参照が壊れていないか (存在しないノードID・接続の取り違え) を
    使い捨てスクリプトではなくテストとして固定する。
    """
    template = load_workflow_template(name)

    validate_structure(template, ALLOWED_WORKFLOWS[name])


@pytest.mark.parametrize("name", sorted(ALLOWED_WORKFLOWS))
def test_clip_set_last_layer_is_inserted_only_for_checkpoint_templates(
    name: str,
) -> None:
    """CLIPSetLastLayerはcheckpoint系にだけ挿入し、DiT系 (unet) には挿入しない。

    DiT系のtext encoderはQwen3で層構造が違うため、通すと条件付けが壊れる。
    clip_skipはそもそもDiT系との併用を検証で拒否している。
    """
    template = load_workflow_template(name)

    if "unet" in name:
        assert "70" not in template
        assert template["6"]["inputs"]["clip"] == ["61", 0]
        assert template["7"]["inputs"]["clip"] == ["61", 0]
        return

    assert template["70"]["class_type"] == "CLIPSetLastLayer"
    assert template["70"]["inputs"]["stop_at_clip_layer"] == -1


def test_detects_bypassed_clip_skip_node() -> None:
    """CLIPTextEncodeがCLIPSetLastLayerを迂回して直結していると検出する。"""
    template = load_workflow_template("txt2img")
    broken = {
        **template,
        "6": {**template["6"], "inputs": {**template["6"]["inputs"], "clip": ["4", 1]}},
    }

    with pytest.raises(Exception, match="clip"):
        validate_structure(broken, TXT2IMG_BINDING)


def test_img2img_clip_skip_structure() -> None:
    template = load_workflow_template("img2img")

    validate_structure(template, IMG2IMG_BINDING)
    assert template["70"]["inputs"]["clip"] == ["4", 1]


def test_img2img_lora_clip_skip_source_is_lora_chain() -> None:
    template = load_workflow_template("img2img_lora")

    validate_structure(template, IMG2IMG_LORA_BINDING)
    assert template["70"]["inputs"]["clip"] == ["22", 1]
