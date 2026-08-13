"""`resolve_workflow_name()` が返す名前が、必ず許可済み集合に含まれることの網羅検査。

`ALLOWED_WORKFLOWS` (adapters.comfyui.workflow.ALL_BINDINGS 由来) と
`resolve_workflow_name()` (workflows.axes 由来) は、どちらも
`workflows.axes.iter_template_specs()` の列挙を基に組み立てているが、
別々の場所 (Specのフィールド判定とテンプレート名の組み立て) で使うため、
実装の書き方次第では食い違いうる。ここではSpecの組み合わせを軸の列挙から
機械的に作り、実際に resolve_workflow_name() へ通して確かめる。

軸を1本足したときに、Specのどのフィールドで軸を選ぶかの分岐
(resolve_workflow_name()) を書き忘れても、このテストが失敗して気づける。
"""

from __future__ import annotations

from typing import Any

import pytest

from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.workflows.axes import (
    AXIS_CONTROLNET,
    AXIS_HIRES,
    AXIS_HIRES_MODEL,
    AXIS_IPADAPTER,
    AXIS_LORA,
    AXIS_UNET,
    AXIS_VAE,
    TemplateSpec,
    iter_template_specs,
)
from agentic_imagegen.workflows.injector import ALLOWED_WORKFLOWS, resolve_workflow_name

CHECKPOINT = "v1-5-pruned-emaonly.safetensors"
EXTERNAL_VAE = "vae-ft-mse-840000-ema-pruned.safetensors"
LORA = {"name": "add_detail.safetensors"}
SEPARATE_MODEL = {
    "unet": "hassakuAnima_v13_int8.safetensors",
    "clip": "qwen_3_06b_base.safetensors",
    "vae": "qwen_image_vae.safetensors",
}
CONTROL = {
    "image": "inputs/pose.png",
    "model": "control_v11p_sd15_canny_fp16.safetensors",
}
REFERENCE = {
    "image": "inputs/character.png",
    "model": "ip-adapter-plus_sd15.safetensors",
    "clip_vision": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
}
UPSCALE_MODEL = "RealESRGAN_x4plus_anime_6B.pth"


def _payload_for(task: str, present_axes: frozenset[str]) -> dict[str, Any]:
    """(task, 軸の集合) から、その組み合わせを再現する最小のSpecペイロードを作る。"""
    payload: dict[str, Any] = {
        "version": "1",
        "task": task,
        "prompt": {"positive": "1girl"},
        "generation": {"seed": 1},
    }

    if AXIS_UNET in present_axes:
        payload["model"] = dict(SEPARATE_MODEL)
    else:
        model: dict[str, Any] = {"checkpoint": CHECKPOINT}
        if AXIS_VAE in present_axes:
            model["vae"] = EXTERNAL_VAE
        if AXIS_LORA in present_axes:
            model["loras"] = [dict(LORA)]
        payload["model"] = model

    if task == "img2img":
        payload["source"] = {"image": "inputs/ref.png"}

    if AXIS_HIRES in present_axes:
        payload["generation"]["upscale"] = {"scale": 1.5}
    elif AXIS_HIRES_MODEL in present_axes:
        payload["generation"]["upscale"] = {"scale": 1.5, "model": UPSCALE_MODEL}

    if AXIS_CONTROLNET in present_axes:
        payload["control"] = dict(CONTROL)
    if AXIS_IPADAPTER in present_axes:
        payload["reference"] = dict(REFERENCE)

    return payload


def _spec_for(task: str, present_axes: frozenset[str]) -> GenerationSpec:
    return GenerationSpec.model_validate(_payload_for(task, present_axes))


def _all_specs() -> list[TemplateSpec]:
    return [TemplateSpec(name="txt2img", task="txt2img", axes=()), *iter_template_specs()]


@pytest.mark.parametrize("template_spec", _all_specs(), ids=lambda spec: spec.name)
def test_resolve_workflow_name_reproduces_every_allowed_combination(
    template_spec: TemplateSpec,
) -> None:
    """axesが列挙する組み合わせごとに、対応するSpecがちょうどその名前へ解決される。"""
    spec = _spec_for(template_spec.task, frozenset(template_spec.axes))

    name = resolve_workflow_name(spec)

    assert name == template_spec.name
    assert name in ALLOWED_WORKFLOWS


def test_every_allowed_workflow_is_reachable_from_some_spec() -> None:
    """ALLOWED_WORKFLOWSの全件が、実際にSpecの組み合わせから再現できることを見る。

    axesの列挙 (ALLOWED_WORKFLOWSの元) と resolve_workflow_name() の判定ロジックが
    将来食い違っても、この2つの集合が一致しなくなることで検知できる。
    """
    reachable = {
        resolve_workflow_name(_spec_for(spec.task, frozenset(spec.axes))) for spec in _all_specs()
    }

    assert reachable == set(ALLOWED_WORKFLOWS)
