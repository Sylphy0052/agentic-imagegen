"""MCP toolとして公開する操作の中身。

MCP層 (mcp_server) は薄いアダプタに留め、ロジックはここへ置く。
CLIと同じ Service / Domain を使うため、経路が違っても検証は同一になる。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic_imagegen.adapters.comfyui.client import ComfyUIClient
from agentic_imagegen.config import Settings
from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.domain.policy import validate_against_limits
from agentic_imagegen.errors import ImageGenError
from agentic_imagegen.services.spec_loader import parse_spec
from agentic_imagegen.workflows.injector import ALLOWED_WORKFLOWS, resolve_workflow_name


def list_workflows() -> list[str]:
    """実行を許可しているWorkflowテンプレート名を返す。"""
    return sorted(ALLOWED_WORKFLOWS)


async def list_models(settings: Settings) -> list[str]:
    """ComfyUIが持っているcheckpoint名を返す。"""
    async with ComfyUIClient(settings) as client:
        return list(await client.available_checkpoints())


async def list_loras(settings: Settings) -> list[str]:
    """ComfyUIが持っているLoRA名を返す。"""
    async with ComfyUIClient(settings) as client:
        return list(await client.available_loras())


def validate_generation(spec: Any, *, settings: Settings, project_root: Path) -> dict[str, Any]:
    """Specを検証し、結果を構造化して返す。

    このtoolの目的は検証結果を得ることなので、不正なSpecでも例外にせず
    `valid: false` と理由を返す。生成は行わない。
    """
    if not isinstance(spec, dict):
        return _failure(["Specはマッピングである必要があります"])

    try:
        parsed = parse_spec(spec, presets_root=settings.presets_root)
        validate_against_limits(parsed, settings)
    except ImageGenError as exc:
        return _failure(str(exc).splitlines())

    return _success(parsed)


def _failure(errors: list[str]) -> dict[str, Any]:
    return {
        "valid": False,
        "errors": [line.strip() for line in errors if line.strip()],
        "workflow": None,
        "resolution": None,
        "checkpoint": None,
        "loras": [],
        "presets": {},
        "prompt": None,
        "source": None,
    }


def _success(spec: GenerationSpec) -> dict[str, Any]:
    params = spec.generation
    # img2imgは入力画像のサイズを使うため、解像度を返しても意味がない
    resolution = (
        None
        if spec.source is not None
        else {
            "width": params.width,
            "height": params.height,
            "batch_size": params.batch_size,
        }
    )

    return {
        "valid": True,
        "errors": [],
        "workflow": resolve_workflow_name(spec),
        "resolution": resolution,
        "checkpoint": spec.model.checkpoint,
        "loras": [lora.model_dump(mode="json") for lora in spec.model.loras],
        "presets": {
            key: value
            for key, value in spec.presets.model_dump(mode="json").items()
            if value is not None
        },
        "prompt": spec.prompt.model_dump(mode="json"),
        "source": spec.source.model_dump(mode="json") if spec.source is not None else None,
    }


__all__ = ["list_loras", "list_models", "list_workflows", "validate_generation"]
