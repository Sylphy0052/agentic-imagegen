"""Workflowテンプレートの読み込みと、許可済みworkflowの管理。

ComfyUI固有の構造検証と注入処理は adapters.comfyui.workflow へ委譲する。
ここが持つのは「どのworkflowを実行してよいか」というポリシーだけである。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from agentic_imagegen.adapters.comfyui.workflow import (
    IMG2IMG_BINDING,
    IMG2IMG_CONTROLNET_BINDING,
    IMG2IMG_HIRES_BINDING,
    IMG2IMG_LORA_BINDING,
    IMG2IMG_LORA_CONTROLNET_BINDING,
    IMG2IMG_LORA_HIRES_BINDING,
    TXT2IMG_BINDING,
    TXT2IMG_CONTROLNET_BINDING,
    TXT2IMG_HIRES_BINDING,
    TXT2IMG_LORA_BINDING,
    TXT2IMG_LORA_CONTROLNET_BINDING,
    TXT2IMG_LORA_HIRES_BINDING,
    WorkflowBinding,
    build_workflow,
    resolve_seed,
)
from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.errors import WorkflowValidationError

#: リポジトリ同梱のWorkflowテンプレート置き場。
WORKFLOWS_DIR: Final = Path(__file__).resolve().parents[3] / "workflows"

#: 実行を許可するworkflow。ユーザー入力から任意のJSONを実行させないための allowlist。
ALLOWED_WORKFLOWS: Final[dict[str, WorkflowBinding]] = {
    "txt2img": TXT2IMG_BINDING,
    "txt2img_lora": TXT2IMG_LORA_BINDING,
    "img2img": IMG2IMG_BINDING,
    "img2img_lora": IMG2IMG_LORA_BINDING,
    "txt2img_hires": TXT2IMG_HIRES_BINDING,
    "txt2img_lora_hires": TXT2IMG_LORA_HIRES_BINDING,
    "img2img_hires": IMG2IMG_HIRES_BINDING,
    "img2img_lora_hires": IMG2IMG_LORA_HIRES_BINDING,
    "txt2img_controlnet": TXT2IMG_CONTROLNET_BINDING,
    "txt2img_lora_controlnet": TXT2IMG_LORA_CONTROLNET_BINDING,
    "img2img_controlnet": IMG2IMG_CONTROLNET_BINDING,
    "img2img_lora_controlnet": IMG2IMG_LORA_CONTROLNET_BINDING,
}


def resolve_workflow_name(spec: GenerationSpec) -> str:
    """Specに対して実際に使うWorkflowテンプレート名を決める。

    `task` は論理的なタスク名であり、テンプレートはそれとLoRA指定の有無で決まる。
    LoRA未指定でLoRA用テンプレートを使う意味はないため、素のテンプレートを選ぶ。
    """
    suffix = ""
    if spec.model.loras:
        suffix += "_lora"
    if spec.generation.upscale is not None:
        suffix += "_hires"
    if spec.control is not None:
        suffix += "_controlnet"
    return f"{spec.task}{suffix}"


def get_binding(name: str) -> WorkflowBinding:
    """許可済みworkflowのbindingを返す。未許可なら拒否する。"""
    binding = ALLOWED_WORKFLOWS.get(name)
    if binding is None:
        allowed = " / ".join(sorted(ALLOWED_WORKFLOWS))
        raise WorkflowValidationError(
            f"workflow {name!r} は許可されていません (許可済み: {allowed})"
        )
    return binding


def load_workflow_template(name: str, *, workflows_dir: Path | None = None) -> dict[str, Any]:
    """許可済みworkflowのテンプレートJSONを読み込む。"""
    binding = get_binding(name)
    directory = workflows_dir if workflows_dir is not None else WORKFLOWS_DIR
    path = directory / f"{binding.name}.json"

    if not path.is_file():
        raise WorkflowValidationError(f"Workflowテンプレートが見つかりません: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkflowValidationError(
            f"WorkflowテンプレートのJSON解析に失敗しました: {path}"
        ) from exc
    except OSError as exc:
        raise WorkflowValidationError(f"Workflowテンプレートを読み込めません: {path}") from exc

    if not isinstance(raw, dict):
        raise WorkflowValidationError(
            f"Workflowテンプレートはマッピングである必要があります: {path}"
        )
    template: dict[str, Any] = raw
    return template


def template_digest(template: dict[str, Any]) -> str:
    """テンプレート内容のダイジェスト。

    ファイルのバイト列ではなく正規化したJSONから取る。インデントや鍵の順序が
    変わっただけの差分でハッシュが動かないようにするため。
    """
    canonical = json.dumps(template, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedWorkflow:
    """実行可能な状態まで組み立てたWorkflow。

    seedとテンプレートのダイジェストは、生成後にmetadataへ記録して
    同じ結果を再現できるようにするために持ち回る。
    """

    workflow: dict[str, Any]
    seed: int
    template_hash: str
    workflow_name: str


def prepare_workflow(
    spec: GenerationSpec,
    *,
    workflows_dir: Path | None = None,
    source_image_name: str | None = None,
    control_image_name: str | None = None,
) -> PreparedWorkflow:
    """Specから実行可能なWorkflowと、解決済みseed・テンプレートのダイジェストを組み立てる。

    seedが -1 の場合はここでランダム値へ解決し、metadataへ記録できるよう返す。
    img2imgでは、ComfyUIへアップロード済みの入力画像名を source_image_name で渡す。
    """
    name = resolve_workflow_name(spec)
    template = load_workflow_template(name, workflows_dir=workflows_dir)
    seed = resolve_seed(spec.generation.seed)
    workflow = build_workflow(
        template,
        spec,
        seed=seed,
        binding=get_binding(name),
        source_image_name=source_image_name,
        control_image_name=control_image_name,
    )
    return PreparedWorkflow(
        workflow=workflow,
        seed=seed,
        template_hash=template_digest(template),
        workflow_name=name,
    )


__all__ = [
    "ALLOWED_WORKFLOWS",
    "WORKFLOWS_DIR",
    "PreparedWorkflow",
    "get_binding",
    "load_workflow_template",
    "prepare_workflow",
    "template_digest",
]
