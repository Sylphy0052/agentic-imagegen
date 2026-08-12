"""Workflowテンプレートの読み込みと、許可済みworkflowの管理。

ComfyUI固有の構造検証と注入処理は adapters.comfyui.workflow へ委譲する。
ここが持つのは「どのworkflowを実行してよいか」というポリシーだけである。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from agentic_imagegen.adapters.comfyui.workflow import (
    TXT2IMG_BINDING,
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
}


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


def prepare_workflow(
    spec: GenerationSpec, *, workflows_dir: Path | None = None
) -> tuple[dict[str, Any], int]:
    """Specから実行可能なWorkflowと、解決済みseedを組み立てる。

    seedが -1 の場合はここでランダム値へ解決し、metadataへ記録できるよう返す。
    """
    template = load_workflow_template(spec.task, workflows_dir=workflows_dir)
    seed = resolve_seed(spec.generation.seed)
    workflow = build_workflow(template, spec, seed=seed, binding=get_binding(spec.task))
    return workflow, seed


__all__ = [
    "ALLOWED_WORKFLOWS",
    "WORKFLOWS_DIR",
    "get_binding",
    "load_workflow_template",
    "prepare_workflow",
]
