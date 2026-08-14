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
    ALL_BINDINGS,
    WorkflowBinding,
    build_workflow,
    resolve_seed,
)
from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.domain.policy import display_path
from agentic_imagegen.errors import WorkflowValidationError
from agentic_imagegen.workflows.axes import (
    AXIS_CONTROLNET,
    AXIS_CONTROLNET_RAW,
    AXIS_HIRES,
    AXIS_HIRES_MODEL,
    AXIS_IPADAPTER,
    AXIS_LORA,
    AXIS_UNET,
    AXIS_VAE,
    suffix_for,
)

#: リポジトリ同梱のWorkflowテンプレート置き場。
WORKFLOWS_DIR: Final = Path(__file__).resolve().parents[3] / "workflows"

#: 実行を許可するworkflow。ユーザー入力から任意のJSONを実行させないための allowlist。
#: 生成しうるテンプレート名と、それを構成するbindingは
#: `adapters.comfyui.workflow.ALL_BINDINGS` (軸の定義は `workflows.axes`) が
#: 一元管理する。ここはその結果をそのまま許可済み集合として受け取るだけ。
ALLOWED_WORKFLOWS: Final[dict[str, WorkflowBinding]] = dict(ALL_BINDINGS)


def resolve_workflow_name(spec: GenerationSpec) -> str:
    """Specに対して実際に使うWorkflowテンプレート名を決める。

    `task` は論理的なタスク名であり、テンプレートはそれとLoRA指定の有無で決まる。
    LoRA未指定でLoRA用テンプレートを使う意味はないため、素のテンプレートを選ぶ。

    どの軸をどう判定するかはSpecの構造に依存するためここに書く。判定した軸の
    並びからテンプレート名を組み立てる部分は `workflows.axes` (`AXIS_ORDER` /
    `suffix_for()`) と共有し、`ALLOWED_WORKFLOWS` の元になる列挙 (`ALL_BINDINGS`)
    と食い違わないようにする。
    """
    present_axes: list[str] = []
    if spec.model.uses_separate_loaders:
        # UNet / CLIP / VAE を別々に読む形式。ローダーの分割はLoRAより手前の軸なので
        # 接尾辞も先頭へ置く。LoRA / ControlNet / IPAdapter との組み合わせは
        # Specのバリデーションで拒否している
        present_axes.append(AXIS_UNET)
    else:
        if spec.model.uses_external_vae:
            # checkpoint同梱ではなく外部VAEを使う指定。VAELoaderもグラフ上流の
            # ローダー段のため、_unet と同じ位置 (LoRAより手前) へ置く
            present_axes.append(AXIS_VAE)
        if spec.model.loras:
            present_axes.append(AXIS_LORA)
    if spec.generation.upscale is not None:
        # 拡大の経路がlatentとpixelで別テンプレートになる
        present_axes.append(AXIS_HIRES_MODEL if spec.generation.upscale.uses_model else AXIS_HIRES)
    if spec.control is not None:
        # 前処理の有無で Canny ノードの有無が変わるため別テンプレートになる
        present_axes.append(
            AXIS_CONTROLNET_RAW if spec.control.skips_preprocessor else AXIS_CONTROLNET
        )
    if spec.reference is not None:
        present_axes.append(AXIS_IPADAPTER)
    return f"{spec.task}{suffix_for(tuple(present_axes))}"


def get_binding(name: str) -> WorkflowBinding:
    """許可済みworkflowのbindingを返す。未許可なら拒否する。"""
    binding = ALLOWED_WORKFLOWS.get(name)
    if binding is None:
        allowed = " / ".join(sorted(ALLOWED_WORKFLOWS))
        raise WorkflowValidationError(
            f"workflow {name!r} は許可されていません (許可済み: {allowed})"
        )
    return binding


def load_workflow_template(
    name: str, *, workflows_dir: Path | None = None, project_root: Path | None = None
) -> dict[str, Any]:
    """許可済みworkflowのテンプレートJSONを読み込む。

    project_root を渡すと、エラーメッセージへ出すテンプレートの位置を作業ルート
    からの相対パスへ丸める (作業ルートの外を指す場合は絶対パスのまま)。
    """
    binding = get_binding(name)
    directory = workflows_dir if workflows_dir is not None else WORKFLOWS_DIR
    path = directory / f"{binding.name}.json"
    shown = display_path(path, project_root)

    if not path.is_file():
        raise WorkflowValidationError(f"Workflowテンプレートが見つかりません: {shown}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkflowValidationError(
            f"WorkflowテンプレートのJSON解析に失敗しました: {shown}"
        ) from exc
    except OSError as exc:
        raise WorkflowValidationError(f"Workflowテンプレートを読み込めません: {shown}") from exc

    if not isinstance(raw, dict):
        raise WorkflowValidationError(
            f"Workflowテンプレートはマッピングである必要があります: {shown}"
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
    project_root: Path | None = None,
    source_image_name: str | None = None,
    control_image_name: str | None = None,
    reference_image_name: str | None = None,
) -> PreparedWorkflow:
    """Specから実行可能なWorkflowと、解決済みseed・テンプレートのダイジェストを組み立てる。

    seedが -1 の場合はここでランダム値へ解決し、metadataへ記録できるよう返す。
    img2imgでは、ComfyUIへアップロード済みの入力画像名を source_image_name で渡す。
    """
    name = resolve_workflow_name(spec)
    template = load_workflow_template(name, workflows_dir=workflows_dir, project_root=project_root)
    seed = resolve_seed(spec.generation.seed)
    workflow = build_workflow(
        template,
        spec,
        seed=seed,
        binding=get_binding(name),
        source_image_name=source_image_name,
        control_image_name=control_image_name,
        reference_image_name=reference_image_name,
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
