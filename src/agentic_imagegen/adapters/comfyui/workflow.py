"""ComfyUI Workflow (API形式JSON) の構造検証とパラメータ注入。

ComfyUIのNode IDとclass_typeの知識はこのモジュールに閉じ込める。
上位層 (services / CLI) はNode IDを一切知らない。

Workflowテンプレートは人間がComfyUI GUIで作成しAPI形式で書き出したものを使う。
LLMがノードや接続を組み立てることは設計上禁止しており、
ここで行うのは「許可されたパラメータの差し替え」だけである。
"""

from __future__ import annotations

import copy
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from agentic_imagegen.domain.models import MAX_SEED, RANDOM_SEED, GenerationSpec
from agentic_imagegen.errors import WorkflowValidationError


@dataclass(frozen=True, slots=True)
class NodeRef:
    """Workflow内の1ノードへの参照。

    node_idだけでなくclass_typeも保持し、テンプレート差し替え時に
    誤ったノードへ値を書き込むことを防ぐ。
    """

    node_id: str
    class_type: str
    required_inputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LinkRef:
    """あるノードの入力が、どのノードから来ているべきかの期待値。"""

    source_node: str
    input_key: str
    expected_role: str


@dataclass(frozen=True, slots=True)
class WorkflowBinding:
    """WorkflowテンプレートとGenerationSpecの対応関係。"""

    name: str
    nodes: Mapping[str, NodeRef]
    links: tuple[LinkRef, ...]


TXT2IMG_BINDING: Final = WorkflowBinding(
    name="txt2img",
    nodes={
        "checkpoint": NodeRef("4", "CheckpointLoaderSimple", ("ckpt_name",)),
        "positive_prompt": NodeRef("6", "CLIPTextEncode", ("text",)),
        "negative_prompt": NodeRef("7", "CLIPTextEncode", ("text",)),
        "latent": NodeRef("5", "EmptyLatentImage", ("width", "height", "batch_size")),
        "ksampler": NodeRef(
            "3",
            "KSampler",
            ("seed", "steps", "cfg", "sampler_name", "scheduler"),
        ),
        "save_image": NodeRef("9", "SaveImage", ("filename_prefix",)),
    },
    links=(
        LinkRef("ksampler", "positive", "positive_prompt"),
        LinkRef("ksampler", "negative", "negative_prompt"),
        LinkRef("ksampler", "latent_image", "latent"),
        LinkRef("ksampler", "model", "checkpoint"),
    ),
)


#: LoRAスロットの役割名。テンプレートのLoraLoaderの段数と一致させる。
LORA_SLOT_ROLES: Final = ("lora_1", "lora_2", "lora_3")

_LORA_INPUTS: Final = ("lora_name", "strength_model", "strength_clip")


def _lora_binding() -> WorkflowBinding:
    """txt2imgの構成に LoraLoader を3段挟んだbindingを組み立てる。

    checkpoint -> lora_1 -> lora_2 -> lora_3 -> KSampler / CLIPTextEncode の順で
    繋がっていることまで検証する。1段でも迂回していればLoRAが効かないため、
    構造検証で落とす。
    """
    nodes = dict(TXT2IMG_BINDING.nodes)
    for index, role in enumerate(LORA_SLOT_ROLES):
        nodes[role] = NodeRef(str(10 + index), "LoraLoader", _LORA_INPUTS)

    links = [link for link in TXT2IMG_BINDING.links if link.input_key != "model"]
    # LoRAチェーンの接続を先頭から順に検証する
    upstream = "checkpoint"
    for role in LORA_SLOT_ROLES:
        links.append(LinkRef(role, "model", upstream))
        links.append(LinkRef(role, "clip", upstream))
        upstream = role
    links.append(LinkRef("ksampler", "model", upstream))
    links.append(LinkRef("positive_prompt", "clip", upstream))
    links.append(LinkRef("negative_prompt", "clip", upstream))

    return WorkflowBinding(name="txt2img_lora", nodes=nodes, links=tuple(links))


TXT2IMG_LORA_BINDING: Final = _lora_binding()


def resolve_seed(seed: int) -> int:
    """seedが -1 ならランダムな値へ解決する。それ以外はそのまま返す。"""
    if seed != RANDOM_SEED:
        return seed
    return secrets.randbelow(MAX_SEED + 1)


def validate_structure(workflow: object, binding: WorkflowBinding) -> None:
    """Workflowが想定構造どおりかを検証する。

    1つでも不一致があれば WorkflowValidationError を送出し、
    誤ったノードへのパラメータ注入を未然に防ぐ (fail-fast)。
    """
    if not isinstance(workflow, dict):
        raise WorkflowValidationError(
            f"Workflow ({binding.name}) はノードIDをキーとするマッピングである必要があります"
        )

    for role, ref in binding.nodes.items():
        node = workflow.get(ref.node_id)
        if node is None:
            raise WorkflowValidationError(
                f"Workflow ({binding.name}) に {role} のノード {ref.node_id} がありません"
            )
        if not isinstance(node, dict):
            raise WorkflowValidationError(f"ノード {ref.node_id} ({role}) の形式が不正です")
        if node.get("class_type") != ref.class_type:
            raise WorkflowValidationError(
                f"ノード {ref.node_id} ({role}) の class_type が想定と異なります "
                f"(期待: {ref.class_type} / 実際: {node.get('class_type')!r})"
            )
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            raise WorkflowValidationError(f"ノード {ref.node_id} ({role}) に inputs がありません")
        for key in ref.required_inputs:
            if key not in inputs:
                raise WorkflowValidationError(
                    f"ノード {ref.node_id} ({role}) の inputs に {key} がありません"
                )

    _validate_links(workflow, binding)


def _validate_links(workflow: Mapping[str, Any], binding: WorkflowBinding) -> None:
    """ノード間の接続が想定どおりかを検証する。

    positive/negativeの取り違えのように、構造は正しくても意味が入れ替わる
    パターンをここで検出する。
    """
    for link in binding.links:
        source = binding.nodes[link.source_node]
        expected_id = binding.nodes[link.expected_role].node_id
        value = workflow[source.node_id]["inputs"].get(link.input_key)

        if not (isinstance(value, list) and len(value) == 2):
            raise WorkflowValidationError(
                f"ノード {source.node_id} の入力 {link.input_key} が接続形式ではありません"
            )
        if value[0] != expected_id:
            raise WorkflowValidationError(
                f"ノード {source.node_id} の入力 {link.input_key} の接続先が想定と異なります "
                f"(期待: {expected_id} / 実際: {value[0]!r})"
            )


def build_workflow(
    template: Mapping[str, Any],
    spec: GenerationSpec,
    *,
    seed: int,
    binding: WorkflowBinding = TXT2IMG_BINDING,
) -> dict[str, Any]:
    """テンプレートへSpecの値を注入した新しいWorkflowを返す。

    テンプレートは書き換えず、deep copyしたものを返す。
    seedは解決済み (0以上) の値である必要がある。
    """
    if seed < 0:
        raise WorkflowValidationError(
            f"seed は解決済みの0以上の値である必要があります (指定値: {seed})"
        )

    validate_structure(template, binding)

    workflow: dict[str, Any] = copy.deepcopy(dict(template))
    params = spec.generation

    def inputs_of(role: str) -> dict[str, Any]:
        node: dict[str, Any] = workflow[binding.nodes[role].node_id]
        node_inputs: dict[str, Any] = node["inputs"]
        return node_inputs

    inputs_of("checkpoint")["ckpt_name"] = spec.model.checkpoint
    inputs_of("positive_prompt")["text"] = spec.prompt.positive
    inputs_of("negative_prompt")["text"] = spec.prompt.negative

    latent = inputs_of("latent")
    latent["width"] = params.width
    latent["height"] = params.height
    latent["batch_size"] = params.batch_size

    ksampler = inputs_of("ksampler")
    ksampler["seed"] = seed
    ksampler["steps"] = params.steps
    ksampler["cfg"] = params.cfg
    ksampler["sampler_name"] = params.sampler
    ksampler["scheduler"] = params.scheduler

    inputs_of("save_image")["filename_prefix"] = spec.output.prefix

    _inject_loras(spec, binding, inputs_of)

    return workflow


def _inject_loras(
    spec: GenerationSpec,
    binding: WorkflowBinding,
    inputs_of: Callable[[str], dict[str, Any]],
) -> None:
    """LoRAスロットを持つテンプレートへ、指定されたLoRAを順に割り当てる。

    余ったスロットは強度0で無効化する。ComfyUIは lora_name に実在するファイル名を
    要求するため空にはできず、直前のLoRA名を使い回す。
    """
    slots = tuple(role for role in LORA_SLOT_ROLES if role in binding.nodes)
    if not slots:
        return

    loras = spec.model.loras
    if not loras:
        raise WorkflowValidationError(
            f"Workflow ({binding.name}) はLoRA用ですが、Specに model.loras が指定されていません"
        )
    if len(loras) > len(slots):
        raise WorkflowValidationError(
            f"Workflow ({binding.name}) のLoRAスロットは{len(slots)}件ですが、"
            f"{len(loras)}件指定されています"
        )

    for index, role in enumerate(slots):
        node = inputs_of(role)
        if index < len(loras):
            lora = loras[index]
            node["lora_name"] = lora.name
            node["strength_model"] = lora.strength_model
            node["strength_clip"] = lora.strength_clip
        else:
            node["lora_name"] = loras[-1].name
            node["strength_model"] = 0.0
            node["strength_clip"] = 0.0


__all__ = [
    "LORA_SLOT_ROLES",
    "TXT2IMG_BINDING",
    "TXT2IMG_LORA_BINDING",
    "LinkRef",
    "NodeRef",
    "WorkflowBinding",
    "build_workflow",
    "resolve_seed",
    "validate_structure",
]
