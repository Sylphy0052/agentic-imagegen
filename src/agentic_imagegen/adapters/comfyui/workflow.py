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


#: UNet / CLIP / VAE を別々に読むテンプレートの役割名。
UNET_LOADER_ROLE: Final = "unet_loader"
CLIP_LOADER_ROLE: Final = "clip_loader"
VAE_LOADER_ROLE: Final = "vae_loader"

TXT2IMG_UNET_BINDING: Final = WorkflowBinding(
    name="txt2img_unet",
    nodes={
        UNET_LOADER_ROLE: NodeRef("60", "UNETLoader", ("unet_name", "weight_dtype")),
        CLIP_LOADER_ROLE: NodeRef("61", "CLIPLoader", ("clip_name", "type")),
        VAE_LOADER_ROLE: NodeRef("62", "VAELoader", ("vae_name",)),
        "positive_prompt": NodeRef("6", "CLIPTextEncode", ("text",)),
        "negative_prompt": NodeRef("7", "CLIPTextEncode", ("text",)),
        "latent": NodeRef("5", "EmptyLatentImage", ("width", "height", "batch_size")),
        "ksampler": NodeRef(
            "3",
            "KSampler",
            ("seed", "steps", "cfg", "sampler_name", "scheduler"),
        ),
        "vae_decode": NodeRef("8", "VAEDecode", ()),
        "save_image": NodeRef("9", "SaveImage", ("filename_prefix",)),
    },
    links=(
        LinkRef("ksampler", "positive", "positive_prompt"),
        LinkRef("ksampler", "negative", "negative_prompt"),
        LinkRef("ksampler", "latent_image", "latent"),
        LinkRef("ksampler", "model", UNET_LOADER_ROLE),
        # 3つのローダーはそれぞれ別の系統を担うため、取り違えると
        # 「動くが指定したモデルが効かない」状態になる。結線まで検証する。
        LinkRef("positive_prompt", "clip", CLIP_LOADER_ROLE),
        LinkRef("negative_prompt", "clip", CLIP_LOADER_ROLE),
        LinkRef("vae_decode", "vae", VAE_LOADER_ROLE),
    ),
)


#: LoRAスロットの役割名。テンプレートのLoraLoaderの段数と一致させる。
LORA_SLOT_ROLES: Final = ("lora_1", "lora_2", "lora_3")

_LORA_INPUTS: Final = ("lora_name", "strength_model", "strength_clip")


def _with_lora_chain(base: WorkflowBinding, *, name: str, first_node_id: int) -> WorkflowBinding:
    """既存のbindingに LoraLoader 3段を挟んだbindingを組み立てる。

    checkpoint -> lora_1 -> lora_2 -> lora_3 -> KSampler / CLIPTextEncode の順で
    繋がっていることまで検証する。1段でも迂回していればLoRAが効かないため、
    構造検証で落とす。

    ノードIDはテンプレートごとに空き番が違うため first_node_id で受ける
    (img2imgは10/11をLoadImageとVAEEncodeで使っている)。
    """
    nodes = dict(base.nodes)
    for index, role in enumerate(LORA_SLOT_ROLES):
        nodes[role] = NodeRef(str(first_node_id + index), "LoraLoader", _LORA_INPUTS)

    # KSamplerのmodelはLoRAの最終段から来るようになるため、元のリンクを差し替える
    links = [link for link in base.links if link.input_key != "model"]
    upstream = "checkpoint"
    for role in LORA_SLOT_ROLES:
        links.append(LinkRef(role, "model", upstream))
        links.append(LinkRef(role, "clip", upstream))
        upstream = role
    links.append(LinkRef("ksampler", "model", upstream))
    links.append(LinkRef("positive_prompt", "clip", upstream))
    links.append(LinkRef("negative_prompt", "clip", upstream))

    return WorkflowBinding(name=name, nodes=nodes, links=tuple(links))


#: hires fix で使うノードの役割名。
UPSCALE_ROLE: Final = "upscale"
HIRES_KSAMPLER_ROLE: Final = "hires_ksampler"

_HIRES_KSAMPLER_INPUTS: Final = (
    "seed",
    "steps",
    "cfg",
    "sampler_name",
    "scheduler",
    "denoise",
)


def _with_hires_fix(base: WorkflowBinding, *, name: str) -> WorkflowBinding:
    """既存のbindingに latent拡大 + 2段目のKSampler を挟んだbindingを組み立てる。

    1段目のKSamplerから拡大ノードへ、そこから2段目のKSamplerへ、最後にVAEDecodeへ
    繋がっていることまで検証する。どこかが元のままだと拡大が効かないまま
    生成が成功してしまい、気づきにくいため。
    """
    nodes = dict(base.nodes)
    nodes[UPSCALE_ROLE] = NodeRef("30", "LatentUpscaleBy", ("upscale_method", "scale_by"))
    nodes[HIRES_KSAMPLER_ROLE] = NodeRef("31", "KSampler", _HIRES_KSAMPLER_INPUTS)
    nodes["vae_decode"] = NodeRef("8", "VAEDecode", ())

    links = [
        *base.links,
        LinkRef(UPSCALE_ROLE, "samples", "ksampler"),
        LinkRef(HIRES_KSAMPLER_ROLE, "latent_image", UPSCALE_ROLE),
        LinkRef("vae_decode", "samples", HIRES_KSAMPLER_ROLE),
    ]
    return WorkflowBinding(name=name, nodes=nodes, links=tuple(links))


TXT2IMG_LORA_BINDING: Final = _with_lora_chain(
    TXT2IMG_BINDING, name="txt2img_lora", first_node_id=10
)


IMG2IMG_BINDING: Final = WorkflowBinding(
    name="img2img",
    nodes={
        "checkpoint": NodeRef("4", "CheckpointLoaderSimple", ("ckpt_name",)),
        "positive_prompt": NodeRef("6", "CLIPTextEncode", ("text",)),
        "negative_prompt": NodeRef("7", "CLIPTextEncode", ("text",)),
        "source_image": NodeRef("10", "LoadImage", ("image",)),
        "vae_encode": NodeRef("11", "VAEEncode", ()),
        "ksampler": NodeRef(
            "3",
            "KSampler",
            ("seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"),
        ),
        "save_image": NodeRef("9", "SaveImage", ("filename_prefix",)),
    },
    links=(
        LinkRef("ksampler", "positive", "positive_prompt"),
        LinkRef("ksampler", "negative", "negative_prompt"),
        # txt2imgと違い、latentは入力画像をVAEEncodeしたものから来る
        LinkRef("ksampler", "latent_image", "vae_encode"),
        LinkRef("ksampler", "model", "checkpoint"),
        LinkRef("vae_encode", "pixels", "source_image"),
        LinkRef("vae_encode", "vae", "checkpoint"),
    ),
)


#: img2imgにLoRAを重ねた構成。VAEはLoraLoaderを通らないため checkpoint 直結のまま。
IMG2IMG_LORA_BINDING: Final = _with_lora_chain(
    IMG2IMG_BINDING, name="img2img_lora", first_node_id=20
)

#: ControlNet で使うノードの役割名。
CONTROL_IMAGE_ROLE: Final = "control_image"
CONTROL_PREPROCESSOR_ROLE: Final = "control_preprocessor"
CONTROL_LOADER_ROLE: Final = "control_loader"
CONTROL_APPLY_ROLE: Final = "control_apply"


def _with_controlnet(base: WorkflowBinding, *, name: str) -> WorkflowBinding:
    """既存のbindingに ControlNet を挟んだbindingを組み立てる。

    ControlNetApplyAdvanced は positive / negative の両方を返すため、
    KSamplerの2つの入力をどちらもここから受け直す。片方だけ繋ぎ替えると
    条件が食い違ったまま生成が成功してしまう。
    """
    nodes = dict(base.nodes)
    nodes[CONTROL_IMAGE_ROLE] = NodeRef("40", "LoadImage", ("image",))
    nodes[CONTROL_PREPROCESSOR_ROLE] = NodeRef("41", "Canny", ("low_threshold", "high_threshold"))
    nodes[CONTROL_LOADER_ROLE] = NodeRef("42", "ControlNetLoader", ("control_net_name",))
    nodes[CONTROL_APPLY_ROLE] = NodeRef(
        "43",
        "ControlNetApplyAdvanced",
        ("strength", "start_percent", "end_percent"),
    )

    # KSamplerのpositive/negativeはControlNet経由になるため、元のリンクを差し替える
    links = [
        link
        for link in base.links
        if not (link.source_node == "ksampler" and link.input_key in {"positive", "negative"})
    ]
    links.extend(
        [
            LinkRef(CONTROL_PREPROCESSOR_ROLE, "image", CONTROL_IMAGE_ROLE),
            LinkRef(CONTROL_APPLY_ROLE, "control_net", CONTROL_LOADER_ROLE),
            LinkRef(CONTROL_APPLY_ROLE, "image", CONTROL_PREPROCESSOR_ROLE),
            LinkRef(CONTROL_APPLY_ROLE, "positive", "positive_prompt"),
            LinkRef(CONTROL_APPLY_ROLE, "negative", "negative_prompt"),
            LinkRef("ksampler", "positive", CONTROL_APPLY_ROLE),
            LinkRef("ksampler", "negative", CONTROL_APPLY_ROLE),
        ]
    )
    return WorkflowBinding(name=name, nodes=nodes, links=tuple(links))


TXT2IMG_CONTROLNET_BINDING: Final = _with_controlnet(TXT2IMG_BINDING, name="txt2img_controlnet")
TXT2IMG_LORA_CONTROLNET_BINDING: Final = _with_controlnet(
    TXT2IMG_LORA_BINDING, name="txt2img_lora_controlnet"
)
IMG2IMG_CONTROLNET_BINDING: Final = _with_controlnet(IMG2IMG_BINDING, name="img2img_controlnet")
IMG2IMG_LORA_CONTROLNET_BINDING: Final = _with_controlnet(
    IMG2IMG_LORA_BINDING, name="img2img_lora_controlnet"
)

TXT2IMG_HIRES_BINDING: Final = _with_hires_fix(TXT2IMG_BINDING, name="txt2img_hires")
TXT2IMG_LORA_HIRES_BINDING: Final = _with_hires_fix(TXT2IMG_LORA_BINDING, name="txt2img_lora_hires")
IMG2IMG_HIRES_BINDING: Final = _with_hires_fix(IMG2IMG_BINDING, name="img2img_hires")
IMG2IMG_LORA_HIRES_BINDING: Final = _with_hires_fix(IMG2IMG_LORA_BINDING, name="img2img_lora_hires")

#: IPAdapter (reference) 用ノードの役割名。
REFERENCE_IMAGE_ROLE: Final = "reference_image"
REFERENCE_LOADER_ROLE: Final = "reference_loader"
REFERENCE_CLIP_VISION_ROLE: Final = "reference_clip_vision"
REFERENCE_APPLY_ROLE: Final = "reference_apply"


def _with_ipadapter(base: WorkflowBinding, *, name: str) -> WorkflowBinding:
    """既存のbindingに IPAdapter を挟んだbindingを組み立てる。

    IPAdapterAdvanced が返すのは MODEL だけなので、KSamplerのmodel入力だけを
    差し替える。ControlNetは positive / negative を差し替えるため、
    両方を同時にかけても互いに干渉しない。

    KSamplerがIPAdapterを経由せずcheckpoint (またはLoRA末尾) から直接MODELを
    受けていると、参照画像が効かないまま生成が成功してしまう。リンクの期待値を
    置いて構造検証で落とす。
    """
    nodes = dict(base.nodes)
    nodes[REFERENCE_IMAGE_ROLE] = NodeRef("50", "LoadImage", ("image",))
    nodes[REFERENCE_LOADER_ROLE] = NodeRef("51", "IPAdapterModelLoader", ("ipadapter_file",))
    nodes[REFERENCE_CLIP_VISION_ROLE] = NodeRef("52", "CLIPVisionLoader", ("clip_name",))
    nodes[REFERENCE_APPLY_ROLE] = NodeRef(
        "53",
        "IPAdapterAdvanced",
        ("weight", "weight_type", "start_at", "end_at"),
    )

    upstream = [
        link for link in base.links if link.source_node == "ksampler" and link.input_key == "model"
    ]
    if len(upstream) != 1:  # pragma: no cover - bindingの組み立て時にしか起きない
        raise ValueError(f"{name}: KSamplerのmodelリンクを一意に特定できません")

    links = [link for link in base.links if link not in upstream]
    links.extend(
        [
            LinkRef(REFERENCE_APPLY_ROLE, "model", upstream[0].expected_role),
            LinkRef(REFERENCE_APPLY_ROLE, "ipadapter", REFERENCE_LOADER_ROLE),
            LinkRef(REFERENCE_APPLY_ROLE, "image", REFERENCE_IMAGE_ROLE),
            LinkRef(REFERENCE_APPLY_ROLE, "clip_vision", REFERENCE_CLIP_VISION_ROLE),
            LinkRef("ksampler", "model", REFERENCE_APPLY_ROLE),
        ]
    )
    return WorkflowBinding(name=name, nodes=nodes, links=tuple(links))


TXT2IMG_IPADAPTER_BINDING: Final = _with_ipadapter(TXT2IMG_BINDING, name="txt2img_ipadapter")
TXT2IMG_LORA_IPADAPTER_BINDING: Final = _with_ipadapter(
    TXT2IMG_LORA_BINDING, name="txt2img_lora_ipadapter"
)
IMG2IMG_IPADAPTER_BINDING: Final = _with_ipadapter(IMG2IMG_BINDING, name="img2img_ipadapter")
IMG2IMG_LORA_IPADAPTER_BINDING: Final = _with_ipadapter(
    IMG2IMG_LORA_BINDING, name="img2img_lora_ipadapter"
)

TXT2IMG_CONTROLNET_IPADAPTER_BINDING: Final = _with_ipadapter(
    TXT2IMG_CONTROLNET_BINDING, name="txt2img_controlnet_ipadapter"
)
TXT2IMG_LORA_CONTROLNET_IPADAPTER_BINDING: Final = _with_ipadapter(
    TXT2IMG_LORA_CONTROLNET_BINDING, name="txt2img_lora_controlnet_ipadapter"
)
IMG2IMG_CONTROLNET_IPADAPTER_BINDING: Final = _with_ipadapter(
    IMG2IMG_CONTROLNET_BINDING, name="img2img_controlnet_ipadapter"
)
IMG2IMG_LORA_CONTROLNET_IPADAPTER_BINDING: Final = _with_ipadapter(
    IMG2IMG_LORA_CONTROLNET_BINDING, name="img2img_lora_controlnet_ipadapter"
)


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
    source_image_name: str | None = None,
    control_image_name: str | None = None,
    reference_image_name: str | None = None,
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

    _inject_model(spec, binding, inputs_of)
    inputs_of("positive_prompt")["text"] = spec.prompt.positive
    inputs_of("negative_prompt")["text"] = spec.prompt.negative

    # img2imgは入力画像のサイズをそのまま使うため、latentノードを持たない
    if "latent" in binding.nodes:
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
    _inject_source_image(spec, binding, inputs_of, source_image_name)
    _inject_upscale(spec, binding, inputs_of, seed=seed)
    _inject_controlnet(spec, binding, inputs_of, control_image_name)
    _inject_reference(spec, binding, inputs_of, reference_image_name)

    return workflow


def _inject_model(
    spec: GenerationSpec,
    binding: WorkflowBinding,
    inputs_of: Callable[[str], dict[str, Any]],
) -> None:
    """テンプレートの形式に合わせてモデルのファイル名を注入する。

    checkpoint 1ファイルのテンプレートと、UNet / CLIP / VAE を別々に読む
    テンプレートで、Spec側の指定の仕方も変わる。食い違ったまま投入すると
    ComfyUI側で分かりにくい失敗になるため、ここで拒否する。
    """
    if UNET_LOADER_ROLE in binding.nodes:
        model = spec.model
        if not model.uses_separate_loaders:
            raise WorkflowValidationError(
                f"Workflow ({binding.name}) は unet / clip / vae の指定を要求しますが、"
                "Specでは checkpoint が指定されています"
            )
        inputs_of(UNET_LOADER_ROLE)["unet_name"] = model.unet
        inputs_of(CLIP_LOADER_ROLE)["clip_name"] = model.clip
        inputs_of(VAE_LOADER_ROLE)["vae_name"] = model.vae
        return

    if spec.model.checkpoint is None:
        raise WorkflowValidationError(
            f"Workflow ({binding.name}) は checkpoint の指定を要求しますが、"
            "Specでは unet / clip / vae が指定されています"
        )
    inputs_of("checkpoint")["ckpt_name"] = spec.model.checkpoint


def _inject_reference(
    spec: GenerationSpec,
    binding: WorkflowBinding,
    inputs_of: Callable[[str], dict[str, Any]],
    reference_image_name: str | None,
) -> None:
    """IPAdapter用テンプレートへ参照画像と各パラメータを注入する。"""
    if REFERENCE_APPLY_ROLE not in binding.nodes:
        return

    reference = spec.reference
    if reference is None:
        raise WorkflowValidationError(
            f"Workflow ({binding.name}) はIPAdapter用ですが、Specに reference が指定されていません"
        )
    if not reference_image_name:
        raise WorkflowValidationError(
            f"Workflow ({binding.name}) の参照画像がComfyUIへアップロードされていません"
        )

    inputs_of(REFERENCE_IMAGE_ROLE)["image"] = reference_image_name
    inputs_of(REFERENCE_LOADER_ROLE)["ipadapter_file"] = reference.model
    inputs_of(REFERENCE_CLIP_VISION_ROLE)["clip_name"] = reference.clip_vision

    apply_node = inputs_of(REFERENCE_APPLY_ROLE)
    apply_node["weight"] = reference.weight
    apply_node["weight_type"] = reference.weight_type
    # ノード側の名前は start_at / end_at。Spec側はControlSpecと揃えて percent にしている
    apply_node["start_at"] = reference.start_percent
    apply_node["end_at"] = reference.end_percent


def _inject_controlnet(
    spec: GenerationSpec,
    binding: WorkflowBinding,
    inputs_of: Callable[[str], dict[str, Any]],
    control_image_name: str | None,
) -> None:
    """ControlNet用テンプレートへ control画像と各パラメータを注入する。"""
    if CONTROL_APPLY_ROLE not in binding.nodes:
        return

    control = spec.control
    if control is None:
        raise WorkflowValidationError(
            f"Workflow ({binding.name}) はControlNet用ですが、Specに control が指定されていません"
        )
    if not control_image_name:
        raise WorkflowValidationError(
            f"Workflow ({binding.name}) のcontrol画像がComfyUIへアップロードされていません"
        )

    inputs_of(CONTROL_IMAGE_ROLE)["image"] = control_image_name
    inputs_of(CONTROL_LOADER_ROLE)["control_net_name"] = control.model

    preprocessor = inputs_of(CONTROL_PREPROCESSOR_ROLE)
    preprocessor["low_threshold"] = control.low_threshold
    preprocessor["high_threshold"] = control.high_threshold

    apply_node = inputs_of(CONTROL_APPLY_ROLE)
    apply_node["strength"] = control.strength
    apply_node["start_percent"] = control.start_percent
    apply_node["end_percent"] = control.end_percent


def _inject_upscale(
    spec: GenerationSpec,
    binding: WorkflowBinding,
    inputs_of: Callable[[str], dict[str, Any]],
    *,
    seed: int,
) -> None:
    """hires fix 用テンプレートへ拡大倍率と2段目の設定を注入する。"""
    if UPSCALE_ROLE not in binding.nodes:
        return

    upscale = spec.generation.upscale
    if upscale is None:
        raise WorkflowValidationError(
            f"Workflow ({binding.name}) はhires fix用ですが、"
            "Specに generation.upscale が指定されていません"
        )

    node = inputs_of(UPSCALE_ROLE)
    node["scale_by"] = upscale.scale
    node["upscale_method"] = upscale.method

    params = spec.generation
    second = inputs_of(HIRES_KSAMPLER_ROLE)
    # 2段目は同じseedを使う。変えると1段目の絵から離れてしまう
    second["seed"] = seed
    second["steps"] = upscale.effective_steps(params.steps)
    second["cfg"] = params.cfg
    second["sampler_name"] = params.sampler
    second["scheduler"] = params.scheduler
    second["denoise"] = upscale.denoise


def _inject_source_image(
    spec: GenerationSpec,
    binding: WorkflowBinding,
    inputs_of: Callable[[str], dict[str, Any]],
    source_image_name: str | None,
) -> None:
    """img2img用テンプレートへ入力画像とdenoiseを注入する。

    LoadImageが参照するのはComfyUIのinput配下の名前であり、Specに書かれた
    リポジトリ内のパスではない。アップロード済みの名前を受け取る前提とする。
    """
    if "source_image" not in binding.nodes:
        return

    if spec.source is None:
        raise WorkflowValidationError(
            f"Workflow ({binding.name}) はimg2img用ですが、Specに source が指定されていません"
        )
    if not source_image_name:
        raise WorkflowValidationError(
            f"Workflow ({binding.name}) の入力画像がComfyUIへアップロードされていません"
        )

    inputs_of("source_image")["image"] = source_image_name
    inputs_of("ksampler")["denoise"] = spec.source.denoise


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
    "CONTROL_APPLY_ROLE",
    "CONTROL_IMAGE_ROLE",
    "CONTROL_LOADER_ROLE",
    "CONTROL_PREPROCESSOR_ROLE",
    "HIRES_KSAMPLER_ROLE",
    "IMG2IMG_BINDING",
    "IMG2IMG_CONTROLNET_BINDING",
    "IMG2IMG_CONTROLNET_IPADAPTER_BINDING",
    "IMG2IMG_HIRES_BINDING",
    "IMG2IMG_IPADAPTER_BINDING",
    "IMG2IMG_LORA_BINDING",
    "IMG2IMG_LORA_CONTROLNET_BINDING",
    "IMG2IMG_LORA_CONTROLNET_IPADAPTER_BINDING",
    "IMG2IMG_LORA_HIRES_BINDING",
    "IMG2IMG_LORA_IPADAPTER_BINDING",
    "LORA_SLOT_ROLES",
    "REFERENCE_APPLY_ROLE",
    "REFERENCE_CLIP_VISION_ROLE",
    "REFERENCE_IMAGE_ROLE",
    "REFERENCE_LOADER_ROLE",
    "TXT2IMG_BINDING",
    "TXT2IMG_CONTROLNET_BINDING",
    "TXT2IMG_CONTROLNET_IPADAPTER_BINDING",
    "TXT2IMG_HIRES_BINDING",
    "TXT2IMG_IPADAPTER_BINDING",
    "TXT2IMG_LORA_BINDING",
    "TXT2IMG_LORA_CONTROLNET_BINDING",
    "TXT2IMG_LORA_CONTROLNET_IPADAPTER_BINDING",
    "TXT2IMG_LORA_HIRES_BINDING",
    "TXT2IMG_LORA_IPADAPTER_BINDING",
    "UPSCALE_ROLE",
    "LinkRef",
    "NodeRef",
    "WorkflowBinding",
    "build_workflow",
    "resolve_seed",
    "validate_structure",
]
