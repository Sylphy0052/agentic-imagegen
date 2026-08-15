"""ComfyUI Workflow (API形式JSON) の構造検証とパラメータ注入。

ComfyUIのNode IDとclass_typeの知識はこのモジュールに閉じ込める。
上位層 (services / CLI) はNode IDを一切知らない。

Workflowテンプレートは人間がComfyUI GUIで作成しAPI形式で書き出したものを使う。
LLMがノードや接続を組み立てることは設計上禁止しており、
ここで行うのは「許可されたパラメータの差し替え」だけである。
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.domain.models import resolve_seed as _resolve_seed
from agentic_imagegen.errors import WorkflowValidationError
from agentic_imagegen.workflows import axes


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


#: clip skip (CLIPSetLastLayer) の役割名。全テンプレートへ無条件に挿入する。
CLIP_SKIP_ROLE: Final = "clip_skip"

TXT2IMG_BINDING: Final = WorkflowBinding(
    name="txt2img",
    nodes={
        "checkpoint": NodeRef("4", "CheckpointLoaderSimple", ("ckpt_name",)),
        CLIP_SKIP_ROLE: NodeRef("70", "CLIPSetLastLayer", ("stop_at_clip_layer",)),
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
        # CLIPTextEncodeはCLIPの供給元へ直結せず、必ずCLIPSetLastLayer経由で受ける。
        LinkRef("positive_prompt", "clip", CLIP_SKIP_ROLE),
        LinkRef("negative_prompt", "clip", CLIP_SKIP_ROLE),
        LinkRef(CLIP_SKIP_ROLE, "clip", "checkpoint"),
    ),
)


#: UNet / CLIP / VAE を別々に読むテンプレートの役割名。
UNET_LOADER_ROLE: Final = "unet_loader"
CLIP_LOADER_ROLE: Final = "clip_loader"
VAE_LOADER_ROLE: Final = "vae_loader"


def _to_separate_loaders(base: WorkflowBinding, *, name: str) -> WorkflowBinding:
    """CheckpointLoaderSimple を UNet / CLIP / VAE の3ローダーへ分けたbindingを組み立てる。

    3つのローダーはそれぞれ別の系統を担うため、取り違えると
    「動くが指定したモデルが効かない」状態になる。結線まで検証する。
    """
    nodes = {role: node for role, node in base.nodes.items() if role != "checkpoint"}
    nodes[UNET_LOADER_ROLE] = NodeRef("60", "UNETLoader", ("unet_name", "weight_dtype"))
    nodes[CLIP_LOADER_ROLE] = NodeRef("61", "CLIPLoader", ("clip_name", "type"))
    nodes[VAE_LOADER_ROLE] = NodeRef("62", "VAELoader", ("vae_name",))
    nodes["vae_decode"] = NodeRef("8", "VAEDecode", ())

    links = [link for link in base.links if link.expected_role != "checkpoint"]
    links += [
        LinkRef("ksampler", "model", UNET_LOADER_ROLE),
        # CLIPTextEncodeは変わらずclip_skip経由。clip_skipの供給元だけ差し替える。
        LinkRef(CLIP_SKIP_ROLE, "clip", CLIP_LOADER_ROLE),
        LinkRef("vae_decode", "vae", VAE_LOADER_ROLE),
    ]
    if "vae_encode" in nodes:
        # img2imgでは入力画像をVAEEncodeする側も同じVAEを見る
        links.append(LinkRef("vae_encode", "vae", VAE_LOADER_ROLE))
    return WorkflowBinding(name=name, nodes=nodes, links=tuple(links))


def _with_external_vae(base: WorkflowBinding, *, name: str) -> WorkflowBinding:
    """既存のbindingへ 外部VAE (VAELoader) を追加し、VAEの参照元を差し替える。

    checkpoint同梱のVAEではなく、色褪せ・眠い線を避けるために使う外部VAE
    (vae-ft-mse-840000 / klF8Anime2VAE など) へ差し替える版を組み立てる。
    DiT系は既に独自の VAELoader ルートを持つため、この関数は checkpoint系の
    bindingにのみ適用する。

    「vaeの入力がcheckpointを指す」と明示されているリンク (img2imgの
    vae_encode、hires (model) 版が増やすVAEDecode / VAEEncodeなど) をまとめて
    VAELoaderへ差し替える。最終段のVAEDecode (node 8) は checkpoint系では
    そもそもリンクとして明示されていない (VAEはLoraLoaderを通らずcheckpoint
    直結のままであることを前提に検証していなかったため) ので、常に差し替え
    対象へ加える。取りこぼすと、そのノードだけ同梱VAEを見たまま残ってしまう。
    """
    nodes = dict(base.nodes)
    nodes[VAE_LOADER_ROLE] = NodeRef("80", "VAELoader", ("vae_name",))
    nodes["vae_decode"] = NodeRef("8", "VAEDecode", ())

    replaced_sources = {
        link.source_node
        for link in base.links
        if link.input_key == "vae" and link.expected_role == "checkpoint"
    }
    replaced_sources.add("vae_decode")

    links = [
        link
        for link in base.links
        if not (link.input_key == "vae" and link.expected_role == "checkpoint")
    ]
    links.extend(LinkRef(source, "vae", VAE_LOADER_ROLE) for source in sorted(replaced_sources))
    return WorkflowBinding(name=name, nodes=nodes, links=tuple(links))


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

    # KSamplerのmodelとclip_skipの供給元はLoRAの最終段から来るようになるため、
    # 元のリンクを差し替える。CLIPTextEncodeは変わらずclip_skip経由のまま。
    links = [
        link
        for link in base.links
        if link.input_key != "model"
        and not (link.source_node == CLIP_SKIP_ROLE and link.input_key == "clip")
    ]
    upstream = "checkpoint"
    for role in LORA_SLOT_ROLES:
        links.append(LinkRef(role, "model", upstream))
        links.append(LinkRef(role, "clip", upstream))
        upstream = role
    links.append(LinkRef("ksampler", "model", upstream))
    links.append(LinkRef(CLIP_SKIP_ROLE, "clip", upstream))

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
    # 2段目のMODELは1段目と同じ供給元から来る。ここを検証しないと、ローダーを
    # 差し替えた構成 (LoRAチェーン / DiT系の3ローダー) で2段目だけが元の
    # CheckpointLoaderを見たまま残っていても気づけない
    model_source = next(
        (
            link.expected_role
            for link in base.links
            if link.source_node == "ksampler" and link.input_key == "model"
        ),
        None,
    )
    if model_source is not None:
        links.append(LinkRef(HIRES_KSAMPLER_ROLE, "model", model_source))
    return WorkflowBinding(name=name, nodes=nodes, links=tuple(links))


#: アップスケールモデルを使うhires fixで増えるノードの役割名。
UPSCALE_MODEL_LOADER_ROLE: Final = "upscale_model_loader"
UPSCALE_MODEL_DECODE_ROLE: Final = "upscale_model_decode"
UPSCALE_MODEL_APPLY_ROLE: Final = "upscale_model_apply"
UPSCALE_MODEL_RESIZE_ROLE: Final = "upscale_model_resize"
UPSCALE_MODEL_ENCODE_ROLE: Final = "upscale_model_encode"


def _with_hires_model_fix(base: WorkflowBinding, *, name: str) -> WorkflowBinding:
    """既存のbindingに アップスケールモデルでの拡大 + 2段目のKSampler を挟む。

    latent拡大との違いは経路だけで、拡大の前後でpixelへ戻して符号化し直す。

        ksampler -> decode -> apply(model) -> resize -> encode -> hires_ksampler

    途中のどれか1つでも元の繋がりが残っていると、拡大が効かないまま生成が
    成功してしまう。VAEの供給元まで含めて結線を検証する。

    VAEの出どころはcheckpointに同梱される場合 (checkpoint系) と単体のVAELoader
    から来る場合 (DiT系) があり、どちらを見るかで拡大前後の符号化が変わる。
    """
    nodes = dict(base.nodes)
    nodes[UPSCALE_MODEL_DECODE_ROLE] = NodeRef("32", "VAEDecode", ())
    nodes[UPSCALE_MODEL_LOADER_ROLE] = NodeRef("33", "UpscaleModelLoader", ("model_name",))
    nodes[UPSCALE_MODEL_APPLY_ROLE] = NodeRef("34", "ImageUpscaleWithModel", ())
    nodes[UPSCALE_MODEL_RESIZE_ROLE] = NodeRef("35", "ImageScaleBy", ("upscale_method", "scale_by"))
    nodes[UPSCALE_MODEL_ENCODE_ROLE] = NodeRef("36", "VAEEncode", ())
    nodes[HIRES_KSAMPLER_ROLE] = NodeRef("31", "KSampler", _HIRES_KSAMPLER_INPUTS)
    nodes["vae_decode"] = NodeRef("8", "VAEDecode", ())

    links = [
        *base.links,
        LinkRef(UPSCALE_MODEL_DECODE_ROLE, "samples", "ksampler"),
        LinkRef(UPSCALE_MODEL_APPLY_ROLE, "image", UPSCALE_MODEL_DECODE_ROLE),
        LinkRef(UPSCALE_MODEL_APPLY_ROLE, "upscale_model", UPSCALE_MODEL_LOADER_ROLE),
        LinkRef(UPSCALE_MODEL_RESIZE_ROLE, "image", UPSCALE_MODEL_APPLY_ROLE),
        LinkRef(UPSCALE_MODEL_ENCODE_ROLE, "pixels", UPSCALE_MODEL_RESIZE_ROLE),
        LinkRef(HIRES_KSAMPLER_ROLE, "latent_image", UPSCALE_MODEL_ENCODE_ROLE),
        LinkRef("vae_decode", "samples", HIRES_KSAMPLER_ROLE),
    ]
    # 増やしたVAEDecode / VAEEncodeは最終段のVAEDecodeと同じVAEを見る。
    # DiT系ではVAELoaderが別にあるため、供給元を追随させないと取り違える
    vae_source = next(
        (
            link.expected_role
            for link in base.links
            if link.source_node == "vae_decode" and link.input_key == "vae"
        ),
        None,
    )
    if vae_source is None and "checkpoint" in nodes:
        # checkpoint系はVAEがcheckpointへ同梱されるため、最終段のVAEDecodeに
        # 供給元のLinkRefが無い。それでも出どころは一意に決まる
        vae_source = "checkpoint"
    if vae_source is not None:
        links += [
            LinkRef(UPSCALE_MODEL_DECODE_ROLE, "vae", vae_source),
            LinkRef(UPSCALE_MODEL_ENCODE_ROLE, "vae", vae_source),
        ]
    # 2段目のMODELの供給元は latent拡大版と同じ理由で検証する
    model_source = next(
        (
            link.expected_role
            for link in base.links
            if link.source_node == "ksampler" and link.input_key == "model"
        ),
        None,
    )
    if model_source is not None:
        links.append(LinkRef(HIRES_KSAMPLER_ROLE, "model", model_source))
    return WorkflowBinding(name=name, nodes=nodes, links=tuple(links))


IMG2IMG_BINDING: Final = WorkflowBinding(
    name="img2img",
    nodes={
        "checkpoint": NodeRef("4", "CheckpointLoaderSimple", ("ckpt_name",)),
        CLIP_SKIP_ROLE: NodeRef("70", "CLIPSetLastLayer", ("stop_at_clip_layer",)),
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
        # CLIPTextEncodeはCLIPの供給元へ直結せず、必ずCLIPSetLastLayer経由で受ける。
        LinkRef("positive_prompt", "clip", CLIP_SKIP_ROLE),
        LinkRef("negative_prompt", "clip", CLIP_SKIP_ROLE),
        LinkRef(CLIP_SKIP_ROLE, "clip", "checkpoint"),
    ),
)


#: ControlNet で使うノードの役割名。
CONTROL_IMAGE_ROLE: Final = "control_image"
CONTROL_PREPROCESSOR_ROLE: Final = "control_preprocessor"
CONTROL_LOADER_ROLE: Final = "control_loader"
CONTROL_APPLY_ROLE: Final = "control_apply"


def _build_controlnet_binding(
    base: WorkflowBinding, *, name: str, preprocess: bool
) -> WorkflowBinding:
    """既存のbindingに ControlNet を挟んだbindingを組み立てる。

    ControlNetApplyAdvanced は positive / negative の両方を返すため、
    KSamplerの2つの入力をどちらもここから受け直す。片方だけ繋ぎ替えると
    条件が食い違ったまま生成が成功してしまう。

    `preprocess` が False のときは Canny を挟まず、LoadImage をそのまま
    ControlNetApplyAdvanced へ繋ぐ (前処理済みの制御画像を渡す経路)。
    """
    nodes = dict(base.nodes)
    nodes[CONTROL_IMAGE_ROLE] = NodeRef("40", "LoadImage", ("image",))
    if preprocess:
        nodes[CONTROL_PREPROCESSOR_ROLE] = NodeRef(
            "41", "Canny", ("low_threshold", "high_threshold")
        )
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
    if preprocess:
        links.append(LinkRef(CONTROL_PREPROCESSOR_ROLE, "image", CONTROL_IMAGE_ROLE))
    links.extend(
        [
            LinkRef(CONTROL_APPLY_ROLE, "control_net", CONTROL_LOADER_ROLE),
            LinkRef(
                CONTROL_APPLY_ROLE,
                "image",
                CONTROL_PREPROCESSOR_ROLE if preprocess else CONTROL_IMAGE_ROLE,
            ),
            LinkRef(CONTROL_APPLY_ROLE, "positive", "positive_prompt"),
            LinkRef(CONTROL_APPLY_ROLE, "negative", "negative_prompt"),
            LinkRef("ksampler", "positive", CONTROL_APPLY_ROLE),
            LinkRef("ksampler", "negative", CONTROL_APPLY_ROLE),
        ]
    )
    return WorkflowBinding(name=name, nodes=nodes, links=tuple(links))


def _with_controlnet(base: WorkflowBinding, *, name: str) -> WorkflowBinding:
    """制御画像を Canny で線画へ変換してから ControlNet へ渡すbinding。"""
    return _build_controlnet_binding(base, name=name, preprocess=True)


def _with_controlnet_raw(base: WorkflowBinding, *, name: str) -> WorkflowBinding:
    """前処理済みの制御画像をそのまま ControlNet へ渡すbinding。"""
    return _build_controlnet_binding(base, name=name, preprocess=False)


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


#: LoRAのノードID帯はtaskごとに空き番が違う (img2imgは10/11をLoadImageと
#: VAEEncodeで使っている)ため、他の軸と違いtask別のテーブルを引く。
_LORA_FIRST_NODE_ID: Final[dict[axes.Task, int]] = {"txt2img": 10, "img2img": 20}

#: 軸ごとのbinding合成関数。`_build_all_bindings()` が `axes.iter_template_specs()`
#: の列挙から軸の並びを受け取り、ここを引いて順に適用する。
#: `lora` だけはtaskごとにノードID帯が違うため、ここには登録せず
#: `_build_all_bindings()` 側で個別に呼ぶ。
#: 軸を1本足したら、対応する `_with_*` 関数をここへ登録する。
AXIS_BINDING_BUILDERS: Final[dict[str, Callable[..., WorkflowBinding]]] = {
    axes.AXIS_UNET: _to_separate_loaders,
    axes.AXIS_VAE: _with_external_vae,
    axes.AXIS_HIRES: _with_hires_fix,
    axes.AXIS_HIRES_MODEL: _with_hires_model_fix,
    axes.AXIS_CONTROLNET: _with_controlnet,
    axes.AXIS_CONTROLNET_RAW: _with_controlnet_raw,
    axes.AXIS_IPADAPTER: _with_ipadapter,
}

_BASE_BINDINGS: Final[dict[axes.Task, WorkflowBinding]] = {
    "txt2img": TXT2IMG_BINDING,
    "img2img": IMG2IMG_BINDING,
}


def _build_all_bindings() -> dict[str, WorkflowBinding]:
    """`axes.iter_template_specs()` の列挙から、生成しうる全bindingを組み立てる。

    合成順は `axes.axes_in_build_order()` に従う。テンプレート名の接尾辞順
    (`axes.AXIS_ORDER`) とは vae の位置だけ異なることに注意する
    (vaeは常に最後に適用する。理由は `axes.AXIS_BUILD_ORDER` のコメントを参照)。
    """
    bindings: dict[str, WorkflowBinding] = {"txt2img": TXT2IMG_BINDING}
    for spec in axes.iter_template_specs():
        binding = _BASE_BINDINGS[spec.task]
        for axis in axes.axes_in_build_order(spec.axes):
            if axis == axes.AXIS_LORA:
                binding = _with_lora_chain(
                    binding, name=spec.name, first_node_id=_LORA_FIRST_NODE_ID[spec.task]
                )
            else:
                binding = AXIS_BINDING_BUILDERS[axis](binding, name=spec.name)
        bindings[spec.name] = binding
    return bindings


#: 生成しうる全テンプレートのbinding。`workflows/injector.py` の `ALLOWED_WORKFLOWS` は
#: この辞書から作る。
ALL_BINDINGS: Final[dict[str, WorkflowBinding]] = _build_all_bindings()

#: 個別のテストから直接参照される代表的なbinding。値は `ALL_BINDINGS` の該当要素と
#: 同一であり、ここでの別名づけは後方互換のためだけに存在する。
TXT2IMG_LORA_BINDING: Final = ALL_BINDINGS["txt2img_lora"]
TXT2IMG_HIRES_BINDING: Final = ALL_BINDINGS["txt2img_hires"]
TXT2IMG_HIRES_MODEL_BINDING: Final = ALL_BINDINGS["txt2img_hires_model"]
TXT2IMG_UNET_BINDING: Final = ALL_BINDINGS["txt2img_unet"]
TXT2IMG_CONTROLNET_BINDING: Final = ALL_BINDINGS["txt2img_controlnet"]
TXT2IMG_IPADAPTER_BINDING: Final = ALL_BINDINGS["txt2img_ipadapter"]
TXT2IMG_VAE_BINDING: Final = ALL_BINDINGS["txt2img_vae"]
TXT2IMG_VAE_LORA_BINDING: Final = ALL_BINDINGS["txt2img_vae_lora"]
TXT2IMG_VAE_HIRES_MODEL_BINDING: Final = ALL_BINDINGS["txt2img_vae_hires_model"]
IMG2IMG_LORA_BINDING: Final = ALL_BINDINGS["img2img_lora"]
IMG2IMG_VAE_BINDING: Final = ALL_BINDINGS["img2img_vae"]


#: seedの解決はバックエンド非依存の取り決めのため domain 側にある。
#: ここからの再輸出は既存のimport経路を保つためのもの。
resolve_seed = _resolve_seed


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
    _inject_clip_skip(spec, binding, inputs_of)
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

    外部VAE (checkpoint + vae) 用のテンプレートは checkpoint に加えて
    VAELoaderへの注入も行う。DiT系のVAELoader注入 (上のブロック) とは
    `UNET_LOADER_ROLE in binding.nodes` の分岐で棲み分けており、
    checkpoint系のVAELoaderがこの分岐へ紛れ込むことはない。
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

    if VAE_LOADER_ROLE in binding.nodes:
        if spec.model.vae is None:
            raise WorkflowValidationError(
                f"Workflow ({binding.name}) は外部VAE (model.vae) の指定を要求しますが、"
                "Specに指定されていません"
            )
        inputs_of(VAE_LOADER_ROLE)["vae_name"] = spec.model.vae


def _inject_clip_skip(
    spec: GenerationSpec,
    binding: WorkflowBinding,
    inputs_of: Callable[[str], dict[str, Any]],
) -> None:
    """CLIPSetLastLayerへclip skipの値を注入する。

    未指定 (None) の場合はテンプレートの既定値 (stop_at_clip_layer=-1、ComfyUI既定と
    同値の素通し) をそのまま使う。これにより model.clip_skip 未指定時の出力は
    現状と完全に一致する。
    """
    if CLIP_SKIP_ROLE not in binding.nodes:
        return

    clip_skip = spec.model.clip_skip
    if clip_skip is None:
        return

    inputs_of(CLIP_SKIP_ROLE)["stop_at_clip_layer"] = -clip_skip


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

    has_preprocessor = CONTROL_PREPROCESSOR_ROLE in binding.nodes
    if has_preprocessor is control.skips_preprocessor:
        # テンプレート選択 (resolve_workflow_name) とSpecの指定が食い違っている。
        # 前処理の有無が黙って入れ替わると、出てくる絵だけが変わって原因が追えない
        expected = "前処理なし" if control.skips_preprocessor else "Canny前処理あり"
        raise WorkflowValidationError(
            f"Workflow ({binding.name}) の前処理の有無が、Specの preprocessor "
            f"({control.preprocessor}, {expected}) と一致しません"
        )

    inputs_of(CONTROL_IMAGE_ROLE)["image"] = control_image_name
    inputs_of(CONTROL_LOADER_ROLE)["control_net_name"] = control.model

    if has_preprocessor:
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
    """hires fix 用テンプレートへ拡大倍率と2段目の設定を注入する。

    latent拡大とアップスケールモデルでは、倍率を書き込む先も意味も違う。

    - latent拡大: LatentUpscaleBy の scale_by へ要求された倍率をそのまま入れる
    - モデル拡大: 倍率はモデル側で決まるため、ImageScaleBy へ要求された倍率へ
      戻すための縮小率を入れる
    """
    uses_model_template = UPSCALE_MODEL_RESIZE_ROLE in binding.nodes
    if UPSCALE_ROLE not in binding.nodes and not uses_model_template:
        return

    upscale = spec.generation.upscale
    if upscale is None:
        raise WorkflowValidationError(
            f"Workflow ({binding.name}) はhires fix用ですが、"
            "Specに generation.upscale が指定されていません"
        )
    if uses_model_template != upscale.uses_model:
        expected = "アップスケールモデル" if uses_model_template else "latent拡大"
        raise WorkflowValidationError(
            f"Workflow ({binding.name}) は{expected}用ですが、"
            "Specの generation.upscale.model の指定が食い違っています"
        )

    if uses_model_template:
        inputs_of(UPSCALE_MODEL_LOADER_ROLE)["model_name"] = upscale.model
        resize = inputs_of(UPSCALE_MODEL_RESIZE_ROLE)
        resize["scale_by"] = upscale.resize_factor()
        resize["upscale_method"] = upscale.method
    else:
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
    "ALL_BINDINGS",
    "CLIP_SKIP_ROLE",
    "CONTROL_APPLY_ROLE",
    "CONTROL_IMAGE_ROLE",
    "CONTROL_LOADER_ROLE",
    "CONTROL_PREPROCESSOR_ROLE",
    "HIRES_KSAMPLER_ROLE",
    "IMG2IMG_BINDING",
    "IMG2IMG_LORA_BINDING",
    "IMG2IMG_VAE_BINDING",
    "LORA_SLOT_ROLES",
    "REFERENCE_APPLY_ROLE",
    "REFERENCE_CLIP_VISION_ROLE",
    "REFERENCE_IMAGE_ROLE",
    "REFERENCE_LOADER_ROLE",
    "TXT2IMG_BINDING",
    "TXT2IMG_CONTROLNET_BINDING",
    "TXT2IMG_HIRES_BINDING",
    "TXT2IMG_HIRES_MODEL_BINDING",
    "TXT2IMG_IPADAPTER_BINDING",
    "TXT2IMG_LORA_BINDING",
    "TXT2IMG_UNET_BINDING",
    "TXT2IMG_VAE_BINDING",
    "TXT2IMG_VAE_HIRES_MODEL_BINDING",
    "TXT2IMG_VAE_LORA_BINDING",
    "UPSCALE_MODEL_APPLY_ROLE",
    "UPSCALE_MODEL_DECODE_ROLE",
    "UPSCALE_MODEL_ENCODE_ROLE",
    "UPSCALE_MODEL_LOADER_ROLE",
    "UPSCALE_MODEL_RESIZE_ROLE",
    "UPSCALE_ROLE",
    "LinkRef",
    "NodeRef",
    "WorkflowBinding",
    "build_workflow",
    "resolve_seed",
    "validate_structure",
]
