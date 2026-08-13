"""GenerationSpec: Agent/Core間の内部API契約。

この層はComfyUIのNode IDやHTTP仕様を一切知らない。
ここで行うのは「設定に依存しない」ハード制約の検証のみで、
環境変数由来の上限値は domain.policy 側で検証する。
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: ComfyUIの comfy/samplers.py SAMPLER_NAMES (KSAMPLER_NAMES + ddim / uni_pc 系) と同じ集合。
#: 並び順もComfyUI側に合わせ、差分を追いやすくする。
SamplerName = Literal[
    "euler",
    "euler_cfg_pp",
    "euler_ancestral",
    "euler_ancestral_cfg_pp",
    "heun",
    "heunpp2",
    "exp_heun_2_x0",
    "exp_heun_2_x0_sde",
    "dpm_2",
    "dpm_2_ancestral",
    "lms",
    "dpm_fast",
    "dpm_adaptive",
    "dpmpp_2s_ancestral",
    "dpmpp_2s_ancestral_cfg_pp",
    "dpmpp_sde",
    "dpmpp_sde_gpu",
    "dpmpp_2m",
    "dpmpp_2m_cfg_pp",
    "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu",
    "dpmpp_2m_sde_heun",
    "dpmpp_2m_sde_heun_gpu",
    "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu",
    "ddpm",
    "lcm",
    "ipndm",
    "ipndm_v",
    "deis",
    "res_multistep",
    "res_multistep_cfg_pp",
    "res_multistep_ancestral",
    "res_multistep_ancestral_cfg_pp",
    "gradient_estimation",
    "gradient_estimation_cfg_pp",
    "er_sde",
    "seeds_2",
    "seeds_3",
    "sa_solver",
    "sa_solver_pece",
    "ddim",
    "uni_pc",
    "uni_pc_bh2",
]

#: ComfyUIの comfy/samplers.py SCHEDULER_HANDLERS と同じ集合。
#: KSamplerが受け付けないもの (BetaSamplingSchedulerノード側の設定を指す beta57 など) は含めない。
SchedulerName = Literal[
    "normal",
    "karras",
    "exponential",
    "sgm_uniform",
    "simple",
    "ddim_uniform",
    "beta",
    "linear_quadratic",
    "kl_optimal",
]

#: 解像度のハード上限。環境変数による実運用上の上限は Settings 側で別途課す。
MIN_DIMENSION: Final = 64
MAX_DIMENSION: Final = 8192
DIMENSION_MULTIPLE: Final = 8

#: seed に -1 を指定した場合は実行時に乱数へ解決する。
RANDOM_SEED: Final = -1
MAX_SEED: Final = 2**63 - 1

ALLOWED_CHECKPOINT_SUFFIXES: Final = frozenset({".safetensors", ".ckpt"})

#: LoRAは学習側の都合で .pt が配布されることがあるため、checkpointより1つ広い。
ALLOWED_LORA_SUFFIXES: Final = frozenset({".safetensors", ".pt", ".ckpt"})

#: 同時に適用できるLoRAの本数。Workflowテンプレート側のLoraLoaderの段数と一致させる。
MAX_LORAS: Final = 3

#: clip_skipの実用上の範囲。ComfyUIのCLIPSetLastLayerは-1から-24を受け付けるが、
#: 実用は1-2 (NovelAI系の慣習で2まで) のため、余裕を持たせつつ絞る。
MIN_CLIP_SKIP: Final = 1
MAX_CLIP_SKIP: Final = 12

#: strengthの実用上の範囲。ComfyUI自体は±100を許すが、事故を防ぐため絞る。
LORA_STRENGTH_LIMIT: Final = 10.0

#: 拡大方法。latent拡大 (LatentUpscaleBy) とpixel拡大 (ImageScaleBy) で
#: 選べる値が違う。bislerp はlatentだけ、lanczos はpixelだけにある。
#: どちらに属するかは UpscaleSpec 側で突き合わせる。
UpscaleMethod = Literal["nearest-exact", "bilinear", "area", "bicubic", "bislerp", "lanczos"]

#: LatentUpscaleBy にしか無い拡大方法。
LATENT_ONLY_UPSCALE_METHODS: Final = frozenset({"bislerp"})

#: ImageScaleBy にしか無い拡大方法。
IMAGE_ONLY_UPSCALE_METHODS: Final = frozenset({"lanczos"})

#: hires fix の拡大倍率の上限。これ以上は生成時間が現実的でない。
MAX_UPSCALE_SCALE: Final = 4.0

#: アップスケールモデルとして受け付ける拡張子。ESRGAN系は .pth 配布が主流。
ALLOWED_UPSCALE_MODEL_SUFFIXES: Final = frozenset({".pth", ".safetensors"})

#: アップスケールモデルの固有倍率として受け付ける範囲。
#: 出回っているのは2x / 4x / 8x で、ImageScaleBy の scale_by 上限も8。
MIN_UPSCALE_MODEL_SCALE: Final = 1.0
MAX_UPSCALE_MODEL_SCALE: Final = 8.0

#: アップスケールモデルの固有倍率の既定値。ESRGAN系で最も多い。
DEFAULT_UPSCALE_MODEL_SCALE: Final = 4.0

#: ControlNetモデルとして受け付ける拡張子。.pth 配布があるためcheckpointより広い。
ALLOWED_CONTROLNET_SUFFIXES: Final = frozenset({".safetensors", ".pth", ".ckpt"})

#: IPAdapterモデルとして受け付ける拡張子。light 系は .bin で配布される。
ALLOWED_IPADAPTER_SUFFIXES: Final = frozenset({".safetensors", ".bin", ".pth", ".ckpt"})

#: CLIP Visionモデルとして受け付ける拡張子。
ALLOWED_CLIP_VISION_SUFFIXES: Final = frozenset({".safetensors", ".bin", ".pt", ".ckpt"})

#: IPAdapterの効かせ方。ComfyUI側 (IPAdapterAdvanced) の選択肢と一致させる。
IPAdapterWeightType = Literal[
    "linear",
    "ease in",
    "ease out",
    "ease in-out",
    "reverse in-out",
    "weak input",
    "weak output",
    "weak middle",
    "strong middle",
    "style transfer",
    "composition",
    "strong style transfer",
    "style and composition",
    "style transfer precise",
    "composition precise",
]

#: weightの実用上の上限。ノード自体は5.0まで許すが、破綻するため絞る。
MAX_REFERENCE_WEIGHT: Final = 3.0

#: img2imgの入力画像として受け付ける拡張子。
ALLOWED_SOURCE_IMAGE_SUFFIXES: Final = frozenset({".png", ".jpg", ".jpeg", ".webp"})

TextAnchor = Literal[
    "top-left",
    "top-center",
    "top-right",
    "middle-left",
    "center",
    "middle-right",
    "bottom-left",
    "bottom-center",
    "bottom-right",
]

TextAlign = Literal["left", "center", "right"]

TextDirection = Literal["horizontal", "vertical"]

#: テキスト合成に使うフォントとして受け付ける拡張子。
ALLOWED_FONT_SUFFIXES: Final = frozenset({".ttf", ".otf", ".ttc"})

#: 1枚へ重ねられるテキストレイヤの上限。これ以上は指定が読めなくなる。
MAX_TEXT_LAYERS: Final = 10

#: 1レイヤに書ける文字数の上限。セグメント列で指定する場合は全セグメント合計で見る。
MAX_TEXT_CONTENT_LENGTH: Final = 500

#: 縦中横 (縦書き中で数文字を横に組んで1セルへ収める組版) の1セルに書ける文字数の上限。
#: 「令和7年」の「7」のように短い数字・アルファベットを想定した値で、
#: これを超えると1セル分の幅へ収まらず可読性が落ちる。
MAX_TATE_CHU_YOKO_LENGTH: Final = 4

#: フォントサイズの上限。解像度の上限に対して十分大きい。
MAX_FONT_SIZE: Final = 512

_PREFIX_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

#: #rgb / #rrggbb / #rrggbbaa の3形式を受け付ける。
_COLOR_PATTERN: Final = re.compile(r"^#(?:[0-9A-Fa-f]{3}|[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})$")


def _validate_relative_image_path(value: str) -> str:
    """リポジトリ配下の画像パスとして安全か検証する。

    checkpointと違い階層の深さは問わない。実体の解決とルート外への脱出検証は
    domain.policy が担う。
    """
    if any(ord(ch) < 32 for ch in value):
        raise ValueError("制御文字を含むパスは指定できません")
    if "\\" in value:
        raise ValueError("バックスラッシュは使用できません")
    if value.startswith(("/", "~")):
        raise ValueError("絶対パスやホームディレクトリ参照は指定できません")

    segments = value.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise ValueError("上位ディレクトリ参照や空のセグメントを含むパスは指定できません")

    if PurePosixPath(segments[-1]).suffix.lower() not in ALLOWED_SOURCE_IMAGE_SUFFIXES:
        allowed = " / ".join(sorted(ALLOWED_SOURCE_IMAGE_SUFFIXES))
        raise ValueError(f"拡張子は {allowed} のいずれかにしてください (指定値: {value})")
    return value


def _validate_model_filename(value: str, *, allowed_suffixes: frozenset[str]) -> str:
    """モデルファイル名にPath Traversalや想定外の拡張子を許さない。

    ComfyUIの各modelsディレクトリ配下を前提とし、サブフォルダは1階層まで許可する。
    checkpoint / LoRA / ControlNet / IPAdapter / CLIP Vision / フォントで共通の規則。
    """
    if any(ord(ch) < 32 for ch in value):
        raise ValueError("制御文字を含む名前は指定できません")
    if "\\" in value:
        raise ValueError("バックスラッシュは使用できません")
    if value.startswith(("/", "~")):
        raise ValueError("絶対パスやホームディレクトリ参照は指定できません")

    segments = value.split("/")
    if len(segments) > 2:
        raise ValueError("サブフォルダは1階層までしか指定できません")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise ValueError("上位ディレクトリ参照を含む名前は指定できません")

    # 実在ファイル名には大文字混じりがある。画像パスの検証と同じく大小は無視する
    if PurePosixPath(segments[-1]).suffix.lower() not in allowed_suffixes:
        allowed = " / ".join(sorted(allowed_suffixes))
        raise ValueError(f"拡張子は {allowed} のいずれかにしてください (指定値: {value})")
    return value


def _reject_control_characters(value: str) -> str:
    """改行以外の制御文字を拒否する。

    TextLayer.content (文字列指定) と TextSegment.text の両方から使う共通ロジック。
    改行だけは複数行の指定として許す。
    """
    if any(ord(ch) < 32 and ch != "\n" for ch in value):
        raise ValueError("改行以外の制御文字は指定できません")
    return value


def _validate_color(value: str) -> str:
    """色指定を検証し、小文字へ正規化する。

    色名 (white 等) は環境やライブラリの版で解釈が揺れるため受け付けない。
    """
    if not _COLOR_PATTERN.fullmatch(value):
        raise ValueError(
            f"色は #rgb / #rrggbb / #rrggbbaa の形式で指定してください (指定値: {value!r})"
        )
    return value.lower()


class _StrictModel(BaseModel):
    """未知キーを拒否し、生成後の変更を禁止する共通設定。"""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class PromptSpec(_StrictModel):
    """プロンプト。"""

    positive: str = Field(min_length=1)
    negative: str = ""


class UpscaleSpec(_StrictModel):
    """hires fix の設定。

    拡大の仕方は2通りある。

    - `model` 未指定: 1段目の生成結果をlatentのまま拡大する (LatentUpscaleBy)。
      追加のモデルが要らない代わりに、拡大時のディテールは2段目のdenoiseだけで補う
    - `model` 指定: 一度pixelへ戻してアップスケールモデルで拡大する
      (VAEDecode -> ImageUpscaleWithModel -> ImageScaleBy -> VAEEncode)。
      拡大の時点で線が補間されるため、denoiseを低く保ったまま解像度を上げられる

    どちらも拡大後に2段目のKSamplerで描き足す点は同じ。
    """

    #: 拡大倍率。1.0以下は拡大にならないため許可しない。
    scale: Annotated[float, Field(gt=1.0, le=MAX_UPSCALE_SCALE)] = 1.5
    #: 2段目のdenoise。低いほど元の絵を保つ。
    denoise: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    #: 2段目のsteps。未指定なら1段目と同じ値を使う。
    steps: Annotated[int, Field(ge=1, le=100)] | None = None
    method: UpscaleMethod = "nearest-exact"
    #: 使うアップスケールモデルのファイル名。未指定ならlatent拡大のまま。
    model: str | None = None
    #: モデルの固有倍率。配布元の表記 (4x なら 4.0) をそのまま書く。
    #: モデルの出力サイズはモデル側で決まるため、`scale` へ合わせるにはこの値が要る。
    #: 未指定は None のまま保つ (既定値を埋めるとlatent拡大のSpecをdumpしたときにも
    #: 値が乗り、metadata.json を読み直せなくなる)。実効値は effective_model_scale。
    model_scale: (
        Annotated[float, Field(ge=MIN_UPSCALE_MODEL_SCALE, le=MAX_UPSCALE_MODEL_SCALE)] | None
    ) = None

    @field_validator("model")
    @classmethod
    def _reject_unsafe_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_model_filename(value, allowed_suffixes=ALLOWED_UPSCALE_MODEL_SUFFIXES)

    @model_validator(mode="after")
    def _check_model_combination(self) -> UpscaleSpec:
        if self.model is None:
            if self.model_scale is not None:
                raise ValueError(
                    "model_scale はアップスケールモデルの固有倍率です。"
                    "latent拡大では意味を持たないため、model と一緒に指定してください"
                )
            if self.method in IMAGE_ONLY_UPSCALE_METHODS:
                raise ValueError(
                    f"method {self.method!r} はアップスケールモデルを使う場合のみ指定できます"
                )
            return self

        if self.method in LATENT_ONLY_UPSCALE_METHODS:
            raise ValueError(
                f"method {self.method!r} はlatent拡大でのみ指定できます "
                "(アップスケールモデルを使う場合はpixel側の拡大方法を選んでください)"
            )
        if self.scale > self.effective_model_scale:
            raise ValueError(
                f"scale ({self.scale}) が model_scale ({self.effective_model_scale}) を"
                "超えています。モデルの出力より大きくは引き伸ばしません"
            )
        return self

    @property
    def uses_model(self) -> bool:
        """アップスケールモデルを使う指定かどうか。"""
        return self.model is not None

    @property
    def effective_model_scale(self) -> float:
        """実際に使うモデルの固有倍率。未指定ならESRGAN系で最も多い4.0とみなす。"""
        return self.model_scale if self.model_scale is not None else DEFAULT_UPSCALE_MODEL_SCALE

    def effective_steps(self, base_steps: int) -> int:
        """2段目で実際に使うsteps。"""
        return self.steps if self.steps is not None else base_steps

    def resize_factor(self) -> float:
        """モデルの出力を `scale` へ合わせるための縮小率。

        アップスケールモデルの倍率は固定 (4x など) のため、要求された倍率が
        それより小さければ拡大後に縮小して合わせる。
        """
        if self.model is None:
            raise ValueError("resize_factor は model を指定した場合にのみ意味を持ちます")
        return self.scale / self.effective_model_scale


class GenerationParams(_StrictModel):
    """生成パラメータ。"""

    width: Annotated[int, Field(ge=MIN_DIMENSION, le=MAX_DIMENSION)] = 512
    height: Annotated[int, Field(ge=MIN_DIMENSION, le=MAX_DIMENSION)] = 512
    steps: Annotated[int, Field(ge=1, le=100)] = 20
    cfg: Annotated[float, Field(ge=0, le=30)] = 7.0
    seed: Annotated[int, Field(ge=RANDOM_SEED, le=MAX_SEED)] = RANDOM_SEED
    batch_size: Annotated[int, Field(ge=1, le=4)] = 1
    sampler: SamplerName = "euler"
    scheduler: SchedulerName = "normal"
    #: 指定するとhires fix (latent拡大 + 2段目のKSampler) を行う。
    upscale: UpscaleSpec | None = None

    @field_validator("width", "height")
    @classmethod
    def _must_be_multiple_of_eight(cls, value: int) -> int:
        if value % DIMENSION_MULTIPLE != 0:
            raise ValueError(f"{DIMENSION_MULTIPLE}の倍数で指定してください (指定値: {value})")
        return value


class LoraSpec(_StrictModel):
    """適用するLoRA1件。"""

    name: str = Field(min_length=1)
    strength_model: Annotated[float, Field(ge=-LORA_STRENGTH_LIMIT, le=LORA_STRENGTH_LIMIT)] = 1.0
    strength_clip: Annotated[float, Field(ge=-LORA_STRENGTH_LIMIT, le=LORA_STRENGTH_LIMIT)] = 1.0

    @field_validator("name")
    @classmethod
    def _reject_unsafe_path(cls, value: str) -> str:
        return _validate_model_filename(value, allowed_suffixes=ALLOWED_LORA_SUFFIXES)


class ModelSpec(_StrictModel):
    """使用するモデル。

    指定の仕方は2通りある。

    - `checkpoint`: UNet / CLIP / VAE を1ファイルに含む従来形式 (SD1.5 / SDXL 系)。
      `vae` を追加で指定すると、checkpoint 同梱の VAE ではなく外部VAE (色褪せ・
      眠い線を避けるために差し替える vae-ft-mse-840000 / klF8Anime2VAE など) を使う
    - `unet` + `clip` + `vae`: 3つを別々に読む形式 (Anima などのDiT系)

    DiT系のモデルはUNet単体で配布され、text encoderとVAEを同梱しないため、
    ローダーを分ける必要がある。`checkpoint` と `unet` / `clip` を同時に書けると
    「どちらが効くのか」が読み取れなくなるため、この2つは排他にしている
    (`checkpoint` と `vae` は併用できる)。
    """

    checkpoint: str | None = None
    #: DiT系モデルのUNet本体。~/ComfyUI/models/diffusion_models/ に置く。
    unet: str | None = None
    #: DiT系モデルのtext encoder。~/ComfyUI/models/text_encoders/ に置く。
    clip: str | None = None
    #: 外部VAE。~/ComfyUI/models/vae/ に置く。`checkpoint` と併用すると同梱VAEの
    #: 代わりに使う外部VAEになり、`unet` + `clip` と併用するとDiT系のVAE (必須) になる。
    vae: str | None = None
    #: 適用順に並べる。Workflowテンプレートの LoraLoader の段へ先頭から割り当てる。
    loras: tuple[LoraSpec, ...] = ()
    #: CLIPの最終層を何層手前で打ち切るか。未指定はComfyUI既定 (clip skip 1相当) のまま。
    #: LoRAがある場合はLoRA適用後のCLIPに対して効かせる。DiT系 (unet/clip/vae) とは
    #: text encoderの構造が異なるため併用できない。
    clip_skip: Annotated[int, Field(ge=MIN_CLIP_SKIP, le=MAX_CLIP_SKIP)] | None = None

    @property
    def uses_separate_loaders(self) -> bool:
        """UNet / CLIP / VAE を別々に読む形式かどうか。"""
        return self.unet is not None

    @property
    def uses_external_vae(self) -> bool:
        """checkpoint と外部VAEを併用する指定かどうか。

        DiT系 (`uses_separate_loaders`) はそもそも `checkpoint` を持たないため、
        こちらには含めない。
        """
        return self.checkpoint is not None and self.vae is not None

    @field_validator("checkpoint", "unet", "clip", "vae")
    @classmethod
    def _reject_unsafe_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_model_filename(value, allowed_suffixes=ALLOWED_CHECKPOINT_SUFFIXES)

    @model_validator(mode="after")
    def _validate_loader_combination(self) -> ModelSpec:
        # unet / clip は checkpoint と排他。vae は checkpoint (外部VAE) /
        # unet+clip (DiT系のVAE、必須) のどちらとも組める独立した軸のため、
        # ここでは分けて扱う。
        separate = {"unet": self.unet, "clip": self.clip}
        specified = [key for key, value in separate.items() if value is not None]

        if self.checkpoint is not None:
            if specified:
                raise ValueError(
                    "checkpoint と {} は同時に指定できません".format(" / ".join(sorted(specified)))
                )
            return self

        if not specified:
            if self.vae is not None:
                raise ValueError(
                    "vae は単体では指定できません。checkpoint と組み合わせるか、"
                    "unet / clip と併せて3つ指定してください"
                )
            raise ValueError("checkpoint、または unet / clip / vae の3つを指定してください")

        missing = sorted(key for key, value in separate.items() if value is None)
        if self.vae is None:
            missing.append("vae")
        if missing:
            raise ValueError(
                "unet / clip / vae は3つセットで指定してください (不足: {})".format(
                    " / ".join(sorted(missing))
                )
            )

        unsupported = []
        if self.loras:
            unsupported.append("LoRA")
        if self.clip_skip is not None:
            unsupported.append("clip_skip")

        if unsupported:
            raise ValueError(
                "unet / clip / vae の指定と{}の併用は未対応です".format(" / ".join(unsupported))
            )

        return self

    @field_validator("loras")
    @classmethod
    def _validate_loras(cls, value: tuple[LoraSpec, ...]) -> tuple[LoraSpec, ...]:
        if len(value) > MAX_LORAS:
            raise ValueError(
                f"LoRAは同時に{MAX_LORAS}件までしか指定できません (指定数: {len(value)})"
            )

        seen: set[str] = set()
        for lora in value:
            if lora.name in seen:
                raise ValueError(f"同じLoRAが重複して指定されています: {lora.name}")
            seen.add(lora.name)
        return value


class OutputSpec(_StrictModel):
    """出力先。

    directory の実体解決とリポジトリ外への脱出検証は domain.policy が担う。
    """

    #: 未指定なら Settings.output_root (既定 outputs) を使う。
    directory: str | None = None
    prefix: str = "imagegen"

    @field_validator("prefix")
    @classmethod
    def _validate_prefix(cls, value: str) -> str:
        if not _PREFIX_PATTERN.fullmatch(value):
            raise ValueError(
                "prefixは英数字で始まり、英数字・ドット・アンダースコア・ハイフンのみ使用できます"
            )
        return value


class SourceSpec(_StrictModel):
    """img2imgの入力画像。

    imageはリポジトリ配下の相対パスで指定する。checkpointと違い階層の深さは問わない。
    実体の解決とリポジトリ外への脱出検証は domain.policy が担う。
    """

    image: str = Field(min_length=1)
    #: 0に近いほど入力画像を保ち、1に近いほど描き直す。
    denoise: Annotated[float, Field(ge=0.0, le=1.0)] = 0.6

    @field_validator("image")
    @classmethod
    def _reject_unsafe_path(cls, value: str) -> str:
        return _validate_relative_image_path(value)


class ControlSpec(_StrictModel):
    """ControlNetの指定。

    control画像は Canny で線画へ変換してから使う。前処理済みの線画を直接渡す
    経路は今のところ持たない (必要になったら足す)。
    """

    #: 構図の元になる画像。リポジトリ配下の相対パス。
    image: str = Field(min_length=1)
    #: ComfyUIの models/controlnet 配下のファイル名。
    model: str = Field(min_length=1)
    strength: Annotated[float, Field(ge=0.0, le=10.0)] = 1.0
    #: 効かせ始める / 終える進行度。構図だけ借りたい場合は end_percent を下げる。
    start_percent: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    end_percent: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    #: Cannyの閾値。低いほど細かい線を拾う。
    low_threshold: Annotated[float, Field(ge=0.01, le=0.99)] = 0.4
    high_threshold: Annotated[float, Field(ge=0.01, le=0.99)] = 0.8

    @field_validator("image")
    @classmethod
    def _reject_unsafe_image(cls, value: str) -> str:
        return _validate_relative_image_path(value)

    @field_validator("model")
    @classmethod
    def _reject_unsafe_model(cls, value: str) -> str:
        return _validate_model_filename(value, allowed_suffixes=ALLOWED_CONTROLNET_SUFFIXES)

    @model_validator(mode="after")
    def _validate_ranges(self) -> ControlSpec:
        if self.low_threshold >= self.high_threshold:
            raise ValueError("low_threshold は high_threshold より小さくしてください")
        if self.start_percent >= self.end_percent:
            raise ValueError("start_percent は end_percent より小さくしてください")
        return self


class StrokeSpec(_StrictModel):
    """文字の縁取り。"""

    width: Annotated[int, Field(ge=1, le=64)] = 2
    color: str = "#000000"

    @field_validator("color")
    @classmethod
    def _normalize_color(cls, value: str) -> str:
        return _validate_color(value)


class ShadowSpec(_StrictModel):
    """文字の影。

    本体を描いたあとにぼかして下へ敷く。
    """

    offset: tuple[
        Annotated[int, Field(ge=-MAX_DIMENSION, le=MAX_DIMENSION)],
        Annotated[int, Field(ge=-MAX_DIMENSION, le=MAX_DIMENSION)],
    ] = (4, 4)
    blur: Annotated[float, Field(ge=0.0, le=64.0)] = 4.0
    color: str = "#000000"
    opacity: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5

    @field_validator("color")
    @classmethod
    def _normalize_color(cls, value: str) -> str:
        return _validate_color(value)


class BoxSpec(_StrictModel):
    """文字の背後へ敷く矩形。

    背景に埋もれて読めなくなるのを避けるために使う。
    """

    color: str = "#000000"
    opacity: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    #: (横, 縦) 方向の余白。
    padding: tuple[Annotated[int, Field(ge=0, le=512)], Annotated[int, Field(ge=0, le=512)]] = (
        16,
        16,
    )
    radius: Annotated[int, Field(ge=0, le=512)] = 0

    @field_validator("color")
    @classmethod
    def _normalize_color(cls, value: str) -> str:
        return _validate_color(value)


class TextSegment(_StrictModel):
    """縦中横などセグメント単位で組版を変える場合の1区間。

    `TextLayer.content` へ文字列の代わりにこの列を渡すと、`tate_chu_yoko=True` の
    区間だけ縦書き中で数文字を横に組んで1セルへ収める (「令和7年」の「7」など)。
    ルビは対象外 (列幅が可変になり別の設計変更が要るため、今回は入れない)。
    """

    text: str = Field(min_length=1)
    tate_chu_yoko: bool = False

    @field_validator("text")
    @classmethod
    def _reject_control_characters_in_text(cls, value: str) -> str:
        return _reject_control_characters(value)

    @model_validator(mode="after")
    def _validate_tate_chu_yoko(self) -> TextSegment:
        if not self.tate_chu_yoko:
            return self
        if len(self.text) > MAX_TATE_CHU_YOKO_LENGTH:
            raise ValueError(
                f"tate_chu_yokoは{MAX_TATE_CHU_YOKO_LENGTH}文字までしか指定できません "
                f"(指定文字数: {len(self.text)})"
            )
        if "\n" in self.text:
            raise ValueError("tate_chu_yokoのセグメントに改行は含められません")
        return self


class TextLayer(_StrictModel):
    """重ねるテキスト1件。

    fontは fonts/ 配下のファイル名で指定する。実体の解決とルート外への脱出検証は
    domain.policy が担う。

    contentは文字列、またはTextSegmentの並びで指定する。文字列指定は従来どおり
    横書き・縦書きとも1文字1セルとして扱う。セグメント列は縦中横 (`tate_chu_yoko`)
    を使いたいときだけ使う書き方で、区切り記号を使ったインライン記法
    (青空文庫式`｜漢字《かんじ》`等) は採らない。`_VERTICAL_ROTATED_CHARS`が
    区切り記号候補すべてと衝突するため。
    """

    content: (
        Annotated[str, Field(min_length=1, max_length=MAX_TEXT_CONTENT_LENGTH)]
        | Annotated[tuple[TextSegment, ...], Field(min_length=1)]
    )
    font: str = Field(min_length=1)
    #: .ttc のようなコレクションの中で使う書体の索引。
    font_index: Annotated[int, Field(ge=0, le=64)] = 0
    size: Annotated[int, Field(ge=1, le=MAX_FONT_SIZE)]
    color: str = "#ffffff"
    anchor: TextAnchor = "center"
    #: anchor からのずれ (px)。
    offset: tuple[
        Annotated[int, Field(ge=-MAX_DIMENSION, le=MAX_DIMENSION)],
        Annotated[int, Field(ge=-MAX_DIMENSION, le=MAX_DIMENSION)],
    ] = (0, 0)
    #: 折り返し幅。1.0以下は画像幅に対する比率、1.0超はpxとして扱う。
    #: 上限は解像度のハード上限 (MAX_DIMENSION) に合わせる。
    max_width: Annotated[float, Field(gt=0.0, le=MAX_DIMENSION)] | None = None
    line_spacing: Annotated[float, Field(ge=0.5, le=5.0)] = 1.2
    align: TextAlign = "center"
    opacity: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    #: 度。反時計回り。
    rotation: Annotated[float, Field(ge=-180.0, le=180.0)] = 0.0
    direction: TextDirection = "horizontal"
    stroke: StrokeSpec | None = None
    shadow: ShadowSpec | None = None
    box: BoxSpec | None = None

    @field_validator("content")
    @classmethod
    def _reject_control_characters_in_content(
        cls, value: str | tuple[TextSegment, ...]
    ) -> str | tuple[TextSegment, ...]:
        # セグメント列側の制御文字チェックは TextSegment._reject_control_characters_in_text
        # で完結しているため、ここでは文字列指定のときだけ検査する。
        if isinstance(value, str):
            return _reject_control_characters(value)
        return value

    @model_validator(mode="after")
    def _validate_content_segments(self) -> TextLayer:
        if not isinstance(self.content, tuple):
            return self

        total_length = sum(len(segment.text) for segment in self.content)
        if total_length > MAX_TEXT_CONTENT_LENGTH:
            raise ValueError(
                f"contentは{MAX_TEXT_CONTENT_LENGTH}文字までしか指定できません "
                f"(指定文字数: {total_length})"
            )

        if self.direction == "horizontal" and any(
            segment.tate_chu_yoko for segment in self.content
        ):
            raise ValueError("direction: horizontalではtate_chu_yokoを指定できません")

        return self

    @field_validator("font")
    @classmethod
    def _reject_unsafe_font(cls, value: str) -> str:
        return _validate_model_filename(value, allowed_suffixes=ALLOWED_FONT_SUFFIXES)

    @field_validator("color")
    @classmethod
    def _normalize_color(cls, value: str) -> str:
        return _validate_color(value)


class TextSpec(_StrictModel):
    """生成後に重ねるテキスト全体。

    layers は指定順に描画し、後のものが上へ重なる。
    """

    layers: tuple[TextLayer, ...] = Field(min_length=1)

    @field_validator("layers")
    @classmethod
    def _validate_layer_count(cls, value: tuple[TextLayer, ...]) -> tuple[TextLayer, ...]:
        if len(value) > MAX_TEXT_LAYERS:
            raise ValueError(
                f"テキストレイヤは{MAX_TEXT_LAYERS}件までしか指定できません (指定数: {len(value)})"
            )
        return value


class ReferenceSpec(_StrictModel):
    """IPAdapterの指定。参照画像から人物や画風の特徴を寄せる。

    ComfyUIの IPAdapterUnifiedLoader は preset名からモデルを暗黙に選ぶが、
    ここでは使わない。checkpoint / LoRA / ControlNet と同じく、
    実際に読み込むファイル名をSpecへ明示する。
    """

    #: 特徴の元になる画像。リポジトリ配下の相対パス。
    image: str = Field(min_length=1)
    #: ComfyUIの models/ipadapter 配下のファイル名。
    model: str = Field(min_length=1)
    #: ComfyUIの models/clip_vision 配下のファイル名。IPAdapterモデルと対応するものを選ぶ。
    clip_vision: str = Field(min_length=1)
    #: 効かせる強さ。1.0前後が目安で、上げすぎると参照画像へ寄りすぎて構図が崩れる。
    weight: Annotated[float, Field(ge=0.0, le=MAX_REFERENCE_WEIGHT)] = 1.0
    weight_type: IPAdapterWeightType = "linear"
    #: 効かせ始める / 終える進行度。
    start_percent: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    end_percent: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0

    @field_validator("image")
    @classmethod
    def _reject_unsafe_image(cls, value: str) -> str:
        return _validate_relative_image_path(value)

    @field_validator("model")
    @classmethod
    def _reject_unsafe_model(cls, value: str) -> str:
        return _validate_model_filename(value, allowed_suffixes=ALLOWED_IPADAPTER_SUFFIXES)

    @field_validator("clip_vision")
    @classmethod
    def _reject_unsafe_clip_vision(cls, value: str) -> str:
        return _validate_model_filename(value, allowed_suffixes=ALLOWED_CLIP_VISION_SUFFIXES)

    @model_validator(mode="after")
    def _validate_ranges(self) -> ReferenceSpec:
        if self.start_percent >= self.end_percent:
            raise ValueError("start_percent は end_percent より小さくしてください")
        return self


class PresetRefs(_StrictModel):
    """適用するpresetの参照。軸ごとに1つまで。

    Specの読み込み時に services.preset_loader が解決し、prompt と generation へ
    展開する。展開後もどのpresetを使ったかは再現のためここに残す。
    下層 (Workflow / Adapter) はこのフィールドを参照しない。
    """

    character: str | None = None
    scene: str | None = None
    style: str | None = None

    def is_empty(self) -> bool:
        return self.character is None and self.scene is None and self.style is None


class GenerationSpec(_StrictModel):
    """画像生成の要求全体。"""

    version: Literal["1"] = "1"
    task: Literal["txt2img", "img2img"] = "txt2img"
    presets: PresetRefs = Field(default_factory=PresetRefs)
    prompt: PromptSpec
    generation: GenerationParams = Field(default_factory=GenerationParams)
    model: ModelSpec
    #: img2imgのときのみ指定する。
    source: SourceSpec | None = None
    #: 指定するとControlNetで構図を制御する。txt2img / img2img のどちらでも使える。
    control: ControlSpec | None = None
    #: 指定するとIPAdapterで参照画像の特徴を寄せる。controlとは併用できる。
    reference: ReferenceSpec | None = None
    #: 指定すると生成後に日本語テキストを合成する。生成そのものの挙動は変わらない。
    text: TextSpec | None = None
    output: OutputSpec = Field(default_factory=OutputSpec)

    @model_validator(mode="after")
    def _validate_task_combination(self) -> GenerationSpec:
        """taskと他フィールドの組み合わせを検証する。

        指定しても効かない項目は黙って無視せず拒否する。書いたのに反映されていない
        状態は、生成結果を見ても原因が分かりにくいため。
        """
        if self.generation.upscale is not None and self.reference is not None:
            # 両方かけると生成時間が現実的でないため、テンプレートを用意していない
            raise ValueError("upscale と reference の同時指定は未対応です")
        if self.model.uses_separate_loaders:
            self._validate_separate_loaders()
        if self.task == "img2img":
            self._validate_img2img()
        elif self.source is not None:
            raise ValueError("source は task が img2img のときにのみ指定できます")
        return self

    def _validate_separate_loaders(self) -> None:
        """unet / clip / vae を使う場合に、テンプレートが無い組み合わせを拒否する。

        ControlNet / IPAdapter のモデルはSD1.5 / SDXL 系向けのものであり、
        DiT系のUNetへはそのまま適用できない。DiT系向けのモデルが出回るまでは
        テンプレートを用意しても使えるモデルが無い。
        img2img と hires fix は組み合わせられる。
        """
        if self.control is not None:
            raise ValueError("unet / clip / vae の指定と control の併用は未対応です")
        if self.reference is not None:
            raise ValueError("unet / clip / vae の指定と reference の併用は未対応です")

    def _validate_img2img(self) -> None:
        if self.source is None:
            raise ValueError("task が img2img のときは source を指定してください")

        # 入力画像のサイズをそのまま使うため、解像度の指定は効かない
        specified = self.generation.model_fields_set
        for field in ("width", "height"):
            if field in specified:
                raise ValueError(f"img2img では入力画像のサイズを使うため {field} は指定できません")

        if self.generation.batch_size > 1:
            raise ValueError("img2img では batch_size は1のみ対応しています")


__all__ = [
    "BoxSpec",
    "ControlSpec",
    "GenerationParams",
    "GenerationSpec",
    "IPAdapterWeightType",
    "ModelSpec",
    "OutputSpec",
    "PresetRefs",
    "PromptSpec",
    "ReferenceSpec",
    "SamplerName",
    "SchedulerName",
    "ShadowSpec",
    "SourceSpec",
    "StrokeSpec",
    "TextAlign",
    "TextAnchor",
    "TextDirection",
    "TextLayer",
    "TextSegment",
    "TextSpec",
    "UpscaleMethod",
    "UpscaleSpec",
]
