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

SamplerName = Literal[
    "euler",
    "euler_ancestral",
    "heun",
    "heunpp2",
    "dpm_2",
    "dpm_2_ancestral",
    "lms",
    "dpm_fast",
    "dpm_adaptive",
    "dpmpp_2s_ancestral",
    "dpmpp_sde",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_3m_sde",
    "ddpm",
    "lcm",
    "ddim",
    "uni_pc",
    "uni_pc_bh2",
]

SchedulerName = Literal[
    "normal",
    "karras",
    "exponential",
    "sgm_uniform",
    "simple",
    "ddim_uniform",
    "beta",
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

#: strengthの実用上の範囲。ComfyUI自体は±100を許すが、事故を防ぐため絞る。
LORA_STRENGTH_LIMIT: Final = 10.0

#: img2imgの入力画像として受け付ける拡張子。
ALLOWED_SOURCE_IMAGE_SUFFIXES: Final = frozenset({".png", ".jpg", ".jpeg", ".webp"})

_PREFIX_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _validate_model_filename(value: str, *, allowed_suffixes: frozenset[str]) -> str:
    """モデルファイル名にPath Traversalや想定外の拡張子を許さない。

    ComfyUIの各modelsディレクトリ配下を前提とし、サブフォルダは1階層まで許可する。
    checkpointとLoRAで共通の規則。
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

    if PurePosixPath(segments[-1]).suffix not in allowed_suffixes:
        allowed = " / ".join(sorted(allowed_suffixes))
        raise ValueError(f"拡張子は {allowed} のいずれかにしてください (指定値: {value})")
    return value


class _StrictModel(BaseModel):
    """未知キーを拒否し、生成後の変更を禁止する共通設定。"""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class PromptSpec(_StrictModel):
    """プロンプト。"""

    positive: str = Field(min_length=1)
    negative: str = ""


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
    """使用するモデル。"""

    checkpoint: str = Field(min_length=1)
    #: 適用順に並べる。Workflowテンプレートの LoraLoader の段へ先頭から割り当てる。
    loras: tuple[LoraSpec, ...] = ()

    @field_validator("checkpoint")
    @classmethod
    def _reject_unsafe_path(cls, value: str) -> str:
        return _validate_model_filename(value, allowed_suffixes=ALLOWED_CHECKPOINT_SUFFIXES)

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
    output: OutputSpec = Field(default_factory=OutputSpec)

    @model_validator(mode="after")
    def _validate_task_combination(self) -> GenerationSpec:
        """taskと他フィールドの組み合わせを検証する。

        指定しても効かない項目は黙って無視せず拒否する。書いたのに反映されていない
        状態は、生成結果を見ても原因が分かりにくいため。
        """
        if self.task == "img2img":
            self._validate_img2img()
        elif self.source is not None:
            raise ValueError("source は task が img2img のときにのみ指定できます")
        return self

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

        if self.model.loras:
            raise ValueError("img2img とLoRAの組み合わせは未対応です")


__all__ = [
    "GenerationParams",
    "GenerationSpec",
    "ModelSpec",
    "OutputSpec",
    "PresetRefs",
    "PromptSpec",
    "SamplerName",
    "SchedulerName",
    "SourceSpec",
]
