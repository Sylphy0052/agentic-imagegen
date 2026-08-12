"""GenerationSpec: Agent/Core間の内部API契約。

この層はComfyUIのNode IDやHTTP仕様を一切知らない。
ここで行うのは「設定に依存しない」ハード制約の検証のみで、
環境変数由来の上限値は domain.policy 側で検証する。
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
_PREFIX_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


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


class ModelSpec(_StrictModel):
    """使用するモデル。"""

    checkpoint: str = Field(min_length=1)

    @field_validator("checkpoint")
    @classmethod
    def _reject_unsafe_path(cls, value: str) -> str:
        """checkpoint名にPath Traversalや想定外の拡張子を許さない。

        ComfyUIのcheckpointsディレクトリ配下を前提とし、サブフォルダは1階層まで許可する。
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

        if PurePosixPath(segments[-1]).suffix not in ALLOWED_CHECKPOINT_SUFFIXES:
            allowed = " / ".join(sorted(ALLOWED_CHECKPOINT_SUFFIXES))
            raise ValueError(f"拡張子は {allowed} のいずれかにしてください (指定値: {value})")
        return value


class OutputSpec(_StrictModel):
    """出力先。

    directory の実体解決とリポジトリ外への脱出検証は domain.policy が担う。
    """

    directory: str = "outputs"
    prefix: str = "imagegen"

    @field_validator("prefix")
    @classmethod
    def _validate_prefix(cls, value: str) -> str:
        if not _PREFIX_PATTERN.fullmatch(value):
            raise ValueError(
                "prefixは英数字で始まり、英数字・ドット・アンダースコア・ハイフンのみ使用できます"
            )
        return value


class GenerationSpec(_StrictModel):
    """画像生成の要求全体。Phase 1では txt2img のみ対応する。"""

    version: Literal["1"] = "1"
    task: Literal["txt2img"] = "txt2img"
    prompt: PromptSpec
    generation: GenerationParams = Field(default_factory=GenerationParams)
    model: ModelSpec
    output: OutputSpec = Field(default_factory=OutputSpec)


__all__ = [
    "GenerationParams",
    "GenerationSpec",
    "ModelSpec",
    "OutputSpec",
    "PromptSpec",
    "SamplerName",
    "SchedulerName",
]
