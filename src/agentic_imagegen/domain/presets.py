"""Preset: GenerationSpecの部分指定を名前で再利用するための仕組み。

presetは入力表現であり、内部API契約 (GenerationSpec) ではない。
services 側で解決したあとは preset の存在が下層へ漏れないようにする。

軸は character / scene / style の3つ。同じ観点の指定が複数箇所に散らないよう、
1軸につき1つまでとする。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from enum import StrEnum
from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentic_imagegen.domain.models import (
    ALLOWED_CHECKPOINT_SUFFIXES,
    MAX_CLIP_SKIP,
    MAX_DIMENSION,
    MAX_SEED,
    MIN_CLIP_SKIP,
    MIN_DIMENSION,
    RANDOM_SEED,
    PresetRefs,
    SamplerName,
    SchedulerName,
    validate_model_filename,
)

#: preset名はファイル名になる。Path Traversalと紛らわしい名前を機械的に排除する。
PRESET_NAME_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class PresetKind(StrEnum):
    """presetの軸。"""

    CHARACTER = "character"
    SCENE = "scene"
    STYLE = "style"

    @property
    def directory(self) -> str:
        """presetルート配下のサブディレクトリ名。"""
        return f"{self.value}s"


#: prompt断片を連結する順序。あとに来るものほどSpec本体に近い。
PRESET_ORDER: Final = (PresetKind.CHARACTER, PresetKind.SCENE, PresetKind.STYLE)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class PresetPrompt(_StrictModel):
    """presetが提供するprompt断片。"""

    positive: str = ""
    negative: str = ""


class PresetGeneration(_StrictModel):
    """生成パラメータの部分指定。

    未指定は None のままにしておき、GenerationParams の既定値を勝手に埋めない。
    埋めてしまうと「presetが明示した値」と「既定値」を区別できなくなる。
    """

    width: Annotated[int, Field(ge=MIN_DIMENSION, le=MAX_DIMENSION)] | None = None
    height: Annotated[int, Field(ge=MIN_DIMENSION, le=MAX_DIMENSION)] | None = None
    steps: Annotated[int, Field(ge=1, le=100)] | None = None
    cfg: Annotated[float, Field(ge=0, le=30)] | None = None
    seed: Annotated[int, Field(ge=RANDOM_SEED, le=MAX_SEED)] | None = None
    batch_size: Annotated[int, Field(ge=1, le=4)] | None = None
    sampler: SamplerName | None = None
    scheduler: SchedulerName | None = None

    def specified(self) -> dict[str, object]:
        """明示的に指定されたフィールドだけを返す。"""
        return {key: value for key, value in self.model_dump().items() if value is not None}


class PresetModel(_StrictModel):
    """モデル設定の部分指定。

    絵柄は checkpoint だけでは決まらない。CLIPをどこで打ち切るか (`clip_skip`) と
    どのVAEでデコードするか (`vae`) でも変わるため、そのcheckpointで検証した値を
    style preset 側へ置けるようにする。Spec側へ手で書く形にすると、書き忘れても
    検証は通り生成も成功し、出来上がった絵だけが静かに変わる。

    `checkpoint` と loader周り (`unet` / `clip` / `loras`) は持たせない。
    どのモデルで描くかは利用者が意識して選ぶ値であり、Spec側の責務として残す。

    未指定は None のままにしておき、ModelSpec の既定値を勝手に埋めない
    (PresetGeneration と同じ理由)。
    """

    clip_skip: Annotated[int, Field(ge=MIN_CLIP_SKIP, le=MAX_CLIP_SKIP)] | None = None
    vae: str | None = None

    @field_validator("vae")
    @classmethod
    def _reject_unsafe_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_model_filename(value, allowed_suffixes=ALLOWED_CHECKPOINT_SUFFIXES)

    def specified(self) -> dict[str, object]:
        """明示的に指定されたフィールドだけを返す。"""
        return {key: value for key, value in self.model_dump().items() if value is not None}


class PresetDocument(_StrictModel):
    """presetファイル1つ分の内容。"""

    description: str = ""
    prompt: PresetPrompt = Field(default_factory=PresetPrompt)
    generation: PresetGeneration = Field(default_factory=PresetGeneration)
    model: PresetModel = Field(default_factory=PresetModel)


def iter_preset_refs(refs: PresetRefs) -> tuple[tuple[PresetKind, str], ...]:
    """指定された軸を PRESET_ORDER の順に返す。

    PresetRefs 自体は GenerationSpec の一部として domain.models にある
    (models が presets を import すると循環するため)。軸の順序はこちらが持つ。
    """
    names: dict[PresetKind, str | None] = {
        PresetKind.CHARACTER: refs.character,
        PresetKind.SCENE: refs.scene,
        PresetKind.STYLE: refs.style,
    }
    return tuple((kind, name) for kind in PRESET_ORDER if (name := names[kind]) is not None)


def merge_prompt_fragments(fragments: Iterable[str]) -> str:
    """prompt断片をカンマ区切りで連結し、重複トークンを取り除く。

    重複判定は大文字小文字と連続空白を無視する。表記は最初に現れたものを残す。
    単純に連結すると preset を重ねるほどトークンが膨らみ、
    CLIPの75トークン制限を圧迫するうえ、意図しない強調が起きる。
    """
    seen: set[str] = set()
    tokens: list[str] = []

    for fragment in fragments:
        for raw in fragment.split(","):
            token = " ".join(raw.split())
            if not token:
                continue
            key = token.casefold()
            if key in seen:
                continue
            seen.add(key)
            tokens.append(token)

    return ", ".join(tokens)


__all__ = [
    "PRESET_NAME_PATTERN",
    "PRESET_ORDER",
    "PresetDocument",
    "PresetGeneration",
    "PresetKind",
    "PresetModel",
    "PresetPrompt",
    "PresetRefs",
    "iter_preset_refs",
    "merge_prompt_fragments",
]
