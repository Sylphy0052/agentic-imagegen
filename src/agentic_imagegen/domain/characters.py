"""キャラクタ台帳の型。

「さっきの子で別の場面を」を再現するには、character preset・style preset・checkpoint・
基準画像・seedの5つが揃っている必要がある。character presetは外見的特徴の軸であり
checkpointに依存しないという責務を保ちたいため、この5つはpresetへは足さず台帳へ置く。

台帳はGenerationSpecではない。生成の可否には関与せず、Specを書くときの手掛かりを
1か所へまとめるだけのもの。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentic_imagegen.domain.models import (
    ALLOWED_CHECKPOINT_SUFFIXES,
    MAX_SEED,
    validate_model_filename,
    validate_relative_image_path,
)
from agentic_imagegen.domain.presets import PRESET_NAME_PATTERN


class _StrictModel(BaseModel):
    """未知キーを拒否し、生成後の変更を禁止する共通設定。"""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class CharacterRecord(_StrictModel):
    """台帳1件分の内容。"""

    description: str = ""
    #: presets/characters/ のファイル名 (拡張子なし)。
    preset: str | None = None
    #: presets/styles/ のファイル名。顔立ちは画風とcheckpointにも依存する。
    style: str | None = None
    #: 基準画像を作ったcheckpoint。変えるとIPAdapterをかけても別人に見える。
    checkpoint: str | None = None
    #: IPAdapterへ渡す基準画像。リポジトリ配下の相対パス。
    reference: str | None = None
    #: 基準画像を作ったときのseed。
    seed: Annotated[int, Field(ge=0, le=MAX_SEED)] | None = None
    notes: str = ""

    @field_validator("preset", "style")
    @classmethod
    def _reject_unsafe_preset_name(cls, value: str | None) -> str | None:
        if value is not None and not PRESET_NAME_PATTERN.fullmatch(value):
            raise ValueError(
                "preset名は英数字で始まり、英数字・ドット・アンダースコア・ハイフンのみ使用できます"
            )
        return value

    @field_validator("checkpoint")
    @classmethod
    def _reject_unsafe_checkpoint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_model_filename(value, allowed_suffixes=ALLOWED_CHECKPOINT_SUFFIXES)

    @field_validator("reference")
    @classmethod
    def _reject_unsafe_reference(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_relative_image_path(value)


@dataclass(frozen=True)
class Character:
    """台帳1件と、その参照先の状態。"""

    name: str
    record: CharacterRecord
    #: 台帳が指しているのに実在しなかったもの。台帳は古びるため、引くたびに確かめる。
    missing: tuple[str, ...] = ()


__all__ = ["Character", "CharacterRecord"]
