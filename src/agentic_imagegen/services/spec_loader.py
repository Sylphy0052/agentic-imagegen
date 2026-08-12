"""GenerationSpecのYAML読み込み。

Pydanticの ValidationError をそのままユーザーへ見せず、
どのフィールドが不正かを示す InvalidGenerationSpec へ変換する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from agentic_imagegen.domain.models import GenerationSpec, TextSpec
from agentic_imagegen.errors import InvalidGenerationSpec
from agentic_imagegen.services.preset_loader import PRESETS_KEY, apply_presets

#: 生成用Specの中でテキスト合成の定義を置くキー。
TEXT_KEY = "text"


def load_spec(path: Path, *, presets_root: Path | None = None) -> GenerationSpec:
    """YAMLファイルからGenerationSpecを読み込んで検証する。

    presets: ブロックがある場合は presets_root 配下のpresetを解決してから検証する。
    """
    return parse_spec(_load_mapping(path), source=path, presets_root=presets_root)


def parse_spec(
    payload: dict[str, Any],
    *,
    source: Path | None = None,
    presets_root: Path | None = None,
) -> GenerationSpec:
    """辞書からGenerationSpecを組み立てる。CLI以外の入力経路からも利用する。

    presetの展開はこの関数を必ず通す。presets: があるのに presets_root が無い場合は
    黙って無視せずエラーにする。展開されないまま下層へ流れると、
    presetを書いたのに効いていない状態に気づけないため。
    """
    resolved = payload
    if PRESETS_KEY in payload:
        if presets_root is None:
            raise InvalidGenerationSpec(
                "presets: が指定されていますが、presetの探索ルートが渡されていません"
            )
        resolved, applied = apply_presets(payload, root=presets_root)
        if applied:
            resolved[PRESETS_KEY] = applied

    try:
        return GenerationSpec.model_validate(resolved)
    except ValidationError as exc:
        raise InvalidGenerationSpec(_format_validation_error(exc, source)) from exc


def load_text_spec(path: Path) -> TextSpec:
    """テキスト合成の定義だけをYAMLから読み込む。

    生成用のSpecをそのまま渡した場合は text: ブロックを読む。テキスト定義だけを
    書いたファイルも受け付ける。同じ内容を2通りの形で書けるようにして、
    生成に使ったSpecを合成のやり直しへ流用できるようにする。
    """
    raw = _load_mapping(path)
    payload = raw.get(TEXT_KEY, raw)

    if not isinstance(payload, dict):
        raise InvalidGenerationSpec(f"text: はマッピングである必要があります: {path}")

    try:
        return TextSpec.model_validate(payload)
    except ValidationError as exc:
        raise InvalidGenerationSpec(_format_validation_error(exc, path)) from exc


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise InvalidGenerationSpec(f"Specファイルが見つかりません: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise InvalidGenerationSpec(f"YAMLの解析に失敗しました: {path}") from exc
    except OSError as exc:
        raise InvalidGenerationSpec(f"Specファイルを読み込めません: {path}") from exc

    if not isinstance(raw, dict):
        raise InvalidGenerationSpec(f"Specのトップレベルはマッピングである必要があります: {path}")
    return raw


def _format_validation_error(exc: ValidationError, source: Path | None) -> str:
    lines = [f"Specの検証に失敗しました: {source}" if source else "Specの検証に失敗しました"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(root)"
        lines.append(f"  - {location}: {error['msg']}")
    return "\n".join(lines)


__all__ = ["load_spec", "load_text_spec", "parse_spec"]
