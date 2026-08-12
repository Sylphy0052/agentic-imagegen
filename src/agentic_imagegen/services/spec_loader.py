"""GenerationSpecのYAML読み込み。

Pydanticの ValidationError をそのままユーザーへ見せず、
どのフィールドが不正かを示す InvalidGenerationSpec へ変換する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.errors import InvalidGenerationSpec


def load_spec(path: Path) -> GenerationSpec:
    """YAMLファイルからGenerationSpecを読み込んで検証する。"""
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

    return parse_spec(raw, source=path)


def parse_spec(payload: dict[str, Any], *, source: Path | None = None) -> GenerationSpec:
    """辞書からGenerationSpecを組み立てる。CLI以外の入力経路からも利用する。"""
    try:
        return GenerationSpec.model_validate(payload)
    except ValidationError as exc:
        raise InvalidGenerationSpec(_format_validation_error(exc, source)) from exc


def _format_validation_error(exc: ValidationError, source: Path | None) -> str:
    lines = [f"Specの検証に失敗しました: {source}" if source else "Specの検証に失敗しました"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(root)"
        lines.append(f"  - {location}: {error['msg']}")
    return "\n".join(lines)


__all__ = ["load_spec", "parse_spec"]
