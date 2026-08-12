"""presetの読み込みと、GenerationSpecへの展開。

apply_presets はSpecのpayload (dict) を受け取り、presetを解決した
payloadを返す。返した時点で presets キーは消えており、以降の層は
presetの存在を知らない。
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from agentic_imagegen.domain.presets import (
    PRESET_NAME_PATTERN,
    PresetDocument,
    PresetKind,
    PresetRefs,
    iter_preset_refs,
    merge_prompt_fragments,
)
from agentic_imagegen.errors import InvalidGenerationSpec

#: Specの presets: ブロックのキー。
PRESETS_KEY = "presets"


def load_preset(kind: PresetKind, name: str, *, root: Path) -> PresetDocument:
    """1つのpresetファイルを読み込む。"""
    if not PRESET_NAME_PATTERN.fullmatch(name):
        raise InvalidGenerationSpec(
            f"preset名は英数字で始まり、英数字・ドット・アンダースコア・ハイフンのみ使用できます "
            f"({kind.value}: {name})"
        )

    path = root / kind.directory / f"{name}.yaml"
    if not path.is_file():
        raise InvalidGenerationSpec(f"presetが見つかりません: {kind.value}/{name} ({path})")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise InvalidGenerationSpec(f"presetのYAML解析に失敗しました: {path}") from exc
    except OSError as exc:
        raise InvalidGenerationSpec(f"presetを読み込めません: {path}") from exc

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise InvalidGenerationSpec(f"presetのトップレベルはマッピングである必要があります: {path}")

    try:
        return PresetDocument.model_validate(raw)
    except ValidationError as exc:
        raise InvalidGenerationSpec(_format_validation_error(exc, path)) from exc


def apply_presets(payload: dict[str, Any], *, root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Specのpayloadにpresetを適用し、(展開後payload, 適用したpreset名) を返す。

    優先順位は spec > style > scene > character。
    prompt は連結、それ以外の生成パラメータは spec の指定を優先して補完する。
    """
    if PRESETS_KEY not in payload:
        return payload, {}

    refs = _parse_refs(payload[PRESETS_KEY])
    documents = [
        (kind, load_preset(kind, name, root=root)) for kind, name in iter_preset_refs(refs)
    ]

    resolved = copy.deepcopy(payload)
    resolved.pop(PRESETS_KEY)

    prompt = resolved.get("prompt")
    if prompt is not None and not isinstance(prompt, dict):
        raise InvalidGenerationSpec("prompt はマッピングである必要があります")
    prompt = dict(prompt or {})

    for field in ("positive", "negative"):
        fragments = [getattr(doc.prompt, field) for _, doc in documents]
        fragments.append(str(prompt.get(field, "")))
        merged = merge_prompt_fragments(fragments)
        if merged:
            prompt[field] = merged
    resolved["prompt"] = prompt

    generation = resolved.get("generation")
    if generation is not None and not isinstance(generation, dict):
        raise InvalidGenerationSpec("generation はマッピングである必要があります")
    merged_generation: dict[str, Any] = {}
    for _, doc in documents:
        merged_generation.update(doc.generation.specified())
    merged_generation.update(dict(generation or {}))
    if merged_generation:
        resolved["generation"] = merged_generation

    applied = {kind.value: name for kind, name in iter_preset_refs(refs)}
    return resolved, applied


def _parse_refs(value: Any) -> PresetRefs:
    if not isinstance(value, dict):
        raise InvalidGenerationSpec("presets はマッピングである必要があります")
    try:
        return PresetRefs.model_validate(value)
    except ValidationError as exc:
        raise InvalidGenerationSpec(_format_validation_error(exc, None)) from exc


def _format_validation_error(exc: ValidationError, source: Path | None) -> str:
    lines = [f"presetの検証に失敗しました: {source}" if source else "presets の指定が不正です"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(root)"
        lines.append(f"  - {location}: {error['msg']}")
    return "\n".join(lines)


__all__ = ["PRESETS_KEY", "apply_presets", "load_preset"]
