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

from agentic_imagegen.domain.policy import display_path
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


def load_preset(
    kind: PresetKind, name: str, *, root: Path, project_root: Path | None = None
) -> PresetDocument:
    """1つのpresetファイルを読み込む。

    project_root を渡すと、エラーメッセージへ出すpresetの位置を作業ルートからの
    相対パスへ丸める (作業ルートの外を指す場合は絶対パスのまま)。
    """
    if not PRESET_NAME_PATTERN.fullmatch(name):
        raise InvalidGenerationSpec(
            f"preset名は英数字で始まり、英数字・ドット・アンダースコア・ハイフンのみ使用できます "
            f"({kind.value}: {name})"
        )

    path = root / kind.directory / f"{name}.yaml"
    shown = display_path(path, project_root)
    if not path.is_file():
        raise InvalidGenerationSpec(f"presetが見つかりません: {kind.value}/{name} ({shown})")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise InvalidGenerationSpec(f"presetのYAML解析に失敗しました: {shown}") from exc
    except OSError as exc:
        raise InvalidGenerationSpec(f"presetを読み込めません: {shown}") from exc

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise InvalidGenerationSpec(
            f"presetのトップレベルはマッピングである必要があります: {shown}"
        )

    try:
        return PresetDocument.model_validate(raw)
    except ValidationError as exc:
        raise InvalidGenerationSpec(_format_validation_error(exc, shown)) from exc


def apply_presets(
    payload: dict[str, Any], *, root: Path, project_root: Path | None = None
) -> tuple[dict[str, Any], dict[str, str]]:
    """Specのpayloadにpresetを適用し、(展開後payload, 適用したpreset名) を返す。

    優先順位は spec > style > scene > character。
    prompt は連結、generation と model は spec の指定を優先して補完する。
    """
    if PRESETS_KEY not in payload:
        return payload, {}

    refs = _parse_refs(payload[PRESETS_KEY])
    documents = [
        (kind, load_preset(kind, name, root=root, project_root=project_root))
        for kind, name in iter_preset_refs(refs)
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

    for key in ("generation", "model"):
        block = _merge_block(resolved, key, documents)
        if block:
            resolved[key] = block

    applied = {kind.value: name for kind, name in iter_preset_refs(refs)}
    return resolved, applied


def _merge_block(
    resolved: dict[str, Any],
    key: str,
    documents: list[tuple[PresetKind, PresetDocument]],
) -> dict[str, Any]:
    """presetの部分指定へSpecの指定を重ねた結果を返す。

    documents は PRESET_ORDER の順に並んでいるため、あとに来た軸が前を上書きし、
    最後にSpecの指定が全てに勝つ (spec > style > scene > character)。
    """
    block = resolved.get(key)
    if block is not None and not isinstance(block, dict):
        raise InvalidGenerationSpec(f"{key} はマッピングである必要があります")

    merged: dict[str, Any] = {}
    for _, document in documents:
        merged.update(getattr(document, key).specified())
    merged.update(dict(block or {}))
    return merged


def _parse_refs(value: Any) -> PresetRefs:
    if not isinstance(value, dict):
        raise InvalidGenerationSpec("presets はマッピングである必要があります")
    try:
        return PresetRefs.model_validate(value)
    except ValidationError as exc:
        raise InvalidGenerationSpec(_format_validation_error(exc, None)) from exc


def _format_validation_error(exc: ValidationError, source: str | None) -> str:
    lines = [f"presetの検証に失敗しました: {source}" if source else "presets の指定が不正です"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(root)"
        lines.append(f"  - {location}: {error['msg']}")
    return "\n".join(lines)


__all__ = ["PRESETS_KEY", "apply_presets", "load_preset"]
