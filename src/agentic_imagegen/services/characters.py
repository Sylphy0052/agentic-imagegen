"""キャラクタ台帳の読み込み。

台帳は `registry/characters/<name>.yaml`。生成には関与せず、
「さっきの子で別の場面を」と言われたときに手掛かりを引くためだけに読む。

台帳は古びる。基準画像を消したりpresetをrenameしても、台帳側は何も知らない。
そのため引くたびに参照先の実在を確かめ、欠けていれば Character.missing へ入れる。
読めること自体は妨げない (seedだけでも分かれば手掛かりになる)。
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from agentic_imagegen.domain.characters import Character, CharacterRecord
from agentic_imagegen.domain.policy import display_path
from agentic_imagegen.domain.presets import PRESET_NAME_PATTERN, PresetKind
from agentic_imagegen.errors import InvalidGenerationSpec

#: 台帳の置き場 (registry_root からの相対)。
CHARACTERS_DIRECTORY = "characters"


def load_character(
    name: str,
    *,
    registry_root: Path,
    presets_root: Path,
    project_root: Path | None = None,
) -> Character:
    """台帳を1件読む。見つからない・壊れている場合は InvalidGenerationSpec。"""
    if not PRESET_NAME_PATTERN.fullmatch(name):
        raise InvalidGenerationSpec(
            "キャラクタ名は英数字で始まり、英数字・ドット・アンダースコア・ハイフンのみ"
            f"使用できます (指定値: {name})"
        )

    path = registry_root / CHARACTERS_DIRECTORY / f"{name}.yaml"
    shown = display_path(path, project_root)
    if not path.is_file():
        raise InvalidGenerationSpec(f"キャラクタが見つかりません: {name} ({shown})")

    record = _read(path, shown)
    return Character(
        name=name,
        record=record,
        missing=_missing(record, presets_root=presets_root, project_root=project_root),
    )


def collect_characters(
    *,
    registry_root: Path,
    presets_root: Path,
    project_root: Path | None = None,
) -> tuple[Character, ...]:
    """読める台帳を名前順に集める。

    1件壊れていても残りは返す。全部が引けなくなるより、壊れた1件が
    一覧から消えるほうが台帳として使える。
    """
    directory = registry_root / CHARACTERS_DIRECTORY
    if not directory.is_dir():
        return ()

    characters: list[Character] = []
    for path in sorted(directory.glob("*.yaml")):
        try:
            characters.append(
                load_character(
                    path.stem,
                    registry_root=registry_root,
                    presets_root=presets_root,
                    project_root=project_root,
                )
            )
        except InvalidGenerationSpec:
            continue
    return tuple(characters)


def _read(path: Path, shown: str) -> CharacterRecord:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise InvalidGenerationSpec(f"台帳のYAML解析に失敗しました: {shown}") from exc
    except OSError as exc:
        raise InvalidGenerationSpec(f"台帳を読み込めません: {shown}") from exc

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise InvalidGenerationSpec(f"台帳のトップレベルはマッピングである必要があります: {shown}")

    try:
        return CharacterRecord.model_validate(raw)
    except ValidationError as exc:
        lines = [f"台帳の検証に失敗しました: {shown}"]
        for error in exc.errors():
            location = ".".join(str(part) for part in error["loc"]) or "(root)"
            lines.append(f"  - {location}: {error['msg']}")
        raise InvalidGenerationSpec("\n".join(lines)) from exc


def _missing(
    record: CharacterRecord, *, presets_root: Path, project_root: Path | None
) -> tuple[str, ...]:
    root = project_root or Path.cwd()
    candidates: list[tuple[str | None, Path]] = [
        (
            record.preset,
            presets_root / PresetKind.CHARACTER.directory / f"{record.preset}.yaml",
        ),
        (record.style, presets_root / PresetKind.STYLE.directory / f"{record.style}.yaml"),
        (record.reference, root / (record.reference or "")),
    ]
    missing: list[str] = []
    for value, path in candidates:
        if value is None or path.is_file():
            continue
        shown = display_path(path, project_root)
        # presetは名前とパスが違うため両方出す。基準画像はパスそのものなので繰り返さない。
        missing.append(shown if shown == value else f"{value} ({shown})")
    return tuple(missing)


__all__ = ["CHARACTERS_DIRECTORY", "collect_characters", "load_character"]
