"""同梱presetが読み込めること、軸ごとの責務を守っていることを検証する。

tests/unit/test_presets.py はpresetの解決規則 (tmp_pathへ書いたpreset) を扱う。
こちらはリポジトリへ実際に置いてあるファイルを総なめする。

presetは増えやすく、追加時に軸の責務 (解像度とseedはSpec側、生成パラメータはstyle側)
が崩れても、参照するSpecが無ければ既存のテストでは気づけない。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_imagegen.domain.presets import PresetDocument, PresetKind
from agentic_imagegen.services.preset_loader import load_preset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRESETS_ROOT = PROJECT_ROOT / "presets"

#: 再現性に直結するため、どの軸のpresetへ書いてもいけないフィールド。
FORBIDDEN_EVERYWHERE = ("width", "height", "seed", "batch_size")

#: style presetが持つべき生成パラメータ。モデルの学習内容に依存するため軸ごと固定する。
REQUIRED_IN_STYLE = ("sampler", "scheduler", "cfg", "steps")


def _presets() -> list[tuple[PresetKind, str]]:
    found: list[tuple[PresetKind, str]] = []
    for kind in PresetKind:
        directory = PRESETS_ROOT / kind.directory
        found.extend((kind, path.stem) for path in sorted(directory.glob("*.yaml")))
    return found


def _load(kind: PresetKind, name: str) -> PresetDocument:
    return load_preset(kind, name, root=PRESETS_ROOT, project_root=PROJECT_ROOT)


def test_presets_exist() -> None:
    """収集が空になっていないことの番人。"""
    assert _presets()


@pytest.mark.parametrize(("kind", "name"), _presets(), ids=lambda v: str(v))
def test_preset_loads(kind: PresetKind, name: str) -> None:
    document = _load(kind, name)

    assert document.description, f"{kind.value}/{name} にdescriptionが無い"
    assert document.prompt.positive or document.prompt.negative


@pytest.mark.parametrize(("kind", "name"), _presets(), ids=lambda v: str(v))
def test_preset_omits_reproducibility_fields(kind: PresetKind, name: str) -> None:
    """解像度とseedはSpec側の責務。presetへ書くと再現性の出所が割れる。"""
    specified = _load(kind, name).generation.specified()

    leaked = [field for field in FORBIDDEN_EVERYWHERE if field in specified]
    assert not leaked, f"{kind.value}/{name} が {leaked} を持っている"


@pytest.mark.parametrize(
    ("kind", "name"),
    [(kind, name) for kind, name in _presets() if kind is not PresetKind.STYLE],
    ids=lambda v: str(v),
)
def test_non_style_preset_has_no_generation(kind: PresetKind, name: str) -> None:
    """サンプラー設定は画風とセットで決まるため、style以外へ置かない。"""
    assert not _load(kind, name).generation.specified()


@pytest.mark.parametrize(
    ("kind", "name"),
    [(kind, name) for kind, name in _presets() if kind is PresetKind.STYLE],
    ids=lambda v: str(v),
)
def test_style_preset_specifies_generation(kind: PresetKind, name: str) -> None:
    """styleを選べばサンプラー設定まで決まる状態にしておく。"""
    specified = _load(kind, name).generation.specified()

    missing = [field for field in REQUIRED_IN_STYLE if field not in specified]
    assert not missing, f"styles/{name} に {missing} が無い"
