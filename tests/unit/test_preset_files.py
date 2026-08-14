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


#: 汎用のアニメ調タグをそのまま共有するSD1.5系のstyle preset。
#: checkpointごとの差はサンプラー設定にあり、プロンプトは同一でよい。
SHARED_PROMPT_STYLES = frozenset(
    {
        "sd15-aom3",
        "sd15-cetusmix",
        "sd15-counterfeit",
        "sd15-darksushi",
        "sd15-hassaku",
        "sd15-meinamix",
        "sd15-perfectdeliberate",
    }
)

#: プロンプトを共有しない理由があるSD1.5系のstyle preset。
#: sd15-anylora はLoRAの画風と競合させないため品質タグだけに絞る。
#: sd15-chilloutmix は写実寄りのモデルのため語彙ごと差し替える。
#: sd15-wai-illustrious はIllustrious系のタグ記法に合わせて very aesthetic を足す。
DIVERGENT_PROMPT_STYLES = frozenset({"sd15-anylora", "sd15-chilloutmix", "sd15-wai-illustrious"})


def _sd15_styles() -> set[str]:
    return {
        name for kind, name in _presets() if kind is PresetKind.STYLE and name.startswith("sd15-")
    }


def test_sd15_styles_are_classified() -> None:
    """SD1.5系を足したら、プロンプトを共有するかどうかをここで決める。

    分類から漏れると、下のテストが新しいpresetを素通りしてしまう。
    """
    assert _sd15_styles() == SHARED_PROMPT_STYLES | DIVERGENT_PROMPT_STYLES


def test_shared_prompt_styles_stay_identical() -> None:
    """共有側のプロンプトが1つだけ書き換わる事故を防ぐ。

    タグの是正はSD1.5系へ一斉に効かせたい。1件だけ直して他が追従しないと、
    どれが最新か分からなくなる。
    """
    prompts = {
        name: (
            _load(PresetKind.STYLE, name).prompt.positive,
            _load(PresetKind.STYLE, name).prompt.negative,
        )
        for name in sorted(SHARED_PROMPT_STYLES)
    }
    distinct = set(prompts.values())

    assert len(distinct) == 1, f"共有しているはずのプロンプトが割れている: {prompts}"


@pytest.mark.parametrize("name", sorted(DIVERGENT_PROMPT_STYLES))
def test_divergent_prompt_styles_actually_differ(name: str) -> None:
    """差分を持つと宣言したpresetが、実際には共有側と同じになっていないこと。"""
    shared = _load(PresetKind.STYLE, sorted(SHARED_PROMPT_STYLES)[0]).prompt
    prompt = _load(PresetKind.STYLE, name).prompt

    assert (prompt.positive, prompt.negative) != (shared.positive, shared.negative)
