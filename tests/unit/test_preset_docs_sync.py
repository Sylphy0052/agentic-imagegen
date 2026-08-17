"""presetの生成パラメータが、それを説明する2つの表と食い違わないことを検証する。

同じ値が3箇所にある。

- `presets/styles/*.yaml` — 実体
- `docs/spec-reference.md` のpreset一覧 — presetが採用した値
- `.claude/skills/prompt-builder/references/models/sd15.md` の「配置済みのSD1.5系モデル」
  — 配布元・利用者の推奨レンジ

役割は違うが、sampler / schedulerは両方の表に出てくるうえ、採用値が推奨レンジから
外れていないことは誰も見ていない。presetを1つ足したときに表の更新を忘れても、
参照するSpecが無ければ既存のテストでは気づけない。ここで機械的に突き合わせる。
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from agentic_imagegen.domain.presets import PresetDocument, PresetKind
from agentic_imagegen.services.preset_loader import load_preset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRESETS_ROOT = PROJECT_ROOT / "presets"
SPEC_REFERENCE = PROJECT_ROOT / "docs" / "spec-reference.md"
SD15_REFERENCE = (
    PROJECT_ROOT / ".claude" / "skills" / "prompt-builder" / "references" / "models" / "sd15.md"
)

#: 「7前後」と書かれた推奨に対して許容する幅。
AROUND_TOLERANCE = 0.2

#: 一覧の model 列の表記と、presetが実際に持つ model の対応。
#: 「外部VAE」はアニメ調のSD1.5系で共通して使う vaeKlF8Anime2。
EXTERNAL_VAE = "vaeKlF8Anime2_klF8Anime2VAE.safetensors"
MODEL_CELLS: dict[str, dict[str, object]] = {
    "clip_skip 2 + 外部VAE": {"clip_skip": 2, "vae": EXTERNAL_VAE},
    "clip_skip 2": {"clip_skip": 2},
    "clip_skip 1": {"clip_skip": 1},
    "-": {},
}

#: sampler / scheduler / cfg / stepsを配布元の推奨ではなく、実機で生成した絵から選んだ
#: style preset。推奨表と食い違っていてよい。
#: sd15-anylora / sd15-meinamix は同じcheckpointをA1111で運用したときの実績設定
#: (.claude/skills/prompt-builder/references/a1111-migration.md)。
#: sd15-hassaku はSD1.5系9種を同一条件で比較して既定に選んだときの設定
#: (同 models/sd15.md の「既定のcheckpointを決める」)。
MEASURED_SETTING_STYLES = frozenset({"sd15-anylora", "sd15-hassaku", "sd15-meinamix"})


def _strip_code(cell: str) -> str:
    return cell.replace("`", "").strip()


def _tables(markdown: Path) -> list[list[list[str]]]:
    """Markdownの表を、行 -> セルの二次元リストとして全て取り出す。"""
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []

    for line in markdown.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if all(set(cell) <= {"-", ":"} for cell in cells):
                continue  # 区切り行
            current.append(cells)
            continue
        if current:
            tables.append(current)
            current = []

    if current:
        tables.append(current)
    return tables


def _table_with_header(markdown: Path, first_header: str) -> list[list[str]]:
    """先頭セルが first_header の表を1つ返す。本文とデータ行だけを含む。"""
    for table in _tables(markdown):
        if table and _strip_code(table[0][0]) == first_header:
            return table[1:]
    raise AssertionError(f"{markdown.name} に先頭セルが {first_header!r} の表が無い")


def _split_sampler(cell: str) -> tuple[str, str]:
    sampler, _, scheduler = _strip_code(cell).partition("/")
    return sampler.strip(), scheduler.strip().replace("`", "")


def _parse_range(text: str) -> tuple[float, float]:
    """「20-60」「6以上」「7前後」「7.5」を下限と上限に開く。"""
    value = _strip_code(text)
    if match := re.fullmatch(r"([\d.]+)-([\d.]+)", value):
        return float(match[1]), float(match[2])
    if match := re.fullmatch(r"([\d.]+)以上", value):
        return float(match[1]), math.inf
    if match := re.fullmatch(r"([\d.]+)前後", value):
        center = float(match[1])
        return center * (1 - AROUND_TOLERANCE), center * (1 + AROUND_TOLERANCE)
    if re.fullmatch(r"[\d.]+", value):
        return float(value), float(value)
    raise AssertionError(f"推奨レンジとして解釈できない: {text!r}")


def _load(name: str) -> PresetDocument:
    return load_preset(PresetKind.STYLE, name, root=PRESETS_ROOT, project_root=PROJECT_ROOT)


def _style_names() -> list[str]:
    return sorted(path.stem for path in (PRESETS_ROOT / "styles").glob("*.yaml"))


def _reference_rows() -> list[list[str]]:
    return _table_with_header(SPEC_REFERENCE, "style preset")


def _guide_rows() -> list[list[str]]:
    return _table_with_header(SD15_REFERENCE, "checkpoint")


def test_tables_are_found() -> None:
    """表の抽出が空になっていないことの番人。"""
    assert _reference_rows()
    assert _guide_rows()
    assert _style_names()


def test_reference_table_covers_every_style_preset() -> None:
    """presetを足して表への追記を忘れると、一覧が実態と合わなくなる。"""
    listed = {_strip_code(row[0]) for row in _reference_rows()}

    assert listed == set(_style_names())


@pytest.mark.parametrize("row", _reference_rows(), ids=lambda row: _strip_code(row[0]))
def test_reference_table_matches_preset(row: list[str]) -> None:
    """spec-reference.md のpreset一覧はpresetの採用値をそのまま載せる。"""
    name, _target, sampler_cell, cfg, steps = row[:5]
    generation = _load(_strip_code(name)).generation
    sampler, scheduler = _split_sampler(sampler_cell)

    assert generation.sampler == sampler
    assert generation.scheduler == scheduler
    assert generation.cfg == pytest.approx(float(_strip_code(cfg)))
    assert generation.steps == int(_strip_code(steps))


@pytest.mark.parametrize("row", _reference_rows(), ids=lambda row: _strip_code(row[0]))
def test_reference_table_matches_applies_to(row: list[str]) -> None:
    """対象列がcheckpointを名指ししているpresetは、それをapplies_toにも持つ。

    applies_to は validate が style preset の取り違えを警告するためだけの情報で、
    Specへは展開されない。書き忘れても生成は成功するため、表との一致で担保する。
    """
    name, target = _strip_code(row[0]), row[1].strip()
    match = re.match(r"`([^`]+)`", target)
    expected = (f"{match[1]}.safetensors",) if match else ()

    assert _load(name).applies_to == expected


@pytest.mark.parametrize("row", _guide_rows(), ids=lambda row: _strip_code(row[0]))
def test_guide_table_preset_exists(row: list[str]) -> None:
    """表が指すstyle presetが実在すること。"""
    name = _strip_code(row[5])

    assert name in _style_names()


@pytest.mark.parametrize("row", _guide_rows(), ids=lambda row: _strip_code(row[0]))
def test_preset_follows_recommended_settings(row: list[str]) -> None:
    """採用値が配布元・利用者の推奨から外れていないこと。"""
    _checkpoint, _tendency, sampler_cell, steps_cell, cfg_cell, name_cell = row[:6]
    name = _strip_code(name_cell)
    if name in MEASURED_SETTING_STYLES:
        return

    generation = _load(name).generation
    sampler, scheduler = _split_sampler(sampler_cell)

    assert generation.sampler == sampler
    assert generation.scheduler == scheduler

    for label, value, cell in (
        ("steps", generation.steps, steps_cell),
        ("cfg", generation.cfg, cfg_cell),
    ):
        assert value is not None
        low, high = _parse_range(cell)
        assert low <= value <= high, f"{name} の{label} {value} が推奨 {cell} の外"


@pytest.mark.parametrize("row", _reference_rows(), ids=lambda row: _strip_code(row[0]))
def test_reference_table_matches_preset_model(row: list[str]) -> None:
    """clip_skipと外部VAEをpresetへ移したあと、一覧の表記が実体とずれないようにする。

    この2つは書き忘れても検証が通り生成も成功するため、表だけが古くなっても
    生成では気づけない。
    """
    name, cell = _strip_code(row[0]), _strip_code(row[5])

    assert cell in MODEL_CELLS, f"{name} の model 列 {cell!r} は表記として未定義"
    assert _load(name).model.specified() == MODEL_CELLS[cell]
