"""同梱サンプルSpecが実際に読み込めることを検証する。

サンプルはドキュメントの一部であり、フィールド名や制約を変えたときに
黙って古くなりやすい。読み込み経路を通して壊れていないことを確認する。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_imagegen.services.spec_loader import load_spec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = PROJECT_ROOT / "specs" / "examples"
PRESETS_ROOT = PROJECT_ROOT / "presets"


def _examples() -> list[Path]:
    return sorted(EXAMPLES_DIR.glob("*.yaml"))


def test_examples_exist() -> None:
    """収集が空になっていないことの番人。"""
    assert _examples()


@pytest.mark.parametrize("example", _examples(), ids=lambda p: p.name)
def test_example_loads(example: Path) -> None:
    spec = load_spec(example, presets_root=PRESETS_ROOT)

    assert spec.prompt.positive
    assert spec.model.checkpoint


@pytest.mark.parametrize("example", _examples(), ids=lambda p: p.name)
def test_example_presets_exist(example: Path) -> None:
    """サンプルが参照するpresetが実在すること。"""
    spec = load_spec(example, presets_root=PRESETS_ROOT)

    for kind, name in (
        ("characters", spec.presets.character),
        ("scenes", spec.presets.scene),
        ("styles", spec.presets.style),
    ):
        if name is not None:
            assert (PRESETS_ROOT / kind / f"{name}.yaml").is_file()
