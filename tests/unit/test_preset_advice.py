"""style presetの選び忘れ・流用を検出する style_warnings のテスト。

生成の可否は変えない。検証は通るが絵柄が静かに変わる類の取りこぼしを、
validateの時点で言葉にすることだけを担う。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.services.preset_advice import style_warnings

HASSAKU = "hassakuSD15_v13.safetensors"
MEINAMIX = "meinamix_v12Final.safetensors"


@pytest.fixture
def presets_root(tmp_path: Path) -> Path:
    root = tmp_path / "presets"
    (root / "styles").mkdir(parents=True)
    (root / "characters").mkdir()
    (root / "scenes").mkdir()

    def write(name: str, document: dict[str, object]) -> None:
        (root / "styles" / f"{name}.yaml").write_text(
            yaml.safe_dump(document, allow_unicode=True), encoding="utf-8"
        )

    write("sd15-hassaku", {"applies_to": [HASSAKU], "model": {"clip_skip": 2}})
    write("sd15-meinamix", {"applies_to": [MEINAMIX], "model": {"clip_skip": 2}})
    write("anime-soft", {"description": "汎用"})
    return root


def _spec(checkpoint: str | None = HASSAKU, style: str | None = None) -> GenerationSpec:
    payload: dict[str, object] = {
        "version": "1",
        "task": "txt2img",
        "prompt": {"positive": "1girl"},
        "model": {"checkpoint": checkpoint},
    }
    if style is not None:
        payload["presets"] = {"style": style}
    return GenerationSpec.model_validate(payload)


class TestMissingStylePreset:
    def test_names_the_matching_preset(self, presets_root: Path) -> None:
        """checkpointに対応するstyle presetがあるなら名指しする。"""
        warnings = style_warnings(_spec(), presets_root=presets_root)

        assert len(warnings) == 1
        assert "sd15-hassaku" in warnings[0]

    def test_mentions_what_is_lost(self, presets_root: Path) -> None:
        """clip skipと外部VAEはstyle presetが持つ。落ちることを言葉にする。"""
        warnings = style_warnings(_spec(), presets_root=presets_root)

        assert "clip skip" in warnings[0]

    def test_warns_even_without_a_matching_preset(self, presets_root: Path) -> None:
        warnings = style_warnings(
            _spec(checkpoint="unknown_model.safetensors"), presets_root=presets_root
        )

        assert len(warnings) == 1
        assert "sd15-hassaku" not in warnings[0]


class TestStylePresetGiven:
    def test_matching_preset_is_silent(self, presets_root: Path) -> None:
        assert style_warnings(_spec(style="sd15-hassaku"), presets_root=presets_root) == ()

    def test_generic_preset_is_silent(self, presets_root: Path) -> None:
        """applies_to を持たないpresetは汎用。どのcheckpointでも咎めない。"""
        assert style_warnings(_spec(style="anime-soft"), presets_root=presets_root) == ()

    def test_mismatched_preset_is_reported(self, presets_root: Path) -> None:
        """別のcheckpoint向けのpresetを流用すると、samplerもcfgも噛み合わない。"""
        warnings = style_warnings(_spec(style="sd15-meinamix"), presets_root=presets_root)

        assert len(warnings) == 1
        assert "sd15-meinamix" in warnings[0]
        assert "sd15-hassaku" in warnings[0]

    def test_mismatched_preset_without_an_alternative(self, presets_root: Path) -> None:
        """行き先が無くても、噛み合っていないことは伝える。"""
        warnings = style_warnings(
            _spec(checkpoint="unknown_model.safetensors", style="sd15-hassaku"),
            presets_root=presets_root,
        )

        assert len(warnings) == 1
        assert "sd15-hassaku" in warnings[0]


class TestRobustness:
    def test_broken_preset_file_does_not_break_validate(self, presets_root: Path) -> None:
        (presets_root / "styles" / "broken.yaml").write_text("{ not yaml:", encoding="utf-8")

        warnings = style_warnings(_spec(), presets_root=presets_root)

        assert "sd15-hassaku" in warnings[0]

    def test_missing_presets_root_is_silent(self, tmp_path: Path) -> None:
        assert style_warnings(_spec(), presets_root=tmp_path / "nope") == ()
