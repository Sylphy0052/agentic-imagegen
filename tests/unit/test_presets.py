"""Preset systemのUnit Test。

presetは「GenerationSpecの部分指定」であり、解決後はpresetの存在を
下層 (Workflow / Adapter) に見せない。ここではその解決規則を検証する。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_imagegen.domain.presets import (
    PresetKind,
    PresetRefs,
    merge_prompt_fragments,
)
from agentic_imagegen.errors import InvalidGenerationSpec
from agentic_imagegen.services.preset_loader import apply_presets, load_preset


def _write(root: Path, kind: str, name: str, body: str) -> Path:
    directory = root / kind
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.yaml"
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture
def presets_root(tmp_path: Path) -> Path:
    root = tmp_path / "presets"
    _write(
        root,
        "characters",
        "kaede",
        """
        description: テスト用キャラクタ
        prompt:
          positive: 1girl, solo, blue hair, blue eyes
          negative: bad anatomy
        """,
    )
    _write(
        root,
        "scenes",
        "rooftop",
        """
        prompt:
          positive: rooftop, sunset, wind
        """,
    )
    _write(
        root,
        "styles",
        "anime-soft",
        """
        prompt:
          positive: anime illustration, masterpiece, best quality
          negative: worst quality
        generation:
          sampler: dpmpp_2m
          scheduler: karras
          cfg: 5.5
        """,
    )
    return root


class TestLoadPreset:
    def test_loads_preset(self, presets_root: Path) -> None:
        preset = load_preset(PresetKind.CHARACTER, "kaede", root=presets_root)

        assert preset.prompt.positive == "1girl, solo, blue hair, blue eyes"
        assert preset.prompt.negative == "bad anatomy"
        assert preset.description == "テスト用キャラクタ"

    def test_keeps_partial_generation(self, presets_root: Path) -> None:
        preset = load_preset(PresetKind.STYLE, "anime-soft", root=presets_root)

        assert preset.generation.sampler == "dpmpp_2m"
        assert preset.generation.cfg == 5.5
        # 未指定のフィールドは None のままで、既定値を勝手に埋めない
        assert preset.generation.steps is None
        assert preset.generation.width is None

    def test_rejects_missing_preset(self, presets_root: Path) -> None:
        with pytest.raises(InvalidGenerationSpec, match="見つかりません"):
            load_preset(PresetKind.CHARACTER, "unknown", root=presets_root)

    def test_rejects_unknown_key(self, presets_root: Path) -> None:
        _write(presets_root, "characters", "broken", "prompt:\n  positve: typo\n")

        with pytest.raises(InvalidGenerationSpec):
            load_preset(PresetKind.CHARACTER, "broken", root=presets_root)

    @pytest.mark.parametrize(
        "name",
        ["../secret", "/etc/passwd", "sub/dir", "..", ".hidden", "with space"],
    )
    def test_rejects_unsafe_name(self, presets_root: Path, name: str) -> None:
        with pytest.raises(InvalidGenerationSpec):
            load_preset(PresetKind.CHARACTER, name, root=presets_root)


class TestMergePromptFragments:
    def test_concatenates_in_order(self) -> None:
        merged = merge_prompt_fragments(["1girl, solo", "rooftop", "masterpiece"])

        assert merged == "1girl, solo, rooftop, masterpiece"

    def test_drops_duplicate_tokens(self) -> None:
        merged = merge_prompt_fragments(["1girl, blue hair", "rooftop", "blue hair, wind"])

        assert merged == "1girl, blue hair, rooftop, wind"

    def test_duplicate_check_ignores_case_and_spacing(self) -> None:
        merged = merge_prompt_fragments(["Blue Hair", "blue   hair", "wind"])

        assert merged == "Blue Hair, wind"

    def test_ignores_empty_fragments(self) -> None:
        merged = merge_prompt_fragments(["", "  ", "1girl", ""])

        assert merged == "1girl"

    def test_drops_empty_tokens(self) -> None:
        merged = merge_prompt_fragments(["1girl,, solo,", "  , rooftop"])

        assert merged == "1girl, solo, rooftop"

    def test_keeps_weight_syntax(self) -> None:
        merged = merge_prompt_fragments(["(blue hair:1.2), solo", "(blue hair:1.2)"])

        assert merged == "(blue hair:1.2), solo"


class TestApplyPresets:
    def test_without_presets_is_noop(self, presets_root: Path) -> None:
        payload = {
            "prompt": {"positive": "1girl"},
            "model": {"checkpoint": "sd15.safetensors"},
        }

        resolved, applied = apply_presets(payload, root=presets_root)

        assert resolved == payload
        assert applied == {}

    def test_merges_in_axis_order(self, presets_root: Path) -> None:
        payload = {
            "presets": {"character": "kaede", "scene": "rooftop", "style": "anime-soft"},
            "prompt": {"positive": "looking at viewer, blue hair"},
            "model": {"checkpoint": "sd15.safetensors"},
        }

        resolved, _ = apply_presets(payload, root=presets_root)

        assert resolved["prompt"]["positive"] == (
            "1girl, solo, blue hair, blue eyes, rooftop, sunset, wind, "
            "anime illustration, masterpiece, best quality, looking at viewer"
        )

    def test_merges_negative_with_same_rule(self, presets_root: Path) -> None:
        payload = {
            "presets": {"character": "kaede", "style": "anime-soft"},
            "prompt": {"positive": "1girl", "negative": "blurry"},
            "model": {"checkpoint": "sd15.safetensors"},
        }

        resolved, _ = apply_presets(payload, root=presets_root)

        assert resolved["prompt"]["negative"] == "bad anatomy, worst quality, blurry"

    def test_removes_presets_key_after_resolution(self, presets_root: Path) -> None:
        payload = {
            "presets": {"character": "kaede"},
            "prompt": {"positive": "1girl"},
            "model": {"checkpoint": "sd15.safetensors"},
        }

        resolved, _ = apply_presets(payload, root=presets_root)

        assert "presets" not in resolved

    def test_generation_prefers_spec_over_preset(self, presets_root: Path) -> None:
        payload = {
            "presets": {"style": "anime-soft"},
            "prompt": {"positive": "1girl"},
            "generation": {"cfg": 8.0},
            "model": {"checkpoint": "sd15.safetensors"},
        }

        resolved, _ = apply_presets(payload, root=presets_root)

        assert resolved["generation"]["cfg"] == 8.0  # spec優先
        assert resolved["generation"]["sampler"] == "dpmpp_2m"  # presetで補完
        assert resolved["generation"]["scheduler"] == "karras"

    def test_returns_applied_preset_names(self, presets_root: Path) -> None:
        payload = {
            "presets": {"character": "kaede", "style": "anime-soft"},
            "prompt": {"positive": "1girl"},
            "model": {"checkpoint": "sd15.safetensors"},
        }

        _, applied = apply_presets(payload, root=presets_root)

        assert applied == {"character": "kaede", "style": "anime-soft"}

    def test_does_not_mutate_input_payload(self, presets_root: Path) -> None:
        payload = {
            "presets": {"character": "kaede"},
            "prompt": {"positive": "1girl"},
            "model": {"checkpoint": "sd15.safetensors"},
        }

        apply_presets(payload, root=presets_root)

        assert payload["presets"] == {"character": "kaede"}
        assert payload["prompt"]["positive"] == "1girl"

    def test_rejects_unknown_axis(self, presets_root: Path) -> None:
        payload = {
            "presets": {"mood": "happy"},
            "prompt": {"positive": "1girl"},
            "model": {"checkpoint": "sd15.safetensors"},
        }

        with pytest.raises(InvalidGenerationSpec):
            apply_presets(payload, root=presets_root)

    def test_rejects_non_mapping_presets(self, presets_root: Path) -> None:
        payload = {
            "presets": ["kaede"],
            "prompt": {"positive": "1girl"},
            "model": {"checkpoint": "sd15.safetensors"},
        }

        with pytest.raises(InvalidGenerationSpec):
            apply_presets(payload, root=presets_root)


class TestPresetRefs:
    def test_unspecified_axis_is_none(self) -> None:
        refs = PresetRefs.model_validate({"character": "kaede"})

        assert refs.character == "kaede"
        assert refs.scene is None
        assert refs.style is None
