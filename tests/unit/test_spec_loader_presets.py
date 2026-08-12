"""spec_loader経由でpresetが解決されることのテスト。

presetの合成規則そのものは test_presets.py が担う。ここでは
「Specの読み込み経路を通したときに展開され、履歴が残る」ことだけを見る。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_imagegen.errors import InvalidGenerationSpec
from agentic_imagegen.services.spec_loader import load_spec

SPEC_WITH_PRESETS = """
version: "1"
task: txt2img

presets:
  character: kaede
  style: anime-soft

prompt:
  positive: looking at viewer

model:
  checkpoint: meinamix_v12Final.safetensors
"""


@pytest.fixture
def presets_root(tmp_path: Path) -> Path:
    root = tmp_path / "presets"
    (root / "characters").mkdir(parents=True)
    (root / "styles").mkdir(parents=True)
    (root / "characters" / "kaede.yaml").write_text(
        "prompt:\n  positive: 1girl, solo, blue hair\n  negative: bad anatomy\n",
        encoding="utf-8",
    )
    (root / "styles" / "anime-soft.yaml").write_text(
        "prompt:\n  positive: anime illustration, masterpiece\ngeneration:\n"
        "  sampler: dpmpp_2m\n  scheduler: karras\n",
        encoding="utf-8",
    )
    return root


def _write_spec(tmp_path: Path, body: str = SPEC_WITH_PRESETS) -> Path:
    path = tmp_path / "spec.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_expands_prompt_from_presets(tmp_path: Path, presets_root: Path) -> None:
    spec = load_spec(_write_spec(tmp_path), presets_root=presets_root)

    assert spec.prompt.positive == (
        "1girl, solo, blue hair, anime illustration, masterpiece, looking at viewer"
    )
    assert spec.prompt.negative == "bad anatomy"


def test_takes_generation_from_preset(tmp_path: Path, presets_root: Path) -> None:
    spec = load_spec(_write_spec(tmp_path), presets_root=presets_root)

    assert spec.generation.sampler == "dpmpp_2m"
    assert spec.generation.scheduler == "karras"


def test_keeps_applied_presets_on_spec(tmp_path: Path, presets_root: Path) -> None:
    spec = load_spec(_write_spec(tmp_path), presets_root=presets_root)

    assert spec.presets.character == "kaede"
    assert spec.presets.style == "anime-soft"
    assert spec.presets.scene is None


def test_missing_presets_root_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(InvalidGenerationSpec, match="探索ルート"):
        load_spec(_write_spec(tmp_path))


def test_spec_without_presets_still_loads(tmp_path: Path, presets_root: Path) -> None:
    body = """
version: "1"
task: txt2img

prompt:
  positive: 1girl

model:
  checkpoint: meinamix_v12Final.safetensors
"""
    spec = load_spec(_write_spec(tmp_path, body), presets_root=presets_root)

    assert spec.prompt.positive == "1girl"
    assert spec.presets.is_empty()


def test_missing_preset_fails_spec_load(tmp_path: Path, presets_root: Path) -> None:
    body = SPEC_WITH_PRESETS.replace("character: kaede", "character: missing")

    with pytest.raises(InvalidGenerationSpec, match="見つかりません"):
        load_spec(_write_spec(tmp_path, body), presets_root=presets_root)
