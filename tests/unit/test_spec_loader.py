"""YAMLからGenerationSpecを読み込む処理のテスト。"""

from pathlib import Path

import pytest

from agentic_imagegen.errors import InvalidGenerationSpec
from agentic_imagegen.services.spec_loader import load_spec

VALID_YAML = """
version: "1"
task: txt2img

prompt:
  positive: 1girl, blue hair, blue eyes
  negative: low quality, blurry

generation:
  width: 512
  height: 768
  steps: 20
  cfg: 5.5
  seed: -1
  batch_size: 1

model:
  checkpoint: v1-5-pruned-emaonly.safetensors

output:
  directory: outputs
  prefix: blue_hair
"""


def test_load_valid_spec(tmp_path: Path) -> None:
    path = tmp_path / "spec.yaml"
    path.write_text(VALID_YAML, encoding="utf-8")

    spec = load_spec(path)

    assert spec.prompt.positive == "1girl, blue hair, blue eyes"
    assert spec.generation.height == 768
    assert spec.output.prefix == "blue_hair"


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(InvalidGenerationSpec, match="見つかりません"):
        load_spec(tmp_path / "absent.yaml")


def test_broken_yaml(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("prompt: [unclosed", encoding="utf-8")

    with pytest.raises(InvalidGenerationSpec, match="YAML"):
        load_spec(path)


def test_non_mapping_yaml(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")

    with pytest.raises(InvalidGenerationSpec):
        load_spec(path)


def test_empty_yaml(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    with pytest.raises(InvalidGenerationSpec):
        load_spec(path)


def test_validation_error_message_names_field(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(VALID_YAML.replace("batch_size: 1", "batch_size: 99"), encoding="utf-8")

    with pytest.raises(InvalidGenerationSpec, match="batch_size"):
        load_spec(path)
