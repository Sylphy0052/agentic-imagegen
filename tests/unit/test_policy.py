"""設定由来のポリシー制約 (上限値・出力先) のテスト。"""

from pathlib import Path
from typing import Any

import pytest

from agentic_imagegen.config import Settings
from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.domain.policy import resolve_output_directory, validate_against_limits
from agentic_imagegen.errors import InvalidGenerationSpec


def _spec(**generation: Any) -> GenerationSpec:
    payload: dict[str, Any] = {
        "prompt": {"positive": "a cat"},
        "model": {"checkpoint": "v1-5-pruned-emaonly.safetensors"},
        "generation": {"width": 512, "height": 512, **generation},
    }
    return GenerationSpec.model_validate(payload)


def _settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "comfyui_base_url": "http://127.0.0.1:8188",
        "max_width": 2048,
        "max_height": 2048,
        "max_pixels": 4194304,
        "max_batch": 4,
        "timeout_seconds": 300,
        "output_root": Path("outputs"),
    }
    return Settings(**{**defaults, **overrides})


def test_within_limits_passes() -> None:
    validate_against_limits(_spec(), _settings())


def test_exceeds_max_width() -> None:
    with pytest.raises(InvalidGenerationSpec, match="width"):
        validate_against_limits(_spec(width=1024), _settings(max_width=512))


def test_exceeds_max_height() -> None:
    with pytest.raises(InvalidGenerationSpec, match="height"):
        validate_against_limits(_spec(height=1024), _settings(max_height=512))


def test_exceeds_max_pixels() -> None:
    with pytest.raises(InvalidGenerationSpec, match="pixel"):
        validate_against_limits(_spec(width=2048, height=2048), _settings(max_pixels=1048576))


def test_exceeds_max_batch() -> None:
    with pytest.raises(InvalidGenerationSpec, match="batch_size"):
        validate_against_limits(_spec(batch_size=4), _settings(max_batch=1))


def test_pixels_counted_per_batch() -> None:
    """batch_sizeを掛けた総ピクセル数で判定する。"""
    with pytest.raises(InvalidGenerationSpec, match="pixel"):
        validate_against_limits(
            _spec(width=1024, height=1024, batch_size=4),
            _settings(max_pixels=2097152),
        )


def test_resolve_output_directory_under_root(tmp_path: Path) -> None:
    resolved = resolve_output_directory("outputs/test", tmp_path)
    assert resolved == (tmp_path / "outputs" / "test").resolve()


@pytest.mark.parametrize(
    "directory",
    ["../outside", "../../etc", "outputs/../../escape", "/etc", "~/elsewhere"],
)
def test_output_directory_escape_rejected(directory: str, tmp_path: Path) -> None:
    with pytest.raises(InvalidGenerationSpec):
        resolve_output_directory(directory, tmp_path)
