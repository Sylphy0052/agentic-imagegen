"""出力ディレクトリの命名規則のテスト。"""

from __future__ import annotations

import re
from pathlib import Path

from agentic_imagegen.config import Settings
from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.services.generation import _prepare_directory


def _spec(prefix: str) -> GenerationSpec:
    return GenerationSpec.model_validate(
        {
            "version": "1",
            "task": "txt2img",
            "prompt": {"positive": "1girl"},
            "model": {"checkpoint": "meinamix_v12Final.safetensors"},
            "output": {"prefix": prefix},
        }
    )


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        comfyui_base_url="http://127.0.0.1:8188",
        max_width=2048,
        max_height=2048,
        max_pixels=4194304,
        max_batch=4,
        timeout_seconds=300,
        output_root=Path("outputs"),
    )


def test_directory_name_starts_with_time(tmp_path: Path) -> None:
    directory = _prepare_directory(_spec("blue_hair"), _settings(tmp_path), tmp_path)

    assert re.fullmatch(r"\d{6}_blue_hair", directory.name), directory.name


def test_directory_is_under_dated_directory(tmp_path: Path) -> None:
    directory = _prepare_directory(_spec("blue_hair"), _settings(tmp_path), tmp_path)

    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", directory.parent.name), directory.parent.name


def test_rerun_in_same_second_does_not_overwrite(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    first = _prepare_directory(_spec("blue_hair"), settings, tmp_path)
    first.mkdir(parents=True)

    second = _prepare_directory(_spec("blue_hair"), settings, tmp_path)

    assert second != first
    assert second.name.startswith(first.name)
