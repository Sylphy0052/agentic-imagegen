"""GenerationSpecのバリデーション (モデル定義上のハード制約)。"""

from typing import Any

import pytest
from pydantic import ValidationError

from agentic_imagegen.domain.models import GenerationSpec


def _spec_dict(**overrides: Any) -> dict[str, Any]:
    """最小限の有効なSpec辞書を作り、指定セクションだけ差し替える。"""
    base: dict[str, Any] = {
        "version": "1",
        "task": "txt2img",
        "prompt": {"positive": "1girl, blue hair", "negative": "low quality"},
        "generation": {
            "width": 512,
            "height": 768,
            "steps": 20,
            "cfg": 5.5,
            "seed": -1,
            "batch_size": 1,
            "sampler": "euler",
            "scheduler": "normal",
        },
        "model": {"checkpoint": "v1-5-pruned-emaonly.safetensors"},
        "output": {"directory": "outputs", "prefix": "blue_hair"},
    }
    for section, values in overrides.items():
        if isinstance(values, dict) and isinstance(base.get(section), dict):
            base[section] = {**base[section], **values}
        else:
            base[section] = values
    return base


def test_valid_generation_spec() -> None:
    spec = GenerationSpec.model_validate(_spec_dict())

    assert spec.version == "1"
    assert spec.task == "txt2img"
    assert spec.prompt.positive == "1girl, blue hair"
    assert spec.generation.width == 512
    assert spec.generation.height == 768
    assert spec.model.checkpoint == "v1-5-pruned-emaonly.safetensors"
    assert spec.output.prefix == "blue_hair"


def test_defaults_are_applied() -> None:
    spec = GenerationSpec.model_validate(
        {
            "prompt": {"positive": "a cat"},
            "model": {"checkpoint": "v1-5-pruned-emaonly.safetensors"},
        }
    )

    assert spec.version == "1"
    assert spec.task == "txt2img"
    assert spec.prompt.negative == ""
    assert spec.generation.width == 512
    assert spec.generation.height == 512
    assert spec.generation.steps == 20
    assert spec.generation.seed == -1
    assert spec.generation.batch_size == 1
    assert spec.output.directory == "outputs"


@pytest.mark.parametrize("width", [0, 32, 63, 100, 513, 8200])
def test_invalid_width(width: int) -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec_dict(generation={"width": width}))


@pytest.mark.parametrize("height", [0, 32, 63, 100, 769, 8200])
def test_invalid_height(height: int) -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec_dict(generation={"height": height}))


@pytest.mark.parametrize("steps", [0, -1, 101, 1000])
def test_invalid_steps(steps: int) -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec_dict(generation={"steps": steps}))


@pytest.mark.parametrize("cfg", [-0.1, -1, 30.1, 100])
def test_invalid_cfg(cfg: float) -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec_dict(generation={"cfg": cfg}))


@pytest.mark.parametrize("batch_size", [0, -1, 5, 100])
def test_invalid_batch_size(batch_size: int) -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec_dict(generation={"batch_size": batch_size}))


@pytest.mark.parametrize("seed", [-2, -100, 2**63])
def test_invalid_seed(seed: int) -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec_dict(generation={"seed": seed}))


@pytest.mark.parametrize("seed", [-1, 0, 42, 2**63 - 1])
def test_valid_seed(seed: int) -> None:
    spec = GenerationSpec.model_validate(_spec_dict(generation={"seed": seed}))
    assert spec.generation.seed == seed


@pytest.mark.parametrize("sampler", ["", "euler_x", "unknown", "EULER"])
def test_invalid_sampler(sampler: str) -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec_dict(generation={"sampler": sampler}))


@pytest.mark.parametrize("scheduler", ["", "karas", "unknown"])
def test_invalid_scheduler(scheduler: str) -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec_dict(generation={"scheduler": scheduler}))


@pytest.mark.parametrize(
    "checkpoint",
    [
        "../model.safetensors",
        "../../model.safetensors",
        "sd/../../model.safetensors",
        "/abs/model.safetensors",
        "..\\model.safetensors",
        "sd\\model.safetensors",
        "~/model.safetensors",
        "sd/1.5/model.safetensors",
        "model.pt",
        "model",
        "",
        "   ",
        "model.safetensors\x00",
    ],
)
def test_checkpoint_path_traversal_rejected(checkpoint: str) -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec_dict(model={"checkpoint": checkpoint}))


@pytest.mark.parametrize(
    "checkpoint",
    [
        "v1-5-pruned-emaonly.safetensors",
        "sd15/anything.safetensors",
        "model.ckpt",
    ],
)
def test_checkpoint_allowed(checkpoint: str) -> None:
    spec = GenerationSpec.model_validate(_spec_dict(model={"checkpoint": checkpoint}))
    assert spec.model.checkpoint == checkpoint


@pytest.mark.parametrize("prefix", ["", "a/b", "../x", "a b", "a\\b", "."])
def test_invalid_output_prefix(prefix: str) -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec_dict(output={"prefix": prefix}))


def test_empty_positive_prompt_rejected() -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec_dict(prompt={"positive": "   "}))


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec_dict(generation={"sampler_name": "euler"}))


@pytest.mark.parametrize("task", ["img2img", "txt2video", "unknown"])
def test_unsupported_task_rejected(task: str) -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec_dict(task=task))


def test_unsupported_version_rejected() -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec_dict(version="2"))


def test_spec_is_immutable() -> None:
    spec = GenerationSpec.model_validate(_spec_dict())
    with pytest.raises(ValidationError):
        spec.generation.width = 1024  # type: ignore[misc]
