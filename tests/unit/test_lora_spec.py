"""LoRA指定 (model.loras) のバリデーション。

checkpointと同じくファイル名がユーザー入力として入るため、
Path Traversalと想定外の拡張子をモデル定義の時点で弾く。
"""

from typing import Any

import pytest
from pydantic import ValidationError

from agentic_imagegen.domain.models import MAX_LORAS, GenerationSpec


def _spec_dict(**model_overrides: Any) -> dict[str, Any]:
    model: dict[str, Any] = {"checkpoint": "v1-5-pruned-emaonly.safetensors"}
    model.update(model_overrides)
    return {
        "version": "1",
        "task": "txt2img",
        "prompt": {"positive": "1girl"},
        "model": model,
    }


def _lora(name: str = "add_detail.safetensors", **overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {"name": name}
    entry.update(overrides)
    return entry


class TestDefaults:
    def test_loras_default_to_empty(self) -> None:
        spec = GenerationSpec.model_validate(_spec_dict())

        assert spec.model.loras == ()

    def test_strength_defaults_to_one(self) -> None:
        spec = GenerationSpec.model_validate(_spec_dict(loras=[_lora()]))

        assert spec.model.loras[0].strength_model == 1.0
        assert spec.model.loras[0].strength_clip == 1.0

    def test_accepts_explicit_strengths(self) -> None:
        spec = GenerationSpec.model_validate(
            _spec_dict(loras=[_lora(strength_model=0.8, strength_clip=0.5)])
        )

        assert spec.model.loras[0].strength_model == 0.8
        assert spec.model.loras[0].strength_clip == 0.5


class TestCount:
    @pytest.mark.parametrize("count", [1, 2, 3])
    def test_accepts_up_to_max(self, count: int) -> None:
        loras = [_lora(f"lora{index}.safetensors") for index in range(count)]

        spec = GenerationSpec.model_validate(_spec_dict(loras=loras))

        assert len(spec.model.loras) == count

    def test_rejects_more_than_max(self) -> None:
        loras = [_lora(f"lora{index}.safetensors") for index in range(MAX_LORAS + 1)]

        with pytest.raises(ValidationError, match=str(MAX_LORAS)):
            GenerationSpec.model_validate(_spec_dict(loras=loras))

    def test_rejects_duplicate_names(self) -> None:
        """同じLoRAを二重に積むと意図しない強度になるため拒否する。"""
        loras = [_lora("add_detail.safetensors"), _lora("add_detail.safetensors")]

        with pytest.raises(ValidationError, match="重複"):
            GenerationSpec.model_validate(_spec_dict(loras=loras))


class TestName:
    @pytest.mark.parametrize(
        "name",
        [
            "../secret.safetensors",
            "/etc/passwd.safetensors",
            "~/lora.safetensors",
            "sub/dir/lora.safetensors",
            "back\\slash.safetensors",
            "./lora.safetensors",
        ],
    )
    def test_rejects_path_traversal(self, name: str) -> None:
        with pytest.raises(ValidationError):
            GenerationSpec.model_validate(_spec_dict(loras=[_lora(name)]))

    @pytest.mark.parametrize("name", ["lora.txt", "lora.bin", "lora", "lora.safetensors.exe"])
    def test_rejects_unexpected_suffix(self, name: str) -> None:
        with pytest.raises(ValidationError):
            GenerationSpec.model_validate(_spec_dict(loras=[_lora(name)]))

    @pytest.mark.parametrize(
        "name",
        [
            "add_detail.safetensors",
            "style/anime.safetensors",
            "legacy.pt",
            "old.ckpt",
            # 実在ファイル名は大文字混じりのことがある
            "Add_Detail.SafeTensors",
        ],
    )
    def test_accepts_valid_names(self, name: str) -> None:
        spec = GenerationSpec.model_validate(_spec_dict(loras=[_lora(name)]))

        assert spec.model.loras[0].name == name

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValidationError):
            GenerationSpec.model_validate(_spec_dict(loras=[_lora("")]))


class TestStrength:
    @pytest.mark.parametrize("value", [-10.5, 10.5, 100.0])
    def test_rejects_out_of_range(self, value: float) -> None:
        with pytest.raises(ValidationError):
            GenerationSpec.model_validate(_spec_dict(loras=[_lora(strength_model=value)]))

    @pytest.mark.parametrize("value", [-10.0, 0.0, 1.0, 10.0])
    def test_accepts_in_range(self, value: float) -> None:
        spec = GenerationSpec.model_validate(_spec_dict(loras=[_lora(strength_clip=value)]))

        assert spec.model.loras[0].strength_clip == value


def test_unknown_key_rejected() -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec_dict(loras=[_lora(strength=1.0)]))


def test_loras_are_immutable() -> None:
    spec = GenerationSpec.model_validate(_spec_dict(loras=[_lora()]))

    with pytest.raises(ValidationError):
        spec.model.loras[0].strength_model = 0.5  # type: ignore[misc]
