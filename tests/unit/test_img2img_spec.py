"""img2img用のSpec拡張 (task / source) のバリデーション。

img2imgは入力画像のサイズをそのまま使うため、width/heightの明示指定は
「書いたのに効かない」状態になる。それを黙って通さないことをここで固定する。
"""

from typing import Any

import pytest
from pydantic import ValidationError

from agentic_imagegen.domain.models import GenerationSpec


def _spec_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "version": "1",
        "task": "img2img",
        "prompt": {"positive": "1girl, blue hair"},
        "source": {"image": "inputs/reference.png"},
        "model": {"checkpoint": "v1-5-pruned-emaonly.safetensors"},
    }
    base.update(overrides)
    return base


class TestTaskAndSource:
    def test_img2img_requires_source(self) -> None:
        payload = _spec_dict()
        del payload["source"]

        with pytest.raises(ValidationError, match="source"):
            GenerationSpec.model_validate(payload)

    def test_txt2img_rejects_source(self) -> None:
        """txt2imgにsourceを書いても効かないため拒否する。"""
        with pytest.raises(ValidationError, match="source"):
            GenerationSpec.model_validate(_spec_dict(task="txt2img"))

    def test_img2img_is_accepted(self) -> None:
        spec = GenerationSpec.model_validate(_spec_dict())

        assert spec.task == "img2img"
        assert spec.source is not None
        assert spec.source.image == "inputs/reference.png"

    def test_txt2img_source_defaults_to_none(self) -> None:
        payload = _spec_dict(task="txt2img")
        del payload["source"]

        spec = GenerationSpec.model_validate(payload)

        assert spec.source is None


class TestDenoise:
    def test_default(self) -> None:
        spec = GenerationSpec.model_validate(_spec_dict())

        assert spec.source is not None
        assert spec.source.denoise == 0.6

    @pytest.mark.parametrize("value", [0.0, 0.35, 1.0])
    def test_accepts_in_range(self, value: float) -> None:
        spec = GenerationSpec.model_validate(
            _spec_dict(source={"image": "inputs/a.png", "denoise": value})
        )

        assert spec.source is not None
        assert spec.source.denoise == value

    @pytest.mark.parametrize("value", [-0.1, 1.1, 2.0])
    def test_rejects_out_of_range(self, value: float) -> None:
        with pytest.raises(ValidationError):
            GenerationSpec.model_validate(
                _spec_dict(source={"image": "inputs/a.png", "denoise": value})
            )


class TestImagePath:
    @pytest.mark.parametrize(
        "image",
        [
            "../outside.png",
            "/etc/passwd.png",
            "~/secret.png",
            "back\\slash.png",
            "./ref.png",
            "",
        ],
    )
    def test_rejects_unsafe_path(self, image: str) -> None:
        with pytest.raises(ValidationError):
            GenerationSpec.model_validate(_spec_dict(source={"image": image}))

    @pytest.mark.parametrize("image", ["ref.txt", "ref", "ref.png.exe", "ref.safetensors"])
    def test_rejects_unexpected_suffix(self, image: str) -> None:
        with pytest.raises(ValidationError):
            GenerationSpec.model_validate(_spec_dict(source={"image": image}))

    @pytest.mark.parametrize(
        "image",
        ["ref.png", "inputs/ref.jpg", "inputs/sub/dir/ref.jpeg", "a.webp"],
    )
    def test_accepts_valid_paths(self, image: str) -> None:
        """checkpointと違い、リポジトリ配下なら階層の深さは問わない。"""
        spec = GenerationSpec.model_validate(_spec_dict(source={"image": image}))

        assert spec.source is not None
        assert spec.source.image == image


class TestUnsupportedCombinations:
    @pytest.mark.parametrize("field", ["width", "height"])
    def test_rejects_explicit_resolution(self, field: str) -> None:
        """img2imgは入力画像のサイズを使う。指定しても効かないため拒否する。"""
        with pytest.raises(ValidationError, match=field):
            GenerationSpec.model_validate(_spec_dict(generation={field: 1024}))

    def test_allows_other_generation_params(self) -> None:
        spec = GenerationSpec.model_validate(
            _spec_dict(generation={"steps": 24, "cfg": 6.0, "seed": 42})
        )

        assert spec.generation.steps == 24
        assert spec.generation.seed == 42

    def test_rejects_batch_size_over_one(self) -> None:
        """テンプレートが単一画像の入力しか持たないため、batchは1に限る。"""
        with pytest.raises(ValidationError, match="batch_size"):
            GenerationSpec.model_validate(_spec_dict(generation={"batch_size": 2}))

    def test_accepts_loras(self) -> None:
        """img2imgでもLoRAを併用できる (専用テンプレートへ切り替わる)。"""
        spec = GenerationSpec.model_validate(
            _spec_dict(
                model={
                    "checkpoint": "v1-5-pruned-emaonly.safetensors",
                    "loras": [{"name": "add_detail.safetensors", "strength_model": 0.7}],
                }
            )
        )

        assert spec.model.loras[0].name == "add_detail.safetensors"
        assert spec.model.loras[0].strength_model == 0.7


def test_unknown_source_key_rejected() -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec_dict(source={"image": "inputs/a.png", "strength": 0.5}))
