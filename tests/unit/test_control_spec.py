"""ControlNet指定 (control) のバリデーション。"""

from typing import Any

import pytest
from pydantic import ValidationError

from agentic_imagegen.domain.models import GenerationSpec

CONTROL: dict[str, Any] = {
    "image": "inputs/pose.png",
    "model": "control_v11p_sd15_canny_fp16.safetensors",
}


def _spec(**control: Any) -> dict[str, Any]:
    merged = {**CONTROL, **control} if control else CONTROL
    return {
        "version": "1",
        "task": "txt2img",
        "prompt": {"positive": "1girl"},
        "control": merged,
        "model": {"checkpoint": "v1-5-pruned-emaonly.safetensors"},
    }


def test_absent_by_default() -> None:
    payload = _spec()
    del payload["control"]

    assert GenerationSpec.model_validate(payload).control is None


def test_defaults() -> None:
    control = GenerationSpec.model_validate(_spec()).control

    assert control is not None
    assert control.strength == 1.0
    assert control.start_percent == 0.0
    assert control.end_percent == 1.0
    assert control.low_threshold == 0.4
    assert control.high_threshold == 0.8


def test_accepts_explicit_values() -> None:
    control = GenerationSpec.model_validate(
        _spec(strength=0.75, start_percent=0.1, end_percent=0.9, low_threshold=0.2)
    ).control

    assert control is not None
    assert control.strength == 0.75
    assert control.start_percent == 0.1
    assert control.end_percent == 0.9
    assert control.low_threshold == 0.2


@pytest.mark.parametrize("value", [-0.1, 10.1])
def test_rejects_out_of_range_strength(value: float) -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec(strength=value))


@pytest.mark.parametrize(("field", "value"), [("start_percent", -0.1), ("end_percent", 1.1)])
def test_rejects_out_of_range_percent(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec(**{field: value}))


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_rejects_out_of_range_threshold(value: float) -> None:
    """Cannyの閾値は 0.01-0.99。"""
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec(low_threshold=value))


def test_rejects_inverted_thresholds() -> None:
    """low >= high では線が拾えない。"""
    with pytest.raises(ValidationError, match="low_threshold"):
        GenerationSpec.model_validate(_spec(low_threshold=0.8, high_threshold=0.4))


def test_rejects_inverted_percents() -> None:
    with pytest.raises(ValidationError, match="start_percent"):
        GenerationSpec.model_validate(_spec(start_percent=0.9, end_percent=0.1))


@pytest.mark.parametrize(
    "image", ["../outside.png", "/etc/passwd.png", "~/x.png", "back\\slash.png", "x.txt"]
)
def test_rejects_unsafe_image(image: str) -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec(image=image))


@pytest.mark.parametrize(
    "model", ["../secret.safetensors", "/abs.safetensors", "sub/dir/x.safetensors", "x.txt"]
)
def test_rejects_unsafe_model(model: str) -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec(model=model))


def test_accepts_pth_model() -> None:
    """ControlNetは .pth で配布されることもある。"""
    control = GenerationSpec.model_validate(_spec(model="control_canny.pth")).control

    assert control is not None
    assert control.model == "control_canny.pth"


def test_accepts_uppercase_suffix() -> None:
    """実在ファイル名は大文字混じりのことがある。拡張子の大小では弾かない。"""
    control = GenerationSpec.model_validate(_spec(model="Control_Canny.SafeTensors")).control

    assert control is not None
    assert control.model == "Control_Canny.SafeTensors"


def test_rejects_unknown_key() -> None:
    with pytest.raises(ValidationError):
        GenerationSpec.model_validate(_spec(guidance="strong"))


class TestControlPreprocessor:
    """Issue #37: 前処理済みの制御画像を Canny を通さずに渡す指定。"""

    def test_defaults_to_canny(self) -> None:
        control = GenerationSpec.model_validate(_spec()).control

        assert control is not None
        assert control.preprocessor == "canny"
        assert control.skips_preprocessor is False

    def test_accepts_none(self) -> None:
        control = GenerationSpec.model_validate(_spec(preprocessor="none")).control

        assert control is not None
        assert control.preprocessor == "none"
        assert control.skips_preprocessor is True

    @pytest.mark.parametrize("value", ["openpose", "depth", "canny "])
    def test_rejects_unsupported_preprocessor(self, value: str) -> None:
        # pose / depth はpreprocessorのカスタムノードが未導入のため受け付けない。
        with pytest.raises(ValidationError):
            GenerationSpec.model_validate(_spec(preprocessor=value))

    @pytest.mark.parametrize("key", ["low_threshold", "high_threshold"])
    def test_rejects_canny_thresholds_when_preprocessor_is_none(self, key: str) -> None:
        # 書いたのに効かない指定は作らない。Cannyを通さないなら閾値は指定できない。
        with pytest.raises(ValidationError, match="preprocessor"):
            GenerationSpec.model_validate(_spec(preprocessor="none", **{key: 0.5}))

    def test_allows_thresholds_with_canny(self) -> None:
        control = GenerationSpec.model_validate(
            _spec(preprocessor="canny", low_threshold=0.2, high_threshold=0.6)
        ).control

        assert control is not None
        assert control.low_threshold == 0.2
        assert control.high_threshold == 0.6

    def test_keeps_threshold_defaults_when_preprocessor_is_none(self) -> None:
        # 明示していなければ既定値のままで通る (Cannyを通さないため使われない)。
        control = GenerationSpec.model_validate(_spec(preprocessor="none")).control

        assert control is not None
        assert control.low_threshold == 0.4
        assert control.high_threshold == 0.8
