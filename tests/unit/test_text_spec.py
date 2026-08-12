"""テキスト合成のSpec定義の検証。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentic_imagegen.domain.models import (
    MAX_DIMENSION,
    MAX_FONT_SIZE,
    MAX_TEXT_CONTENT_LENGTH,
    MAX_TEXT_LAYERS,
    BoxSpec,
    GenerationSpec,
    ShadowSpec,
    StrokeSpec,
    TextLayer,
    TextSpec,
)


def _layer(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {"content": "秋葉原駅", "font": "NotoSansJP-Bold.ttf", "size": 64}
    base.update(overrides)
    return base


def _spec(text: dict[str, object] | None) -> dict[str, object]:
    spec: dict[str, object] = {
        "prompt": {"positive": "a street"},
        "model": {"checkpoint": "meinamix_v12Final.safetensors"},
    }
    if text is not None:
        spec["text"] = text
    return spec


class TestTextLayerDefaults:
    def test_builds_with_required_fields_only(self) -> None:
        layer = TextLayer.model_validate(_layer())

        assert layer.content == "秋葉原駅"
        assert layer.font == "NotoSansJP-Bold.ttf"
        assert layer.size == 64
        assert layer.font_index == 0
        assert layer.color == "#ffffff"
        assert layer.anchor == "center"
        assert layer.offset == (0, 0)
        assert layer.max_width is None
        assert layer.line_spacing == pytest.approx(1.2)
        assert layer.align == "center"
        assert layer.opacity == pytest.approx(1.0)
        assert layer.rotation == pytest.approx(0.0)
        assert layer.direction == "horizontal"
        assert layer.stroke is None
        assert layer.shadow is None
        assert layer.box is None

    def test_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError):
            TextLayer.model_validate(_layer(unknown="x"))


class TestTextLayerContent:
    def test_rejects_empty_string(self) -> None:
        with pytest.raises(ValidationError):
            TextLayer.model_validate(_layer(content=""))

    def test_accepts_limit_exactly(self) -> None:
        layer = TextLayer.model_validate(_layer(content="あ" * MAX_TEXT_CONTENT_LENGTH))

        assert len(layer.content) == MAX_TEXT_CONTENT_LENGTH

    def test_rejects_over_limit(self) -> None:
        with pytest.raises(ValidationError, match=str(MAX_TEXT_CONTENT_LENGTH)):
            TextLayer.model_validate(_layer(content="あ" * (MAX_TEXT_CONTENT_LENGTH + 1)))

    def test_allows_newlines(self) -> None:
        layer = TextLayer.model_validate(_layer(content="一行目\n二行目"))

        assert layer.content == "一行目\n二行目"

    def test_rejects_control_characters(self) -> None:
        with pytest.raises(ValidationError, match="制御文字"):
            TextLayer.model_validate(_layer(content="改\x00行"))


class TestTextLayerFont:
    @pytest.mark.parametrize("name", ["NotoSansJP.ttf", "sub/NotoSansJP.otf", "yu.ttc"])
    def test_accepts_allowed_suffixes(self, name: str) -> None:
        assert TextLayer.model_validate(_layer(font=name)).font == name

    @pytest.mark.parametrize(
        "name",
        [
            "font.woff2",
            "/abs/font.ttf",
            "~/font.ttf",
            "../font.ttf",
            "a/b/c/font.ttf",
            "font\\x.ttf",
        ],
    )
    def test_rejects_unsafe_path_and_suffix(self, name: str) -> None:
        with pytest.raises(ValidationError):
            TextLayer.model_validate(_layer(font=name))

    def test_rejects_negative_font_index(self) -> None:
        with pytest.raises(ValidationError):
            TextLayer.model_validate(_layer(font_index=-1))


class TestTextLayerSize:
    @pytest.mark.parametrize("size", [1, MAX_FONT_SIZE])
    def test_accepts_bounds(self, size: int) -> None:
        assert TextLayer.model_validate(_layer(size=size)).size == size

    @pytest.mark.parametrize("size", [0, MAX_FONT_SIZE + 1])
    def test_rejects_out_of_range(self, size: int) -> None:
        with pytest.raises(ValidationError):
            TextLayer.model_validate(_layer(size=size))


class TestColorFormat:
    @pytest.mark.parametrize("color", ["#fff", "#FFFFFF", "#00000080", "#a1b2c3"])
    def test_accepted_formats(self, color: str) -> None:
        assert TextLayer.model_validate(_layer(color=color)).color == color.lower()

    @pytest.mark.parametrize("color", ["white", "#ffff", "ffffff", "#gggggg", ""])
    def test_rejected_formats(self, color: str) -> None:
        with pytest.raises(ValidationError):
            TextLayer.model_validate(_layer(color=color))


class TestTextLayerGeometry:
    @pytest.mark.parametrize("rotation", [-180.0, 0.0, 180.0])
    def test_accepts_rotation_bounds(self, rotation: float) -> None:
        assert TextLayer.model_validate(_layer(rotation=rotation)).rotation == rotation

    @pytest.mark.parametrize("rotation", [-180.1, 180.1])
    def test_rejects_rotation_out_of_range(self, rotation: float) -> None:
        with pytest.raises(ValidationError):
            TextLayer.model_validate(_layer(rotation=rotation))

    def test_accepts_negative_offset(self) -> None:
        assert TextLayer.model_validate(_layer(offset=[-10, -48])).offset == (-10, -48)

    def test_accepts_offset_at_bounds(self) -> None:
        layer = TextLayer.model_validate(_layer(offset=[-MAX_DIMENSION, MAX_DIMENSION]))

        assert layer.offset == (-MAX_DIMENSION, MAX_DIMENSION)

    @pytest.mark.parametrize("offset", [[-MAX_DIMENSION - 1, 0], [0, MAX_DIMENSION + 1]])
    def test_rejects_offset_out_of_range(self, offset: list[int]) -> None:
        with pytest.raises(ValidationError):
            TextLayer.model_validate(_layer(offset=offset))

    def test_rejects_non_positive_max_width(self) -> None:
        with pytest.raises(ValidationError):
            TextLayer.model_validate(_layer(max_width=0))

    def test_accepts_max_width_at_upper_bound(self) -> None:
        layer = TextLayer.model_validate(_layer(max_width=MAX_DIMENSION))

        assert layer.max_width == MAX_DIMENSION

    def test_rejects_max_width_over_upper_bound(self) -> None:
        with pytest.raises(ValidationError):
            TextLayer.model_validate(_layer(max_width=MAX_DIMENSION + 1))

    @pytest.mark.parametrize("anchor", ["top-left", "center", "bottom-right"])
    def test_accepts_anchor(self, anchor: str) -> None:
        assert TextLayer.model_validate(_layer(anchor=anchor)).anchor == anchor

    def test_rejects_unknown_anchor(self) -> None:
        with pytest.raises(ValidationError):
            TextLayer.model_validate(_layer(anchor="middle-middle"))

    def test_accepts_vertical_direction(self) -> None:
        assert TextLayer.model_validate(_layer(direction="vertical")).direction == "vertical"


class TestDecorations:
    def test_accepts_stroke(self) -> None:
        layer = TextLayer.model_validate(_layer(stroke={"width": 3, "color": "#000000"}))

        assert layer.stroke == StrokeSpec(width=3, color="#000000")

    def test_stroke_width_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            TextLayer.model_validate(_layer(stroke={"width": 0}))

    def test_accepts_shadow(self) -> None:
        layer = TextLayer.model_validate(_layer(shadow={"offset": [4, 4], "blur": 6}))

        assert layer.shadow == ShadowSpec(offset=(4, 4), blur=6.0)

    def test_rejects_negative_blur(self) -> None:
        with pytest.raises(ValidationError):
            TextLayer.model_validate(_layer(shadow={"blur": -1}))

    def test_shadow_accepts_offset_at_bounds(self) -> None:
        layer = TextLayer.model_validate(_layer(shadow={"offset": [-MAX_DIMENSION, MAX_DIMENSION]}))

        assert layer.shadow is not None
        assert layer.shadow.offset == (-MAX_DIMENSION, MAX_DIMENSION)

    def test_shadow_rejects_offset_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            TextLayer.model_validate(_layer(shadow={"offset": [0, MAX_DIMENSION + 1]}))

    def test_accepts_box(self) -> None:
        layer = TextLayer.model_validate(_layer(box={"padding": [16, 24], "radius": 12}))

        assert layer.box == BoxSpec(padding=(16, 24), radius=12)

    def test_rejects_negative_box_padding(self) -> None:
        with pytest.raises(ValidationError):
            TextLayer.model_validate(_layer(box={"padding": [-1, 0]}))

    @pytest.mark.parametrize("opacity", [-0.1, 1.1])
    def test_rejects_opacity_out_of_range(self, opacity: float) -> None:
        with pytest.raises(ValidationError):
            TextLayer.model_validate(_layer(opacity=opacity))


class TestTextSpec:
    def test_requires_at_least_one_layer(self) -> None:
        with pytest.raises(ValidationError):
            TextSpec.model_validate({"layers": []})

    def test_accepts_limit_exactly(self) -> None:
        spec = TextSpec.model_validate({"layers": [_layer()] * MAX_TEXT_LAYERS})

        assert len(spec.layers) == MAX_TEXT_LAYERS

    def test_rejects_over_limit(self) -> None:
        with pytest.raises(ValidationError, match=str(MAX_TEXT_LAYERS)):
            TextSpec.model_validate({"layers": [_layer()] * (MAX_TEXT_LAYERS + 1)})

    def test_keeps_declared_order(self) -> None:
        spec = TextSpec.model_validate(
            {"layers": [_layer(content="下"), _layer(content="上")]},
        )

        assert [layer.content for layer in spec.layers] == ["下", "上"]


class TestGenerationSpecIntegration:
    def test_text_is_optional(self) -> None:
        assert GenerationSpec.model_validate(_spec(None)).text is None

    def test_accepts_text(self) -> None:
        spec = GenerationSpec.model_validate(_spec({"layers": [_layer()]}))

        assert spec.text is not None
        assert spec.text.layers[0].content == "秋葉原駅"

    def test_allowed_for_img2img(self) -> None:
        payload = _spec({"layers": [_layer()]})
        payload["task"] = "img2img"
        payload["source"] = {"image": "inputs/base.png"}

        spec = GenerationSpec.model_validate(payload)

        assert spec.text is not None
