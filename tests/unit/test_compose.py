"""テキスト合成サービスの検証。"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from agentic_imagegen.domain.models import TextSpec
from agentic_imagegen.errors import TextCompositionError
from agentic_imagegen.services.compose import (
    anchor_origin,
    compose_text,
    parse_color,
    wrap_lines,
)

CANVAS: tuple[int, int] = (400, 300)


@pytest.fixture
def base_image(tmp_path: Path) -> Path:
    path = tmp_path / "base.png"
    Image.new("RGB", CANVAS, (0, 0, 0)).save(path)
    return path


def _spec(**overrides: object) -> TextSpec:
    layer: dict[str, object] = {
        "content": "ABC",
        "font": "test.ttf",
        "size": 32,
        "color": "#ff0000",
    }
    layer.update(overrides)
    return TextSpec.model_validate({"layers": [layer]})


def _color_counts(path: Path) -> dict[tuple[int, int, int], int]:
    with Image.open(path) as image:
        counts = image.convert("RGB").getcolors(maxcolors=CANVAS[0] * CANVAS[1])
    assert counts is not None
    return {color: count for count, color in counts}


def _opaque_pixels(path: Path) -> int:
    counts = _color_counts(path)
    return sum(count for color, count in counts.items() if color != (0, 0, 0))


def _bounding_box(path: Path) -> tuple[int, int, int, int] | None:
    with Image.open(path) as image:
        return image.convert("RGB").getbbox()


def _row_bounding_box(path: Path, y0: int, y1: int) -> tuple[int, int, int, int] | None:
    """縦方向を [y0, y1) へ絞ったうえでbounding boxを取る。

    複数行のうち1行だけを見たいとき (align の検証など) に使う。
    """
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return rgb.crop((0, y0, rgb.width, y1)).getbbox()


def _column_bounding_box(path: Path, x0: int, x1: int) -> tuple[int, int, int, int] | None:
    """横方向を [x0, x1) へ絞ったうえでbounding boxを取る。

    縦書きの列を1本だけ見たいとき (align の検証など) に使う。
    """
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return rgb.crop((x0, 0, x1, rgb.height)).getbbox()


class TestParseColor:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("#fff", (255, 255, 255, 255)),
            ("#000", (0, 0, 0, 255)),
            ("#ff0000", (255, 0, 0, 255)),
            ("#0000ff80", (0, 0, 255, 128)),
        ],
    )
    def test_parses(self, value: str, expected: tuple[int, int, int, int]) -> None:
        assert parse_color(value) == expected

    def test_multiplies_opacity(self) -> None:
        assert parse_color("#ffffff", opacity=0.5) == (255, 255, 255, 128)

    def test_zero_opacity_is_transparent(self) -> None:
        assert parse_color("#ffffff", opacity=0.0) == (255, 255, 255, 0)


class TestAnchorOrigin:
    @pytest.mark.parametrize(
        ("anchor", "expected"),
        [
            ("top-left", (0, 0)),
            ("top-center", (45, 0)),
            ("top-right", (90, 0)),
            ("middle-left", (0, 40)),
            ("center", (45, 40)),
            ("middle-right", (90, 40)),
            ("bottom-left", (0, 80)),
            ("bottom-center", (45, 80)),
            ("bottom-right", (90, 80)),
        ],
    )
    def test_computes_origin(self, anchor: str, expected: tuple[int, int]) -> None:
        origin = anchor_origin(
            anchor,  # type: ignore[arg-type]
            canvas=(100, 100),
            block=(10, 20),
            offset=(0, 0),
        )

        assert origin == expected

    def test_adds_offset(self) -> None:
        origin = anchor_origin(
            "bottom-center",
            canvas=(100, 100),
            block=(10, 20),
            offset=(5, -8),
        )

        assert origin == (50, 72)


class TestWrapLines:
    def test_splits_only_on_newlines_without_max_width(self) -> None:
        assert wrap_lines("一行目\n二行目", measure=len, max_width=None) == ["一行目", "二行目"]

    def test_wraps_when_exceeding_width(self) -> None:
        assert wrap_lines("abcdef", measure=len, max_width=3) == ["abc", "def"]

    def test_wraps_inside_a_word(self) -> None:
        # 日本語は単語境界を持たないため、文字単位で折り返す
        assert wrap_lines("あいうえお", measure=len, max_width=2) == ["あい", "うえ", "お"]

    def test_combines_with_explicit_newline(self) -> None:
        assert wrap_lines("abcd\nef", measure=len, max_width=3) == ["abc", "d", "ef"]

    def test_keeps_char_that_alone_exceeds_width(self) -> None:
        assert wrap_lines("abc", measure=lambda s: len(s) * 10, max_width=5) == ["a", "b", "c"]


class TestComposeText:
    def test_writes_output_without_touching_source(
        self, base_image: Path, fonts_root: Path
    ) -> None:
        original = base_image.read_bytes()
        output = base_image.parent / "out.png"

        result = compose_text(image=base_image, spec=_spec(), fonts_root=fonts_root, output=output)

        assert result.output == output
        assert output.is_file()
        assert base_image.read_bytes() == original

    def test_keeps_original_resolution(self, base_image: Path, fonts_root: Path) -> None:
        output = base_image.parent / "out.png"

        compose_text(image=base_image, spec=_spec(), fonts_root=fonts_root, output=output)

        with Image.open(output) as image:
            assert image.size == CANVAS

    def test_draws_with_given_color(self, base_image: Path, fonts_root: Path) -> None:
        output = base_image.parent / "out.png"

        compose_text(image=base_image, spec=_spec(), fonts_root=fonts_root, output=output)

        colors = set(_color_counts(output))
        assert (255, 0, 0) in colors

    def test_returns_resolved_font(self, base_image: Path, fonts_root: Path) -> None:
        output = base_image.parent / "out.png"

        result = compose_text(image=base_image, spec=_spec(), fonts_root=fonts_root, output=output)

        assert len(result.fonts) == 1
        assert result.fonts[0].name == "test.ttf"
        assert result.fonts[0].path == fonts_root / "test.ttf"

    def test_draws_nothing_at_zero_opacity(self, base_image: Path, fonts_root: Path) -> None:
        output = base_image.parent / "out.png"

        compose_text(
            image=base_image, spec=_spec(opacity=0.0), fonts_root=fonts_root, output=output
        )

        assert _opaque_pixels(output) == 0

    def test_anchor_moves_drawn_position(self, base_image: Path, fonts_root: Path) -> None:
        top = base_image.parent / "top.png"
        bottom = base_image.parent / "bottom.png"

        compose_text(
            image=base_image, spec=_spec(anchor="top-left"), fonts_root=fonts_root, output=top
        )
        compose_text(
            image=base_image,
            spec=_spec(anchor="bottom-right"),
            fonts_root=fonts_root,
            output=bottom,
        )

        top_box = _bounding_box(top)
        bottom_box = _bounding_box(bottom)
        assert top_box is not None
        assert bottom_box is not None
        assert top_box[1] < bottom_box[1]
        assert top_box[0] < bottom_box[0]

    def test_wrapping_grows_vertically(self, base_image: Path, fonts_root: Path) -> None:
        wide = base_image.parent / "wide.png"
        narrow = base_image.parent / "narrow.png"
        content = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        compose_text(
            image=base_image,
            spec=_spec(content=content, anchor="top-left"),
            fonts_root=fonts_root,
            output=wide,
        )
        compose_text(
            image=base_image,
            spec=_spec(content=content, anchor="top-left", max_width=0.3),
            fonts_root=fonts_root,
            output=narrow,
        )

        wide_box = _bounding_box(wide)
        narrow_box = _bounding_box(narrow)
        assert wide_box is not None
        assert narrow_box is not None
        assert narrow_box[3] > wide_box[3]
        assert narrow_box[2] < wide_box[2]

    def test_wrapped_width_stays_within_limit(self, base_image: Path, fonts_root: Path) -> None:
        output = base_image.parent / "out.png"

        compose_text(
            image=base_image,
            spec=_spec(content="ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 3, max_width=0.5),
            fonts_root=fonts_root,
            output=output,
        )

        box = _bounding_box(output)
        assert box is not None
        assert box[2] - box[0] <= CANVAS[0] * 0.5 + 2

    def test_vertical_grows_downward(self, base_image: Path, fonts_root: Path) -> None:
        horizontal = base_image.parent / "h.png"
        vertical = base_image.parent / "v.png"

        compose_text(
            image=base_image, spec=_spec(content="ABCD"), fonts_root=fonts_root, output=horizontal
        )
        compose_text(
            image=base_image,
            spec=_spec(content="ABCD", direction="vertical"),
            fonts_root=fonts_root,
            output=vertical,
        )

        h_box = _bounding_box(horizontal)
        v_box = _bounding_box(vertical)
        assert h_box is not None
        assert v_box is not None
        assert (v_box[3] - v_box[1]) > (h_box[3] - h_box[1])
        assert (v_box[2] - v_box[0]) < (h_box[2] - h_box[0])

    def test_stroke_increases_drawn_area(self, base_image: Path, fonts_root: Path) -> None:
        plain = base_image.parent / "plain.png"
        stroked = base_image.parent / "stroked.png"

        compose_text(image=base_image, spec=_spec(), fonts_root=fonts_root, output=plain)
        compose_text(
            image=base_image,
            spec=_spec(stroke={"width": 4, "color": "#00ff00"}),
            fonts_root=fonts_root,
            output=stroked,
        )

        assert _opaque_pixels(stroked) > _opaque_pixels(plain)

    def test_shadow_increases_drawn_area(self, base_image: Path, fonts_root: Path) -> None:
        plain = base_image.parent / "plain.png"
        shadowed = base_image.parent / "shadowed.png"

        compose_text(image=base_image, spec=_spec(), fonts_root=fonts_root, output=plain)
        compose_text(
            image=base_image,
            # 背景と同じ黒だと差が出ないため、影にも判別できる色を与える
            spec=_spec(shadow={"offset": [6, 6], "blur": 2, "color": "#0000ff", "opacity": 1.0}),
            fonts_root=fonts_root,
            output=shadowed,
        )

        assert _opaque_pixels(shadowed) > _opaque_pixels(plain)

    def test_draws_box_behind_text(self, base_image: Path, fonts_root: Path) -> None:
        output = base_image.parent / "out.png"

        compose_text(
            image=base_image,
            spec=_spec(box={"color": "#0000ff", "opacity": 1.0, "padding": [20, 20]}),
            fonts_root=fonts_root,
            output=output,
        )

        colors = set(_color_counts(output))
        assert (0, 0, 255) in colors

    def test_rotation_changes_bounding_box(self, base_image: Path, fonts_root: Path) -> None:
        straight = base_image.parent / "straight.png"
        rotated = base_image.parent / "rotated.png"

        compose_text(
            image=base_image, spec=_spec(content="ABCDEF"), fonts_root=fonts_root, output=straight
        )
        compose_text(
            image=base_image,
            spec=_spec(content="ABCDEF", rotation=45.0),
            fonts_root=fonts_root,
            output=rotated,
        )

        straight_box = _bounding_box(straight)
        rotated_box = _bounding_box(rotated)
        assert straight_box is not None
        assert rotated_box is not None
        assert (rotated_box[3] - rotated_box[1]) > (straight_box[3] - straight_box[1])

    def test_later_layer_is_drawn_on_top(self, base_image: Path, fonts_root: Path) -> None:
        output = base_image.parent / "out.png"
        spec = TextSpec.model_validate(
            {
                "layers": [
                    {
                        "content": "A",
                        "font": "test.ttf",
                        "size": 200,
                        "color": "#ff0000",
                        "box": {"color": "#ff0000", "opacity": 1.0},
                    },
                    {
                        "content": "A",
                        "font": "test.ttf",
                        "size": 200,
                        "color": "#00ff00",
                        "box": {"color": "#00ff00", "opacity": 1.0},
                    },
                ]
            }
        )

        compose_text(image=base_image, spec=spec, fonts_root=fonts_root, output=output)

        colors = set(_color_counts(output))
        assert (0, 255, 0) in colors
        assert (255, 0, 0) not in colors

    def test_writes_jpeg_as_rgb(self, tmp_path: Path, fonts_root: Path) -> None:
        source = tmp_path / "base.jpg"
        Image.new("RGB", CANVAS, (10, 10, 10)).save(source)
        output = tmp_path / "out.jpg"

        compose_text(image=source, spec=_spec(), fonts_root=fonts_root, output=output)

        with Image.open(output) as image:
            assert image.mode == "RGB"

    def test_fails_for_unsupported_output_extension(
        self, base_image: Path, fonts_root: Path
    ) -> None:
        output = base_image.parent / "out.txt"

        with pytest.raises(TextCompositionError, match="対応していない出力形式"):
            compose_text(image=base_image, spec=_spec(), fonts_root=fonts_root, output=output)

        assert not output.exists()

    def test_does_not_overwrite_existing_output(self, base_image: Path, fonts_root: Path) -> None:
        output = base_image.parent / "out.png"
        output.write_bytes(b"existing")

        with pytest.raises(TextCompositionError, match="既に存在"):
            compose_text(image=base_image, spec=_spec(), fonts_root=fonts_root, output=output)

        assert output.read_bytes() == b"existing"

    def test_fails_when_font_missing(self, base_image: Path, tmp_path: Path) -> None:
        output = base_image.parent / "out.png"

        with pytest.raises(TextCompositionError, match="フォントが見つかりません"):
            compose_text(
                image=base_image, spec=_spec(), fonts_root=tmp_path / "empty", output=output
            )

    def test_fails_when_image_unreadable(self, tmp_path: Path, fonts_root: Path) -> None:
        broken = tmp_path / "broken.png"
        broken.write_bytes(b"not an image")

        with pytest.raises(TextCompositionError, match="画像"):
            compose_text(
                image=broken, spec=_spec(), fonts_root=fonts_root, output=tmp_path / "out.png"
            )

    def test_creates_output_directory(self, base_image: Path, fonts_root: Path) -> None:
        output = base_image.parent / "nested" / "dir" / "out.png"

        compose_text(image=base_image, spec=_spec(), fonts_root=fonts_root, output=output)

        assert output.is_file()

    def test_fails_when_font_file_is_corrupt(self, base_image: Path, tmp_path: Path) -> None:
        corrupt_root = tmp_path / "corrupt_fonts"
        corrupt_root.mkdir()
        (corrupt_root / "broken.ttf").write_bytes(b"not a font file" * 8)
        output = base_image.parent / "out.png"

        with pytest.raises(TextCompositionError, match="フォントを読み込めません"):
            compose_text(
                image=base_image,
                spec=_spec(font="broken.ttf"),
                fonts_root=corrupt_root,
                output=output,
            )

        assert not output.exists()

    def test_fails_when_image_exceeds_max_pixels(self, base_image: Path, fonts_root: Path) -> None:
        output = base_image.parent / "out.png"
        max_pixels = CANVAS[0] * CANVAS[1] - 1

        with pytest.raises(TextCompositionError, match="大きすぎます"):
            compose_text(
                image=base_image,
                spec=_spec(),
                fonts_root=fonts_root,
                output=output,
                max_pixels=max_pixels,
            )

        assert not output.exists()

    def test_allows_image_within_max_pixels(self, base_image: Path, fonts_root: Path) -> None:
        output = base_image.parent / "out.png"
        max_pixels = CANVAS[0] * CANVAS[1]

        compose_text(
            image=base_image,
            spec=_spec(),
            fonts_root=fonts_root,
            output=output,
            max_pixels=max_pixels,
        )

        assert output.is_file()

    def test_does_not_write_through_dangling_symlink(
        self, base_image: Path, fonts_root: Path
    ) -> None:
        target = base_image.parent / "target.png"
        link = base_image.parent / "link.png"
        link.symlink_to(target)

        with pytest.raises(TextCompositionError, match="既に存在"):
            compose_text(image=base_image, spec=_spec(), fonts_root=fonts_root, output=link)

        assert not target.exists()


class TestComposeTextDisplayPath:
    """project_root を渡したときの各エラーメッセージのパス表示。

    絶対パスは実行環境のユーザー名とディレクトリ構成を露出する。PR #32のセキュリティ
    監査指摘を受け、resolve_font と同じ方式 (作業ルート配下なら相対パス) を
    compose_text / _open_image / _save 由来のメッセージにも揃える。
    """

    def test_shows_relative_path_when_output_already_exists(
        self, base_image: Path, fonts_root: Path
    ) -> None:
        project_root = base_image.parent
        output = project_root / "out.png"
        output.write_bytes(b"existing")

        with pytest.raises(TextCompositionError) as excinfo:
            compose_text(
                image=base_image,
                spec=_spec(),
                fonts_root=fonts_root,
                output=output,
                project_root=project_root,
            )

        message = str(excinfo.value)
        assert "out.png" in message
        assert str(output) not in message

    def test_shows_relative_path_when_image_exceeds_max_pixels(
        self, base_image: Path, fonts_root: Path
    ) -> None:
        project_root = base_image.parent
        max_pixels = CANVAS[0] * CANVAS[1] - 1

        with pytest.raises(TextCompositionError) as excinfo:
            compose_text(
                image=base_image,
                spec=_spec(),
                fonts_root=fonts_root,
                output=project_root / "out.png",
                max_pixels=max_pixels,
                project_root=project_root,
            )

        message = str(excinfo.value)
        assert base_image.name in message
        assert str(base_image) not in message

    def test_shows_relative_path_when_image_unreadable(
        self, fonts_root: Path, tmp_path: Path
    ) -> None:
        project_root = tmp_path
        broken = project_root / "broken.png"
        broken.write_bytes(b"not an image")

        with pytest.raises(TextCompositionError) as excinfo:
            compose_text(
                image=broken,
                spec=_spec(),
                fonts_root=fonts_root,
                output=project_root / "out.png",
                project_root=project_root,
            )

        message = str(excinfo.value)
        assert "broken.png" in message
        assert str(broken) not in message

    def test_shows_relative_path_when_output_exists_through_dangling_symlink(
        self, base_image: Path, fonts_root: Path
    ) -> None:
        # display_path は resolve() を経由するため、シンボリックリンク自体の名前
        # ではなくリンク先の名前で表示される。それでもリンクの絶対パス自体は
        # 露出しないことを確認する (表示名がリンク先へ変わること自体は許容する)。
        project_root = base_image.parent
        target = project_root / "target.png"
        link = project_root / "link.png"
        link.symlink_to(target)

        with pytest.raises(TextCompositionError) as excinfo:
            compose_text(
                image=base_image,
                spec=_spec(),
                fonts_root=fonts_root,
                output=link,
                project_root=project_root,
            )

        message = str(excinfo.value)
        assert "既に存在" in message
        assert str(link) not in message
        assert str(target) not in message

    def test_shows_relative_path_when_save_fails(
        self, base_image: Path, fonts_root: Path, tmp_path: Path
    ) -> None:
        project_root = tmp_path
        output_dir = project_root / "readonly"
        output_dir.mkdir()
        output = output_dir / "out.png"
        output_dir.chmod(0o500)
        try:
            with pytest.raises(TextCompositionError) as excinfo:
                compose_text(
                    image=base_image,
                    spec=_spec(),
                    fonts_root=fonts_root,
                    output=output,
                    project_root=project_root,
                )
        finally:
            output_dir.chmod(0o700)

        message = str(excinfo.value)
        assert "readonly/out.png" in message
        assert str(output) not in message

    def test_keeps_absolute_path_when_project_root_omitted(
        self, base_image: Path, fonts_root: Path
    ) -> None:
        output = base_image.parent / "out.png"
        output.write_bytes(b"existing")

        with pytest.raises(TextCompositionError, match=str(output)):
            compose_text(image=base_image, spec=_spec(), fonts_root=fonts_root, output=output)

    def test_keeps_absolute_path_when_outside_project_root(
        self, base_image: Path, fonts_root: Path, tmp_path: Path
    ) -> None:
        project_root = tmp_path / "elsewhere"
        project_root.mkdir()
        output = base_image.parent / "out.png"
        output.write_bytes(b"existing")

        with pytest.raises(TextCompositionError, match=str(output)):
            compose_text(
                image=base_image,
                spec=_spec(),
                fonts_root=fonts_root,
                output=output,
                project_root=project_root,
            )


class TestComposeTextAlign:
    """align (`left` / `center` / `right`) が描画位置を変えることの検証。

    行によって長さの異なる2行を用意し、短い行 (1行目) だけを切り出して
    align ごとに位置が動くことを見る。
    """

    CONTENT = "A\nABCDEFGHIJ"

    def _first_row_box(
        self, base_image: Path, fonts_root: Path, output: Path, *, direction: str
    ) -> dict[str, tuple[int, int, int, int] | None]:
        boxes: dict[str, tuple[int, int, int, int] | None] = {}
        for align in ("left", "center", "right"):
            path = output.with_stem(f"{output.stem}_{align}")
            compose_text(
                image=base_image,
                spec=_spec(
                    content=self.CONTENT, anchor="top-left", align=align, direction=direction
                ),
                fonts_root=fonts_root,
                output=path,
            )
            # 1行目 (短い方) だけを切り出す。line_height = round(size * line_spacing) = 38
            boxes[align] = _row_bounding_box(path, 0, 36)
        return boxes

    def test_horizontal_align_moves_short_first_line(
        self, base_image: Path, fonts_root: Path
    ) -> None:
        output = base_image.parent / "out.png"
        boxes = self._first_row_box(base_image, fonts_root, output, direction="horizontal")

        assert boxes["left"] is not None
        assert boxes["center"] is not None
        assert boxes["right"] is not None
        assert boxes["left"][0] < boxes["center"][0] < boxes["right"][0]

    def test_vertical_align_moves_short_first_column(
        self, base_image: Path, fonts_root: Path
    ) -> None:
        output = base_image.parent / "out.png"
        boxes: dict[str, tuple[int, int, int, int] | None] = {}
        for align in ("left", "center", "right"):
            path = output.with_stem(f"{output.stem}_{align}")
            compose_text(
                image=base_image,
                spec=_spec(
                    content=self.CONTENT, anchor="top-left", align=align, direction="vertical"
                ),
                fonts_root=fonts_root,
                output=path,
            )
            # 列は右から左へ進む。1列目 (短い "A") は column_width=38 * 2列ぶんの
            # 右端 [38, 76) にある。2列目 ("ABCDEFGHIJ") が支配的な全体bboxではなく、
            # この列だけを切り出して縦位置を比較する
            boxes[align] = _column_bounding_box(path, 38, 76)

        assert boxes["left"] is not None
        assert boxes["center"] is not None
        assert boxes["right"] is not None
        # 縦書きでは align が縦位置を動かす。上端 (y) が align ごとに変わる
        assert boxes["left"][1] < boxes["center"][1] < boxes["right"][1]


class TestComposeTextVerticalWrap:
    def test_max_width_wraps_vertical_columns(self, base_image: Path, fonts_root: Path) -> None:
        content = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        tall = base_image.parent / "tall.png"
        wrapped = base_image.parent / "wrapped.png"

        compose_text(
            image=base_image,
            spec=_spec(content=content, anchor="top-left", direction="vertical"),
            fonts_root=fonts_root,
            output=tall,
        )
        compose_text(
            image=base_image,
            spec=_spec(content=content, anchor="top-left", direction="vertical", max_width=0.3),
            fonts_root=fonts_root,
            output=wrapped,
        )

        tall_box = _bounding_box(tall)
        wrapped_box = _bounding_box(wrapped)
        assert tall_box is not None
        assert wrapped_box is not None
        # 縦書きの折り返しは画像の高さが基準。折り返すと列が増えて横へ広がり、
        # 1列あたりの高さは短くなる
        assert wrapped_box[2] > tall_box[2]
        assert wrapped_box[3] < tall_box[3]
