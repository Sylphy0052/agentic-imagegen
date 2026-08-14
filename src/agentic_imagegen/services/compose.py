"""生成後のテキスト合成。

SD1.5 / SDXL 系のモデルは日本語をほぼ描けないため、文字は生成に任せず
ここで描画する。ComfyUI へは依存しないので、生成を伴わない単体の合成にも使える。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from PIL import Image, ImageDraw, ImageFilter, ImageFont, UnidentifiedImageError

from agentic_imagegen.domain.models import TextAnchor, TextLayer, TextSegment, TextSpec
from agentic_imagegen.domain.policy import display_path, resolve_font
from agentic_imagegen.errors import TextCompositionError

#: max_width をこの値以下で指定した場合は画像サイズに対する比率として扱う。
_RATIO_THRESHOLD: Final = 1.0

#: アルファを持てない形式。合成後に RGB へ落とす。
_OPAQUE_SUFFIXES: Final = frozenset({".jpg", ".jpeg"})

_HORIZONTAL_ANCHORS: Final = {
    "left": 0.0,
    "center": 0.5,
    "right": 1.0,
}

_VERTICAL_ANCHORS: Final = {
    "top": 0.0,
    "middle": 0.5,
    "bottom": 1.0,
}

#: 縦書きで右上へ寄せる句読点。全角枠の左下に寄っている横書き用の字形を補正する。
_VERTICAL_PUNCTUATION_CHARS: Final = frozenset("、。，．")

#: 縦書きで右上へ寄せる小書き文字 (捨て仮名)。句読点より補正量は小さい。
_VERTICAL_SMALL_KANA_CHARS: Final = frozenset("ぁぃぅぇぉっゃゅょゎゕゖァィゥェォッャュョヮヵヶ")

#: 句読点の移動量 (dx, dy)。em (layer.size) に対する比率。
_VERTICAL_PUNCTUATION_OFFSET: Final = (0.5, -0.5)

#: 小書き文字の移動量 (dx, dy)。em (layer.size) に対する比率。
_VERTICAL_SMALL_KANA_OFFSET: Final = (0.08, -0.08)

#: 縦書きで時計回りに90度回転させる約物。長音・ダッシュ類、括弧類、
#: 三点/二点リーダ、コロン・セミコロンをまとめる。
_VERTICAL_ROTATED_CHARS: Final = frozenset(
    "ー〜～−—―‐-（）()「」『』【】〔〕〈〉《》［］[]｛｝{}…‥：；"
)

#: ルビの文字サイズ。親文字 (layer.size) に対する比率。日本語組版で
#: 一般的な「親文字の半分」を採る。
_RUBY_SIZE_RATIO: Final = 0.5


@dataclass(frozen=True, slots=True)
class ResolvedFont:
    """レイヤが実際に使ったフォント。

    metadata へ残し、同じSpecで見た目が変わったときにフォントの差し替えを
    切り分けられるようにする。
    """

    name: str
    path: Path
    index: int


@dataclass(frozen=True, slots=True)
class ComposeResult:
    """1回の合成の結果。"""

    output: Path
    fonts: tuple[ResolvedFont, ...]


@dataclass(frozen=True, slots=True)
class TextCell:
    """縦書き1列・横書き1行を構成する最小単位。

    通常は1文字=1セルだが、縦中横 (`tate_chu_yoko=True`) とルビ付き
    (`ruby` が空でない) では複数文字が1セルへまとまる。折り返しはセル単位で
    行うことで、縦中横セルやルビの掛かる親文字を途中で分割しないようにする。

    縦中横は複数文字を字送り1つぶんへ収めるのに対し、ルビ付きセルは親文字を
    1文字ずつ並べたままルビを添えるだけなので、占める字送りの数が異なる
    (`rows`)。
    """

    text: str
    tate_chu_yoko: bool = False
    ruby: str = ""

    @property
    def rows(self) -> int:
        """このセルが占める字送りの数。

        縦中横は何文字でも1セル (字送り1つ) へ収める。それ以外は1文字1字送りで、
        ルビ付きセルだけが複数文字を持ちうる。
        """
        if self.tate_chu_yoko:
            return 1
        return len(self.text)


def _content_to_cell_paragraphs(content: str | tuple[TextSegment, ...]) -> list[list[TextCell]]:
    """content を段落 (`\\n` 区切り) ごとのセル列へ正規化する。

    文字列指定は1文字=1セルとして展開し、既存の縦書き (1文字1セル) の挙動と
    完全に一致させる。セグメント列指定は `tate_chu_yoko=True` のセグメントと
    ルビ付きのセグメントを1セグメント=1セルとし、どちらでもないセグメントは
    文字列指定と同じく1文字ずつ1セルへ展開したうえで隣接するセグメントを
    同じ段落として連結する。

    ルビ付きセグメントを1セルへまとめるのは、ルビを親文字の全長へ均等割りする
    ためにその全長を1つの描画単位として持つ必要があるため。改行は domain 側で
    拒否済みのため、段落を跨ぐ親文字は来ない。
    """
    if isinstance(content, str):
        return [[TextCell(text=char) for char in paragraph] for paragraph in content.split("\n")]

    paragraphs: list[list[TextCell]] = [[]]
    for segment in content:
        if segment.tate_chu_yoko or segment.ruby is not None:
            paragraphs[-1].append(
                TextCell(
                    text=segment.text,
                    tate_chu_yoko=segment.tate_chu_yoko,
                    ruby=segment.ruby or "",
                )
            )
            continue
        for index, part in enumerate(segment.text.split("\n")):
            if index > 0:
                paragraphs.append([])
            paragraphs[-1].extend(TextCell(text=char) for char in part)
    return paragraphs


def _wrap_cell_lines(
    paragraphs: Sequence[list[TextCell]],
    *,
    measure: Callable[[list[TextCell]], float],
    max_width: float,
) -> list[list[TextCell]]:
    """`wrap_lines` と同じ考え方をセル列に適用し、幅を超えたセルの手前で折り返す。

    セル境界でのみ折り返すため、縦中横セルが文字の途中で割れることはない。
    1セルだけで幅を超える場合も、そのセルを捨てずに1行として残す
    (`wrap_lines` と同じ規則)。
    """
    lines: list[list[TextCell]] = []
    for paragraph in paragraphs:
        if not paragraph:
            lines.append([])
            continue

        current: list[TextCell] = []
        for cell in paragraph:
            candidate = [*current, cell]
            if current and measure(candidate) > max_width:
                lines.append(current)
                current = [cell]
            else:
                current = candidate
        lines.append(current)
    return lines


def parse_color(value: str, *, opacity: float = 1.0) -> tuple[int, int, int, int]:
    """`#rgb` / `#rrggbb` / `#rrggbbaa` を RGBA へ変換する。

    形式の検証は TextLayer 側で済んでいる。opacity はアルファへ掛け合わせる。
    """
    body = value.removeprefix("#")
    if len(body) == 3:
        body = "".join(ch * 2 for ch in body)
    if len(body) == 6:
        body = f"{body}ff"

    red = int(body[0:2], 16)
    green = int(body[2:4], 16)
    blue = int(body[4:6], 16)
    alpha = int(body[6:8], 16)
    return red, green, blue, round(alpha * opacity)


def anchor_origin(
    anchor: TextAnchor,
    *,
    canvas: tuple[int, int],
    block: tuple[int, int],
    offset: tuple[int, int],
) -> tuple[int, int]:
    """アンカーとオフセットから、描画ブロックの左上座標を求める。"""
    vertical, _, horizontal = anchor.partition("-")
    if not horizontal:
        # "center" は縦横ともに中央を指す
        vertical, horizontal = "middle", "center"

    x = round((canvas[0] - block[0]) * _HORIZONTAL_ANCHORS[horizontal]) + offset[0]
    y = round((canvas[1] - block[1]) * _VERTICAL_ANCHORS[vertical]) + offset[1]
    return x, y


def wrap_lines(
    content: str,
    *,
    measure: Callable[[str], float],
    max_width: float | None,
) -> list[str]:
    """明示的な改行で分けたうえで、幅を超える行を文字単位で折り返す。

    日本語は単語境界を持たないため、空白ではなく文字単位で折り返す。
    1文字だけで幅を超える場合は、その文字を捨てずに1行として残す。
    """
    paragraphs = content.split("\n")
    if max_width is None:
        return paragraphs

    lines: list[str] = []
    for paragraph in paragraphs:
        if not paragraph:
            lines.append("")
            continue

        current = ""
        for char in paragraph:
            candidate = current + char
            if current and measure(candidate) > max_width:
                lines.append(current)
                current = char
            else:
                current = candidate
        lines.append(current)
    return lines


def compose_text(
    *,
    image: Path,
    spec: TextSpec,
    fonts_root: Path,
    output: Path,
    max_pixels: int | None = None,
    project_root: Path | None = None,
) -> ComposeResult:
    """画像へテキストを合成し、別ファイルとして書き出す。

    入力画像は変更しない。出力先が既にある場合は上書きせずに失敗する。

    project_root はエラーメッセージに出すパスの表示にのみ使う (詳細は resolve_font
    を参照)。作業ルート配下のパスは相対パスへ丸め、実行環境のディレクトリ構成を
    露出しないようにする。省略時は従来どおり絶対パスで表示する。
    """
    if output.exists() or output.is_symlink():
        # exists() はdangling symlinkをFalseとして扱う。放置すると _save の書き込みが
        # リンク先へ実体を作ってしまうため、symlinkであること自体も拒否する。
        # 実際の排他性は _save の O_EXCL 書き込みで保証する (ここはメッセージ品質のため)。
        raise TextCompositionError(f"出力先が既に存在します: {display_path(output, project_root)}")

    fonts = tuple(
        ResolvedFont(
            name=layer.font,
            path=resolve_font(layer.font, fonts_root, project_root=project_root),
            index=layer.font_index,
        )
        for layer in spec.layers
    )

    canvas, source_mode = _open_image(image, project_root=project_root)
    if max_pixels is not None and canvas.width * canvas.height > max_pixels:
        pixels = canvas.width * canvas.height
        raise TextCompositionError(
            f"画像が大きすぎます ({pixels} pixels > {max_pixels} pixels): "
            f"{display_path(image, project_root)}"
        )

    for layer, font in zip(spec.layers, fonts, strict=True):
        canvas = Image.alpha_composite(canvas, _render_layer(layer, font, canvas.size))

    output.parent.mkdir(parents=True, exist_ok=True)
    _save(canvas, output, source_mode, project_root=project_root)
    return ComposeResult(output=output, fonts=fonts)


def _open_image(path: Path, *, project_root: Path | None = None) -> tuple[Image.Image, str]:
    try:
        with Image.open(path) as opened:
            return opened.convert("RGBA"), opened.mode
    except (UnidentifiedImageError, OSError) as exc:
        raise TextCompositionError(
            f"画像を開けません: {display_path(path, project_root)} "
            f"({_redact_path_in_exc(exc, path, project_root)})"
        ) from exc


def _redact_path_in_exc(exc: Exception, path: Path, project_root: Path | None) -> str:
    """例外の文字列表現に埋め込まれた絶対パスを表示用パスへ置き換える。

    open() 等のOS例外は引数のパスをそのまま文字列化に含める (例:
    `[Errno 13] Permission denied: '/abs/path'`)。呼び出し側で prefix を
    display_path で丸めても、この部分をそのまま連結すると絶対パスが漏れ戻る
    ため、同じ変換をここでも当てる。
    """
    text = str(exc)
    display = display_path(path, project_root)
    for candidate in {str(path), str(path.resolve())}:
        if candidate in text:
            text = text.replace(candidate, display)
    return text


def _save(
    canvas: Image.Image,
    output: Path,
    source_mode: str,
    *,
    project_root: Path | None = None,
) -> None:
    opaque_output = output.suffix.lower() in _OPAQUE_SUFFIXES
    keep_alpha = not opaque_output and source_mode in {"RGBA", "LA", "P", "RGB"}
    image = canvas if keep_alpha else canvas.convert("RGB")

    save_format = Image.registered_extensions().get(output.suffix.lower())
    if save_format is None:
        raise TextCompositionError(f"対応していない出力形式です: {output.suffix}")

    # 存在確認から書き込みまでの間に他プロセスが同じパスへ作成する競合 (TOCTOU) や、
    # dangling symlink 経由でリンク先へ書き込んでしまう事態を、O_EXCL相当の排他生成で塞ぐ。
    try:
        with output.open("xb") as fh:
            image.save(fh, format=save_format)
    except FileExistsError as exc:
        raise TextCompositionError(
            f"出力先が既に存在します: {display_path(output, project_root)}"
        ) from exc
    except (OSError, ValueError) as exc:
        # 書き込み中の失敗で作りかけのファイルを残さない。
        output.unlink(missing_ok=True)
        raise TextCompositionError(
            f"画像を書き出せません: {display_path(output, project_root)} "
            f"({_redact_path_in_exc(exc, output, project_root)})"
        ) from exc


def _load_font(font: ResolvedFont, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(font.path), size=size, index=font.index)
    except (OSError, ValueError) as exc:
        raise TextCompositionError(f"フォントを読み込めません: {font.name} ({exc})") from exc


def _render_layer(
    layer: TextLayer, font: ResolvedFont, canvas_size: tuple[int, int]
) -> Image.Image:
    """レイヤ1件を、キャンバスと同じ大きさの透明な画像へ描く。

    回転と不透明度をレイヤ単位で閉じるため、レイヤごとに別の画像を作る。
    """
    loaded = _load_font(font, layer.size)
    lines = _wrap_layer(layer, loaded, canvas_size)
    block = _block_size(layer, loaded, lines)
    origin = anchor_origin(layer.anchor, canvas=canvas_size, block=block, offset=layer.offset)

    rendered = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    if layer.box is not None:
        _draw_box(rendered, layer, origin, block)
    if layer.shadow is not None:
        rendered = Image.alpha_composite(
            rendered, _render_shadow(layer, loaded, lines, origin, block, canvas_size)
        )
    _draw_text(
        rendered,
        layer,
        loaded,
        lines,
        origin,
        block,
        fill=parse_color(layer.color),
        stroke=True,
    )

    if layer.rotation:
        center = (origin[0] + block[0] / 2, origin[1] + block[1] / 2)
        rendered = rendered.rotate(layer.rotation, resample=Image.Resampling.BICUBIC, center=center)
    if layer.opacity < 1.0:
        rendered.putalpha(rendered.getchannel("A").point(_alpha_scaler(layer.opacity)))
    return rendered


def _alpha_scaler(opacity: float) -> Callable[[int], int]:
    def scale(value: int) -> int:
        return round(value * opacity)

    return scale


def _wrap_layer(
    layer: TextLayer, font: ImageFont.FreeTypeFont, canvas_size: tuple[int, int]
) -> list[list[TextCell]]:
    """content をセル単位の行 (横書き) / 列 (縦書き) へ折り返す。

    縦中横セルは `_wrap_cell_lines` がセル境界でしか折り返さないため、
    ここでは常にセル単位のリストを返す。横書きは tate_chu_yoko セルを
    持てない (domain 側で検証済み) ため、後続の描画は文字列へ戻して
    既存ロジックをそのまま使える。
    """
    paragraphs = _content_to_cell_paragraphs(layer.content)
    if layer.max_width is None:
        return paragraphs

    # 縦書きでは行が縦へ伸びるため、折り返しの基準も画像の高さになる
    axis = canvas_size[1] if layer.direction == "vertical" else canvas_size[0]
    limit = layer.max_width * axis if layer.max_width <= _RATIO_THRESHOLD else layer.max_width

    if layer.direction == "vertical":
        advance = _vertical_advance(layer)
        return _wrap_cell_lines(
            paragraphs,
            measure=lambda cells: sum(cell.rows for cell in cells) * advance,
            max_width=limit,
        )
    return _wrap_cell_lines(
        paragraphs,
        measure=lambda cells: font.getlength("".join(cell.text for cell in cells)),
        max_width=limit,
    )


def _line_height(layer: TextLayer) -> int:
    return round(layer.size * layer.line_spacing)


def _vertical_advance(layer: TextLayer) -> int:
    """縦書きの字送り。列の間隔とは別に、文字は等間隔で置く。"""
    return layer.size


def _ruby_size(layer: TextLayer) -> int:
    """ルビの文字サイズ (px)。"""
    return max(1, round(layer.size * _RUBY_SIZE_RATIO))


def _ruby_band_width(layer: TextLayer, lines: Sequence[list[TextCell]]) -> int:
    """列幅のうちルビへ充てる帯の幅 (px)。ルビが1つも無ければ 0。

    レイヤ内に1つでもルビがあれば、ルビの付かない列も含めて**全列を一律**で
    広げる。列ごとに幅を変えると列の間隔が揃わず、縦組みとして読みにくくなる。
    """
    if layer.direction != "vertical":
        return 0
    if not any(cell.ruby for line in lines for cell in line):
        return 0
    return _ruby_size(layer)


def _vertical_column_width(layer: TextLayer, lines: Sequence[list[TextCell]]) -> tuple[int, int]:
    """縦書きの (親文字帯の幅, 列全体の幅) を返す。

    列は右から左へ進み、列の右側がルビ帯になる。
    """
    glyph_band = _line_height(layer)
    return glyph_band, glyph_band + _ruby_band_width(layer, lines)


def _block_size(
    layer: TextLayer, font: ImageFont.FreeTypeFont, lines: Sequence[list[TextCell]]
) -> tuple[int, int]:
    if layer.direction == "vertical":
        _, column_width = _vertical_column_width(layer, lines)
        # rows は字送りの数。縦中横セルは何文字でも1、ルビ付きセルは親文字の
        # 文字数ぶんを数えるため、ルビの有無で列の高さは変わらない。
        rows = max((sum(cell.rows for cell in line) for line in lines), default=0)
        return column_width * len(lines), _vertical_advance(layer) * rows

    width = max(
        (font.getlength("".join(cell.text for cell in line)) for line in lines), default=0.0
    )
    return round(width), _line_height(layer) * len(lines)


def _draw_box(
    canvas: Image.Image,
    layer: TextLayer,
    origin: tuple[int, int],
    block: tuple[int, int],
) -> None:
    box = layer.box
    assert box is not None  # noqa: S101 - 呼び出し側で確認済み

    pad_x, pad_y = box.padding
    rectangle = (
        origin[0] - pad_x,
        origin[1] - pad_y,
        origin[0] + block[0] + pad_x,
        origin[1] + block[1] + pad_y,
    )
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        rectangle,
        radius=box.radius,
        fill=parse_color(box.color, opacity=box.opacity),
    )


def _render_shadow(
    layer: TextLayer,
    font: ImageFont.FreeTypeFont,
    lines: Sequence[list[TextCell]],
    origin: tuple[int, int],
    block: tuple[int, int],
    canvas_size: tuple[int, int],
) -> Image.Image:
    shadow = layer.shadow
    assert shadow is not None  # noqa: S101 - 呼び出し側で確認済み

    rendered = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    shifted = (origin[0] + shadow.offset[0], origin[1] + shadow.offset[1])
    _draw_text(
        rendered,
        layer,
        font,
        lines,
        shifted,
        block,
        fill=parse_color(shadow.color, opacity=shadow.opacity),
        stroke=False,
    )
    if shadow.blur > 0:
        rendered = rendered.filter(ImageFilter.GaussianBlur(shadow.blur))
    return rendered


def _draw_text(
    canvas: Image.Image,
    layer: TextLayer,
    font: ImageFont.FreeTypeFont,
    lines: Sequence[list[TextCell]],
    origin: tuple[int, int],
    block: tuple[int, int],
    *,
    fill: tuple[int, int, int, int],
    stroke: bool,
) -> None:
    draw = ImageDraw.Draw(canvas)
    stroke_width = layer.stroke.width if stroke and layer.stroke is not None else 0
    stroke_fill = parse_color(layer.stroke.color) if stroke and layer.stroke is not None else None

    if layer.direction == "vertical":
        _draw_vertical(
            canvas, draw, layer, font, lines, origin, block, fill, stroke_width, stroke_fill
        )
        return
    _draw_horizontal(draw, layer, font, lines, origin, block, fill, stroke_width, stroke_fill)


def _draw_horizontal(
    draw: ImageDraw.ImageDraw,
    layer: TextLayer,
    font: ImageFont.FreeTypeFont,
    lines: Sequence[list[TextCell]],
    origin: tuple[int, int],
    block: tuple[int, int],
    fill: tuple[int, int, int, int],
    stroke_width: int,
    stroke_fill: tuple[int, int, int, int] | None,
) -> None:
    # 横書きは tate_chu_yoko セルを持たない (domain 側で検証済み) ため、
    # セルを文字列へ戻せば既存のロジックと完全に同じ結果になる。
    line_height = _line_height(layer)
    for index, line in enumerate(lines):
        text = "".join(cell.text for cell in line)
        offset = _align_offset(layer.align, block[0], font.getlength(text))
        draw.text(
            (origin[0] + offset, origin[1] + index * line_height),
            text,
            font=font,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )


def _draw_vertical(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    layer: TextLayer,
    font: ImageFont.FreeTypeFont,
    lines: Sequence[list[TextCell]],
    origin: tuple[int, int],
    block: tuple[int, int],
    fill: tuple[int, int, int, int],
    stroke_width: int,
    stroke_fill: tuple[int, int, int, int] | None,
) -> None:
    """縦書きを1セルずつ配置して描く。

    Pillow は縦書きを持たないため自前で並べる。列は右から左へ進める。
    句読点・小書き文字は右上へ寄せ、長音や括弧などの約物は時計回りに90度回転させ、
    縦中横セル (`tate_chu_yoko=True`) は数文字を横に組んで1セルへ収める
    (常時適用、GenerationSpec 側にON/OFFの指定はない)。

    ルビ付きセル (`ruby` が空でない) は、親文字を通常どおり1文字ずつ並べたうえで、
    列の右側のルビ帯へルビを添える。ルビ帯は `_ruby_band_width` が全列一律で
    確保するため、ルビの有無で列の間隔は変わらない。
    """
    glyph_band, column_width = _vertical_column_width(layer, lines)
    advance = _vertical_advance(layer)
    for column, line in enumerate(lines):
        x = origin[0] + block[0] - (column + 1) * column_width
        column_offset = _align_offset(
            layer.align, block[1], advance * sum(cell.rows for cell in line)
        )
        row = 0
        for cell in line:
            y = origin[1] + column_offset + row * advance
            if cell.tate_chu_yoko:
                _draw_tate_chu_yoko_cell(
                    canvas,
                    font,
                    cell.text,
                    (x + glyph_band / 2, y + advance / 2),
                    glyph_band,
                    advance,
                    fill,
                    stroke_width,
                    stroke_fill,
                )
            else:
                for index, char in enumerate(cell.text):
                    _draw_vertical_char(
                        canvas,
                        draw,
                        font,
                        char,
                        (x, y + index * advance),
                        band_width=glyph_band,
                        size=layer.size,
                        advance=advance,
                        fill=fill,
                        stroke_width=stroke_width,
                        stroke_fill=stroke_fill,
                    )
            if cell.ruby:
                _draw_ruby(
                    canvas,
                    draw,
                    font,
                    cell.ruby,
                    layer=layer,
                    band=(x + glyph_band, column_width - glyph_band),
                    span=(y, cell.rows * advance),
                    fill=fill,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_fill,
                )
            row += cell.rows


def _draw_vertical_char(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    char: str,
    position: tuple[float, float],
    *,
    band_width: int,
    size: int,
    advance: int,
    fill: tuple[int, int, int, int],
    stroke_width: int,
    stroke_fill: tuple[int, int, int, int] | None,
) -> None:
    """縦書きの1文字を、幅 `band_width` の帯の中央へ置く。

    親文字にもルビにも同じ規則 (約物の回転・句読点や小書き文字の位置補正) を
    当てるため、`size` / `advance` を引数で受けてレイヤのフォントサイズから
    切り離してある。
    """
    x, y = position
    if char in _VERTICAL_ROTATED_CHARS:
        _draw_rotated_vertical_char(
            canvas,
            font,
            char,
            (x + band_width / 2, y + advance / 2),
            size=size,
            advance=advance,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
        return
    centered = (band_width - font.getlength(char)) / 2
    dx, dy = _vertical_glyph_offset(char, size)
    draw.text(
        (x + centered + dx, y + dy),
        char,
        font=font,
        fill=fill,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )


def _draw_ruby(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    ruby: str,
    *,
    layer: TextLayer,
    band: tuple[float, int],
    span: tuple[float, int],
    fill: tuple[int, int, int, int],
    stroke_width: int,
    stroke_fill: tuple[int, int, int, int] | None,
) -> None:
    """親文字の右のルビ帯へ、ルビを縦に並べて描く。

    `band` は (帯の左端x, 帯の幅)、`span` は (親文字の上端y, 親文字の高さ)。

    ルビが親文字の高さへ収まる場合は親文字の全長へ**均等割り**する
    (JIS X 4051 のモノルビと同じ振り方)。収まらない場合は詰めずに、
    親文字の前後へ均等に**はみ出す**。同じ列で隣り合うセルにもルビがあると
    はみ出したぶんが重なりうるが、ルビを詰めて読めなくするよりは良いと判断した。
    """
    if not ruby:
        return

    band_left, band_width = band
    top, height = span
    size = _ruby_size(layer)
    ruby_font = font.font_variant(size=size)
    # 親文字と同じ比率で縁取りを細くする。親文字ぶんの太さのままでは
    # 半分の大きさのルビが縁取りで潰れる。
    ruby_stroke = round(stroke_width * _RUBY_SIZE_RATIO)

    total = len(ruby) * size
    if total <= height:
        pitch = height / len(ruby)
        first_center = top + pitch / 2
    else:
        pitch = float(size)
        first_center = top + height / 2 - total / 2 + size / 2

    for index, char in enumerate(ruby):
        center = first_center + index * pitch
        _draw_vertical_char(
            canvas,
            draw,
            ruby_font,
            char,
            (band_left, center - size / 2),
            band_width=band_width,
            size=size,
            advance=size,
            fill=fill,
            stroke_width=ruby_stroke,
            stroke_fill=stroke_fill,
        )


def _vertical_glyph_offset(char: str, size: int) -> tuple[float, float]:
    """縦書きで右上へ寄せる補正量 (px)。対象外の文字は (0, 0)。

    横書き用の字形は全角枠の左下に寄っているため、句読点・小書き文字を
    右上へずらして縦中の見た目を整える。移動量は em (size) に対する固定比率で、
    フォント実測 (font.getbbox()) は使わない。
    """
    if char in _VERTICAL_PUNCTUATION_CHARS:
        dx, dy = _VERTICAL_PUNCTUATION_OFFSET
    elif char in _VERTICAL_SMALL_KANA_CHARS:
        dx, dy = _VERTICAL_SMALL_KANA_OFFSET
    else:
        return 0.0, 0.0
    return dx * size, dy * size


def _draw_rotated_vertical_char(
    canvas: Image.Image,
    font: ImageFont.FreeTypeFont,
    char: str,
    cell_center: tuple[float, float],
    *,
    size: int,
    advance: int,
    fill: tuple[int, int, int, int],
    stroke_width: int,
    stroke_fill: tuple[int, int, int, int] | None,
) -> None:
    """約物を時計回りに90度回転させて描く。

    1文字ぶんの正方形タイルへ文字を (stroke込みで) 描き、タイルごと回転してから
    親キャンバスへ合成する。stroke_width / stroke_fill をタイルへ描く段階で
    渡さないと、縁取りだけ回転せず残ってしまう。

    ルビからも呼ぶため、文字サイズはレイヤではなく `size` で受ける。
    """
    tile_side = size * 2 + stroke_width * 2
    tile = Image.new("RGBA", (tile_side, tile_side), (0, 0, 0, 0))
    tile_draw = ImageDraw.Draw(tile)
    tile_draw.text(
        (tile_side / 2 - font.getlength(char) / 2, tile_side / 2 - advance / 2),
        char,
        font=font,
        fill=fill,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )

    # Pillow の rotate は反時計回りが正のため、時計回り90度は -90 を渡す。
    rotated = tile.rotate(
        -90, resample=Image.Resampling.BICUBIC, center=(tile_side / 2, tile_side / 2)
    )
    dest = (round(cell_center[0] - tile_side / 2), round(cell_center[1] - tile_side / 2))
    canvas.alpha_composite(rotated, dest=dest)


def _draw_tate_chu_yoko_cell(
    canvas: Image.Image,
    font: ImageFont.FreeTypeFont,
    text: str,
    cell_center: tuple[float, float],
    column_width: int,
    advance: int,
    fill: tuple[int, int, int, int],
    stroke_width: int,
    stroke_fill: tuple[int, int, int, int] | None,
) -> None:
    """縦中横 (数文字を横に組んで1セルへ収める組版) の1セルを描く。

    _draw_rotated_vertical_char と同じく、タイルへ (stroke込みで) 描いてから
    親キャンバスへ合成する。回転ではなくセル幅を超えた場合の縮小が異なる点が
    違うだけで、stroke_width / stroke_fill をタイル描画の段階で渡さないと
    縁取りが縮小されず残ってしまう罠は同じ。
    """
    tile_width = max(round(font.getlength(text)) + stroke_width * 2, 1)
    # 字送り1つ分の高さでは `g` / `j` / `p` / `q` の descender がタイル下端で切れる
    # (フォントの ascender + descender は 1em を超える)。_draw_rotated_vertical_char が
    # 正方形タイルを2倍取っているのと同じ理由で、ここも余裕を持たせる。
    # はみ出した分は透明のまま合成されるので、行の高さも列の幅も変わらない。
    tile_height = advance * 2 + stroke_width * 2
    tile = Image.new("RGBA", (tile_width, tile_height), (0, 0, 0, 0))
    tile_draw = ImageDraw.Draw(tile)
    tile_draw.text(
        (stroke_width, tile_height / 2 - advance / 2),
        text,
        font=font,
        fill=fill,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )

    # 行の高さも列の幅も変えない。テキストが1セル幅に収まらない場合だけ、
    # 横方向へ縮小して1セルへ収める (縦中横は「セルへ収める」組版のため)。
    if tile.width > column_width:
        scale = column_width / tile.width
        tile = tile.resize(
            (column_width, max(1, round(tile.height * scale))),
            resample=Image.Resampling.BICUBIC,
        )

    dest = (round(cell_center[0] - tile.width / 2), round(cell_center[1] - tile.height / 2))
    canvas.alpha_composite(tile, dest=dest)


def _align_offset(align: str, block_size: float, line_size: float) -> float:
    if align == "center":
        return (block_size - line_size) / 2
    if align == "right":
        return block_size - line_size
    return 0.0


__all__ = [
    "ComposeResult",
    "ResolvedFont",
    "anchor_origin",
    "compose_text",
    "parse_color",
    "wrap_lines",
]
