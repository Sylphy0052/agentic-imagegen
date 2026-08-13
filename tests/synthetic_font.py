"""テキスト合成のテスト専用に、その場で組み立てる最小のTrueTypeフォント。

リポジトリへフォントを同梱するとライセンス確認が必要になるため、`fontTools` で
バイナリを生成する。全てのグリフを同じ幅の塗り潰し矩形にすることで、
どの文字を描いても結果が決定的になり、ホスト環境にあるフォントへ依存しなくなる。

テキスト合成のテストはグリフの見た目ではなくレイアウト (幅・bounding box・
画素差分) を検証しているため、実在の書体を積む必要はない。
"""

from __future__ import annotations

from io import BytesIO

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

#: フォントの座標系の基準。値そのものに意味はなく、下記の値との比率だけが効く。
_UNITS_PER_EM = 1000

#: 塗り潰し矩形グリフの送り幅。em の全幅を使うことで
#: `size` 指定に対してぴったり `size` px 分だけ進む、扱いやすい値にする。
_ADVANCE_WIDTH = _UNITS_PER_EM

#: 縦方向のメトリクス。矩形グリフがベースラインの上下にまたがる程度の値にする。
_ASCENT = 900
_DESCENT = -200

#: cmapに詰める文字範囲。ASCII全体と、テストで使う日本語 (ひらがな・カタカナ・
#: 常用漢字を含むCJK統合漢字・縦書きの約物) を1文字も漏らさず同じグリフへ割り当てる。
#: cmap format 4 は多対一マッピングを1つの連続レンジにまとめて扱う都合上、
#: 全Unicode平面を1レンジにすると生成テーブルが上限 (65535 bytes) を超えて
#: 失敗するため、範囲を分けて指定する。
_CMAP_RANGES: tuple[tuple[int, int], ...] = (
    (0x20, 0x7E),  # ASCII印字可能文字
    (0x2010, 0x2026),  # 一般句読点 (ハイフン・ダッシュ類、三点/二点リーダ)
    (0x2212, 0x2212),  # 数学記号のマイナス符号 (縦書きで回転させる約物として使う)
    (0x3001, 0x301C),  # CJK記号・句読点 (読点/句点、括弧類、波ダッシュ)
    (0x3040, 0x30FF),  # ひらがな + カタカナ (長音符ーを含む)
    (0x4E00, 0x9FFF),  # CJK統合漢字
    (0xFF08, 0xFF5E),  # 半角・全角形 (全角括弧・全角句読点・全角コロン等)
)

_NOTDEF_GLYPH = ".notdef"
_BLOCK_GLYPH = "block"


def _block_glyph() -> object:
    """emのほぼ全域を占める塗り潰し矩形のグリフを作る。

    輪郭を持たない空グリフにすると描画画素がゼロになり、「文字が描かれたこと」
    を確認しているテストが成立しなくなるため、必ず1つの矩形輪郭を持たせる。
    """
    pen = TTGlyphPen(None)
    margin = _UNITS_PER_EM // 10
    top = _ASCENT - margin
    bottom = _DESCENT + margin
    right = _UNITS_PER_EM - margin
    left = margin
    pen.moveTo((left, bottom))
    pen.lineTo((right, bottom))
    pen.lineTo((right, top))
    pen.lineTo((left, top))
    pen.closePath()
    return pen.glyph()


def build_ttf_bytes() -> bytes:
    """最小のTrueTypeフォントをその場で組み立て、バイト列として返す。"""
    builder = FontBuilder(_UNITS_PER_EM, isTTF=True)
    glyph_order = [_NOTDEF_GLYPH, _BLOCK_GLYPH]
    builder.setupGlyphOrder(glyph_order)

    mapping = {
        codepoint: _BLOCK_GLYPH
        for start, end in _CMAP_RANGES
        for codepoint in range(start, end + 1)
    }
    builder.setupCharacterMap(mapping)

    empty_pen = TTGlyphPen(None)
    glyphs = {_NOTDEF_GLYPH: empty_pen.glyph(), _BLOCK_GLYPH: _block_glyph()}
    builder.setupGlyf(glyphs)

    metrics = dict.fromkeys(glyph_order, (_ADVANCE_WIDTH, 0))
    builder.setupHorizontalMetrics(metrics)
    builder.setupHorizontalHeader(ascent=_ASCENT, descent=_DESCENT)
    builder.setupNameTable({"familyName": "AgenticImagegenTestBlock", "styleName": "Regular"})
    builder.setupOS2(
        sTypoAscender=_ASCENT,
        sTypoDescender=_DESCENT,
        usWinAscent=_ASCENT,
        usWinDescent=-_DESCENT,
    )
    builder.setupPost()

    buffer = BytesIO()
    builder.save(buffer)
    return buffer.getvalue()


__all__ = ["build_ttf_bytes"]
