"""promptに書かれた `embedding:<name>` 記法の抽出。

Textual Inversion embeddingは、ComfyUIのCLIPTextEncodeが `embedding:<name>` という
プレフィックス付きトークンを見つけたときだけ解決を試みる (comfy/sd1_clip.py の
SDTokenizer.tokenize_with_weights)。プレフィックスの無い素の単語 (`easynegative` など)
は普通のテキストとして扱われ、embeddingとしては一切解決されない。
本モジュールはComfyUIと同じ構文だけを対象に、実在チェック用の名前を取り出す。

ComfyUI自身は未配置のembeddingを見つけても例外を出さず、警告ログを残して
黙って無視するだけ (生成そのものは成功し、embeddingが効かないだけになる)。
それではユーザーが気づけないため、生成前にここで検出する
(実在チェックは services.generation が担う。ここでは名前の抽出のみ)。
"""

from __future__ import annotations

import re
from typing import Final

#: 名前は空白・カンマ・括弧・コロンの手前までを1トークンとして取り出す。
#: コロンを区切りに含めるのは `(embedding:name:1.2)` のような重み指定と衝突しないため。
_NAME: Final = r"([^\s,()<>:\[\]]+)"

#: ComfyUIが実際に解決する位置だけを対象にする。
#: ComfyUIは `(?<=\s)embedding:` でテキストを分割し、分割後の各wordが
#: `embedding:` で始まるかを見る (comfy/sd1_clip.py の SDTokenizer.tokenize_with_weights)。
#: したがって解決されるのは「テキストの先頭」か「空白の直後」に限られる。
#: 括弧の直後を許すのは、重み指定が先に剥がされてからトークン化されるため。
_EMBEDDING_TOKEN_PATTERN: Final = re.compile(r"(?:^|(?<=[\s(\[]))embedding:" + _NAME)

#: 同じ `embedding:` でも、区切り以外の文字の直後にあるものはComfyUIが解決しない。
#: 例: `1girl,embedding:easynegative` はカンマの直後で空白が無いため、
#: ComfyUIにとっては `1girl,embedding:easynegative` という1つの語であり、
#: embeddingとして扱われない (警告すら出ずに素のテキストとして扱われる)。
_UNRESOLVED_TOKEN_PATTERN: Final = re.compile(r"(?<=[^\s(\[])embedding:" + _NAME)

#: ComfyUIの `GET /embeddings` は拡張子を落とした名前を返す (server.py の
#: `os.path.splitext(a)[0]`) 一方、`embedding:easynegative.safetensors` のように
#: 拡張子付きで書いても load_embed はファイルを見つけて解決する。
#: 突き合わせる前にここで揃える。
_EMBEDDING_EXTENSIONS: Final = (".safetensors", ".pt", ".bin")


def strip_embedding_extension(name: str) -> str:
    """`/embeddings` が返す名前と突き合わせるために拡張子を落とす。"""
    lowered = name.lower()
    for extension in _EMBEDDING_EXTENSIONS:
        if lowered.endswith(extension):
            return name[: -len(extension)]
    return name


def _collect(pattern: re.Pattern[str], texts: tuple[str, ...]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for text in texts:
        for match in pattern.finditer(text):
            name = match.group(1)
            if name:
                seen[name] = None
    return tuple(seen)


def extract_embedding_names(*texts: str) -> tuple[str, ...]:
    """promptの文字列群から、ComfyUIが解決するembedding名を重複なく取り出す。

    複数のtextsにまたがって出現しても、初出順を保ったまま重複を除いて返す。
    該当が無ければ空タプル。
    """
    return _collect(_EMBEDDING_TOKEN_PATTERN, texts)


def extract_unresolvable_embedding_refs(*texts: str) -> tuple[str, ...]:
    """書いてはあるがComfyUIが解決しない `embedding:` 参照を取り出す。

    区切り文字の直後に空白が無い書き方 (`1girl,embedding:easynegative` など) は、
    ComfyUIにとってはただの1語であり、embeddingとして解決されない。
    未配置のときと違って警告すら出ないため、生成前に気づけるようにする。
    """
    return _collect(_UNRESOLVED_TOKEN_PATTERN, texts)


__all__ = [
    "extract_embedding_names",
    "extract_unresolvable_embedding_refs",
    "strip_embedding_extension",
]
