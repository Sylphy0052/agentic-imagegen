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

#: ComfyUIの embedding_identifier ("embedding:") と同じ記法だけを対象にする。
#: 名前は空白・カンマ・括弧・コロンの手前までを1トークンとして取り出す。
#: コロンを区切りに含めるのは `(embedding:name:1.2)` のような重み指定と衝突しないため。
#: `\b` により "someembedding:x" のような語の途中の一致は除外する。
_EMBEDDING_TOKEN_PATTERN: Final = re.compile(r"\bembedding:([^\s,()<>:]+)")


def extract_embedding_names(*texts: str) -> tuple[str, ...]:
    """promptの文字列群から参照されているembedding名を重複なく取り出す。

    複数のtextsにまたがって出現しても、初出順を保ったまま重複を除いて返す。
    該当が無ければ空タプル。
    """
    seen: dict[str, None] = {}
    for text in texts:
        for match in _EMBEDDING_TOKEN_PATTERN.finditer(text):
            name = match.group(1)
            if name:
                seen[name] = None
    return tuple(seen)


__all__ = ["extract_embedding_names"]
