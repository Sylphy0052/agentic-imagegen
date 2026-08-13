"""promptからの `embedding:<name>` 抽出のテスト。"""

from __future__ import annotations

import pytest

from agentic_imagegen.domain.embeddings import (
    extract_embedding_names,
    extract_unresolvable_embedding_refs,
    strip_embedding_extension,
)


def test_extracts_single_embedding() -> None:
    names = extract_embedding_names("embedding:easynegative, worst quality, low quality")

    assert names == ("easynegative",)


def test_extracts_multiple_embeddings_preserving_order() -> None:
    names = extract_embedding_names("embedding:easynegative, embedding:badhandv4, worst quality")

    assert names == ("easynegative", "badhandv4")


def test_deduplicates_across_multiple_texts() -> None:
    names = extract_embedding_names(
        "1girl, embedding:easynegative",
        "embedding:easynegative, embedding:badhandv4",
    )

    assert names == ("easynegative", "badhandv4")


def test_bare_word_without_prefix_is_not_extracted() -> None:
    """ComfyUI自身も `embedding:` プレフィックスの無い語はembeddingとして解決しない。"""
    names = extract_embedding_names("easynegative, worst quality")

    assert names == ()


def test_does_not_match_prefix_inside_another_word() -> None:
    """語の途中に \"embedding:\" が現れても、先頭ではないので拾わない。"""
    names = extract_embedding_names("someembedding:test")

    assert names == ()


def test_handles_weighted_syntax() -> None:
    """`(embedding:name:1.2)` のような重み付け表記では、コロンの手前までを名前とする。"""
    names = extract_embedding_names("(embedding:easynegative:1.2), worst quality")

    assert names == ("easynegative",)


def test_embedding_at_start_of_string() -> None:
    names = extract_embedding_names("embedding:easynegative")

    assert names == ("easynegative",)


def test_no_texts_returns_empty() -> None:
    assert extract_embedding_names() == ()


def test_empty_strings_return_empty() -> None:
    assert extract_embedding_names("", "") == ()


def test_comma_without_space_is_not_resolved() -> None:
    """ComfyUIは空白の直後の `embedding:` しか分割しない。

    `1girl,embedding:easynegative` はComfyUIにとって1つの語であり、
    `embedding:` で始まらないためembeddingとして解決されない
    (comfy/sd1_clip.py は空白の直後の `embedding:` でしか分割しない)。
    """
    assert extract_embedding_names("1girl,embedding:easynegative") == ()


def test_comma_without_space_is_reported_as_unresolvable() -> None:
    """解決されない書き方は、警告すら出ないので別に拾って知らせる。"""
    names = extract_unresolvable_embedding_refs("1girl,embedding:easynegative")

    assert names == ("easynegative",)


@pytest.mark.parametrize(
    "text",
    [
        "embedding:easynegative",
        "1girl, embedding:easynegative",
        "(embedding:easynegative:1.2)",
    ],
)
def test_resolvable_positions_are_not_reported_as_unresolvable(text: str) -> None:
    assert extract_unresolvable_embedding_refs(text) == ()


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("easynegative.safetensors", "easynegative"),
        ("badhandv4.pt", "badhandv4"),
        ("foo.bin", "foo"),
        ("easynegative", "easynegative"),
        ("ng_deepnegative_v1_75t", "ng_deepnegative_v1_75t"),
    ],
)
def test_strip_embedding_extension(name: str, expected: str) -> None:
    """`GET /embeddings` は拡張子を落とした名前を返すため、比較前に揃える。"""
    assert strip_embedding_extension(name) == expected
