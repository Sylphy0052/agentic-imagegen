"""promptからの `embedding:<name>` 抽出のテスト。"""

from __future__ import annotations

from agentic_imagegen.domain.embeddings import extract_embedding_names


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
