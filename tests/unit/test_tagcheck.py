"""prompt-builder skillのtagcheck.pyを検証する。

このスクリプトは `src/` の外 (`.claude/skills/`) にあるためパッケージとして
importできない。ファイルパスから直接ロードする。

実ネットワークへは接続しない。`urlopen` を差し替えて応答を組み立てる。
"""

from __future__ import annotations

import importlib.util
import io
import json
import types
import urllib.error
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / ".claude" / "skills" / "prompt-builder" / "scripts" / "tagcheck.py"


def _load_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("tagcheck", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tagcheck() -> types.ModuleType:
    return _load_module()


class _FakeResponse(io.BytesIO):
    """`with urllib.request.urlopen(...) as response` を満たす最小の応答。"""

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _respond(body: str) -> Any:
    def _urlopen(*_: object, **__: object) -> _FakeResponse:
        return _FakeResponse(body.encode("utf-8"))

    return _urlopen


def _raise(error: Exception) -> Any:
    def _urlopen(*_: object, **__: object) -> _FakeResponse:
        raise error

    return _urlopen


# --- normalize -----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1girl", "1girl"),
        ("  solo  ", "solo"),
        ("Hair Over One Eye", "hair_over_one_eye"),
        # プロンプトをシェルへ貼るとカンマが各トークンへ残る。
        # 落とさないと実在タグを「存在しない」と誤判定する。
        ("1girl,", "1girl"),
        ("solo, ", "solo"),
        (",night", "night"),
        ("oversized clothes,", "oversized_clothes"),
    ],
)
def test_normalize(tagcheck: types.ModuleType, raw: str, expected: str) -> None:
    assert tagcheck.normalize(raw) == expected


# --- strip_weight --------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # 重みを付けたまま問い合わせると実在するタグでも「存在しない」と誤報する。
        ("(1girl:1.3)", "1girl"),
        ("(masterpiece:1.2)", "masterpiece"),
        ("[night:0.8]", "night"),
        ("(solo)", "solo"),
        ("((solo))", "solo"),
        ("(hair over one eye:1.1)", "hair over one eye"),
        # 括弧が無ければ触らない。Danbooruに実在する顔文字タグを壊さないため。
        (":3", ":3"),
        (":d", ":d"),
        ("1girl", "1girl"),
        ("^_^", "^_^"),
    ],
)
def test_strip_weight(tagcheck: types.ModuleType, raw: str, expected: str) -> None:
    assert tagcheck.strip_weight(raw) == expected


def test_normalize_drops_weight(tagcheck: types.ModuleType) -> None:
    assert tagcheck.normalize("(Hair Over One Eye:1.3),") == "hair_over_one_eye"


def test_normalize_keeps_emoticon_tag(tagcheck: types.ModuleType) -> None:
    assert tagcheck.normalize(":3,") == ":3"


# --- verdict -------------------------------------------------------------


def test_verdict_existing(tagcheck: types.ModuleType) -> None:
    assert tagcheck.verdict("1girl", 8_276_580) == "実在する"


def test_verdict_weak(tagcheck: types.ModuleType) -> None:
    assert tagcheck.verdict("extra_digits", 525) == tagcheck.WEAK


def test_verdict_boundary_is_not_weak(tagcheck: types.ModuleType) -> None:
    assert tagcheck.verdict("borderline", tagcheck.WEAK_COUNT) == "実在する"


@pytest.mark.parametrize("count", [None, 0])
def test_verdict_missing(tagcheck: types.ModuleType, count: int | None) -> None:
    assert tagcheck.verdict("athletic", count) == tagcheck.MISSING


def test_verdict_quality_label_ignores_count(tagcheck: types.ModuleType) -> None:
    assert "品質ラベル" in tagcheck.verdict("masterpiece", 0)


@pytest.mark.parametrize("score", [f"score_{n}" for n in range(1, 10)])
def test_verdict_covers_all_aesthetic_scores(tagcheck: types.ModuleType, score: str) -> None:
    """references/tag-replacements.md は score_1 から score_9 を品質ラベルと書いている。"""
    assert "品質ラベル" in tagcheck.verdict(score, 0)


def test_verdict_legacy_tag_ignores_count(tagcheck: types.ModuleType) -> None:
    assert "学習時点" in tagcheck.verdict("bangs", 0)


# --- parse_tags ----------------------------------------------------------


def _parse(tagcheck: types.ModuleType, argv: list[str]) -> list[str]:
    parser = tagcheck.build_parser()
    return tagcheck.parse_tags(parser, parser.parse_args(argv))


def test_parse_tags_from_prompt(tagcheck: types.ModuleType) -> None:
    assert _parse(tagcheck, ["--prompt", "1girl, solo, oversized"]) == [
        "1girl",
        "solo",
        "oversized",
    ]


def test_parse_tags_from_positional(tagcheck: types.ModuleType) -> None:
    assert _parse(tagcheck, ["1girl", "solo"]) == ["1girl", "solo"]


def test_parse_tags_positional_with_commas(tagcheck: types.ModuleType) -> None:
    assert _parse(tagcheck, ["1girl,", "solo,", "night"]) == ["1girl", "solo", "night"]


def test_parse_tags_rejects_both_sources(tagcheck: types.ModuleType) -> None:
    """位置引数を黙って捨てると、確認したつもりの語が確認されないまま残る。"""
    with pytest.raises(SystemExit):
        _parse(tagcheck, ["1girl", "--prompt", "solo"])


def test_parse_tags_drops_empty(tagcheck: types.ModuleType) -> None:
    assert _parse(tagcheck, ["--prompt", "1girl, , solo,"]) == ["1girl", "solo"]


# --- build_url / extract_count -------------------------------------------


def test_build_url_targets_danbooru(tagcheck: types.ModuleType) -> None:
    url = tagcheck.build_url("hair_over_one_eye")
    assert url.startswith("https://danbooru.donmai.us/tags.json?")
    assert "hair_over_one_eye" in url


def test_build_url_rejects_foreign_host(
    tagcheck: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tagcheck, "API", "https://example.com/tags.json")
    with pytest.raises(ValueError, match="想定外の問い合わせ先"):
        tagcheck.build_url("1girl")


def test_extract_count_reads_head(tagcheck: types.ModuleType) -> None:
    assert tagcheck.extract_count([{"post_count": 42}]) == 42


def test_extract_count_empty_is_none(tagcheck: types.ModuleType) -> None:
    assert tagcheck.extract_count([]) is None


@pytest.mark.parametrize("payload", [{"post_count": 42}, ["1girl"], "1girl"])
def test_extract_count_rejects_unexpected_shape(tagcheck: types.ModuleType, payload: Any) -> None:
    with pytest.raises(TypeError):
        tagcheck.extract_count(payload)


# --- main ----------------------------------------------------------------


def test_main_reports_counts(
    tagcheck: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        tagcheck.urllib.request, "urlopen", _respond(json.dumps([{"post_count": 8}]))
    )
    monkeypatch.setattr(tagcheck.time, "sleep", lambda _: None)
    monkeypatch.setattr("sys.argv", ["tagcheck.py", "--prompt", "1girl, solo"])

    assert tagcheck.main() == 0
    out = capsys.readouterr().out
    assert "1girl\t8\t" in out
    assert "2件を確認、うち要対応 2件" in out


def test_main_without_tags_returns_usage_code(
    tagcheck: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.argv", ["tagcheck.py"])
    assert tagcheck.main() == 2
    assert "usage" in capsys.readouterr().out


def test_main_continues_after_network_error(
    tagcheck: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        tagcheck.urllib.request, "urlopen", _raise(urllib.error.URLError("unreachable"))
    )
    monkeypatch.setattr(tagcheck.time, "sleep", lambda _: None)
    monkeypatch.setattr("sys.argv", ["tagcheck.py", "--prompt", "1girl, solo"])

    assert tagcheck.main() == 0
    out = capsys.readouterr().out
    assert out.count("確認できなかった") == 2
    assert "2件を確認、うち要対応 2件" in out


def test_main_continues_after_non_json_response(
    tagcheck: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """メンテナンス中のHTML等を返されても、残りのタグの確認まで落とさない。"""
    monkeypatch.setattr(tagcheck.urllib.request, "urlopen", _respond("<html>maintenance</html>"))
    monkeypatch.setattr(tagcheck.time, "sleep", lambda _: None)
    monkeypatch.setattr("sys.argv", ["tagcheck.py", "--prompt", "1girl, solo"])

    assert tagcheck.main() == 0
    out = capsys.readouterr().out
    assert out.count("確認できなかった") == 2


def test_main_continues_after_unexpected_shape(
    tagcheck: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(tagcheck.urllib.request, "urlopen", _respond(json.dumps({"post_count": 8})))
    monkeypatch.setattr(tagcheck.time, "sleep", lambda _: None)
    monkeypatch.setattr("sys.argv", ["tagcheck.py", "--prompt", "1girl"])

    assert tagcheck.main() == 0
    assert "確認できなかった" in capsys.readouterr().out


def test_main_error_output_hides_stack_detail(
    tagcheck: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """エラー表示に絶対パスや内部メッセージを載せない。"""
    monkeypatch.setattr(
        tagcheck.urllib.request,
        "urlopen",
        _raise(urllib.error.URLError(f"failed at {SCRIPT_PATH}")),
    )
    monkeypatch.setattr("sys.argv", ["tagcheck.py", "1girl"])

    assert tagcheck.main() == 0
    out = capsys.readouterr().out
    assert str(SCRIPT_PATH) not in out
    assert "URLError" in out
