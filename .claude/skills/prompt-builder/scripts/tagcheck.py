#!/usr/bin/env python3
"""Danbooruのタグが実在するかをまとめて確認する。

使い方:

    python3 .claude/skills/prompt-builder/scripts/tagcheck.py 1girl solo oversized_clothes
    python3 .claude/skills/prompt-builder/scripts/tagcheck.py --prompt "1girl, solo, oversized"

`--prompt` はカンマ区切りのプロンプトをそのまま渡す。スペースはアンダースコアへ
変換してから問い合わせる (Danbooruのタグ名はアンダースコア表記で登録されている)。

post_count の判定基準と、0でも残す例外 (品質ラベル / 学習時点にのみ存在したタグ) は
docs/prompting-guide.md の「タグの実在を確認する」を一次情報とする。
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

API = "https://danbooru.donmai.us/tags.json"
ALLOWED_SCHEME = "https"
ALLOWED_HOST = "danbooru.donmai.us"
USER_AGENT = "agentic-imagegen-prompt-builder"
TIMEOUT_SECONDS = 20
INTERVAL_SECONDS = 0.2

# これを下回る件数は学習への寄与が小さく、置換を検討する。
WEAK_COUNT = 1000

# Danbooruタグではないが学習時のラベルとして効くもの。post_countでは判定しない。
QUALITY_LABELS = frozenset(
    {
        "masterpiece",
        "best_quality",
        "high_quality",
        "normal_quality",
        "low_quality",
        "worst_quality",
        "high_score",
        "great_score",
        "score_1",
        "score_2",
        "score_3",
        "score_7",
        "score_8",
        "score_9",
    }
)

# 現在のDanbooruでは整理されているが、古い世代のcheckpointの学習データには存在したもの。
LEGACY_TAGS = frozenset({"bangs"})

MISSING = "存在しない (置換または削除)"
WEAK = "件数が少なく効きにくい (置換を検討)"

# 到達失敗・想定外レスポンスをまとめて「確認できなかった」に倒すための例外。
# 1件の失敗で残りのタグの確認まで落とさない。
FETCH_ERRORS = (
    urllib.error.URLError,
    TimeoutError,
    OSError,
    json.JSONDecodeError,
    ValueError,
    TypeError,
    KeyError,
    IndexError,
)


def normalize(tag: str) -> str:
    """問い合わせ用のタグ名へ揃える。

    プロンプトをシェルへ貼ると `1girl, solo` のようにカンマが各トークンへ残る。
    そのまま問い合わせると実在するタグを「存在しない」と誤判定するため落とす。
    """
    return tag.strip().strip(",").strip().lower().replace(" ", "_")


def build_url(tag: str) -> str:
    """問い合わせ先URLを組み立て、想定のスキームとホストであることを確かめる。"""
    query = urllib.parse.urlencode({"search[name]": tag, "limit": 1})
    url = f"{API}?{query}"
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != ALLOWED_SCHEME or parsed.netloc != ALLOWED_HOST:
        raise ValueError(f"想定外の問い合わせ先: {parsed.scheme}://{parsed.netloc}")
    return url


def extract_count(payload: Any) -> int | None:
    """tags.jsonのレスポンスからpost_countを取り出す。

    APIの仕様変更や、JSONではない応答 (メンテナンス中のHTML等) を素通しすると
    呼び出し側が「確認できた」と誤解する。形が違えばここで弾く。
    """
    if not isinstance(payload, list):
        raise TypeError("tags.jsonの応答がリストではない")
    if not payload:
        return None
    head = payload[0]
    if not isinstance(head, dict):
        raise TypeError("tags.jsonの要素がオブジェクトではない")
    count = head.get("post_count")
    if count is None:
        return None
    return int(count)


def fetch_count(tag: str) -> int | None:
    """post_countを返す。タグが登録されていなければNone。"""
    # S310を抑制する根拠は build_url でのスキーム / ホスト検証にある。
    # 接続先はハードコードした定数であり、タグはクエリ値としてのみ埋め込む。
    request = urllib.request.Request(build_url(tag), headers={"User-Agent": USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
        payload = json.load(response)
    return extract_count(payload)


def verdict(tag: str, count: int | None) -> str:
    if tag in QUALITY_LABELS:
        return "品質ラベル (post_countでは判定しない)"
    if tag in LEGACY_TAGS:
        return "学習時点にのみ存在 (古い世代のモデルでは残す)"
    if not count:
        return MISSING
    if count < WEAK_COUNT:
        return WEAK
    return "実在する"


def parse_tags(parser: argparse.ArgumentParser, args: argparse.Namespace) -> list[str]:
    if args.prompt and args.tags:
        parser.error("--prompt と位置引数は同時に指定できない")
    raw = args.prompt.split(",") if args.prompt else args.tags
    tags = [normalize(tag) for tag in raw]
    return [tag for tag in tags if tag]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Danbooruのタグの実在をまとめて確認する")
    parser.add_argument("tags", nargs="*", help="確認するタグ (スペース区切り)")
    parser.add_argument("--prompt", help="カンマ区切りのプロンプトをそのまま渡す")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    tags = parse_tags(parser, args)
    if not tags:
        parser.print_usage()
        return 2

    needs_action = 0
    for index, tag in enumerate(tags):
        if index:
            time.sleep(INTERVAL_SECONDS)
        try:
            count = fetch_count(tag)
        except FETCH_ERRORS as error:
            print(f"{tag}\t-\t確認できなかった ({type(error).__name__})")
            needs_action += 1
            continue
        shown = "-" if count is None else f"{count:,}"
        note = verdict(tag, count)
        if note in (MISSING, WEAK):
            needs_action += 1
        print(f"{tag}\t{shown}\t{note}")

    print(f"\n{len(tags)}件を確認、うち要対応 {needs_action}件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
