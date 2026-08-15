#!/usr/bin/env python3
"""Danbooruのタグが実在するかをまとめて確認する。

使い方:

    python3 .claude/skills/prompt-builder/scripts/tagcheck.py 1girl solo oversized_clothes
    python3 .claude/skills/prompt-builder/scripts/tagcheck.py --prompt "1girl, solo, oversized"

`--prompt` はカンマ区切りのプロンプトをそのまま渡す。スペースはアンダースコアへ
変換してから問い合わせる (Danbooruのタグ名はアンダースコア表記で登録されている)。

一度引いた結果は `.cache/tagcheck.json` へ30日残す。同じタグは何度も出てくるうえ、
post_countは日単位でしか動かないため、問い合わせ直す意味が薄い。取り直すときは
`--refresh`、キャッシュを触らせたくないときは `--no-cache`。

post_count の判定基準と、0でも残す例外 (品質ラベル / 学習時点にのみ存在したタグ) は
docs/prompting-guide.md の「タグの実在を確認する」を一次情報とする。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

API = "https://danbooru.donmai.us/tags.json"
ALLOWED_SCHEME = "https"
ALLOWED_HOST = "danbooru.donmai.us"
USER_AGENT = "agentic-imagegen-prompt-builder"
TIMEOUT_SECONDS = 20
INTERVAL_SECONDS = 0.2

# これを下回る件数は学習への寄与が小さく、置換を検討する。
WEAK_COUNT = 1000

# 一度引いた結果の置き場。リポジトリ直下の .cache/ (git管理外)。
CACHE_ENV = "IMAGEGEN_TAGCHECK_CACHE"
DEFAULT_CACHE_PATH = Path(__file__).resolve().parents[4] / ".cache" / "tagcheck.json"

# 形式を変えたら上げる。読み込み側は一致しないキャッシュを捨てる。
CACHE_VERSION = 1

# 有効期限。実在するか否かはこの期間では動かない。
CACHE_TTL_SECONDS = 30 * 24 * 60 * 60

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
    }
    # Anima系のaesthetic score。references/tag-replacements.md の記述
    # (score_1 から score_9) と食い違わないよう連番で生成する。
    | {f"score_{n}" for n in range(1, 10)}
)

# 現在のDanbooruでは整理されているが、古い世代のcheckpointの学習データには存在したもの。
LEGACY_TAGS = frozenset({"bangs"})

# A1111系の重み付け記法。`(tag:1.3)` / `(tag)` / `[tag]` を1段ずつ剥がす。
_WEIGHTED = re.compile(r"[(\[]\s*(?P<tag>.+?)\s*(?::\s*[0-9]*\.?[0-9]+\s*)?[)\]]")

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


def strip_weight(tag: str) -> str:
    """`(tag:1.3)` のような重み付け記法からタグ名だけを取り出す。

    重みを付けたまま問い合わせると実在するタグでも「存在しない」と誤報する。
    外すのは括弧で囲まれている場合だけにする。`:3` や `:d` はDanbooruに実在する
    顔文字タグであり、コロン以降を無条件に落とすとこれらを壊す。
    """
    while True:
        stripped = _WEIGHTED.fullmatch(tag.strip())
        if stripped is None:
            return tag.strip()
        tag = stripped.group("tag")


def normalize(tag: str) -> str:
    """問い合わせ用のタグ名へ揃える。

    プロンプトをシェルへ貼ると `1girl, solo` のようにカンマが各トークンへ残る。
    そのまま問い合わせると実在するタグを「存在しない」と誤判定するため落とす。
    """
    return strip_weight(tag.strip().strip(",")).lower().replace(" ", "_")


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


def cache_path() -> Path:
    override = os.environ.get(CACHE_ENV, "").strip()
    return Path(override) if override else DEFAULT_CACHE_PATH


def load_cache(path: Path) -> dict[str, Any]:
    """キャッシュを読む。読めない・形が違う場合は空として扱う。

    キャッシュは補助であり、壊れていたら捨てて取り直せばよい。
    ここで失敗させるとタグの確認そのものができなくなる。
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict) or raw.get("version") != CACHE_VERSION:
        return {}
    entries = raw.get("entries")
    return entries if isinstance(entries, dict) else {}


def cached_count(entries: dict[str, Any], tag: str, now: float) -> tuple[bool, int | None]:
    """(キャッシュに当たったか, post_count) を返す。"""
    entry = entries.get(tag)
    if not isinstance(entry, dict):
        return False, None
    fetched_at = entry.get("fetched_at")
    if not isinstance(fetched_at, int | float) or now - fetched_at > CACHE_TTL_SECONDS:
        return False, None
    count = entry.get("count")
    if count is not None and not isinstance(count, int):
        return False, None
    return True, count


def save_cache(path: Path, entries: dict[str, Any]) -> None:
    """書けなければ黙って諦める。確認結果はキャッシュの成否に依らない。"""
    payload = {"version": CACHE_VERSION, "entries": entries}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # 途中で落ちた書きかけを読ませないよう、別名で書いてから差し替える。
        temporary = path.with_name(f"{path.name}.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        return


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
    parser.add_argument("--no-cache", action="store_true", help="キャッシュを読み書きしない")
    parser.add_argument("--refresh", action="store_true", help="取り直してキャッシュを上書きする")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    tags = parse_tags(parser, args)
    if not tags:
        parser.print_usage()
        return 2

    path = cache_path()
    use_cache = not args.no_cache
    entries = load_cache(path) if use_cache else {}
    now = time.time()

    needs_action = 0
    hits = 0
    fetched = 0
    stored = False
    for tag in tags:
        hit, count = (
            cached_count(entries, tag, now) if use_cache and not args.refresh else (False, None)
        )
        if hit:
            hits += 1
        else:
            # ウェイトは問い合わせの間隔。キャッシュから答えた分では待たない。
            if fetched:
                time.sleep(INTERVAL_SECONDS)
            fetched += 1
            try:
                count = fetch_count(tag)
            except FETCH_ERRORS as error:
                # 到達できなかった結果を覚えると、復旧してもしばらく確認できない。
                print(f"{tag}\t-\t確認できなかった ({type(error).__name__})")
                needs_action += 1
                continue
            if use_cache:
                entries[tag] = {"count": count, "fetched_at": now}
                stored = True

        shown = "-" if count is None else f"{count:,}"
        note = verdict(tag, count)
        if note in (MISSING, WEAK):
            needs_action += 1
        print(f"{tag}\t{shown}\t{note}")

    if stored:
        save_cache(path, entries)

    summary = f"\n{len(tags)}件を確認、うち要対応 {needs_action}件"
    if hits:
        summary += f" (キャッシュ {hits}件)"
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
