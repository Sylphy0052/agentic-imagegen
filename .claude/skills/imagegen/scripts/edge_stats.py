#!/usr/bin/env python3
"""画像の高周波比を出す。条件を振って比べるときの破綻検出に使う。

使い方:

    python3 .claude/skills/imagegen/scripts/edge_stats.py outputs/2026-08-15/*/
    python3 .claude/skills/imagegen/scripts/edge_stats.py outputs/.../image_0001.png --json

定義: グレースケール化 -> `ImageFilter.FIND_EDGES` -> しきい値 (既定24) を超えた画素の割合。
値はしきい値と画像サイズで動くため、**同じ条件で撮った1回の比較の中でだけ**見比べる。
別の日の値や別のしきい値の値と直接比べない。

見るのは破綻の有無だけで、絵の良し悪しの順位付けには使わない。
アニメ調のSD1.5系を512x768 -> hires x2.0で出した場合の実測は0.04-0.11で、
0.2を超えるものはVAE不整合などで絵が壊れている
(判断の基準は docs/prompting-guide.md の「既定のcheckpointを決める」を一次情報とする)。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageFilter, UnidentifiedImageError

#: エッジとみなす強度のしきい値 (0-255)。変えると値が動くため既定を固定する。
DEFAULT_THRESHOLD = 24
#: ディレクトリを渡されたときに拾う拡張子。
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def edge_ratio(path: Path, threshold: int = DEFAULT_THRESHOLD) -> float:
    """エッジ強度がしきい値を超えた画素の割合を返す。"""
    with Image.open(path) as image:
        edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
    # FIND_EDGES は最外周を常に強いエッジとして返す。含めると小さい画像ほど
    # 底上げされ、解像度の違う比較が成り立たなくなるため1px落とす。
    edges = edges.crop((1, 1, max(edges.width - 1, 1), max(edges.height - 1, 1)))
    # モード "L" なので1画素1バイト。getdata はPillow 14で消えるため使わない。
    values = edges.tobytes()
    if not values:
        return 0.0
    return sum(1 for value in values if value > threshold) / len(values)


def collect_images(paths: list[Path]) -> list[Path]:
    """ファイルはそのまま、ディレクトリは直下の画像を名前順で拾う。"""
    collected: list[Path] = []
    for path in paths:
        if path.is_dir():
            collected.extend(
                sorted(child for child in path.iterdir() if child.suffix.lower() in IMAGE_SUFFIXES)
            )
        else:
            collected.append(path)
    return collected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="画像の高周波比を出す (破綻検出用)")
    parser.add_argument("paths", nargs="+", help="画像ファイルか、画像を含むディレクトリ")
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD,
        help=f"エッジとみなす強度 (既定: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="機械可読な形で出す")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    targets = collect_images([Path(value) for value in args.paths])

    results: list[dict[str, object]] = []
    failed = False
    for path in targets:
        try:
            ratio = edge_ratio(path, args.threshold)
        except (FileNotFoundError, IsADirectoryError, OSError, UnidentifiedImageError) as error:
            print(f"読めない: {path} ({error})", file=sys.stderr)
            failed = True
            continue
        results.append({"path": str(path), "edge_ratio": ratio, "threshold": args.threshold})

    if args.as_json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for result in results:
            print(f"{result['edge_ratio']:.3f}  {result['path']}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
