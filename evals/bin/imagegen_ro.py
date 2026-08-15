#!/usr/bin/env python3
"""evalから叩くimagegen CLIの読み取り専用ラッパー。

evalでは在庫や台帳を引く必要がある一方、生成は走らせたくない。
`Bash(uv run imagegen:*)` のような許可では `generate` まで通ってしまい、
`Bash(uv run imagegen character:*)` のように絞ると、呼び方が少し違うだけで
(`uv run --offline --no-sync imagegen character list` など) 前方一致から外れる。

そこで許可はこのスクリプト1つに与え、通すサブコマンドをここで決める。

    python3 evals/bin/imagegen_ro.py character show aoi
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
EXECUTABLE = REPOSITORY / ".venv" / "bin" / "imagegen"

#: 状態を変えないサブコマンドだけを通す。generate / batch / compose は通さない。
READ_ONLY = frozenset({"catalog", "validate", "history", "character", "health"})


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in READ_ONLY:
        allowed = ", ".join(sorted(READ_ONLY))
        print(f"このラッパーが通すのは {allowed} だけです", file=sys.stderr)
        return 2
    if not EXECUTABLE.is_file():
        print(f"{EXECUTABLE} がありません。uv sync を先に実行してください", file=sys.stderr)
        return 2
    # uvを経由しない。`uv run` はサンドボックス下で同期に入って止まることがある。
    completed = subprocess.run(  # noqa: S603
        [str(EXECUTABLE), *argv], cwd=REPOSITORY, env=os.environ.copy(), check=False
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
